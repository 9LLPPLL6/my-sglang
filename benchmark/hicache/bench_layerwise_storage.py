"""Storage-side microbenchmark for the layerwise (L2/L3 fused) HiCache tier.

Answers the two questions the layerwise plan gates on before any end-to-end
number is meaningful:

1. What does this filesystem actually deliver with Direct I/O at a given queue
   depth?  That is the ceiling; nothing above it is achievable.
2. How much of that ceiling survives splitting a page into per-layer-group
   range reads?  Smaller extents mean more IOPS for the same bytes, so this is
   where a badly chosen group size shows up.

It also reports the group-0 latency, which is the part of an L3 hit that no
computation can hide and therefore lands directly in TTFT.

    python3 benchmark/hicache/bench_layerwise_storage.py \\
        --root /mnt/parallel-fs/sglang-hicache --pages 256 --group-size 4
"""

import argparse
import os
import shutil
import statistics
import time

import torch

from sglang.srt.mem_cache.layerwise_storage.aio_engine import (
    AlignedBuffer,
    LinuxAioContext,
    probe_alignment,
)
from sglang.srt.mem_cache.layerwise_storage.file_backend import LayerwiseFileBackend
from sglang.srt.mem_cache.layerwise_storage.page_format import (
    PageIdentity,
    build_page_layout,
    model_fingerprint,
)
from sglang.srt.mem_cache.layerwise_storage.page_writer import PageFileWriter
from sglang.srt.mem_cache.layerwise_storage.plan_builder import build_read_plan
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost

_GIB = 1 << 30


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/tmp/sglang-layerwise-bench")
    parser.add_argument("--pages", type=int, default=128)
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--kv-heads", type=int, default=4, help="per-TP-rank KV heads")
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--first-group-layers", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--queue-depth", type=int, default=128)
    parser.add_argument("--read-ahead-groups", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--keep", action="store_true", help="keep the written pages")
    parser.add_argument(
        "--allow-buffered",
        action="store_true",
        help="run without O_DIRECT; results are not a valid storage baseline",
    )
    return parser.parse_args()


def _build_host(args: argparse.Namespace) -> MHATokenToKVPoolHost:
    dtype = getattr(torch, args.dtype)
    host = MHATokenToKVPoolHost.__new__(MHATokenToKVPoolHost)
    host.layout = "page_first_direct"
    host.page_num = args.pages
    host.layer_num = args.layers
    host.page_size = args.page_size
    host.head_num = args.kv_heads
    host.head_dim = args.head_dim
    host.size = args.pages * args.page_size
    host.dtype = dtype
    host.kv_buffer = torch.zeros(
        (2, args.pages, args.layers, args.page_size, args.kv_heads, args.head_dim),
        dtype=dtype,
    )
    return host


def _build_identity(args: argparse.Namespace, host) -> PageIdentity:
    fingerprint = model_fingerprint(
        model_name="layerwise-bench",
        dtype_name=args.dtype,
        layer_num=args.layers,
        page_size=args.page_size,
        local_kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        tp_size=1,
    )
    return PageIdentity(
        fingerprint=fingerprint,
        tp_size=1,
        tp_rank=0,
        dtype_name=args.dtype,
        layer_num=args.layers,
        page_size=args.page_size,
        local_kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        element_size=host.dtype.itemsize,
    )


def _publish_pages(writer: PageFileWriter, *, pages: int, region_nbytes: int):
    keys = []
    payload = os.urandom(region_nbytes)
    for page in range(pages):
        key = f"bench{page:06d}"
        if not writer.write_page_bytes(key, k_bytes=payload, v_bytes=payload):
            raise RuntimeError(f"failed to publish {key}")
        keys.append(key)
    return keys


def _bench_whole_page_baseline(*, writer, keys, layout, queue_depth, iters):
    """Ceiling: read each page's two regions as two large aligned extents."""
    context = LinuxAioContext(queue_depth=queue_depth)
    region_nbytes = layout.v_offset - layout.k_offset
    buffers = [
        AlignedBuffer(region_nbytes, alignment=layout.alignment)
        for _ in range(queue_depth)
    ]
    descriptors = [os.open(writer.page_path(key), _read_flags(writer)) for key in keys]
    try:
        durations = []
        for _ in range(iters):
            requests = []
            for index, fd in enumerate(descriptors):
                for part, offset in (("k", layout.k_offset), ("v", layout.v_offset)):
                    requests.append(
                        (
                            fd,
                            buffers[len(requests) % queue_depth].ptr,
                            region_nbytes,
                            offset,
                            len(requests),
                        )
                    )
            durations.append(_run_to_completion(context, requests))
        total_nbytes = len(descriptors) * 2 * region_nbytes
        return _summarize(durations, total_nbytes)
    finally:
        context.close()
        for fd in descriptors:
            os.close(fd)


def _bench_layerwise(*, backend, host, keys, layout, args):
    """The real path: one transaction, groups submitted in order."""
    host_indices = torch.arange(len(keys) * host.page_size, dtype=torch.int64)
    plan, target = build_read_plan(
        host_pool=host,
        host_indices=host_indices,
        page_keys=keys,
        layout=layout,
        first_group_layers=args.first_group_layers,
        group_size=args.group_size,
    )
    total_nbytes = plan.total_io_nbytes
    durations = []
    group0_latencies = []
    for iteration in range(args.iters):
        handle = backend.begin_read(
            transaction_id=f"bench-{iteration}",
            generation=iteration,
            plan=plan,
            target=target,
        )
        start = time.perf_counter()
        # Group 0 is submitted and awaited alone: it gates admission, so its
        # latency is measured without any read-ahead hiding it.
        backend.submit_group(
            handle=handle, group=plan.groups[0], priority=0, deadline_s=None
        )
        remaining = {0: len(plan.groups[0].extents)}
        while remaining[0] > 0:
            for completion in backend.poll(handle=handle):
                remaining[completion.group_id] -= 1
        group0_latencies.append(time.perf_counter() - start)

        # Steady state keeps read_ahead_groups + 1 groups in flight, which is
        # what the pipeline does; a strictly serial loop would understate it.
        window = args.read_ahead_groups + 1
        next_submit = 1
        outstanding = 0
        while next_submit < len(plan.groups) or outstanding > 0:
            while next_submit < len(plan.groups) and outstanding < window:
                group = plan.groups[next_submit]
                remaining[next_submit] = len(group.extents)
                backend.submit_group(
                    handle=handle, group=group, priority=2, deadline_s=None
                )
                next_submit += 1
                outstanding += 1
            for completion in backend.poll(handle=handle):
                remaining[completion.group_id] -= 1
                if remaining[completion.group_id] == 0:
                    outstanding -= 1
        durations.append(time.perf_counter() - start)
        backend.close(handle=handle)
    return _summarize(durations, total_nbytes), group0_latencies, plan


def _run_to_completion(context: LinuxAioContext, requests) -> float:
    start = time.perf_counter()
    pending = list(requests)
    outstanding = 0
    while pending or outstanding:
        if pending:
            accepted = context.submit_reads(pending)
            outstanding += accepted
            pending = pending[accepted:]
        outstanding -= len(context.poll())
    return time.perf_counter() - start


def _summarize(durations, total_nbytes):
    best = min(durations)
    median = statistics.median(durations)
    return {
        "total_gib": total_nbytes / _GIB,
        "best_gibps": total_nbytes / _GIB / best,
        "median_gibps": total_nbytes / _GIB / median,
        "median_s": median,
    }


def _read_flags(writer: PageFileWriter) -> int:
    return os.O_RDONLY | (os.O_DIRECT if writer.direct_io else 0)


def main() -> None:
    args = _parse_args()
    host = _build_host(args)
    identity = _build_identity(args, host)

    profile = probe_alignment(args.root, require_direct=not args.allow_buffered)
    layout = build_page_layout(identity, alignment=profile.alignment)
    region_nbytes = layout.v_offset - layout.k_offset

    print(f"root              : {args.root}")
    print(
        f"direct io         : {profile.direct_io_available} "
        f"(alignment {profile.alignment} B)"
    )
    print(f"page payload      : {layout.physical_nbytes / (1 << 20):.2f} MiB")
    print(f"layer stride      : {layout.layer_stride} B")
    print(f"pages             : {args.pages}")

    writer = PageFileWriter(
        root=args.root,
        identity=identity,
        alignment_profile=profile,
        require_direct_io=not args.allow_buffered,
    )
    write_start = time.perf_counter()
    keys = _publish_pages(writer, pages=args.pages, region_nbytes=region_nbytes)
    write_s = time.perf_counter() - write_start
    written_gib = args.pages * layout.file_nbytes / _GIB
    print(f"write-through     : {written_gib / write_s:.2f} GiB/s")

    baseline = _bench_whole_page_baseline(
        writer=writer,
        keys=keys,
        layout=layout,
        queue_depth=args.queue_depth,
        iters=args.iters,
    )

    backend = LayerwiseFileBackend(
        root=args.root,
        identity=identity,
        queue_depth=args.queue_depth,
        max_inflight_bytes=1 << 30,
        alignment_profile=profile,
        require_direct_io=not args.allow_buffered,
    )
    try:
        layerwise, group0_latencies, plan = _bench_layerwise(
            backend=backend, host=host, keys=keys, layout=layout, args=args
        )
    finally:
        backend.shutdown()

    extents_per_group = len(plan.groups[0].extents)
    print()
    print(
        f"groups            : {len(plan.groups)} "
        f"(first {args.first_group_layers} layers, then {args.group_size})"
    )
    print(f"extents per group : {extents_per_group}")
    print(
        f"baseline  (whole) : {baseline['median_gibps']:.2f} GiB/s median, "
        f"{baseline['best_gibps']:.2f} GiB/s best"
    )
    print(
        f"layerwise (range) : {layerwise['median_gibps']:.2f} GiB/s median, "
        f"{layerwise['best_gibps']:.2f} GiB/s best"
    )

    ratio = layerwise["median_gibps"] / baseline["median_gibps"]
    gate = "PASS" if ratio >= 0.8 else "FAIL"
    print(f"ratio vs baseline : {ratio * 100:.1f}% ({gate}, gate is 80%)")
    print(
        f"group 0 latency   : {statistics.median(group0_latencies) * 1e3:.3f} ms "
        f"median, {max(group0_latencies) * 1e3:.3f} ms max"
    )
    if not profile.direct_io_available:
        print("WARNING: buffered I/O was used; this is not a storage baseline")

    if not args.keep:
        shutil.rmtree(args.root, ignore_errors=True)


if __name__ == "__main__":
    main()
