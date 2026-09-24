#!/usr/bin/env python3
"""PCIe host-to-device bandwidth, to compare against the GPFS read bandwidth.

An L3 hit moves the KV twice, storage to host and host to device; an L2 hit
only does the second move. Whether the pipeline can hide the extra move depends
on how the two rates compare, and this script supplies the host-to-device one.
The storage side comes from the layerwise backend's own read report.

This measures the copy alone, so it is an upper bound rather than the rate
hicache reaches under load: the real rate can only be inferred from a forward,
and a forward also contains compute.
"""

from __future__ import annotations

import argparse
import time

import torch


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def measure(nbytes: int, chunk_bytes: int, repeats: int, pin: bool) -> dict:
    """Copy `nbytes` host to device, `chunk_bytes` per copy."""
    host = torch.empty(nbytes, dtype=torch.uint8, device="cpu", pin_memory=pin)
    dev = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    chunks = [
        (off, min(chunk_bytes, nbytes - off)) for off in range(0, nbytes, chunk_bytes)
    ]

    spans = []
    for _ in range(repeats + 1):
        torch.cuda.synchronize()
        started = time.perf_counter()
        for offset, size in chunks:
            dev[offset : offset + size].copy_(
                host[offset : offset + size], non_blocking=True
            )
        torch.cuda.synchronize()
        spans.append((time.perf_counter() - started) * 1e3)
    spans = spans[1:]  # Drop the first pass; it carries allocation and warm-up.

    del dev, host
    torch.cuda.empty_cache()
    span_ms = mean(spans)
    return {
        "chunk_mib": chunk_bytes / 2**20,
        "chunks": len(chunks),
        "span_ms": round(span_ms, 2),
        "gibps": round(nbytes / span_ms * 1e3 / 2**30, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hit-tokens", type=int, default=16384)
    parser.add_argument("--kv-bytes-per-token", type=int, default=2 * 80 * 8 * 128 * 1)
    parser.add_argument(
        "--layer-groups",
        type=int,
        default=10,
        help="按 layerwise 流水线的分组数切块，看粒度影不影响带宽",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--storage-gibps",
        type=float,
        default=39.58,
        help="对照用的 L3->L2 读带宽，取自 layerwise 后端的读报告",
    )
    args = parser.parse_args()

    total = args.hit_tokens * args.kv_bytes_per_token
    print(f"一条 {args.hit_tokens}-token 前缀的 KV = {total / 2**30:.2f} GiB")
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    print(f"{'内存':<8}{'块大小MiB':>11}{'块数':>7}{'耗时ms':>10}{'GiB/s':>9}")
    print("-" * 46)
    rows = []
    for pin, label in ((True, "pinned"), (False, "pageable")):
        for groups in (1, args.layer_groups):
            row = measure(total, total // groups, args.repeats, pin)
            rows.append((label, row))
            print(
                f"{label:<8}{row['chunk_mib']:>11.1f}{row['chunks']:>7}"
                f"{row['span_ms']:>10.2f}{row['gibps']:>9.2f}"
            )

    best = max(r["gibps"] for _, r in rows if _ == "pinned")
    print()
    print("=" * 46)
    print(f"L2 -> L1  (PCIe, pinned 上界)   {best:>8.2f} GiB/s")
    print(f"L3 -> L2  (GPFS, 实测读报告)     {args.storage_gibps:>8.2f} GiB/s")
    print(f"                        比值   {args.storage_gibps / best:>8.2f}x")
    print("=" * 46)
    print("注：上面是纯搬运的上界。hicache 在负载下的实际 H2D 速率更低，")
    print("    要从 L2 命中的 forward 反推，而 forward 里还含计算。")


if __name__ == "__main__":
    main()
