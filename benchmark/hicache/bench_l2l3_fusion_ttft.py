"""TTFT of a fused L2/L3 hit against a pure L2 hit, with the tier proven per measurement.

The question: when the suffix gives the model enough compute to hide the
L3->L2 read, how much slower is the layerwise fused path than a pure L2 (host)
hit, and where does the old L3 path land?

Nothing here trusts that a request was served by the tier it was supposed to be
served by. Every measurement carries three independent probes and is discarded
if any of them disagrees:

  tier      meta_info.cached_tokens_details -- what the scheduler says it matched
  disk      /proc/<pid>/io read_bytes over the whole server process tree, sampled
            around the timed request. This counts bytes the kernel really fetched
            from the block layer, so a page-cache hit reads as 0 and cannot be
            mistaken for an NVMe read. It is also the only thing that separates a
            genuine L2 hit from a streamed L3 hit on the fused server, because the
            bridge books its staged prefetch as a host hit either way.
  resident  mincore() over the whole L3 store just before the timed request, so a
            "cold" read is only called cold once the pages are demonstrably gone

Arms:

  l1        prefix left in device HBM                      -> floor
  l2        prefix pushed out of HBM by fillers, no L3      -> the reference
  l2_fused  same, on the fused server, so the L2 and L3 numbers come from one
            binary and one config and differ only in which tier served them
  l3_old    stock `file` backend, wait_complete
  l3_nopipe `layerwise_file`, full_wait   (new backend, pipeline off)
  l3_fused  `layerwise_file`, layerwise   (new backend, pipeline on)

Every measurement gets a fresh prefix. That matters: a reused prefix makes the
warm-up request an L3 *read*, which repopulates the page cache a second before
the timed read and silently subsidises the buffered `file` backend. With a fresh
prefix the warm-up is always a write, so sync+fadvise leaves the store genuinely
cold -- and `resident` proves it did.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import random
import re
import shutil
import statistics
import subprocess
import time
from pathlib import Path

import psutil
import requests

HERE = Path(__file__).resolve().parent
REPO_PY = str(HERE.parents[1] / "python")          # <repo>/python
RESULTS_ROOT = HERE / "results" / "l2l3_fusion"

PAGE = os.sysconf("SC_PAGE_SIZE")
_libc = ctypes.CDLL("libc.so.6", use_errno=True)
# mmap through libc rather than the mmap module: a PROT_READ mapping is not a
# writable Python buffer, so ctypes cannot take its address.
_libc.mmap.restype = ctypes.c_void_p
_libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                       ctypes.c_int, ctypes.c_int, ctypes.c_long]
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                          ctypes.POINTER(ctypes.c_ubyte)]
_PROT_READ, _MAP_SHARED = 0x1, 0x01
_MAP_FAILED = ctypes.c_void_p(-1).value
# mincore reports residency in bit 0; keep only that bit, then count the ones.
_LOW_BIT = bytes(i & 1 for i in range(256))

ARMS = ("l1", "l2", "l2_fused", "l3_old", "l3_nopipe", "l3_fused")
# `l3_nopipe_t<N>` is the same arm at --hicache-storage-io-threads=N. Each N is
# its own server (the flag is a startup argument), and each gets its own store
# so one thread count can never read pages another one left in a cache.
_NOPIPE_THREADS = re.compile(r"^l3_nopipe_t(\d+)$")
SERVER_OF_ARM = {
    "l1": "mem",
    "l2": "mem",
    "l3_old": "l3_old",
    "l3_nopipe": "l3_nopipe",
    "l3_fused": "l3_fused",
    "l2_fused": "l3_fused",
}
STORE_OF_SERVER = {"l3_old": "file", "l3_nopipe": "layerwise", "l3_fused": "layerwise"}


def arm_is_known(arm: str) -> bool:
    return arm in ARMS or _NOPIPE_THREADS.match(arm) is not None


def server_of_arm(arm: str) -> str:
    # One server per thread count, named after the arm it serves.
    return arm if _NOPIPE_THREADS.match(arm) else SERVER_OF_ARM[arm]


def store_of_server(server: str) -> str | None:
    if _NOPIPE_THREADS.match(server):
        return f"layerwise-{server}"
    return STORE_OF_SERVER.get(server)


def io_threads_of_server(server: str, default: int) -> int:
    match = _NOPIPE_THREADS.match(server)
    return int(match.group(1)) if match else default


def is_nopipe(arm: str) -> bool:
    return arm == "l3_nopipe" or _NOPIPE_THREADS.match(arm) is not None


def log(msg: str) -> None:
    print(f"[l2l3] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# probes
# --------------------------------------------------------------------------- #


def tree_read_bytes(pid: int) -> int:
    """Bytes this server really pulled from the block layer, whole process tree.

    /proc/<pid>/io read_bytes excludes anything served from the page cache, which
    is exactly the distinction the L3 arms need to prove."""
    total = 0
    try:
        root = psutil.Process(pid)
        procs = [root] + root.children(recursive=True)
    except psutil.Error:
        return total
    for p in procs:
        try:
            total += p.io_counters().read_bytes
        except (psutil.Error, AttributeError):
            pass
    return total


def resident_bytes(path: str) -> int:
    """Bytes of this file currently held in the page cache, via mincore().

    Returns -1 if residency could not be determined, so a failed probe is never
    silently counted as "cold"."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return 0
    try:
        size = os.fstat(fd).st_size
        if size == 0:
            return 0
        addr = _libc.mmap(None, size, _PROT_READ, _MAP_SHARED, fd, 0)
        if not addr or addr == _MAP_FAILED:
            return -1
        try:
            npages = (size + PAGE - 1) // PAGE
            vec = (ctypes.c_ubyte * npages)()
            if _libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(size), vec) != 0:
                return -1
            return bytes(vec).translate(_LOW_BIT).count(1) * PAGE
        finally:
            _libc.munmap(ctypes.c_void_p(addr), ctypes.c_size_t(size))
    finally:
        os.close(fd)


def store_walk(root: Path):
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            yield os.path.join(dirpath, name)


def store_bytes(root: Path) -> tuple[int, int]:
    total = files = 0
    for path in store_walk(root):
        try:
            total += os.stat(path).st_size
            files += 1
        except OSError:
            pass
    return total, files


def store_resident(root: Path) -> tuple[int, int]:
    """(bytes of the store still in page cache, files whose probe failed)."""
    total = failed = 0
    for path in store_walk(root):
        r = resident_bytes(path)
        if r < 0:
            failed += 1
        else:
            total += r
    return total, failed


def fadvise_all(root: Path) -> int:
    """Drop clean pages of every store file. sync() first: DONTNEED skips dirty
    pages, and write-through leaves the freshly written prefix dirty."""
    os.sync()
    dropped = 0
    for path in store_walk(root):
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            dropped += 1
        except OSError:
            pass
        finally:
            os.close(fd)
    return dropped


def fs_type(path: Path) -> str:
    try:
        return subprocess.run(["stat", "-f", "-c", "%T", str(path)],
                              capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def churn(path: Path, gib: float) -> float:
    """Read a junk file to push a filesystem-managed cache off the store.

    The only lever left on GPFS, whose pagepool ignores fadvise. It is not
    verifiable, so a churned measurement still has to earn its keep through the
    read_bytes probe on the timed request."""
    nbytes = int(gib * (1 << 30))
    if nbytes <= 0:
        return 0.0
    if not path.exists() or path.stat().st_size < nbytes:
        path.parent.mkdir(parents=True, exist_ok=True)
        block = os.urandom(1 << 22)
        with path.open("wb") as f:
            written = 0
            while written < nbytes:
                f.write(block)
                written += len(block)
            f.flush()
            os.fsync(f.fileno())
    t0 = time.perf_counter()
    read = 0
    with path.open("rb", buffering=0) as f:
        while read < nbytes:
            chunk = f.read(1 << 22)
            if not chunk:
                break
            read += len(chunk)
    return (time.perf_counter() - t0) * 1e3


def make_store_cold(root: Path, *, budget_bytes: int, attempts: int = 6,
                    churn_path: Path | None = None, churn_gib: float = 0.0) -> dict:
    """Get the store out of any read cache, and report how well that can be shown.

    On a page-cache filesystem this is fadvise plus a mincore check, so "cold"
    is demonstrated. GPFS caches file data in its own pagepool instead: mincore
    always reports 0 and fadvise is a no-op, so neither can show or clear what
    is cached. There the only real evidence is the read_bytes probe taken around
    the timed request, and `page_cache_governed` records that so validate() does
    not treat a vacuous mincore reading as proof."""
    fs = fs_type(root)
    if fs == "gpfs":
        churn_ms = churn(churn_path, churn_gib) if (churn_path and churn_gib) else 0.0
        return {"fs": fs, "page_cache_governed": False,
                "churn_gib": churn_gib, "churn_ms": round(churn_ms, 1),
                "cold": True}

    resident, failed = store_resident(root)
    for i in range(attempts):
        if resident <= budget_bytes and failed == 0:
            return {"fs": fs, "page_cache_governed": True, "attempts": i,
                    "resident_bytes": resident, "probe_failures": failed,
                    "cold": True}
        fadvise_all(root)
        time.sleep(0.3)
        resident, failed = store_resident(root)
    return {"fs": fs, "page_cache_governed": True, "attempts": attempts,
            "resident_bytes": resident, "probe_failures": failed,
            "cold": resident <= budget_bytes and failed == 0}


# --------------------------------------------------------------------------- #
# server
# --------------------------------------------------------------------------- #


def base_args(args, port: int) -> list[str]:
    return [
        "--model-path", args.model,
        "--tp-size", str(args.tp_size),
        "--page-size", str(args.page_size),
        "--attention-backend", args.attention_backend,
        "--cuda-graph-backend-decode", "disabled",
        "--cuda-graph-backend-prefill", "disabled",
        "--disable-overlap-schedule",
        # The installed flashinfer is older than this checkout expects and its
        # autotuner import fails during FP8 warmup. Autotuning only picks kernel
        # variants, so skipping it costs throughput and nothing in correctness.
        "--disable-flashinfer-autotune",
        "--chunked-prefill-size", "-1",
        "--max-prefill-tokens", str(args.max_prefill_tokens),
        *(["--context-length", str(args.context_length)] if args.context_length else []),
        "--max-total-tokens", str(args.max_total_tokens),
        "--max-running-requests", "1",
        "--mem-fraction-static", str(args.mem_fraction_static),
        "--host", "127.0.0.1",
        "--port", str(port),
        "--enable-hierarchical-cache",
        "--hicache-ratio", str(args.hicache_ratio),
        "--hicache-write-policy", "write_through",
        "--hicache-io-backend", "direct",
        "--hicache-mem-layout", "page_first_direct",
        "--hicache-host-memory-mode", "cache",
    ]


def server_spec(server: str, args, port: int) -> tuple[list[str], dict]:
    argv = base_args(args, port)
    env: dict[str, str] = {}
    if server == "mem":
        return argv, env

    argv += ["--hicache-storage-prefetch-policy", "wait_complete"]
    if server == "l3_old":
        argv += ["--hicache-storage-backend", "file"]
        env["SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"] = str(Path(args.store_root) / "file")
        env["SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE"] = args.store_max_size
        env["SGLANG_HICACHE_FILE_BACKEND_MIN_FREE_SPACE"] = args.store_min_free
        return argv, env

    argv += ["--hicache-storage-backend", "layerwise_file"]
    if server.startswith("l3_nopipe"):
        argv += ["--hicache-storage-load-mode", "full_wait",
                 "--hicache-storage-io-threads",
                 str(io_threads_of_server(server, args.io_threads))]
        env["SGLANG_HICACHE_LAYERWISE_ROOT"] = str(
            Path(args.store_root) / store_of_server(server))
        env["SGLANG_HICACHE_LAYERWISE_MAX_SIZE"] = args.store_max_size
        env["SGLANG_HICACHE_LAYERWISE_MIN_FREE_SPACE"] = args.store_min_free
        return argv, env
    if server == "l3_fused":
        argv += [
            "--hicache-storage-load-mode", "layerwise",
            "--hicache-storage-group-size", str(args.group_size),
            "--hicache-storage-max-concurrent-streams",
            str(args.max_concurrent_streams),
            "--hicache-storage-group-timeout-ms", str(args.group_timeout_ms),
        ]
    else:
        argv += ["--hicache-storage-load-mode", "full_wait"]
    # Sharded submission applies to both read paths now; it used to be refused
    # on the streaming one, which left the pipeline reading single-context.
    if args.io_threads > 1:
        argv += ["--hicache-storage-io-threads", str(args.io_threads)]
    env["SGLANG_HICACHE_LAYERWISE_ROOT"] = str(Path(args.store_root) / "layerwise")
    env["SGLANG_HICACHE_LAYERWISE_MAX_SIZE"] = args.store_max_size
    env["SGLANG_HICACHE_LAYERWISE_MIN_FREE_SPACE"] = args.store_min_free
    return argv, env


def launch(server: str, args, port: int, log_path: Path):
    argv, extra_env = server_spec(server, args, port)
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO_PY + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK"] = "1"
    if args.context_length:
        # H+S can exceed the checkpoint's trained context. RoPE then extrapolates
        # on the tail positions, which changes what the model would say -- and
        # nothing about the storage path, the KV byte count, or the timings this
        # benchmark reports.
        env["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
    if args.storage_batch_size:
        env["SGLANG_HICACHE_STORAGE_BATCH_SIZE"] = str(args.storage_batch_size)
    # The JIT kernels shell out to nvcc, and /usr/bin/nvcc is CUDA 12.0, which
    # cannot target sm_120a. Put a CUDA 13 toolkit matching torch ahead of it.
    env["PATH"] = (os.path.expanduser("~/.local/bin") + ":" + args.cuda_bin_dir
                   + ":" + env.get("PATH", ""))
    env["CUDA_HOME"] = str(Path(args.cuda_bin_dir).parent)
    env["CUDA_VISIBLE_DEVICES"] = args.gpus
    env.update(extra_env)

    cmd = [args.python, "-m", "sglang.launch_server"] + argv
    log_path.write_text(
        "$ PYTHONPATH=" + REPO_PY + " "
        + " ".join(f"{k}={v}" for k, v in sorted(extra_env.items()))
        + " " + " ".join(cmd) + "\n\n")
    fh = log_path.open("a")
    log(f"launching '{server}' on port {port}")
    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    return proc, fh


def wait_health(port: int, proc, timeout: float = 1200.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.returncode}")
        try:
            if requests.get(f"http://127.0.0.1:{port}/health_generate",
                            timeout=5).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(2.0)
    raise TimeoutError("server did not become healthy")


def shutdown(proc, fh) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=90)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)
    fh.close()


# --------------------------------------------------------------------------- #
# requests
# --------------------------------------------------------------------------- #


def gen_ids(n: int, rng: random.Random) -> list[int]:
    # A safe slice of the vocab: no specials, no EOS (Llama specials are >=128000).
    return [rng.randrange(1000, 100000) for _ in range(n)]


def generate(port: int, input_ids: list[int], *, timeout: float = 900.0) -> dict:
    payload = {"input_ids": input_ids,
               "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
               "stream": True}
    t0 = time.perf_counter()
    ttft = None
    meta = {}
    with requests.post(f"http://127.0.0.1:{port}/generate", json=payload,
                       stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            body = raw[len("data:"):].strip()
            if body == "[DONE]":
                break
            if ttft is None:
                ttft = (time.perf_counter() - t0) * 1e3
            try:
                meta = json.loads(body).get("meta_info", {}) or meta
            except json.JSONDecodeError:
                pass
    if ttft is None:
        raise RuntimeError("no chunk received")
    return {"ttft_ms": ttft, "meta": meta}


def flush_cache(port: int) -> None:
    for _ in range(30):
        r = requests.post(f"http://127.0.0.1:{port}/flush_cache", timeout=30)
        if r.status_code == 200 and "not flushed" not in r.text.lower():
            time.sleep(0.5)
            return
        time.sleep(1.0)
    raise RuntimeError("flush_cache never succeeded")


def tier(meta: dict) -> dict:
    d = meta.get("cached_tokens_details") or {}
    return {"cached": meta.get("cached_tokens", 0), "device": d.get("device", 0),
            "host": d.get("host", 0), "storage": d.get("storage", 0),
            "backend": d.get("storage_backend", "none")}


LOG_MARKERS = {"declined": "Layerwise streaming declined",
               "dropped": "Layerwise streaming dropped",
               "disabled": "Layerwise storage streaming disabled",
               "traceback": "Traceback (most recent call last)"}


def log_counts(path: Path) -> dict:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return {k: 0 for k in LOG_MARKERS}
    return {k: text.count(v) for k, v in LOG_MARKERS.items()}


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #


def wait_writeback(root: Path, *, min_bytes: int, timeout: float = 300.0) -> int:
    deadline = time.time() + timeout
    last, stable = -1, 0
    while time.time() < deadline:
        cur, _ = store_bytes(root)
        if cur >= min_bytes and cur == last:
            stable += 1
            if stable >= 2:
                return cur
        else:
            stable = 0
        last = cur
        time.sleep(1.0)
    raise TimeoutError(f"write-through never reached {min_bytes} bytes (last {last})")


def evict_device(port: int, args, rng: random.Random) -> int:
    """Fill the device pool with fresh tokens so LRU pushes the prefix out of HBM.
    The host pool is hicache_ratio times larger, so the host copy survives."""
    pushed = 0
    chunk = min(args.filler_tokens, args.max_prefill_tokens - 64)
    while pushed < args.max_total_tokens:
        generate(port, gen_ids(chunk, rng))
        pushed += chunk
    return pushed


# Match the whole log line so the rank tag can be read off it. An optional
# leading group would not work: the engine simply skips it and every line comes
# back as rank 0, which is exactly the collapse this parsing has to avoid.
_READ_LINE = re.compile(
    r"^.*layerwise_file read: pages=(\d+) bytes=(\d+) ms=([\d.]+) "
    r"open_ms=([\d.]+) io_ms=([\d.]+) GiB/s=([\d.]+) threads=(\d+).*$",
    re.MULTILINE,
)
_RANK_TAG = re.compile(r"TP(\d+)\]")


def read_lines_since(log_path: Path, offset: int) -> tuple[list[dict], int]:
    """Parse the backend's per-batch read reports appended since ``offset``.

    The backend is the only place that sees both the bytes and the wall time of
    the storage read; differencing two TTFTs can only bound it. Reading by byte
    offset keeps a measurement's batches separate from the warm-up's.
    """
    with log_path.open("rb") as handle:
        handle.seek(offset)
        chunk = handle.read()
        end = handle.tell()
    rows = []
    for match in _READ_LINE.finditer(chunk.decode("utf-8", "replace")):
        pages, nbytes, ms, open_ms, io_ms, gibps, threads = match.groups()
        rank_tag = _RANK_TAG.search(match.group(0))
        rows.append({"rank": int(rank_tag.group(1)) if rank_tag else 0,
                     "pages": int(pages), "bytes": int(nbytes), "ms": float(ms),
                     "open_ms": float(open_ms), "io_ms": float(io_ms),
                     "gibps": float(gibps), "threads": int(threads)})
    return rows, end


def summarize_reads(rows: list[dict]) -> dict:
    """Fold one request's batches into a single L3 -> L2 transfer figure.

    The controller runs batches one after another on a single thread, so the
    per-batch times add up to the transfer's wall time rather than overlapping.
    """
    if not rows:
        return {}
    # Every TP rank reads its own shard of the same prefix, concurrently, into
    # its own host pool. Batches WITHIN a rank are serial and add up; the ranks
    # themselves overlap, so the request waits for the slowest of them. Summing
    # across ranks would report a two-rank read as taking twice as long as it
    # does, and would divide two ranks' bytes by two ranks' seconds -- which
    # silently yields the per-rank rate while looking like an aggregate.
    per_rank: dict[int, dict] = {}
    for row in rows:
        acc = per_rank.setdefault(row["rank"], {"ms": 0.0, "open_ms": 0.0,
                                                "io_ms": 0.0, "bytes": 0, "pages": 0,
                                                "batches": 0})
        for field in ("ms", "open_ms", "io_ms", "bytes", "pages"):
            acc[field] += row[field]
        acc["batches"] += 1

    critical = max(per_rank.values(), key=lambda a: a["ms"])
    total_bytes = sum(a["bytes"] for a in per_rank.values())
    rank_gibps = [a["bytes"] / (1 << 30) / (a["io_ms"] / 1e3)
                  for a in per_rank.values() if a["io_ms"] > 0]
    return {
        "ranks": len(per_rank),
        "batches_per_rank": critical["batches"],
        "l3_pages": sum(a["pages"] for a in per_rank.values()),
        "l3_bytes": total_bytes,
        # The slowest rank's wall clock: what the request actually waits for.
        "l3_ms": round(critical["ms"], 2),
        "l3_open_ms": round(critical["open_ms"], 2),
        "l3_io_ms": round(critical["io_ms"], 2),
        # Per rank, transfer only -- the figure that describes one rank's share
        # of the storage, and the one that has to scale with io_threads.
        "l3_gibps": round(statistics.median(rank_gibps), 2) if rank_gibps else 0.0,
        # Every rank's bytes over the critical path: what the filesystem as a
        # whole delivered to this request.
        "l3_agg_gibps": (round(total_bytes / (1 << 30) / (critical["io_ms"] / 1e3), 2)
                         if critical["io_ms"] else 0.0),
    }


def warm_prefix(port: int, prefix: list[int], rng: random.Random, step: int) -> int:
    """Write the prefix into the store, extending it a step at a time.

    One 131k-token prefill would be a different experiment: it needs a prefill
    budget and an activation footprint the measured request never uses. Growing
    the prefix in steps keeps every prefill small, and each step is still a cold
    miss that writes -- it never reads the store, so it cannot warm any cache
    under the timed read.
    """
    sent = 0
    while sent < len(prefix):
        sent = min(sent + step, len(prefix))
        generate(port, prefix[:sent] + gen_ids(8, rng))
    return sent


def measure(arm: str, port: int, proc, prefix: list[int], suffix_len: int,
            args, rng: random.Random, store_root, log_path: Path) -> dict:
    prep: dict = {}
    expect_kv = len(prefix) * args.kv_bytes_per_token

    flush_cache(port)
    # Warm-up. The prefix is fresh every measurement, so this is always a cold
    # miss that WRITES the prefix. It never pre-reads the store, so it cannot
    # warm the page cache under the timed read.
    before_store = store_bytes(store_root)[0] if store_root is not None else 0
    warm_started = time.perf_counter()
    warm_prefix(port, prefix, rng, args.warm_step)
    prep["warm_ms"] = round((time.perf_counter() - warm_started) * 1e3, 1)

    if store_root is not None:
        # Gate on the GROWTH this warm-up caused, not on an absolute size: the
        # store already holds every earlier measurement's prefix, so an absolute
        # threshold is satisfied before this prefix has reached the disk at all.
        prep["store_grew"] = wait_writeback(
            store_root, min_bytes=before_store + int(expect_kv * 0.9)) - before_store

    if arm == "l1":
        pass                                    # prefix stays in HBM
    elif arm in ("l2", "l2_fused"):
        prep["evicted_tokens"] = evict_device(port, args, rng)
    else:
        flush_cache(port)                       # drop the HBM and host copies
        prep["cold"] = make_store_cold(
            store_root, budget_bytes=int(expect_kv * 0.02),
            churn_path=Path(args.churn_file), churn_gib=args.churn_gib)

    before_io = tree_read_bytes(proc.pid)
    before_log = log_counts(log_path)
    log_offset = log_path.stat().st_size
    res = generate(port, prefix + (gen_ids(suffix_len, rng) if suffix_len else []))
    read_delta = tree_read_bytes(proc.pid) - before_io
    after_log = log_counts(log_path)
    reads, _ = read_lines_since(log_path, log_offset)

    rec = {"arm": arm, "hit_tokens": len(prefix), "suffix_tokens": suffix_len,
           "ttft_ms": round(res["ttft_ms"], 1), "tier": tier(res["meta"]),
           "disk_read_bytes": read_delta, "expect_kv_bytes": expect_kv,
           "disk_read_frac": round(read_delta / expect_kv, 3),
           "log_delta": {k: after_log[k] - before_log.get(k, 0) for k in after_log},
           "prep": prep}
    rec.update(summarize_reads(reads))
    return rec


def validate(arm: str, rec: dict) -> str | None:
    """Reject anything that was not served by the tier this arm is about."""
    t, h = rec["tier"], rec["hit_tokens"]
    tol = max(64, h // 100)
    frac = rec["disk_read_frac"]

    if t["cached"] < h - tol:
        return f"prefix not fully cached ({t['cached']} < {h})"
    for k in ("declined", "dropped", "disabled", "traceback"):
        if rec["log_delta"].get(k, 0):
            return f"server logged {k} during this request"

    if arm == "l1":
        if t["device"] < h - tol:
            return f"device hit {t['device']} < {h}"
        if frac > 0.05:
            return f"read {frac:.2f} of the KV from disk on an HBM hit"
        return None

    if arm in ("l2", "l2_fused"):
        if t["device"] > tol:
            return f"HBM contamination: device hit {t['device']}"
        if t["host"] < h - tol:
            return f"host hit {t['host']} < {h}"
        if t["storage"] > tol:
            return f"storage hit {t['storage']} > 0"
        # The decisive one: a real L2 hit touches no block device.
        if frac > 0.05:
            return f"L3 contamination: read {frac:.2f} of the KV from disk"
        return None

    # L3 arms
    if t["device"] > tol:
        return f"HBM contamination: device hit {t['device']}"
    if arm == "l3_fused":
        # The bridge books its staged prefetch as a host hit and never sets
        # storage_hit_length, so a storage hit here means it fell back.
        if t["storage"] >= h - tol:
            return "fell back to the blocking read"
        if t["host"] < h - tol:
            return f"staged host hit {t['host']} < {h}"
    else:
        if t["host"] > tol:
            return f"L2 contamination: host hit {t['host']}"
        if t["storage"] < h - tol:
            return f"storage hit {t['storage']} < {h}"
    cold = rec["prep"].get("cold", {})
    if cold.get("page_cache_governed", True):
        if cold.get("probe_failures"):
            return f"residency probe failed on {cold['probe_failures']} store files"
        if not cold.get("cold", False):
            return f"store not cold: {cold.get('resident_bytes')} bytes still resident"
    # The gate that holds on every filesystem: the timed request had to pull the
    # prefix off the block device. Anything served from a read cache -- the Linux
    # page cache, or a GPFS pagepool that fadvise cannot touch -- shows up here as
    # a low fraction and is refused rather than reported.
    if frac < 0.8:
        return (f"only {frac:.2f} of the KV came off the block device; the rest was "
                f"served from a read cache ({cold.get('fs', '?')})")
    return None


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def t3(t: dict) -> str:
    return f"{t['device']}/{t['host']}/{t['storage']}"


def run(args) -> Path:
    run_dir = RESULTS_ROOT / (args.run_id or time.strftime("%Y%m%d-%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    records = (run_dir / "records.jsonl").open("a")
    log(f"run dir {run_dir}")

    arms = [a for a in args.arms.split(",") if a]
    for a in arms:
        if not arm_is_known(a):
            raise SystemExit(f"unknown arm {a}; pick from {ARMS} or l3_nopipe_t<N>")
    suffixes = [int(s) for s in args.suffixes.split(",")]
    # H changes nothing about how a server is launched, so every cached-prefix
    # length is swept inside one server rather than costing its own model load.
    hits = [int(h) for h in str(args.hit_tokens).split(",")]

    by_server: dict = {}
    for a in arms:
        by_server.setdefault(server_of_arm(a), []).append(a)

    rng = random.Random(args.seed)
    counter = 0
    for server, server_arms in by_server.items():
        log_path = run_dir / f"server-{server}.log"
        store_root = None
        if store_of_server(server) is not None:
            store_root = Path(args.store_root) / store_of_server(server)
            shutil.rmtree(store_root, ignore_errors=True)
            store_root.mkdir(parents=True, exist_ok=True)

        proc, fh = launch(server, args, args.port, log_path)
        try:
            wait_health(args.port, proc)
            log(f"'{server}' healthy")
            for rep in range(args.reps):
                for arm in server_arms:
                    for h in hits:
                        for s in suffixes:
                            counter += 1
                            prefix = gen_ids(h, random.Random(args.seed + 7919 * counter))
                            rec = measure(arm, args.port, proc, prefix, s, args, rng,
                                          store_root, log_path)
                            rec.update(rep=rep, server=server, ts=time.time())
                            rec["complaint"] = validate(arm, rec)
                            records.write(json.dumps(rec) + "\n")
                            records.flush()
                            flag = "" if rec["complaint"] is None else f"  !! {rec['complaint']}"
                            l3 = (f"L3->L2={rec['l3_ms']:7.1f} ms "
                                  f"(open {rec['l3_open_ms']:6.1f} + io {rec['l3_io_ms']:7.1f}) "
                                  f"{rec['l3_gibps']:6.2f} GiB/s/rank "
                                  f"({rec['l3_agg_gibps']:6.2f} agg)  "
                                  if rec.get("l3_ms") else "")
                            log(f"{arm:13s} H={h:<7d} S={s:<5d} "
                                f"TTFT={rec['ttft_ms']:8.1f} ms  {l3}"
                                f"disk={rec['disk_read_frac']:.2f}x  "
                                f"tier(d/h/s)={t3(rec['tier'])}{flag}")
        finally:
            shutdown(proc, fh)
    records.close()
    summarize(run_dir)
    return run_dir


def summarize(run_dir: Path) -> None:
    lines = (run_dir / "records.jsonl").read_text().splitlines()
    recs = [json.loads(l) for l in lines if l.strip()]
    good = [r for r in recs if r.get("complaint") is None]
    bad = [r for r in recs if r.get("complaint") is not None]

    cells: dict = {}
    disk: dict = {}
    l3ms: dict = {}
    l3bw: dict = {}
    l3open: dict = {}
    l3io: dict = {}
    for r in good:
        key = (r["arm"], r["hit_tokens"], r["suffix_tokens"])
        cells.setdefault(key, []).append(r["ttft_ms"])
        disk.setdefault(key, []).append(r["disk_read_frac"])
        if r.get("l3_ms"):
            l3ms.setdefault(key, []).append(r["l3_ms"])
            l3bw.setdefault(key, []).append(r["l3_gibps"])
            l3open.setdefault(key, []).append(r["l3_open_ms"])
            l3io.setdefault(key, []).append(r["l3_io_ms"])

    seen = {k[0] for k in cells}
    # Keep the canonical order, then append the l3_nopipe_t<N> family by N.
    arms = [a for a in ARMS if a in seen]
    arms += sorted((a for a in seen if _NOPIPE_THREADS.match(a)),
                   key=lambda a: int(_NOPIPE_THREADS.match(a).group(1)))
    hits = sorted({k[1] for k in cells})
    combos = sorted({(k[1], k[2]) for k in cells})

    def med(arm, h, s, table=cells):
        v = table.get((arm, h, s))
        return statistics.median(v) if v else None

    def block(title, rows, fmt):
        print()
        print(title)
        head = f"{'H':>7} {'S':>6} {'hit%':>7} " + " ".join(f"{a:>11}" for a in rows)
        print(head)
        print("-" * len(head))
        for h, s in combos:
            line = f"{h:>7} {s:>6} {h/(h+s):>6.2%} "
            line += " ".join(fmt(a, h, s) for a in rows)
            print(line)

    block("TTFT median (ms)", arms,
          lambda a, h, s: (f"{med(a, h, s):>8.0f}[{len(cells.get((a, h, s), []))}]"
                           if med(a, h, s) is not None else f"{'-':>11}"))

    block("fraction of the prefix KV actually read from the block device", arms,
          lambda a, h, s: (f"{med(a, h, s, disk):>11.2f}"
                           if med(a, h, s, disk) is not None else f"{'-':>11}"))

    if l3ms:
        l3_arms = [a for a in arms if any(k[0] == a for k in l3ms)]
        block("L3 -> L2 transfer, measured inside the backend (ms)", l3_arms,
              lambda a, h, s: (f"{med(a, h, s, l3ms):>11.0f}"
                               if med(a, h, s, l3ms) is not None else f"{'-':>11}"))
        block("  of which page open() (ms)", l3_arms,
              lambda a, h, s: (f"{med(a, h, s, l3open):>11.0f}"
                               if med(a, h, s, l3open) is not None else f"{'-':>11}"))
        block("  of which transfer (ms)", l3_arms,
              lambda a, h, s: (f"{med(a, h, s, l3io):>11.0f}"
                               if med(a, h, s, l3io) is not None else f"{'-':>11}"))
        block("L3 read bandwidth per rank, transfer only (GiB/s)", l3_arms,
              lambda a, h, s: (f"{med(a, h, s, l3bw):>11.2f}"
                               if med(a, h, s, l3bw) is not None else f"{'-':>11}"))

    for ref in ("l2", "l2_fused"):
        if ref not in arms:
            continue
        others = [a for a in arms if a != ref]

        def gap(a, h, s, ref=ref):
            base, v = med(ref, h, s), med(a, h, s)
            return f"{v - base:>11.0f}" if (base is not None and v is not None) else f"{'-':>11}"

        block(f"gap vs {ref} (ms)", others, gap)

    if len(hits) > 1:
        print()
        print(f"KV loaded per hit length: " + ", ".join(
            f"H={h} → {h * 163840 / (1 << 20):.0f} MiB" for h in hits)
            + "   (at 160 KiB/token; pass --kv-bytes-per-token for another model)")

    if bad:
        print()
        print(f"{len(bad)} record(s) rejected:")
        for r in bad:
            print(f"  {r['arm']:13s} H={r['hit_tokens']:<7d} S={r['suffix_tokens']:<6d} "
                  f"{r['complaint']}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--analyze", type=Path, help="summarize an existing run dir and exit")
    p.add_argument("--run-id")
    p.add_argument("--arms", default="l1,l2,l3_old,l3_nopipe,l3_fused,l2_fused")
    p.add_argument("--hit-tokens", default="8832",
                   help="H, comma-separated; each a multiple of page size. Swept "
                        "inside one server, so extra values cost no model loads")
    p.add_argument("--suffixes", default="64,2688,4416,8832")
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--seed", type=int, default=20260910)

    p.add_argument("--model", default="/home/lpl/models/Llama-3.3-70B-Instruct-FP8")
    p.add_argument("--python", default="/home/lpl/sglangtest/.venv/bin/python",
                   help="interpreter that has torch; the sglang code under test "
                        "comes from THIS checkout via PYTHONPATH, not from the venv")
    p.add_argument("--kv-bytes-per-token", type=int, default=2 * 80 * 8 * 128 * 1,
                   help="2*layers*kv_heads*head_dim*dtype_bytes, summed over TP "
                        "ranks. Llama-3.3-70B-FP8 has an FP8 KV cache -> 1 byte")
    p.add_argument("--tp-size", type=int, default=2)
    p.add_argument("--gpus", default="0,1")
    p.add_argument("--port", type=int, default=31921)
    p.add_argument("--page-size", type=int, default=64)
    p.add_argument("--attention-backend", default="triton")
    p.add_argument("--cuda-bin-dir", default="/usr/local/cuda-13.0/bin")
    p.add_argument("--mem-fraction-static", type=float, default=0.85)
    p.add_argument("--max-total-tokens", type=int, default=32768,
                   help="device KV pool; small enough that fillers can evict the prefix")
    p.add_argument("--max-prefill-tokens", type=int, default=40960)
    p.add_argument("--hicache-ratio", type=float, default=4.0)
    p.add_argument("--filler-tokens", type=int, default=8192)

    p.add_argument("--store-root", default="/zion0/kv-aio-bench/l3-run-llama",
                   help="where the L3 tier lives. /zion0 is GPFS: the O_DIRECT "
                        "layerwise backend is measurable there, but the buffered "
                        "stock `file` backend is served from the GPFS pagepool "
                        "and its measurements will be refused")
    p.add_argument("--store-max-size", default="100Gi")
    p.add_argument("--store-min-free", default="100Gi")
    p.add_argument("--churn-gib", type=float, default=0.0,
                   help="buffered-read this much junk before an L3 read, to push a "
                        "filesystem-managed cache (GPFS pagepool) off the store")
    p.add_argument("--churn-file", default="/zion0/kv-aio-bench/churn.bin")

    p.add_argument("--io-threads", type=int, default=1,
                   help="--hicache-storage-io-threads for the l3_nopipe server. "
                        "Shards one whole-prefix read across that many AIO "
                        "contexts; a parallel filesystem needs it to reach its "
                        "aggregate bandwidth. Refused by the layerwise path, so "
                        "it is passed to the full_wait server only")
    p.add_argument("--warm-step", type=int, default=16384,
                   help="grow the prefix this many tokens per warm-up request. "
                        "Keeps every write-phase prefill small enough to stay "
                        "inside --max-prefill-tokens for a very long prefix")
    p.add_argument("--context-length", type=int, default=0,
                   help="override the model's context length (0 = leave it). "
                        "Needed when H+S exceeds what the checkpoint was trained "
                        "for; it changes generated text, not the storage path")
    p.add_argument("--storage-batch-size", type=int, default=0,
                   help="SGLANG_HICACHE_STORAGE_BATCH_SIZE: pages per batched "
                        "storage call (0 = leave the 128 default). The controller "
                        "waits for each batch before starting the next, so this "
                        "caps how much a parallel backend can have in flight")
    p.add_argument("--max-concurrent-streams", type=int, default=1,
                   help="layerwise transactions allowed to stream at once; "
                        "the rest fall back to the blocking whole-prefix read")
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--group-timeout-ms", type=int, default=1000,
                   help="--hicache-storage-group-timeout-ms. A group that misses "
                        "it aborts the transaction mid-forward and, with no "
                        "poison/replay yet, takes the scheduler down. Long "
                        "prefixes need more than the 1000 ms default")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.analyze:
        summarize(args.analyze)
        return
    run(args)


if __name__ == "__main__":
    main()
