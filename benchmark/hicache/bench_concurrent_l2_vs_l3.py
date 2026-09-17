#!/usr/bin/env python3
"""Concurrent L2-vs-L3 TTFT on one layerwise server.

One server (`l3_fused`), two arms that differ only in where the prefix lives:

  l2  -- warm the prefix, evict HBM only, so the hit is served from host memory
  l3  -- warm the prefix, flush HBM+host, so the hit must come from storage

Every request in a wave gets its OWN prefix, so a request never reads a prefix
another request just pulled into L2.  Concurrency comes from a thread pool; the
server sees them as it would see real traffic.

Reuses the single-request bench's helpers so the tier attribution, the warm-up
and the cold-store handling are the audited ones.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "/home/lpl/sglang/benchmark/hicache")
import bench_l2l3_fusion_ttft as B  # noqa: E402

SERVER = "l3_fused"


def patch_base_args(concurrency: int) -> None:
    """The single-request bench pins the server to --max-running-requests 1.

    That is the right choice when the point is to attribute one request's TTFT
    to one storage tier, and exactly wrong here: it would serialize the wave and
    measure nothing about concurrency. Raise it, and leave everything else the
    audited bench chose.
    """
    original = B.base_args

    def patched(args, port):
        argv = original(args, port)
        i = argv.index("--max-running-requests")
        argv[i + 1] = str(max(concurrency, 1))
        return argv

    B.base_args = patched


def pct(values, p):
    v = sorted(values)
    if not v:
        return float("nan")
    k = (len(v) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)


def warm_all(port, prefixes, rng, step):
    """Write every prefix through to storage, one at a time."""
    for i, prefix in enumerate(prefixes):
        B.warm_prefix(port, prefix, rng, step)
        if (i + 1) % 4 == 0:
            B.log(f"  warmed {i + 1}/{len(prefixes)}")


def fire(port, prefixes, suffix_len, rng, conc):
    """Fire one request per prefix at the given concurrency."""
    suffixes = [B.gen_ids(suffix_len, rng) if suffix_len else [] for _ in prefixes]
    results = [None] * len(prefixes)

    def one(i):
        t0 = time.perf_counter()
        res = B.generate(port, list(prefixes[i]) + suffixes[i])
        return i, {
            "ttft_ms": round(res["ttft_ms"], 1),
            "queued_ms": round((t0 - wave_start) * 1e3, 1),
            "tier": B.tier(res["meta"]),
        }

    wave_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=conc) as pool:
        for i, rec in pool.map(one, range(len(prefixes))):
            results[i] = rec
    wall = time.perf_counter() - wave_start
    return results, wall


def run_arm(arm, port, proc, args, rng, store_root, log_path):
    hit = args.hit_tokens
    prefixes = [B.gen_ids(hit, rng) for _ in range(args.requests)]
    expect_kv = hit * args.kv_bytes_per_token * args.requests

    B.log(f"[{arm}] warming {args.requests} x {hit}-token prefixes")
    B.flush_cache(port)
    before_store = B.store_bytes(store_root)[0]
    warm_all(port, prefixes, rng, args.warm_step)
    grew = (
        B.wait_writeback(
            store_root, min_bytes=before_store + int(expect_kv * 0.9), timeout=900.0
        )
        - before_store
    )
    B.log(f"[{arm}] store grew {grew / 2**30:.2f} GiB")

    prep = {"store_grew_bytes": grew}
    if arm == "l2":
        prep["evicted_tokens"] = B.evict_device(port, args, rng)
    else:
        B.flush_cache(port)
        prep["cold"] = B.make_store_cold(
            store_root,
            budget_bytes=int(expect_kv * 0.02),
            churn_path=Path(args.churn_file),
            churn_gib=args.churn_gib,
        )

    before_io = B.tree_read_bytes(proc.pid)
    before_log = B.log_counts(log_path)
    log_offset = log_path.stat().st_size

    B.log(f"[{arm}] firing {args.requests} requests at concurrency {args.concurrency}")
    results, wall = fire(port, prefixes, args.suffix_tokens, rng, args.concurrency)

    reads, _ = B.read_lines_since(log_path, log_offset)
    return {
        "arm": arm,
        "hit_tokens": hit,
        "suffix_tokens": args.suffix_tokens,
        "concurrency": args.concurrency,
        "requests": args.requests,
        "wall_s": round(wall, 2),
        "disk_read_bytes": B.tree_read_bytes(proc.pid) - before_io,
        "expect_kv_bytes": expect_kv,
        "log_delta": {
            k: B.log_counts(log_path)[k] - before_log.get(k, 0) for k in before_log
        },
        "prep": prep,
        "reads": B.summarize_reads(reads),
        "per_request": results,
    }


def report(records):
    print()
    print("=" * 96)
    hdr = (
        f"{'arm':<5}{'n':>4}{'  TTFT p50':>11}{'p90':>9}{'p99':>9}{'max':>9}"
        f"{'  wall s':>9}{'  命中层级 (device/host/storage)':>34}"
    )
    print(hdr)
    print("-" * 96)
    base = None
    for rec in records:
        t = [r["ttft_ms"] for r in rec["per_request"]]
        tiers = {"device": 0, "host": 0, "storage": 0}
        for r in rec["per_request"]:
            for k in tiers:
                tiers[k] += r["tier"][k]
        n = len(t)
        avg = {k: v / max(n, 1) for k, v in tiers.items()}
        print(
            f"{rec['arm']:<5}{n:>4}{st.median(t):>11.1f}{pct(t,90):>9.1f}"
            f"{pct(t,99):>9.1f}{max(t):>9.1f}{rec['wall_s']:>9.2f}"
            f"{avg['device']:>12.0f}{avg['host']:>11.0f}{avg['storage']:>11.0f}"
        )
        if rec["arm"] == "l2":
            base = st.median(t)
    if base is not None:
        print()
        for rec in records:
            if rec["arm"] == "l2":
                continue
            t = [r["ttft_ms"] for r in rec["per_request"]]
            d = st.median(t) - base
            print(
                f"  {rec['arm']} 相对 L2:  +{d:.0f} ms  "
                f"({st.median(t) / base:.2f}x)   "
                f"读盘 {rec['disk_read_bytes'] / 2**30:.2f} GiB / "
                f"预期 KV {rec['expect_kv_bytes'] / 2**30:.2f} GiB"
            )
            r = rec["reads"]
            if r:
                print(f"    后端自报读取: {json.dumps(r, ensure_ascii=False)}")
            print(f"    日志计数: {rec['log_delta']}")
    print("=" * 96)


def main():
    # B.parse_args() reads sys.argv and rejects anything it does not declare,
    # so take this script's own flags out of argv before handing it over.
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--concurrency", type=int, default=8)
    extra.add_argument("--requests", type=int, default=24)
    extra.add_argument("--suffix-tokens", type=int, default=1280)
    known, rest = extra.parse_known_args()
    sys.argv = [sys.argv[0]] + rest

    args = B.parse_args()
    patch_base_args(known.concurrency)
    args.concurrency = known.concurrency
    args.requests = known.requests
    args.suffix_tokens = known.suffix_tokens
    args.hit_tokens = int(str(args.hit_tokens).split(",")[0])

    rng = random.Random(args.seed)
    run_dir = B.RESULTS_ROOT / (args.run_id or f"conc-{time.strftime('%Y%m%d-%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / f"{SERVER}.log"
    store_root = Path(args.store_root) / B.store_of_server(SERVER)

    B.log(f"run dir {run_dir}")
    proc, fh = B.launch(SERVER, args, args.port, log_path)
    records = []
    try:
        B.wait_health(args.port, proc)
        B.log("server healthy")
        for arm in ("l2", "l3"):
            rec = run_arm(arm, args.port, proc, args, rng, store_root, log_path)
            records.append(rec)
            (run_dir / f"{arm}.json").write_text(json.dumps(rec, indent=2))
            B.log(f"[{arm}] done, wall {rec['wall_s']}s")
    finally:
        B.shutdown(proc, fh)

    (run_dir / "summary.json").write_text(json.dumps(records, indent=2))
    report(records)
    print(f"\n原始数据: {run_dir}")


if __name__ == "__main__":
    main()
