#!/usr/bin/env python3
"""One burst of probes where some hit L2 and the rest read L3, arm by arm.

The question is whether attaching L3 slows down the requests that were already
hitting L2. The control is therefore a server with no storage backend at all
(`l2_no_l3` -> `mem`), not the same server with the reads turned off, so every
arm launches its own server and uses its own store directory.

The tier split needs no per-prefix eviction, only the primitives that already
exist. Warm every prefix (write-through puts them all on disk), flush so L1 and
L2 are empty while the files survive, read back only the ones that should hit
L2, then evict the device pool. The host pool is `hicache_ratio` times larger
than the device pool, so those host copies survive the eviction.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_l2l3_fusion_ttft as B  # noqa: E402
from bench_steady_state_l3_probe import Background, patch_base_args  # noqa: E402

# arm -> (server, fraction of probes that must read L3). `l2_no_l3` is the only
# arm without a storage backend, so l2_no_l3 -> l2_with_l3 isolates the cost of
# attaching L3 and l2_with_l3 -> l3_all isolates the cost of reading from it.
ARM_SPEC = {
    "l2_no_l3": ("mem", 0.0),
    "l2_with_l3": ("l3_nixl", 0.0),
    "l3_all": ("l3_nixl", 1.0),
    "mixed_aio": ("l3_fused", 0.5),
    "mixed_nixl": ("l3_nixl", 0.5),
}
# Older spelling, kept so earlier command lines still resolve.
ARM_SPEC["l2_only"] = ARM_SPEC["l2_no_l3"]
ARM_SERVER = {arm: spec[0] for arm, spec in ARM_SPEC.items()}

_REQ_STATS = re.compile(
    r"ReqTimeStats\(rid=([^,]+), input_len=(\d+),.*?"
    r"queue_duration=([\d.]+)ms, forward_duration=([\d.]+)ms",
    re.DOTALL,
)
# The streaming path reports its read under a different prefix than the
# whole-prefix backend, so `B.summarize_reads` sees nothing on a fused arm.
_STREAM_READ = re.compile(
    r"layerwise_stream read: txn=(\S+) pages=(\d+) extents=(\d+) bytes=(\d+) "
    r"ms=([\d.]+) open_ms=([\d.]+) GiB/s=([\d.]+)"
)


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def log_tail(log_path: Path, offset: int) -> str:
    with log_path.open("rb") as handle:
        handle.seek(offset)
        return handle.read().decode("utf-8", "replace")


def req_stats_by_rid(tail: str, input_len: int) -> dict[str, tuple[float, float]]:
    """rid -> (queue, forward). input_len keeps the background requests out."""
    return {
        rid: (float(queue), float(forward))
        for rid, length, queue, forward in _REQ_STATS.findall(tail)
        if int(length) == input_len
    }


def stream_reads(tail: str) -> dict:
    """Read bandwidth, the direct comparison between engines; one per rank."""
    rows = _STREAM_READ.findall(tail)
    if not rows:
        return {}
    spans = [float(r[4]) for r in rows]
    return {
        "n": len(rows),
        "bytes": sum(int(r[3]) for r in rows),
        "span_ms_mean": round(mean(spans), 2),
        "open_ms_mean": round(mean([float(r[5]) for r in rows]), 2),
        "gibps_mean": round(mean([float(r[6]) for r in rows]), 2),
    }


def check_tiers(arm, probes, hit_tokens, l3_count):  # noqa: ARG001
    """Reject any probe that was not served by the tier its arm is about."""
    tol = max(64, hit_tokens // 100)
    complaints = []
    for index, probe in enumerate(probes):
        tier = probe["tier"]
        want_l3 = l3_count > 0 and index >= len(probes) - l3_count
        if tier["cached"] < hit_tokens - tol:
            complaints.append(f"探针{index} 前缀没全命中: cached={tier['cached']}")
        if tier["device"] > tol:
            complaints.append(f"探针{index} HBM 污染: device={tier['device']}")
        if want_l3:
            if tier["storage"] < hit_tokens - tol:
                complaints.append(
                    f"探针{index} 本该走 L3，storage={tier['storage']} < {hit_tokens}"
                )
        else:
            if tier["storage"] > tol:
                complaints.append(f"探针{index} 本该命中 L2，storage={tier['storage']}")
            if tier["host"] < hit_tokens - tol:
                complaints.append(
                    f"探针{index} 本该命中 L2，host={tier['host']} < {hit_tokens}"
                )
        probe["served_by"] = "l3" if want_l3 else "l2"
    return complaints


def prepare(arm, port, args, rng, store_root, prefixes, l3_count):
    """Leave the first N-l3_count prefixes in host memory and the rest on disk."""
    hit = args.hit_tokens
    expect_kv = hit * args.kv_bytes_per_token * len(prefixes)
    prep = {}

    B.log(f"[{arm}] 预热 {len(prefixes)} x {hit}-token 前缀")
    B.flush_cache(port)
    before_store = B.store_bytes(store_root)[0] if store_root else 0
    for index, prefix in enumerate(prefixes):
        B.warm_prefix(port, prefix, rng, args.warm_step)
        if (index + 1) % 4 == 0:
            B.log(f"  预热 {index + 1}/{len(prefixes)}")

    if store_root is not None:
        grew = (
            B.wait_writeback(
                store_root,
                min_bytes=before_store + int(expect_kv * 0.9),
                timeout=1800.0,
            )
            - before_store
        )
        prep["store_grew_bytes"] = grew
        B.log(f"[{arm}] 存储增长 {grew / 2**30:.2f} GiB")

        # Drops L1 and L2 together while the files on disk survive; the page
        # cache is dropped too, or a "storage" read is served from memory.
        B.flush_cache(port)
        prep["cold"] = B.make_store_cold(
            store_root,
            budget_bytes=int(expect_kv * 0.02),
            churn_path=Path(args.churn_file),
            churn_gib=args.churn_gib,
        )
        l2_prefixes = prefixes[: len(prefixes) - l3_count]
        B.log(f"[{arm}] 把 {len(l2_prefixes)} 条从 L3 读回 L2")
        for prefix in l2_prefixes:
            B.warm_prefix(port, prefix, rng, args.warm_step)

    prep["evicted_tokens"] = B.evict_device(port, args, rng)
    return prep


def run_arm(arm, port, proc, args, rng, store_root, log_path):
    hit = args.hit_tokens
    l3_count = round(ARM_SPEC[arm][1] * args.probes)
    rng_arm = random.Random(
        args.seed
    )  # Every arm draws the same prefixes, or the arms are not comparable.
    prefixes = [B.gen_ids(hit, rng_arm) for _ in range(args.probes)]

    prep = prepare(arm, port, args, rng, store_root, prefixes, l3_count)

    background = Background(
        port,
        args.background_requests,
        args.background_input_tokens,
        args.background_output_tokens,
        rng,
    )
    B.log(f"[{arm}] 启动 {args.background_requests} 路背景请求")
    background.start()
    time.sleep(args.background_warmup_s)

    before_io = B.tree_read_bytes(proc.pid)
    before_log = B.log_counts(log_path)
    log_offset = log_path.stat().st_size

    suffixes = [
        B.gen_ids(args.suffix_tokens, rng) if args.suffix_tokens else []
        for _ in prefixes
    ]
    probes = [None] * len(prefixes)

    def fire(index):
        res = B.generate(port, list(prefixes[index]) + suffixes[index])
        return index, {
            "ttft_ms": round(res["ttft_ms"], 1),
            "tier": B.tier(res["meta"]),
            "rid": res["meta"].get("id"),
        }

    B.log(f"[{arm}] 并发发出 {len(prefixes)} 个探针（{l3_count} 个走 L3）")
    with ThreadPoolExecutor(max_workers=len(prefixes)) as pool:
        for index, rec in pool.map(fire, range(len(prefixes))):
            probes[index] = rec

    bg_stats = background.stats()
    background.shutdown()

    complaints = check_tiers(arm, probes, hit, l3_count)
    for complaint in complaints:
        B.log(f"  !! [{arm}] {complaint}")

    tail = log_tail(log_path, log_offset)
    stats = req_stats_by_rid(tail, hit + args.suffix_tokens)
    matched = 0
    for probe in probes:
        entry = stats.get(probe.get("rid"))
        if entry is not None:
            probe["queue_ms"], probe["forward_ms"] = round(entry[0], 1), round(
                entry[1], 1
            )
            matched += 1
    reads, _ = B.read_lines_since(log_path, log_offset)

    def subset(key, served):
        return round(
            mean([p[key] for p in probes if p["served_by"] == served and key in p]), 1
        )

    return {
        "arm": arm,
        "server": ARM_SERVER[arm],
        "hit_tokens": hit,
        "suffix_tokens": args.suffix_tokens,
        "probes": args.probes,
        "l3_probes": l3_count,
        "background": bg_stats,
        "per_probe": probes,
        "ttft_mean": round(mean([p["ttft_ms"] for p in probes]), 1),
        "ttft_mean_l2": round(
            mean([p["ttft_ms"] for p in probes if p["served_by"] == "l2"]), 1
        ),
        "ttft_mean_l3": round(
            mean([p["ttft_ms"] for p in probes if p["served_by"] == "l3"]), 1
        ),
        "queue_mean": round(mean([q for q, _ in stats.values()]), 1),
        "forward_mean": round(mean([f for _, f in stats.values()]), 1),
        "queue_mean_l2": subset("queue_ms", "l2"),
        "queue_mean_l3": subset("queue_ms", "l3"),
        "forward_mean_l2": subset("forward_ms", "l2"),
        "forward_mean_l3": subset("forward_ms", "l3"),
        "req_stats_n": len(stats),
        "req_stats_matched": matched,
        "stream_reads": stream_reads(tail),
        "read_paths": {
            "streamed": tail.count("layerwise_admit:"),
            "fell_back": tail.count("layerwise_file read:"),
            "ranks": args.tp_size,
        },
        "disk_read_bytes": B.tree_read_bytes(proc.pid) - before_io,
        "log_delta": {
            k: B.log_counts(log_path)[k] - before_log.get(k, 0) for k in before_log
        },
        "prep": prep,
        "reads": B.summarize_reads(reads),
        "complaints": complaints,
    }


def report(records):
    print()
    print("=" * 96)
    print("【总体】每个臂全部探针的均值")
    print(
        f"{'臂':<12}{'服务端':<10}{'探针':>5}{'走L3':>5}"
        f"{'TTFT':>9}{'queue':>9}{'forward':>9}{'背景tok/s':>10}"
    )
    print("-" * 96)
    for rec in records:
        print(
            f"{rec['arm']:<12}{rec['server']:<10}{rec['probes']:>5}"
            f"{rec['l3_probes']:>5}{rec['ttft_mean']:>9.1f}"
            f"{rec['queue_mean']:>9.1f}{rec['forward_mean']:>9.1f}"
            f"{rec['background']['tokens_per_s']:>10.1f}"
        )

    print()
    print("【分子集】命中 L2 的那批 / 走 L3 的那批，分开看")
    print(
        f"{'臂':<12}{'L2:TTFT':>10}{'L2:queue':>10}{'L2:forward':>12}"
        f"{'L3:TTFT':>10}{'L3:queue':>10}{'L3:forward':>12}"
    )
    print("-" * 96)
    for rec in records:
        print(
            f"{rec['arm']:<12}{rec['ttft_mean_l2']:>10.1f}"
            f"{rec['queue_mean_l2']:>10.1f}{rec['forward_mean_l2']:>12.1f}"
            f"{rec['ttft_mean_l3']:>10.1f}{rec['queue_mean_l3']:>10.1f}"
            f"{rec['forward_mean_l3']:>12.1f}"
        )

    streams = [(r["arm"], r["stream_reads"]) for r in records if r.get("stream_reads")]
    if streams:
        print()
        print("【L3 读取】流式 range read 自报（每 rank 一条）")
        print(
            f"{'臂':<12}{'条数':>6}{'总字节':>14}{'span均值ms':>12}"
            f"{'open均值ms':>12}{'GiB/s均值':>11}"
        )
        print("-" * 96)
        for arm, read in streams:
            print(
                f"{arm:<12}{read['n']:>6}{read['bytes']:>14}"
                f"{read['span_ms_mean']:>12.2f}{read['open_ms_mean']:>12.2f}"
                f"{read['gibps_mean']:>11.2f}"
            )

    base = next((r for r in records if r["arm"] in ("l2_no_l3", "l2_only")), None)
    if base is not None:
        print()
        print("【相对 l2_no_l3 的差距】")
        for rec in records:
            if rec is base:
                continue
            print(
                f"  {rec['arm']:<12} 整体 TTFT {rec['ttft_mean'] - base['ttft_mean']:+8.1f} ms"
                f"  ({rec['ttft_mean'] / base['ttft_mean']:.2f}x)"
                f"   整体 queue {rec['queue_mean'] - base['queue_mean']:+8.1f} ms"
            )
            print(
                f"  {'':<12} 其中命中 L2 的那批 TTFT "
                f"{rec['ttft_mean_l2'] - base['ttft_mean_l2']:+8.1f} ms"
                f"   queue {rec['queue_mean_l2'] - base['queue_mean_l2']:+8.1f} ms"
            )
        print()
        print(
            "  注意 forward 会被批次大小影响：全命中的臂探针挤在同一个 prefill 批次里，"
        )
        print("  混合臂的 L2/L3 两批分开进，批次更小。比较 queue 比比较 TTFT 干净。")

    for rec in records:
        bad = {k: v for k, v in rec["log_delta"].items() if v}
        miss = rec["probes"] - rec["req_stats_matched"]
        if bad or rec["complaints"] or miss:
            print(
                f"  !! {rec['arm']}: 日志计数 {bad} 归属异常 {len(rec['complaints'])} "
                f"未匹配到时间统计的探针 {miss}"
            )
    print("=" * 96)


def main():
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--server-arms", default="l2_no_l3,l2_with_l3,l3_all")
    extra.add_argument("--probes", type=int, default=8)
    extra.add_argument(
        "--l3-probes",
        type=int,
        default=4,
        help="已废弃：走 L3 的探针数现在由 ARM_SPEC 里每个臂的占比决定",
    )
    extra.add_argument("--suffix-tokens", type=int, default=512)
    extra.add_argument("--background-requests", type=int, default=6)
    extra.add_argument("--background-input-tokens", type=int, default=512)
    extra.add_argument("--background-output-tokens", type=int, default=20000)
    extra.add_argument("--background-warmup-s", type=float, default=8.0)
    known, rest = extra.parse_known_args()
    sys.argv = [sys.argv[0]] + rest

    args = B.parse_args()
    for key, value in vars(known).items():
        setattr(args, key, value)
    args.hit_tokens = int(str(args.hit_tokens).split(",")[0])
    patch_base_args(args.background_requests + args.probes + 1)

    arms = [a.strip() for a in args.server_arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ARM_SPEC]
    if unknown:
        raise SystemExit(f"未知的臂: {unknown}，可选 {sorted(ARM_SPEC)}")

    rng = random.Random(args.seed)
    run_dir = B.RESULTS_ROOT / (
        args.run_id or f"mixed-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    B.log(f"运行目录 {run_dir}")

    records = []
    for arm in arms:
        server = ARM_SERVER[arm]
        log_path = run_dir / f"{arm}.log"
        # One store per arm. Two arms can share a server type, hence the
        # directory `store_of_server` hands out, and every arm warms the same
        # prefixes: the second arm would find them already written, see zero
        # growth, and `wait_writeback` gates on growth, so it would time out.
        arm_args = copy.copy(args)
        arm_args.store_root = str(Path(args.store_root) / arm)
        store_name = B.store_of_server(server)
        store_root = Path(arm_args.store_root) / store_name if store_name else None
        B.log(f"=== 臂 {arm}（服务端 {server}）===")
        proc, fh = B.launch(server, arm_args, args.port, log_path)
        try:
            B.wait_health(args.port, proc)
            B.log("服务端就绪")
            rec = run_arm(arm, args.port, proc, arm_args, rng, store_root, log_path)
        finally:
            B.shutdown(proc, fh)
        records.append(rec)
        (run_dir / f"{arm}.json").write_text(json.dumps(rec, indent=2))

    (run_dir / "summary.json").write_text(json.dumps(records, indent=2))
    report(records)
    print(f"\n原始数据: {run_dir}")


if __name__ == "__main__":
    main()
