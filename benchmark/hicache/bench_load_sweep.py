#!/usr/bin/env python3
"""How the L3-versus-L2 TTFT gap moves as the offered load rises.

A prefetch starts when the request is enqueued, so a request that has to queue
anyway may get its read for free; only an idle scheduler, where a request is
batched the moment it arrives, makes L3 look expensive. Sweeping the load says
which regime the gap belongs to.

The three streams differ by working set size, not by a switch, which is also
what separates L2 from L3 in production: a small prefix set stays resident in
the host pool, a set larger than the pool is always evicted before it comes
round again. Every request is classified by the tier the server reports, so the
split is measured rather than assumed.

Load is swept closed-loop (N clients, each sending the next request as soon as
the previous one returns). A fixed arrival rate above capacity would let the
queue diverge, which measures how long the run lasted rather than the system.

All three streams share one server with the storage backend attached; whether
attaching it costs anything is a separate experiment.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_l2l3_fusion_ttft as B  # noqa: E402
from bench_steady_state_l3_probe import Background, patch_base_args  # noqa: E402

_REQ_STATS = re.compile(
    r"ReqTimeStats\(rid=([^,]+), input_len=(\d+),.*?"
    r"queue_duration=([\d.]+)ms, forward_duration=([\d.]+)ms",
    re.DOTALL,
)


def parse_size(text: str) -> int:
    """'100Gi' / '400G' / '1Ti' -> bytes. Unparseable returns 0, meaning unchecked."""
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMGTP]?)i?B?\s*", str(text or ""))
    if not match:
        return 0
    scale = {"": 1, "K": 2**10, "M": 2**20, "G": 2**30, "T": 2**40, "P": 2**50}
    return int(float(match.group(1)) * scale[match.group(2).upper()])


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def pct(values, p):
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * p / 100)))]


def req_stats_by_rid(log_path: Path, offset: int, input_len: int):
    with log_path.open("rb") as handle:
        handle.seek(offset)
        tail = handle.read().decode("utf-8", "replace")
    return {
        rid: (float(queue), float(forward))
        for rid, length, queue, forward in _REQ_STATS.findall(tail)
        if int(length) == input_len
    }


class Picker:
    """Which prefix the next request uses. Cycled in order; random would make
    eviction nondeterministic and the tier split unreproducible."""

    def __init__(self, policy, hot, cold):
        self.policy = policy
        self.hot = hot
        self.cold = cold
        self.lock = threading.Lock()
        self.n = 0

    def next(self):
        with self.lock:
            index = self.n
            self.n += 1
        if self.policy == "l2":
            return self.hot[index % len(self.hot)]
        if self.policy == "l3":
            return self.cold[index % len(self.cold)]
        # mixed alternates: even draws the resident set, odd the evicted one.
        if index % 2 == 0:
            return self.hot[(index // 2) % len(self.hot)]
        return self.cold[(index // 2) % len(self.cold)]


def drive(port, picker, rng, suffix_tokens, conc, count, record):
    """`conc` clients send `count` requests back to back.

    With record=False nothing is measured; the calls only establish the working
    set for the phase that follows.
    """
    results = []
    lock = threading.Lock()
    issued = [0]

    def worker():
        while True:
            with lock:
                if issued[0] >= count:
                    return
                issued[0] += 1
            prefix = picker.next()
            suffix = B.gen_ids(suffix_tokens, rng) if suffix_tokens else []
            res = B.generate(port, list(prefix) + suffix)
            if record:
                with lock:
                    results.append(
                        {
                            "ttft_ms": round(res["ttft_ms"], 1),
                            "tier": B.tier(res["meta"]),
                            "rid": res["meta"].get("id"),
                        }
                    )

    with ThreadPoolExecutor(max_workers=conc) as pool:
        for _ in range(conc):
            pool.submit(worker)
    return results


def warmup_count(policy, hot, cold, override):
    """Unmeasured requests needed to bring the working set to steady state.

    The previous phase left something else resident, so each phase rebuilds.
    The two streams want opposite things: `l2` needs its whole set to be the
    most recently used, so it cycles the set once; `l3` needs the prefixes it
    is about to measure to be absent, so it cycles the first half of its set
    and the measured requests land on the second half.
    """
    if override:
        return override
    if policy == "l2":
        return len(hot)
    if policy == "l3":
        return len(cold) // 2
    return 2 * len(hot)


def run_phase(policy, conc, port, args, rng, hot, cold, log_path):
    picker = Picker(policy, hot, cold)
    warmup = warmup_count(policy, hot, cold, args.warmup_requests)
    drive(port, picker, rng, args.suffix_tokens, conc, warmup, False)

    log_offset = log_path.stat().st_size
    started = time.perf_counter()
    records = drive(port, picker, rng, args.suffix_tokens, conc, args.requests, True)
    span = time.perf_counter() - started

    stats = req_stats_by_rid(log_path, log_offset, args.hit_tokens + args.suffix_tokens)
    for rec in records:
        entry = stats.get(rec["rid"])
        if entry is not None:
            rec["queue_ms"], rec["forward_ms"] = round(entry[0], 1), round(entry[1], 1)
        tier = rec["tier"]
        # Classify by the tier the server reported, not by what the policy
        # intended. cached=0 is a full recompute and gets its own bucket:
        # folding it into L1 makes the slowest case look like the fastest.
        if tier["cached"] <= 0:
            rec["served_by"] = "miss"
        elif tier["storage"] >= max(tier["host"], tier["device"]):
            rec["served_by"] = "l3"
        elif tier["host"] >= tier["device"]:
            rec["served_by"] = "l2"
        else:
            rec["served_by"] = "l1"

    def group(served):
        return [r for r in records if r["served_by"] == served]

    def agg(rows):
        ttft = [r["ttft_ms"] for r in rows]
        return {
            "n": len(rows),
            "ttft_mean": round(mean(ttft), 1),
            "ttft_p50": round(pct(ttft, 50), 1),
            "ttft_p99": round(pct(ttft, 99), 1),
            "queue_mean": round(
                mean([r["queue_ms"] for r in rows if "queue_ms" in r]), 1
            ),
            "forward_mean": round(
                mean([r["forward_ms"] for r in rows if "forward_ms" in r]), 1
            ),
        }

    return {
        "policy": policy,
        "concurrency": conc,
        "warmup_requests": warmup,
        "span_s": round(span, 2),
        "throughput_req_s": round(len(records) / span, 2) if span > 0 else 0.0,
        "all": agg(records),
        "l2": agg(group("l2")),
        "l3": agg(group("l3")),
        "l1": agg(group("l1")),
        "miss": agg(group("miss")),
        "stats_matched": sum(1 for r in records if "queue_ms" in r),
        "per_request": records,
    }


def report(records, args):
    print()
    print("=" * 108)
    print(
        f"【负载扫描】命中前缀 {args.hit_tokens}，后缀 {args.suffix_tokens}，"
        f"每档 {args.requests} 个请求"
    )
    print()
    print(
        f"{'并发':>4} {'流':<7}{'实测L1/L2/L3/miss':>18}{'TTFT均值':>10}{'TTFT P99':>10}"
        f"{'queue均值':>10}{'forward均值':>12}{'吞吐req/s':>11}"
    )
    print("-" * 108)
    last_conc = None
    for rec in records:
        if last_conc is not None and rec["concurrency"] != last_conc:
            print()
        last_conc = rec["concurrency"]
        split = (
            f"{rec['l1']['n']}/{rec['l2']['n']}/{rec['l3']['n']}" f"/{rec['miss']['n']}"
        )
        a = rec["all"]
        print(
            f"{rec['concurrency']:>4} {rec['policy']:<7}{split:>18}"
            f"{a['ttft_mean']:>10.1f}{a['ttft_p99']:>10.1f}"
            f"{a['queue_mean']:>10.1f}{a['forward_mean']:>12.1f}"
            f"{rec['throughput_req_s']:>11.2f}"
        )

    print()
    print("【差距】同一并发档下，相对 l2 流的 TTFT 均值")
    print(f"{'并发':>4}{'mixed - l2':>14}{'l3 - l2':>12}{'l3/l2':>9}")
    print("-" * 108)
    by_conc = {}
    for rec in records:
        by_conc.setdefault(rec["concurrency"], {})[rec["policy"]] = rec
    for conc in sorted(by_conc):
        row = by_conc[conc]
        if "l2" not in row:
            continue
        base = row["l2"]["all"]["ttft_mean"]
        mixed = row.get("mixed", {}).get("all", {}).get("ttft_mean", float("nan"))
        l3 = row.get("l3", {}).get("all", {}).get("ttft_mean", float("nan"))
        print(f"{conc:>4}{mixed - base:>+14.1f}{l3 - base:>+12.1f}{l3 / base:>9.2f}")

    print()
    print("【分层】只看那些实测走了 L3 的请求，对照同档 l2 流")
    print(
        f"{'并发':>4}{'l2流:TTFT':>12}{'l3流内L3:TTFT':>15}{'差':>9}"
        f"{'l2流:forward':>14}{'l3流内L3:forward':>18}{'差':>9}"
    )
    print("-" * 108)
    for conc in sorted(by_conc):
        row = by_conc[conc]
        if "l2" not in row or "l3" not in row:
            continue
        a, b = row["l2"]["all"], row["l3"]["l3"]
        if not b["n"]:
            continue
        print(
            f"{conc:>4}{a['ttft_mean']:>12.1f}{b['ttft_mean']:>15.1f}"
            f"{b['ttft_mean'] - a['ttft_mean']:>+9.1f}"
            f"{a['forward_mean']:>14.1f}{b['forward_mean']:>18.1f}"
            f"{b['forward_mean'] - a['forward_mean']:>+9.1f}"
        )

    print()
    missed = [r for r in records if r["miss"]["n"]]
    for rec in missed:
        print(
            f"  !! {rec['policy']}@{rec['concurrency']}: {rec['miss']['n']} 个请求"
            f"彻底没命中（前缀不在任何一层）——这一档的数不能用"
        )
    if missed:
        print(
            "     两个已知原因：① 存储上限小于前缀池，驱逐器把早写进去的页删了"
            "（启动时已校验，一般不会）；② host 暂存不够——并发数 x 命中前缀"
            "超过 host 池时，prefetch 拿不到 host 内存就被静默撤销"
            "（revoke_pending_prefetch 不打日志），请求退化成全量重算。"
        )
    for rec in records:
        if rec["stats_matched"] < len(rec["per_request"]):
            print(
                f"  !! {rec['policy']}@{rec['concurrency']}: 只匹配到 "
                f"{rec['stats_matched']}/{len(rec['per_request'])} 条时间统计"
            )
    print("=" * 108)


def main():
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--server", default="l3_nixl", choices=("l3_fused", "l3_nixl"))
    extra.add_argument("--concurrency", default="1,2,4,8")
    extra.add_argument("--policies", default="l2,mixed,l3")
    extra.add_argument(
        "--hot-prefixes",
        type=int,
        default=15,
        help="比设备池大（不是 L1 命中）、比 host 池小（留在 L2）",
    )
    extra.add_argument(
        "--cold-prefixes",
        type=int,
        default=60,
        help="host 池的两倍；循环前半之后，后半必然已被驱逐",
    )
    extra.add_argument("--requests", type=int, default=24, help="每档计入统计的请求数")
    extra.add_argument(
        "--warmup-requests",
        type=int,
        default=0,
        help="每档开头不计入统计的请求；0 表示按策略自动算",
    )
    extra.add_argument("--suffix-tokens", type=int, default=512)
    extra.add_argument("--background-requests", type=int, default=6)
    extra.add_argument("--background-input-tokens", type=int, default=512)
    extra.add_argument("--background-output-tokens", type=int, default=100000)
    extra.add_argument("--background-warmup-s", type=float, default=8.0)
    known, rest = extra.parse_known_args()
    sys.argv = [sys.argv[0]] + rest

    args = B.parse_args()
    for key, value in vars(known).items():
        setattr(args, key, value)
    args.hit_tokens = int(str(args.hit_tokens).split(",")[0])
    concurrency = [int(c) for c in args.concurrency.split(",") if c]
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    patch_base_args(args.background_requests + max(concurrency) + 2)

    rng = random.Random(args.seed)
    run_dir = B.RESULTS_ROOT / (
        args.run_id or f"sweep-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / f"{args.server}.log"
    store_root = Path(args.store_root) / B.store_of_server(args.server)
    B.log(f"运行目录 {run_dir}")

    pool_rng = random.Random(args.seed + 1)
    hot = [B.gen_ids(args.hit_tokens, pool_rng) for _ in range(args.hot_prefixes)]
    cold = [B.gen_ids(args.hit_tokens, pool_rng) for _ in range(args.cold_prefixes)]

    proc, fh = B.launch(args.server, args, args.port, log_path)
    records = []
    try:
        B.wait_health(args.port, proc)
        B.log("服务端就绪")
        total = len(hot) + len(cold)
        expect_kv = args.hit_tokens * args.kv_bytes_per_token * total
        limit = parse_size(args.store_max_size)
        if limit and expect_kv > limit * 0.9:
            raise SystemExit(
                f"前缀池要 {expect_kv / 2**30:.0f} GiB，而 --store-max-size 是 "
                f"{limit / 2**30:.0f} GiB。驱逐器会在预热过程中就把早写进去的页"
                f"删掉，于是计入统计的请求变成彻底 miss。把上限调大。"
            )
        B.log(
            f"预热 {total} 条前缀（{args.hot_prefixes} 热 + {args.cold_prefixes} 冷）"
        )
        B.flush_cache(args.port)
        before_store = B.store_bytes(store_root)[0]
        for index, prefix in enumerate(hot + cold):
            B.warm_prefix(args.port, prefix, rng, args.warm_step)
            if (index + 1) % 8 == 0:
                B.log(f"  预热 {index + 1}/{total}")
        grew = (
            B.wait_writeback(
                store_root,
                min_bytes=before_store + int(expect_kv * 0.9),
                timeout=3600.0,
            )
            - before_store
        )
        B.log(f"存储增长 {grew / 2**30:.2f} GiB")

        background = Background(
            args.port,
            args.background_requests,
            args.background_input_tokens,
            args.background_output_tokens,
            rng,
        )
        B.log(f"启动 {args.background_requests} 路背景请求")
        background.start()
        time.sleep(args.background_warmup_s)

        for conc in concurrency:
            for policy in policies:
                B.log(f"--- 并发 {conc}，流 {policy} ---")
                rec = run_phase(policy, conc, args.port, args, rng, hot, cold, log_path)
                a = rec["all"]
                B.log(
                    f"    L1/L2/L3={rec['l1']['n']}/{rec['l2']['n']}/{rec['l3']['n']}"
                    f"  TTFT均值={a['ttft_mean']:.0f}ms"
                    f"  queue={a['queue_mean']:.0f}  forward={a['forward_mean']:.0f}"
                )
                records.append(rec)
                (run_dir / "summary.json").write_text(json.dumps(records, indent=2))
        background.shutdown()
    finally:
        B.shutdown(proc, fh)

    (run_dir / "summary.json").write_text(json.dumps(records, indent=2))
    report(records, args)
    print(f"\n原始数据: {run_dir}")


if __name__ == "__main__":
    main()
