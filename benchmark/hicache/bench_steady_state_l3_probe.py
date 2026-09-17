#!/usr/bin/env python3
"""L3 prefetch against a server that is already busy.

The burst benchmark (`bench_concurrent_l2_vs_l3.py`) fires every request at
once with `max_new_tokens=1`, so nothing is ever mid-inference: each request's
prefetch has no ongoing work to hide behind and the GPU idles while storage is
read.  That is the wrong shape for asking whether an L3 read can be overlapped.

Here a background population decodes continuously, and probes arrive one at a
time into that:

    background  ████ decode ████████ decode ████████ decode ████
    probe            ^ enqueued, prefetch starts
                     |     <---- GPFS read ---->        the GPU is busy
                     +-- prefetch done -> admitted -> prefill -> first token

`--hicache-storage-prefetch-policy wait_complete` (set by the shared bench)
makes the probe wait in the queue until its prefetch finishes, so the read time
*is* the queue time -- and the background decode is what fills it.

Two arms differ only in where the probe's prefix lives:

  l2  -- warm it, evict HBM only, so the hit is served from host memory
  l3  -- warm it, flush HBM+host, so the hit must come from storage

Each probe gets its own prefix, so one probe never reads a prefix another just
pulled in.  The tier attribution of every probe is asserted, because the thing
most likely to invalidate this run is the background load evicting the probe
prefixes out of L2.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
import threading
import time
from pathlib import Path

import requests

sys.path.insert(0, "/home/lpl/sglang/benchmark/hicache")
import bench_l2l3_fusion_ttft as B  # noqa: E402


def patch_base_args(max_running: int) -> None:
    """The shared bench pins --max-running-requests to 1 for single-request
    attribution; the background population alone needs more than that."""
    original = B.base_args

    def patched(args, port):
        argv = original(args, port)
        argv[argv.index("--max-running-requests") + 1] = str(max_running)
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


class Background:
    """A population of long-generation requests, kept decoding on its own threads.

    Each worker holds one streaming request open and abandons it when asked to
    stop, so the population is steady for the whole probe phase rather than
    re-prefilling on a loop.
    """

    def __init__(self, port, workers, input_tokens, output_tokens, rng):
        self.port = port
        self.workers = workers
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.rng = rng
        self.stop = threading.Event()
        self.threads = []
        self.lock = threading.Lock()
        self.tokens = 0
        self.first_token_s = None
        self.last_token_s = None

    def _worker(self, input_ids):
        payload = {
            "input_ids": input_ids,
            "sampling_params": {
                "max_new_tokens": self.output_tokens,
                "temperature": 0.0,
                "ignore_eos": True,
            },
            "stream": True,
        }
        try:
            with requests.post(
                f"http://127.0.0.1:{self.port}/generate",
                json=payload,
                stream=True,
                timeout=3600,
            ) as resp:
                resp.raise_for_status()
                for raw in resp.iter_lines(decode_unicode=True):
                    if self.stop.is_set():
                        break
                    if not raw or not raw.startswith("data:"):
                        continue
                    now = time.perf_counter()
                    with self.lock:
                        self.tokens += 1
                        if self.first_token_s is None:
                            self.first_token_s = now
                        self.last_token_s = now
        except requests.RequestException:
            pass

    def start(self):
        for _ in range(self.workers):
            input_ids = B.gen_ids(self.input_tokens, self.rng)
            thread = threading.Thread(target=self._worker, args=(input_ids,))
            thread.daemon = True
            thread.start()
            self.threads.append(thread)

    def shutdown(self):
        """Stop generating, and make the server stop too.

        Closing the client connection is not enough: the request keeps
        generating server-side for the rest of its max_new_tokens, and the next
        arm's `flush_cache` refuses to run while anything is in flight. That is
        what failed the first run of this benchmark.
        """
        self.stop.set()
        try:
            requests.post(
                f"http://127.0.0.1:{self.port}/abort_request",
                json={"abort_all": True},
                timeout=30,
            )
        except requests.RequestException:
            pass
        for thread in self.threads:
            thread.join(timeout=30)
        # flush_cache only succeeds once the engine is idle, and it already
        # retries; use it as the drain barrier rather than a second idle probe.
        B.flush_cache(self.port)

    def stats(self):
        with self.lock:
            span = (
                (self.last_token_s - self.first_token_s)
                if self.first_token_s is not None and self.last_token_s is not None
                else 0.0
            )
            return {
                "workers": self.workers,
                "tokens": self.tokens,
                "span_s": round(span, 2),
                # Tokens per second across the whole population; divide by the
                # worker count for a per-request rate.
                "tokens_per_s": round(self.tokens / span, 1) if span > 0 else 0.0,
            }


def check_tiers(arm, probes, hit_tokens):
    """The background load evicting a probe prefix out of L2 is the failure this
    run is most exposed to, and it is silent: the probe simply reads from a
    lower tier. Say so loudly instead of reporting its TTFT as an L2 number."""
    complaints = []
    for index, probe in enumerate(probes):
        tier = probe["tier"]
        if arm == "l2":
            if tier["storage"] > 0:
                complaints.append(
                    f"probe {index}: storage={tier['storage']} -- the prefix was "
                    "evicted out of L2, so this is not an L2 measurement"
                )
            if tier["host"] < hit_tokens:
                complaints.append(f"probe {index}: host={tier['host']} < {hit_tokens}")
        else:
            if tier["device"] > 0:
                complaints.append(f"probe {index}: device={tier['device']} -- HBM leak")
            if tier["host"] + tier["storage"] < hit_tokens:
                complaints.append(
                    f"probe {index}: hit {tier['host'] + tier['storage']} "
                    f"< {hit_tokens}"
                )
    return complaints


def run_arm(arm, port, proc, args, rng, store_root, log_path):
    hit = args.hit_tokens
    prefixes = [B.gen_ids(hit, rng) for _ in range(args.probes)]
    expect_kv = hit * args.kv_bytes_per_token * args.probes

    B.log(f"[{arm}] warming {args.probes} x {hit}-token probe prefixes")
    B.flush_cache(port)
    before_store = B.store_bytes(store_root)[0]
    for i, prefix in enumerate(prefixes):
        B.warm_prefix(port, prefix, rng, args.warm_step)
        if (i + 1) % 4 == 0:
            B.log(f"  warmed {i + 1}/{len(prefixes)}")
    grew = (
        B.wait_writeback(
            store_root, min_bytes=before_store + int(expect_kv * 0.9), timeout=1800.0
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

    # Background starts only now: before the arm prep its writes would be
    # flushed away, and during the prep they would muddy the tier attribution.
    background = Background(
        port,
        args.background_requests,
        args.background_input_tokens,
        args.background_output_tokens,
        rng,
    )
    B.log(f"[{arm}] starting {args.background_requests} background generations")
    background.start()
    time.sleep(args.background_warmup_s)

    before_io = B.tree_read_bytes(proc.pid)
    before_log = B.log_counts(log_path)
    log_offset = log_path.stat().st_size

    B.log(f"[{arm}] firing {args.probes} probes, one at a time")
    probes = []
    for index, prefix in enumerate(prefixes):
        suffix = B.gen_ids(args.suffix_tokens, rng) if args.suffix_tokens else []
        res = B.generate(port, list(prefix) + suffix)
        probes.append(
            {"ttft_ms": round(res["ttft_ms"], 1), "tier": B.tier(res["meta"])}
        )
        B.log(f"  probe {index + 1}/{args.probes}: {probes[-1]['ttft_ms']:.0f} ms")
        time.sleep(args.probe_gap_s)

    bg_stats = background.stats()
    background.shutdown()

    reads, _ = B.read_lines_since(log_path, log_offset)
    complaints = check_tiers(arm, probes, hit)
    for complaint in complaints:
        B.log(f"  !! [{arm}] {complaint}")

    return {
        "arm": arm,
        "hit_tokens": hit,
        "suffix_tokens": args.suffix_tokens,
        "probes": args.probes,
        "background": bg_stats,
        "disk_read_bytes": B.tree_read_bytes(proc.pid) - before_io,
        "expect_kv_bytes": expect_kv,
        "log_delta": {
            k: B.log_counts(log_path)[k] - before_log.get(k, 0) for k in before_log
        },
        "prep": prep,
        "reads": B.summarize_reads(reads),
        "complaints": complaints,
        "per_probe": probes,
    }


def report(records):
    print()
    print("=" * 100)
    print(
        f"{'arm':<5}{'n':>4}{'  probe TTFT p50':>17}{'p90':>9}{'max':>9}"
        f"{'  背景 tok/s':>14}{'  读盘':>10}{'  归属异常':>10}"
    )
    print("-" * 100)
    base = None
    for rec in records:
        t = [p["ttft_ms"] for p in rec["per_probe"]]
        print(
            f"{rec['arm']:<5}{len(t):>4}{st.median(t):>17.1f}{pct(t, 90):>9.1f}"
            f"{max(t):>9.1f}{rec['background']['tokens_per_s']:>14.1f}"
            f"{rec['disk_read_bytes'] / 2**30:>8.2f}G{len(rec['complaints']):>10}"
        )
        if rec["arm"] == "l2":
            base = st.median(t)
    if base is not None:
        print()
        for rec in records:
            if rec["arm"] == "l2":
                continue
            t = [p["ttft_ms"] for p in rec["per_probe"]]
            median = st.median(t)
            print(
                f"  {rec['arm']} 相对 L2: {median - base:+.0f} ms "
                f"({median / base:.2f}x)   "
                f"读盘 {rec['disk_read_bytes'] / 2**30:.2f} GiB / "
                f"预期 {rec['expect_kv_bytes'] / 2**30:.2f} GiB"
            )
            if rec["reads"]:
                print(f"    后端自报读取: {json.dumps(rec['reads'])}")
            print(f"    日志计数: {rec['log_delta']}")
    for rec in records:
        for complaint in rec["complaints"]:
            print(f"  !! [{rec['arm']}] {complaint}")
    print("=" * 100)


def main():
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument(
        "--server",
        default="l3_fused",
        choices=("l3_fused", "l3_nopipe"),
        help="l3_fused streams layer groups; l3_nopipe reads the whole prefix "
        "and is the control for how much the pipeline itself costs",
    )
    extra.add_argument("--probes", type=int, default=8)
    extra.add_argument("--suffix-tokens", type=int, default=5120)
    extra.add_argument("--background-requests", type=int, default=6)
    extra.add_argument("--background-input-tokens", type=int, default=512)
    extra.add_argument("--background-output-tokens", type=int, default=20000)
    extra.add_argument("--background-warmup-s", type=float, default=8.0)
    extra.add_argument("--probe-gap-s", type=float, default=2.0)
    known, rest = extra.parse_known_args()
    sys.argv = [sys.argv[0]] + rest

    args = B.parse_args()
    for key, value in vars(known).items():
        setattr(args, key, value)
    args.hit_tokens = int(str(args.hit_tokens).split(",")[0])
    patch_base_args(args.background_requests + 2)

    rng = random.Random(args.seed)
    run_dir = B.RESULTS_ROOT / (
        args.run_id or f"steady-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / f"{args.server}.log"
    store_root = Path(args.store_root) / B.store_of_server(args.server)

    B.log(f"run dir {run_dir}")
    proc, fh = B.launch(args.server, args, args.port, log_path)
    records = []
    try:
        B.wait_health(args.port, proc)
        B.log("server healthy")
        for arm in ("l2", "l3"):
            rec = run_arm(arm, args.port, proc, args, rng, store_root, log_path)
            records.append(rec)
            (run_dir / f"{arm}.json").write_text(json.dumps(rec, indent=2))
    finally:
        B.shutdown(proc, fh)

    (run_dir / "summary.json").write_text(json.dumps(records, indent=2))
    report(records)
    print(f"\n原始数据: {run_dir}")


if __name__ == "__main__":
    main()
