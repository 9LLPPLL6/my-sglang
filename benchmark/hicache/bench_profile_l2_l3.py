#!/usr/bin/env python3
"""Capture one torch profiler trace per tier, for comparing the two code paths.

Same steady-state scenario as `bench_steady_state_l3_probe.py` -- background
requests decoding, probes arriving into that -- but the point here is not a
number. It is two traces taken under identical server settings that differ
only in which tier the probes' prefix is read from, so a diff between them is
a diff between the L2 and L3 code paths and nothing else.

`with_stack` is on, which is the whole reason to run this and also what makes
the run useless for timing: the overhead is large and uneven. Read these
traces for what code runs, not for how long it takes.

The profiling window is opened immediately before the probes are fired and
closed as soon as the last one returns, because the background population
keeps decoding the whole time and every one of its iterations lands in the
trace too.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, "/home/lpl/sglang/benchmark/hicache")
import bench_l2l3_fusion_ttft as B  # noqa: E402
import bench_steady_state_l3_probe as S  # noqa: E402

SERVER = "l3_fused"


def start_profile(port: int, output_dir: Path, prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "output_dir": str(output_dir),
        "activities": ["CPU", "GPU"],
        "with_stack": True,
        "record_shapes": True,
        "merge_profiles": True,
        "profile_prefix": prefix,
    }
    resp = requests.post(
        f"http://127.0.0.1:{port}/start_profile", json=payload, timeout=120
    )
    resp.raise_for_status()
    B.log(f"  profiler started -> {output_dir}")


def stop_profile(port: int) -> None:
    # Export plus merge, with stacks, over a few seconds of two ranks: this is
    # the slow part of the whole script, not the probes.
    resp = requests.post(f"http://127.0.0.1:{port}/stop_profile", timeout=3600)
    resp.raise_for_status()
    B.log("  profiler stopped, traces written")


def run_arm(arm: str, port: int, args, rng, store_root: Path, trace_root: Path) -> dict:
    hit = args.hit_tokens
    prefixes = [B.gen_ids(hit, rng) for _ in range(args.probes)]
    expect_kv = hit * args.kv_bytes_per_token * args.probes

    B.log(f"[{arm}] warming {args.probes} x {hit}-token prefixes")
    B.flush_cache(port)
    before_store = B.store_bytes(store_root)[0]
    for i, prefix in enumerate(prefixes):
        B.warm_prefix(port, prefix, rng, args.warm_step)
    grew = (
        B.wait_writeback(
            store_root, min_bytes=before_store + int(expect_kv * 0.9), timeout=1800.0
        )
        - before_store
    )
    B.log(f"[{arm}] store grew {grew / 2**30:.2f} GiB")

    if arm == "l2":
        B.evict_device(port, args, rng)
    else:
        B.flush_cache(port)
        B.make_store_cold(
            store_root,
            budget_bytes=int(expect_kv * 0.02),
            churn_path=Path(args.churn_file),
            churn_gib=args.churn_gib,
        )

    background = S.Background(
        port,
        args.background_requests,
        args.background_input_tokens,
        args.background_output_tokens,
        rng,
    )
    B.log(f"[{arm}] starting {args.background_requests} background generations")
    background.start()
    time.sleep(args.background_warmup_s)

    suffixes = [B.gen_ids(args.suffix_tokens, rng) for _ in prefixes]
    probes: list = [None] * len(prefixes)

    def fire(index: int):
        res = B.generate(port, list(prefixes[index]) + suffixes[index])
        return index, {
            "ttft_ms": round(res["ttft_ms"], 1),
            "tier": B.tier(res["meta"]),
        }

    start_profile(port, trace_root / f"trace-{arm}", arm)
    B.log(f"[{arm}] firing {args.probes} probes, all at once")
    wall = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.probes) as pool:
        for index, rec in pool.map(fire, range(len(prefixes))):
            probes[index] = rec
    wall = time.perf_counter() - wall
    stop_profile(port)

    bg = background.stats()
    background.shutdown()
    complaints = S.check_tiers(arm, probes, hit)
    for complaint in complaints:
        B.log(f"  !! [{arm}] {complaint}")
    B.log(f"[{arm}] probe wall {wall:.2f}s, TTFT {[p['ttft_ms'] for p in probes]}")
    return {
        "arm": arm,
        "probe_wall_s": round(wall, 2),
        "background": bg,
        "per_probe": probes,
        "complaints": complaints,
    }


def main() -> None:
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--probes", type=int, default=8)
    extra.add_argument("--suffix-tokens", type=int, default=512)
    extra.add_argument("--background-requests", type=int, default=6)
    extra.add_argument("--background-input-tokens", type=int, default=512)
    extra.add_argument("--background-output-tokens", type=int, default=20000)
    extra.add_argument("--background-warmup-s", type=float, default=8.0)
    extra.add_argument(
        "--trace-root",
        default="/zion0/kv-aio-bench/traces",
        help="traces are large with with_stack; keep them off the root disk",
    )
    known, rest = extra.parse_known_args()
    sys.argv = [sys.argv[0]] + rest

    args = B.parse_args()
    for key, value in vars(known).items():
        setattr(args, key, value)
    args.hit_tokens = int(str(args.hit_tokens).split(",")[0])
    S.patch_base_args(args.background_requests + args.probes + 1)

    rng = random.Random(args.seed)
    run_dir = B.RESULTS_ROOT / (args.run_id or f"prof-{time.strftime('%Y%m%d-%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / f"{SERVER}.log"
    store_root = Path(args.store_root) / B.store_of_server(SERVER)
    trace_root = Path(args.trace_root) / run_dir.name

    B.log(f"run dir {run_dir}")
    B.log(f"trace root {trace_root}")
    proc, fh = B.launch(SERVER, args, args.port, log_path)
    records = []
    try:
        B.wait_health(args.port, proc)
        B.log("server healthy")
        for arm in ("l2", "l3"):
            records.append(
                run_arm(arm, args.port, args, rng, store_root, trace_root)
            )
    finally:
        B.shutdown(proc, fh)

    (run_dir / "summary.json").write_text(json.dumps(records, indent=2))
    print(f"\n原始数据: {run_dir}\ntrace: {trace_root}")


if __name__ == "__main__":
    main()
