# Layerwise HiCache storage (L2/L3 fusion)

Goal: make an L3 hit feel like an L2 hit. Not by moving data into DRAM, but by
making the storage read fast enough and overlapped enough that nothing above
HiCache has to know an L3 tier exists.

Two paths live here, sharing one on-disk format:

| path | status | what it does |
|---|---|---|
| `--hicache-storage-backend layerwise_file` | working, measured | whole-prefix L3 reads with `O_DIRECT` + Linux AIO, landing straight in the host KV pool |
| `--hicache-storage-load-mode layerwise` | working for one streaming request at a time; validated on TP=1 and TP=2 | per-layer-group streaming so the storage read, H2D and forward overlap |

Stage write-ups with the full methodology live in `L2L3fusion-docs/`.

## Measured, Qwen3-8B, page 64, 8.8k-token cached prefix

Page cache dropped before every L3 read (`posix_fadvise(DONTNEED)`), otherwise a
buffered backend measures RAM, not storage.

Backend comparison, TP=1, short suffix:

| | TTFT |
|---|---|
| recompute the prefix | 931 ms |
| L3 hit, `file` backend | 924 ms |
| L3 hit, `layerwise_file` backend | 444 ms |
| L1 / L2 hit (indistinguishable) | 51 ms |

The stock file backend saved nothing over recomputing. `layerwise_file` is 2.1x
faster than it and 2.1x faster than recompute.

Pipeline A/B with the same backend on both sides, so the difference is the
pipeline alone:

| suffix | TP=1 full_wait | TP=1 layerwise | TP=2 full_wait | TP=2 layerwise |
|---|---|---|---|---|
| ~6 tok | 462 ms | 390 ms | 416 ms | 384 ms |
| ~2100 tok | 731 ms | 460 ms | 598 ms | 438 ms |
| ~4200 tok | 1058 ms | 751 ms | | |

Correctness is checked as L1-hit vs L3-streamed, not recompute vs L3: recompute
and a cached-prefix run are not bit-identical (different kernel paths), so
comparing against recompute reports false failures. L1 vs L3 was identical 3/3
on both TP=1 and TP=2.

## When the read is hidden

Per layer the pipeline needs read <= compute; multiplied by the layer count that
collapses to "total read time <= suffix compute time":

```
 S       b * R
--- >=  -------
 H        BW
```

`H` = hit tokens, `S` = newly computed tokens, `b` = KV bytes per token per rank
(`2 * layers * local_kv_heads * head_dim * dtype_bytes`), `R` = suffix prefill
rate, `BW` = per-rank read bandwidth. The right side does not involve the prefix
length, so the condition is a fixed ratio.

Measured with `H` pinned at 8832: the cost of going through L3 over pure compute
falls from 346 ms to a floor of ~90 ms, knee at `S/H ~ 0.3`, which is what the
formula predicts for this box (0.24-0.29). The ~90 ms floor is group 0 plus
admission plus the last group's tail -- about 27% of the full read, so the
pipeline hides at best 73% of it.

TP barely moves the ratio: `b` shrinks and `R` grows together. Only real
per-rank storage bandwidth lowers it.

## Picking `--hicache-storage-group-size`

This box's root NVMe tops out at 3.45 GiB/s and ~28.2k IOPS (fio, read-only,
flat across 128k/512k/2m at QD>=16). It is PCIe-limited, not flash-limited: the
Samsung 990 PRO negotiated Gen3 x4 against its Gen4 capability.

| group_size | extent size | achieved | of ceiling |
|---|---|---|---|
| 1 | 64 KiB | 1.72 GiB/s | 50% |
| 2 | 128 KiB | 1.91 GiB/s | 55% |
| 4 | 256 KiB | 2.76 GiB/s | 80% |
| 8+ | 512 KiB+ | 3.26 GiB/s | 94% |
| whole page | 2.36 MiB | 3.44 GiB/s | 99.7% |

`group_size=1` lands exactly on the IOPS ceiling (28.2k x 64 KiB = 1.72 GiB/s):
small extents hit the IOPS wall instead of the bandwidth wall. The knob's job is
to grow the extent past that. Beyond 8 layers there is nothing left to gain
here. Re-run `benchmark/hicache/bench_layerwise_storage.py` on the target
storage; the best point differs per device.

## On-disk format

One logical page is one file. Its payload is byte-identical to the host pool's
`page_first_direct` flat page, so nothing repacks KV on either path:

```
[aligned header]
K[layer 0 .. N-1]     each layer: page_size x local_kv_heads x head_dim
[padding to alignment]
V[layer 0 .. N-1]
[padding to alignment]
```

Splitting a page into a K extent and a V extent is what makes the read
zero-copy: a page's K and V are half a buffer apart in the host pool, so one
read can never fill both, but each side alone is contiguous in memory and on
disk. Verified aligned for TP=1/2/4/8 at every layer-group boundary, so the
Direct I/O target is the KV pool address itself and no bounce buffer is used.

Padding exists only on disk and is never treated as host capacity. Identity
(model fingerprint, TP size/rank, dtype, geometry, offsets, checksum) lives in
the header, never in the filename, so a format change never renames files.

The read path does not verify that header yet. It relies on the directory tree
for identity and computes offsets from the running process's layout, so a page
moved between trees -- or one whose rename became visible before its payload
reached the device, since the writer does not fsync -- would be loaded as valid
KV. `decode_header` and `PageFileWriter.read_layout` implement the check and
have no callers.

Path layout — the directory tree partitions only on what it must:

```
<root>/format-v1/<fingerprint>/tp-<size>/rank-<rank>/<bucket>/<page-hash>.kv
```

## Modules

| module | responsibility |
|---|---|
| `page_format.py` | header, geometry, path layout, fingerprint |
| `page_writer.py` | write-through publish (temp file + atomic rename) |
| `page_store_evictor.py` | byte cap and free-space watermark over the page tree |
| `aio_engine.py` | `O_DIRECT` probing, aligned buffers, raw Linux AIO syscalls |
| `io_arbiter.py` | priority arbitration: admission > demand > read-ahead > write-through |
| `file_backend.py` | `LayerwiseStorageBackend` over the above |
| `plan_builder.py` | host geometry + file geometry -> ordered layer-group read plan |
| `types.py` / `backend.py` | value types and the async backend interface |
| `state_machine.py` | transaction and group lifecycles, private-buffer ownership |
| `pipeline.py` | the streaming driver: drain, agree, hand off, read ahead |
| `controller.py` | ties the pipeline to one page store and one H2D session |
| `radix_bridge.py` | the `UnifiedRadixCache` glue: take over a hit, gate admission, consume at load-back, release |
| `consensus.py` | the per-group agreement interface. Only `SingleRankConsensus` is used: cross-rank agreement happens once at admission, in `radix_bridge`, off the forward path. `TorchDistGroupConsensus` has no callers |

## Configuration

```
--hicache-storage-backend layerwise_file        # the fast whole-prefix path
--hicache-io-backend direct
--hicache-mem-layout page_first_direct
--hicache-write-policy write_through
--hicache-host-memory-mode cache

# add these for layer-group streaming
--hicache-storage-load-mode layerwise
--hicache-storage-first-group-layers 1          # the one group nothing can hide
--hicache-storage-group-size 8
--hicache-storage-read-ahead-groups 2           # in-flight groups = this + 1
--disable-overlap-schedule
--chunked-prefill-size -1

SGLANG_HICACHE_LAYERWISE_ROOT=/mnt/parallel-fs/sglang-hicache
SGLANG_HICACHE_LAYERWISE_MAX_SIZE=2Ti          # empty or 0 = unlimited
SGLANG_HICACHE_LAYERWISE_MIN_FREE_SPACE=8Gi    # 0 disables the watermark
```

The free-space watermark defaults to non-zero on purpose: losing a cached page
costs a recompute, filling the device kills the scheduler process.

Streaming requires the `layerwise_file` backend, since the reader and the
write-through writer have to agree on page identity and on-disk layout. Startup
logs `Layerwise storage streaming enabled: ...`; a
`Layerwise storage streaming disabled: unsupported ...` line means it fell
closed to the ordinary blocking read, and says why.

Streaming falls closed for a non-MHA host pool, side or sidecar pools, more than
one cache component, buffer-only host memory, or any other backend.

## What streaming still needs

1. **More than one streaming transaction at a time.** A second concurrent
   storage hit silently uses the blocking path. Verified not to hang or diverge
   ranks under three concurrent hits on TP=2, but it does not accelerate.
2. **A storage failure after admission.** The device slots are already
   published and partly filled, so the pump raises rather than serving KV it
   never read. The plan's poison/replay is not implemented.
3. **Fault injection on real ranks.** The admission decision is MIN-reduced
   across ranks and unit-tested against a simulated peer, but a mid-transaction
   read failure has not been injected on two live ranks.
4. **Mixing with running decode**, and a real mixed read/write load through the
   arbiter. Both are untested.
5. **MLA / SWA / Mamba**, held out by the fail-closed guard.

Items 1 and 2 are the hard gates before this is serving-ready.
