# Layerwise HiCache storage (L2/L3 fusion)

Goal: make an L3 hit feel like an L2 hit. Not by moving data into DRAM, but by
making the storage read fast enough and overlapped enough that nothing above
HiCache has to know an L3 tier exists.

Two paths live here, sharing one on-disk format:

| path | status | what it does |
|---|---|---|
| `--hicache-storage-backend layerwise_file` | working, measured | whole-prefix L3 reads with `O_DIRECT` + Linux AIO, landing straight in the host KV pool |
| `--hicache-storage-load-mode layerwise` | components built and unit-tested; not yet wired into `UnifiedRadixCache` | per-layer-group streaming so storage read, H2D and forward overlap |

## Measured, Qwen3-8B TP=1, page 64, 8.8k-token cached prefix

Page cache dropped before every L3 read (`posix_fadvise(DONTNEED)`), otherwise a
buffered backend is measuring RAM, not storage.

| | TTFT |
|---|---|
| recompute the prefix | 931 ms |
| L3 hit, `file` backend | 924 ms |
| L3 hit, `layerwise_file` backend | 444 ms |
| L1/L2 hit | 51 ms |

The stock file backend saved nothing over recomputing. `layerwise_file` is 2.1x
faster than it and 2.1x faster than recompute. The remaining 444 -> 51 ms is
what the streaming path targets.

Storage microbenchmark: `benchmark/hicache/bench_layerwise_storage.py`. On one
NVMe the layer-range read reaches 94% of the raw whole-page Direct AIO ceiling
at `--group-size 8`, and collapses to 53% at `--group-size 2` — group size is
the knob that decides whether range reads cost anything.

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

Padding exists only on disk and is never treated as host capacity. Identity
(model fingerprint, TP size/rank, dtype, geometry, offsets, checksum) lives in
the header, never in the filename, so a format change never renames files. A
reader that cannot fully identify a page fails closed and recomputes.

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
| `consensus.py` | non-blocking per-group cross-rank agreement |
| `pipeline.py` | the streaming driver: drain, agree, hand off, read ahead |
| `controller.py` | scheduler-facing entry point tying it all together |

## Configuration

```
--hicache-storage-backend layerwise_file        # the fast whole-prefix path
--hicache-io-backend direct
--hicache-mem-layout page_first_direct
--hicache-write-policy write_through
--hicache-host-memory-mode cache

SGLANG_HICACHE_LAYERWISE_ROOT=/mnt/parallel-fs/sglang-hicache
SGLANG_HICACHE_LAYERWISE_MAX_SIZE=2Ti          # empty or 0 = unlimited
SGLANG_HICACHE_LAYERWISE_MIN_FREE_SPACE=8Gi    # 0 disables the watermark
```

The free-space watermark defaults to non-zero on purpose: losing a cached page
costs a recompute, filling the device kills the scheduler process.

Streaming knobs (`--hicache-storage-load-mode layerwise`) are validated but
inert until the pipeline is wired into the radix cache:
`--hicache-storage-first-group-layers`, `--hicache-storage-group-size`,
`--hicache-storage-read-ahead-groups`, `--hicache-storage-group-timeout-ms`,
`--hicache-storage-admission-budget-ms`, `--hicache-storage-max-inflight-bytes`,
`--hicache-storage-slow-fallback`.

## What the streaming path still needs

The pipeline, its backend, the H2D session (`L2TransferEngine`
`begin/submit_range/finish`) and the generation-aware layer gate all exist and
are tested. What is missing is the wiring in `UnifiedRadixCache`:

1. Route a storage hit to `LayerwiseStorageController.begin` instead of the
   blocking prefetch thread, and gate admission on group 0 only.
2. Keep the host staging out of the radix tree until every group is agreed and
   the forward pass consumed it, then publish it as an ordinary L2 node.
3. Release device slots through `LoadBackOwnership.REQUEST` on abort, and host
   staging only once the backend reports the read terminal.
4. Multi-request streaming, mixing with running decode, and the poison/replay
   path for a failure after admission.
