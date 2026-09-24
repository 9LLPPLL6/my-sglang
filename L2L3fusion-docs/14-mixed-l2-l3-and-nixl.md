# 阶段 14：混合命中（一半 L2、一半 L3）对照纯 L2，以及把 GPFS 接上 NIXL

| | |
|---|---|
| 日期 | 2026-09-23 |
| 模型 | Llama-3.3-70B-Instruct-FP8，TP=2 |
| 存储 | GPFS `/zion0`，O_DIRECT |
| 配置 | page 512、命中前缀 16384、后缀 512、8 探针并发、其中 4 个走 L3、背景 6 路、`io_threads` 8、`max_concurrent_streams` 8、`max_total_tokens` 163840 |
| 脚本 | `benchmark/hicache/bench_mixed_l2_l3.py` |
| 前一阶段 | [13 并发准入](13-why-all-eight-admit-together.md) |

> **结论一**：**挂上 L3 不会拖慢那些本来就命中 L2 的请求。** 它们的排队时间
> 三个臂完全一致：7.8 / 7.9 / 7.8 ms。
>
> **结论二**：混合臂的 TTFT 看起来"更快"（1727 vs 2112）是**批次假象**——
> 同样 8 个请求被拆成了两批，而 forward 与批次大小成正比（H2D 带宽受限）。
>
> **结论三**：混合的真实代价落在走 L3 的那 4 个请求身上（queue 1250 ms），
> 而其中 **89% 是"读还没被提交"**——阶段 B 被 L2 那批的 forward 堵在调度线程上。
> 读本身只要 31.7 ms。
>
> **结论四**：**NIXL 与 Linux AIO 基本持平**（带宽 39.58 vs 40.81 GiB/s），
> 只有 open 时间 NIXL 明显更好（3.10 vs 7.64 ms）。

---

## 1. 三个臂

| 臂 | 服务端 | 说明 |
|---|---|---|
| `l2_only` | `mem` | L1+L2，**不开存储后端**。8 个探针全部命中 L2。 |
| `mixed_aio` | `l3_fused` | L1+L2+L3，layerwise 流水线，range read 走 Linux AIO。4 个 L2 + 4 个 L3。 |
| `mixed_nixl` | `l3_nixl` | 同上，range read 走 **NIXL POSIX**。 |

对照组是**根本没有 L3 后端**的服务端，不是"同一个服务端但不读 L3"——
所以每个臂各起一次服务端。

`mixed_aio` 是我自己加的对照，不在原始需求里：NIXL 引擎是新写的，
没有它的话一旦混合臂变慢，分不清是"混合"造成的还是"nixl"造成的。

### 混合命中怎么造出来

不需要按前缀逐条驱逐，用现有原语就能拼：

1. `flush_cache`，预热全部 8 条前缀（`write_through` 把它们写进 L3）
2. 等写回完成（实测正好 20.00 GiB）
3. `flush_cache` —— L1 和 L2 清空，**L3 上的 8 条还在**
4. `make_store_cold` —— 打冷页缓存，否则"读 L3"读的是内存
5. 重新预热前 4 条 —— 它们从 L3 读回，重新进入 L1+L2
6. `evict_device` —— 把这 4 条挤出 HBM，host 池是 6 倍大所以副本留在 L2

归属是逐探针校验的，8/8 精确：前 4 个 `host=16384 storage=0`，
后 4 个 `storage=16384 host=0`，零 HBM 污染，`streamed=8 fell_back=0`，
磁盘读正好 10.00 GiB（= 4 × 2.5 GiB），一分不差。

---

## 2. 总体（均值，n=8）

| 臂 | TTFT | queue | forward | 背景 tok/s |
|---|---|---|---|---|
| `l2_only` | 2112.1 | **7.8** | 2082.8 | 59.4 |
| `mixed_aio` | 1727.3 | 641.7 | 1046.8 | 41.8 |
| `mixed_nixl` | 1754.9 | 628.4 | 1049.5 | 43.1 |

**混合臂的 TTFT 更低，但这不是好事，是个假象。** 见 §4。

## 3. 分子集（均值，n=4）

| 臂 | L2:TTFT | **L2:queue** | L2:forward | L3:TTFT | **L3:queue** | L3:forward |
|---|---|---|---|---|---|---|
| `l2_only` | 2112.1 | **7.8** | 2082.8 | — | — | — |
| `mixed_aio` | 1126.7 | **7.9** | 1075.4 | 2328.0 | **1275.5** | 1018.2 |
| `mixed_nixl` | 1164.1 | **7.8** | 1074.4 | 2345.7 | **1249.0** | 1024.5 |

### 这就是问题的答案

**命中 L2 的请求，排队时间一毫秒没变：7.8 → 7.9 / 7.8。**
挂上 L3 后端、并且同一批里有 4 个请求正在读 GPFS，也没有推迟它们进批次。

逐个探针看也一样（`l2_only` 是 7.7–7.9，`mixed_nixl` 是 7.6–8.0），不是均值抹平的。

---

## 4. 为什么混合臂 TTFT 反而更低：forward 被批次大小决定

`l2_only` 的 8 个探针并发发出、排队只有 7.8 ms，于是**挤进同一个 prefill 批次**。
混合臂里 L2 那 4 个先进一批，L3 那 4 个晚 1.2 秒进另一批。

| | 批内探针数 | 要搬的 KV | forward | 折合 H2D |
|---|---|---|---|---|
| `l2_only` | 8 | 20 GiB | 2082.8 ms | 9.6 GB/s |
| 混合臂 L2 批 | 4 | 10 GiB | 1075 ms | 9.3 GB/s |

**2082.8 / 1075.4 = 1.94 ≈ 2**，正好是批次大小之比。forward 被 L2→L1 的 H2D
带宽限制，和批内 token 数成正比。

所以"混合更快"只是因为同样 8 个请求被拆成了两批，**每批各自更快，但总时间更长**。
比较 TTFT 会得到反直觉的结论，**比较 queue 才干净**——queue 只含调度排队，
不含批次大小的影响。

---

## 5. 混合的真实代价：L3 那批的 1250 ms

用阶段 13 的埋点把它拆开（均值，n=4，相对最早入队时刻的毫秒）：

| 段 | `mixed_aio` | `mixed_nixl` |
|---|---|---|
| 入队 → **读提交** | **1120** | **1115** |
| 读提交 → 宣布准入 | 48 | 49 |
| 宣布准入 → 实际准入 | 108 | 85 |
| 合计 | 1276 | 1250 |

**89% 花在"读还没被提交"上。** 而读本身只要 31.7 ms。

机制：阶段 B（分配 host 内存 + 提交读）跑在 `check_hicache_events` 里，
也就是**调度线程**上。L2 那 4 个请求先被准入，它们的 prefill forward 要 1075 ms，
这段时间调度线程不回来，L3 的读就提交不了。

```
1120 ms (入队→读提交)  ≈  1075 ms (L2 批次的 forward)  +  ~45 ms (那一轮剩下的部分)
```

对照阶段 13：**全部走 L3** 的并发 8 场景里，同样三段是 215 / 94 / 217 = 526 ms。
那里前面没有 L2 批次挡路。差出来的约 700 ms 就是"等 L2 那批做完 forward"。

**这是阶段 13 结论的直接推论**：阶段 B 在调度线程上，所以任何 forward 都会堵住它。
混合场景把这一点放大了，因为 L2 那批的 forward 又长（H2D 受限）又必然排在前面
（它们排队只要 7.8 ms）。

---

## 6. NIXL 对 Linux AIO

两个臂只差传输层：分片、优先级仲裁、bounce buffer、fd 缓存、读统计全部是同一份代码
（`NixlIoContext` 实现了 `LinuxAioContext` 那 6 个方法的接口）。

| 指标 | `mixed_aio` | `mixed_nixl` | |
|---|---|---|---|
| 流式读 span | 31.67 ms | 31.77 ms | 持平 |
| 带宽 | 40.81 GiB/s | 39.58 GiB/s | NIXL −3% |
| **open 时间** | 7.64 ms | **3.10 ms** | **NIXL 好 2.5 倍** |
| L3 queue | 1275.5 | 1249.0 | NIXL −26 ms |
| 宣布准入→实际准入 | 108 ms | 85 ms | NIXL −23 ms |
| L3 TTFT | 2328.0 | 2345.7 | NIXL +18 ms |
| L2 TTFT | 1126.7 | 1164.1 | NIXL +37 ms |

**结论是持平。** 带宽差 3%、TTFT 差 18–37 ms，都在 n=4 的噪声里；
唯一稳定的差异是 open 时间，NIXL 只要 2.5 分之一。

L2 子集那 37 ms 的差**解释不了**：两臂的 queue（7.8 vs 7.9）和 forward
（1074.4 vs 1075.4）几乎相同，差值落在 TTFT 减去这两项的剩余部分里
（客户端往返、detokenize）。n=4，不深究。

### 引擎对拍

上线前做过字节级对拍：同一批 16 个 2 MiB range read 分别走两个引擎，
**结果逐字节相同**，且与预期内容一致。

---

## 7. 背景吞吐降了 28%，但归因不干净

| 臂 | 背景 tokens | span | tok/s |
|---|---|---|---|
| `l2_only` | 468 | 7.89 s | **59.4** |
| `mixed_aio` | 330 | 7.90 s | **41.8** |
| `mixed_nixl` | 342 | 7.94 s | **43.1** |

span 三臂相同，所以这个比较在时长上是公平的。

**但有一个混淆项本实验排除不了**：混合臂的背景生成本身也要 `write_through` 到 L3，
`l2_only` 完全没有这笔开销。所以这 28% 里有多少是预取抢了调度线程、
有多少是背景请求自己的写回，这里分不开。

> **2026-09-23 补**：[阶段 15 §2](15-is-l3-as-fast-as-l2.md) 加了一个
> 「开着后端但探针全命中 L2」的臂，把这两段分开了：背景吞吐
> **59.2 → 44.2 → 43.3 tok/s**。降幅几乎全部来自**写回**（开后端那一步），
> 真正读 L3 只再降 2%。

---

## 8. 不确定的地方

1. **§7 的归因分不开**（写回 vs 预取）。要分开得加一个"关掉写回但保留读"的臂。
2. **每个臂单次运行，n=8 探针**（分子集后 n=4）。NIXL 与 AIO 的差异都在这个噪声带内。
3. **§6 里 L2 子集那 37 ms 的差没有解释**，只知道不在 queue 也不在 forward。
4. **混合比例只测了 50/50。** 少数 L3（比如 8 选 2）会不会让 §5 的阻塞变短，没测。
5. **`l2_only` 用的是 `mem` 服务端**，它和混合臂不只差"有没有 L3"——
   还差写回开销（§7）。**已在[阶段 15 §2](15-is-l3-as-fast-as-l2.md) 补上那个臂，结论见上。**
6. **绝对 TTFT 不代表生产性能**：沿用单请求基准配置（关闭分块预填充、无 CUDA graph）。

---

## 9. 代码改动

全部增量，默认行为不变（`engine` 默认 `"aio"`）：

| 文件 | 性质 |
|---|---|
| `mem_cache/layerwise_storage/nixl_engine.py` | 新增，`NixlIoContext` |
| `mem_cache/layerwise_storage/file_backend.py` | `engine=` / `nixl_plugin=` / `pinned_regions=` 参数 |
| `mem_cache/layerwise_storage/controller.py` | 传引擎选择，并把 host KV 池的两块 buffer 作为 pinned region |
| `server_args.py` | `--hicache-storage-layerwise-engine {aio,nixl}`、`--hicache-storage-layerwise-nixl-plugin` |
| `benchmark/hicache/bench_l2l3_fusion_ttft.py` | 加 `l3_nixl` 服务端预设 |
| `benchmark/hicache/bench_mixed_l2_l3.py` | 新增混合驱动 |

环境：`sglangtest` venv 里装了 `nixl-cu13==1.4.1` 和 `nixl==1.4.1`（meta 转发包，
`--no-deps` 装以避免拖进用不上的 `nixl-cu12`）。撤销：`uv pip uninstall nixl nixl-cu13`。

---

## 10. 怎么复现

```bash
cd /home/lpl/sglang/benchmark/hicache
/home/lpl/sglangtest/.venv/bin/python bench_mixed_l2_l3.py \
  --server-arms l2_only,mixed_aio,mixed_nixl --probes 8 --l3-probes 4 \
  --page-size 512 --hit-tokens 16384 --suffix-tokens 512 \
  --background-requests 6 --io-threads 8 --max-concurrent-streams 8 \
  --max-total-tokens 163840 --hicache-ratio 6.0 \
  --store-root /zion0/kv-aio-bench/<独立目录> --run-id <名字>
```

两个 L3 臂各用自己的 store 子目录（`layerwise` 和 `layerwise-nixl`），
所以一个引擎写的页永远不会被另一个引擎读回。

原始数据：`benchmark/hicache/results/l2l3_fusion/mixed-v1/`
