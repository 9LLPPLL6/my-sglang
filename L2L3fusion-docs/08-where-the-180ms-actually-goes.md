# 阶段 08：那 180 ms 到底是谁的——准入，还是 forward

| | |
|---|---|
| 阶段 | 08 |
| 日期 | 2026-09-18 |
| 前一阶段 | [07-steady-state-and-what-the-pipeline-costs.md](07-steady-state-and-what-the-pipeline-costs.md) |
| 状态 | 六组实测；**结论收敛**，并推翻阶段 07 的两个假设和一条归因 |
| 模型 | Llama-3.3-70B-Instruct-FP8，TP=2 |
| 存储 | GPFS `/zion0` |
| 脚本 | `benchmark/hicache/bench_steady_state_l3_probe.py` |
| 分支 | `L2-L3fusion-continuous-read` |

> 阶段 07 测出流水线相对 L2 净亏 ~180 ms，但**没能归因**，列了两个竞争假设就停住了。
>
> 这一阶段把这 180 ms 拆开了。答案是**两个假设都不对**：
> forward 侧零暴露（层门从没卡住计算），读也几乎无关（读快 4.9 倍只换回 27 ms）。
> 真正的代价是**准入要等两趟调度线程的会合点**，这两个窗口加起来约 300 ms，
> 而且**与读多快无关**。

---

## 1. 一句话结果

L3 相对 L2 多出来的时间，六组配置全部落在准入前，forward 里一毫秒都没有：

| 配置 | 读时长 | 读带宽 | **Δ准入** | **Δforward** |
|---|---|---|---|---|
| `nopipe` 阻塞读 | 35 ms | 38 GiB/s | **+100 ms** | −2 ms |
| 流水线 `group_size=80`（单组） | 35 ms | 35.8 GiB/s | **+162 ms** | −12 ms |
| 流水线 `group_size=40` | 153 ms | 8.2 GiB/s | **+170 ms** | −18 ms |
| 流水线 `group_size=20` | 158 ms | 7.9 GiB/s | **+171 ms** | −13 ms |
| 流水线 `group_size=8` | 171 ms | 7.3 GiB/s | **+189 ms** | −12 ms |
| 流水线 `group_size=8`（复跑） | — | — | **+186 ms** | −16 ms |

读时长在这六组里变化 4.9 倍（35 → 171 ms），Δ准入只从 162 动到 189——**27 ms**。

---

## 2. 拆法：不需要新埋点

上游早就打好了点。`SchedulerReqTimeStats` 记了三个时刻
（`python/sglang/srt/observability/req_time_stats.py`）：

```
wait_queue_entry_time   入队（scheduler.py:2823，紧跟在 _prefetch_kvcache 后面，
                        中间只隔一行 waiting_queue.append）
forward_entry_time      准入进 prefill 批（scheduler.py:3526）
completion_time         完成
```

于是
- **`queue_duration` = 入队 → 准入**，L3 臂的预取等待全在这里
- **`forward_duration` = 准入 → 完成**，层门阻塞只可能在这里

`--enable-request-time-stats-logging` 让 rank 0 给每个完成的请求打一行
（`scheduler_components/output_streamer.py:232`）：

```
ReqTimeStats(rid=633a3c27..., input_len=21504, cached_input_len=16384, output_len=1,
             attempts=0, type=unified): queue_duration=306.43ms,
             forward_duration=2492.00ms, entry_time=...
```

`input_len=21504`（= H 16384 + S 5120）把探针和热身、filler、背景请求干净地分开了。
bench 的改动只有加这一个 flag。

### 为看清读什么时候开始，补了两个绝对时间戳

流式读的自报行原来只有 span，没有可对齐的参照物，我因此把它读错过两次
（见 §5）。现在打出 span 的两端，用的是和 `entry_time` 同一个墙钟：

```
layerwise_stream read: txn=633a3c27... pages=32 extents=640 bytes=1342177280
                       ms=166.23 open_ms=2.56 GiB/s=7.52 threads=8
                       start=... end=... shard_bytes=...
```

`txn` 就等于 rid，两条日志直接可join。**注意这行是在 transaction close 时打的**，
对流式读来说远晚于读结束，所以日志自己的时间戳不能当作 span 的任何一端用。

---

## 3. 三条结论

### ① forward 侧零暴露

六组全部是 −2 到 −18 ms。层门（`LayerDoneCounter.wait_until`，由
`memory_pool.py:2342` 的 `get_key_buffer` 逐层调用）在 prefill 里**没有可测的阻塞**。

这一点值得强调，因为它否掉了一个很自然的担心："计算第 N 层时第 N 层的
L3→L2 还没到"。在这个负载下它没有发生：prefill 本身 2.5 秒，而整条前缀的读
最慢 171 ms，流水线对第 1..N−1 组的掩盖是充分的。

顺带说明 L2 臂**也走层门**——`ready_to_load_host_cache() → start_loading()`
（`cache_controller.py:1080`）一样占 producer slot、一样驱动同一个计数器。
所以 L3 不是"多了一道层门"，是层门等的东西从 PCIe 变成了 GPFS。两臂 forward
相等，说明这个替换本身不要钱。

### ② 准入被调度线程的会合点量化

L3 的预取要经过三段：A 查询在预取线程，**B 分配 host 内存 + 提交读，在调度主线程**，
C 才是 IO。B 的位置是：

```
scheduler.py:3311            check_hicache_events()        ← 一轮只调一次
  └─ _drain_storage_control_queues_impl
      └─ _drain_and_alloc_storage_hit
          └─ _try_alloc_storage_hit
              └─ layerwise_bridge.start(operation)   unified_radix_cache.py:2231
                                                     ← 读到这里才开始
scheduler.py:3442            check_prefetch_progress()     ← 同一趟里，晚几微秒
```

所以流式路径**和阻塞路径走同一条三段式队列**，读不是入队就开始的。实测：

| 配置 | 入队 → 读开始 | 读开始 → 准入 |
|---|---|---|
| `group_size=8` | 148.0 ms | 169.7 ms |
| `group_size=20` | 135.5 ms | 167.0 ms |
| `group_size=40` | 133.0 ms | 168.6 ms |
| `group_size=80` | 129.5 ms | 166.1 ms |

两个窗口都近乎常数，**加起来约 300 ms**。而 L2 臂的 queue 是 130 ms
（它只需要等一趟）。差值就是 Δ准入。

### ③ 读时长被这两个窗口吸收掉了

把"读开始 → 准入"再拆一层，这是本阶段最干净的一个数：

| 配置 | 读时长 | 读完 → 准入 | **两者之和** |
|---|---|---|---|
| `group_size=8` | 171.1 ms | −1.4 ms | **169.7** |
| `group_size=20` | 158.3 ms | 8.7 ms | **167.0** |
| `group_size=40` | 153.4 ms | 15.2 ms | **168.6** |
| `group_size=80` | 34.9 ms | 131.2 ms | **166.1** |

**恒定 167 ± 2 ms。** 读快了 4.9 倍，省下的 136 ms 一分不少地变成了干等。

这就是为什么 ② 是结论而 ①③ 是它的推论：请求等的不是数据，是调度线程下一次
来看它。在这个量化窗口里把 IO 做快，等于把时间从"读"这一栏搬到"等"那一栏。

---

## 4. 于是流水线在这个场景下换不回东西

设计意图是"第 0 组落地就准入，后续组在 forward 后面流式补齐"。实测它没有兑现：

```
group_size=8：  读开始 148.0    准入 319.3    读结束 319.0
                                              ↑ 准入和整条前缀读完重合（8 个探针逐个：
                                                −23.6 +2.8 +3.5 +1.3 −0.9 −14.5 +3.8 +1.6）
```

请求等的是**整条前缀读完**，不是第 0 组。原因在 ②③：第 0 组再快，也要等下一个
会合点，而那个窗口（167 ms）比整条前缀的读（171 ms）只短 4 ms。

归一化之后流水线甚至略亏于阻塞读：

| | Δ准入 / T | 读时长 / T |
|---|---|---|
| `nopipe` | 0.59 | 0.21 |
| `group_size=80` | 0.70 | 0.15 |
| `group_size=8` | 0.80 | 0.73 |

（`T` = 调度轮周期估计，见 §7 局限 2。）多出的 ~0.11 T 是流式状态机自己的
advance / consensus 步骤——读时间完全相同（都 35 ms）的 `nopipe` 和 `g80`
之间，差的就是这个。

---

## 5. 推翻阶段 07 的三处

### 5.1 §6 的两个假设都不对

- **假设 A（暴露是准入延迟）**：方向对了，但机制说错了。我当时说"第 0 组落地后
  调度线程要到下一个心跳才注意到"，隐含前提是第 0 组早就落地了。实际上读根本
  **还没开始**——A 漏掉了"读要等调度线程来启动"这一段，它比我说的那一段更大。
- **假设 B（暴露就是读）**：直接否掉。forward 零暴露，且读快 4.9 倍只省 27 ms。

判别实验我原计划是"把背景请求从 6 降到 2，看暴露跟不跟着变"。**没有做，也不必做了**：
`queue_duration` / `forward_duration` 的拆分比改负载更直接，而且一次就把两个假设
同时排除了。

### 5.2 §7 第 4 条的归因是错的

原文：

> **流式路径的 extent 粒度（2 MiB × 640 个）是它带宽只有阻塞路径 1/6 的直接原因**

**错了。粒度不是原因。** `group_size=80` 用同样的 2 MiB extent、同样 640 个，
跑出 **35.8 GiB/s**，和阻塞路径持平。真正的原因是 `_advance_read_ahead`
**一组一组提交**：`group_size=8` 时一次只喂 64 个 extent，GPFS 队列太浅。

（`begin_read` 早就把每个 page 分派到固定 shard 了，分片也完全均衡——
8 个各 160 MiB，一字不差。所以从来不是分片的问题。）

### 5.3 §4 对那 166 ms 的读法

原文把 166 ms 当成"分组机制的代价，落在关键路径上"。按 ③，它**大部分不落在
关键路径上**：同样的配置下把读压到 35 ms，TTFT 只改善 27 ms。

阶段 07 结尾写的"在这个实验做完之前，不要根据本阶段的数字去改流水线"——
这句是对的，而且现在可以说得更强：**按阶段 07 的数字去改流水线（调大
group_size、改 extent 粒度）最多值 27 ms。**

---

## 6. 那么杠杆在哪

不在读，也不在流水线。在**把那两个会合窗口从关键路径上去掉**：

1. **入队 → 读开始（~135 ms）**：读要等调度主线程跑到
   `_drain_and_alloc_storage_hit`。这一段正好是阶段 05 量的 **W1**
   （tokenize 结束 → `match_prefix` 开始）所对应的位置——如果在 tokenizer
   之后就开始前缀匹配和预取，这一段可以整段消失。阶段 05 测的 W1 ≈ 一个调度轮，
   量级吻合。
2. **读完 → 准入（`group_size=80` 时 131 ms）**：预取完成时主动让调度线程重查，
   而不是等它下一趟自己来看。

这两条都不需要动流水线，也不需要读得更快。

**本阶段不提出改法，只交归因。** 上面两条是方向，不是结论——尤其第 1 条会动
tokenizer 到 scheduler 的边界，代价和收益都还没量过。

---

## 7. 口径局限

1. **"入队 → 读开始"这一段没有直接埋点**，是用 `start − entry_time` 算出来的。
   两个量同一进程、同一 `perf_counter` 基准，但中间经过了 stage A（GPFS 元数据
   查询）和一次 TP-MIN all_reduce，二者各占多少**没有拆开**。
2. **调度轮周期 T 有两个估计值且不一致**：从日志时间戳直接量（相邻 `Decode batch`
   行差 40 轮）得 **250 ms**；用 `6 / 背景tok/s` 估得 **231 ms**。后者偏低，因为
   背景吞吐是客户端在整个 span 上算的，而 span 里含 8 次每次 2.5 秒的探针 prefill。
   §4 的 `/T` 一列用的是后者，只用于同一轮内的相对比较。
3. **"读完 → 准入 = 167 ms 恒定"这个窗口，我没能用 T 解释清楚**。若它是"等下一趟
   调度"，`group_size=80` 的读在会合点后 35 ms 就结束，应该再等 T−35 ≈ 215 ms，
   实测 131 ms。相位模型缺一块。这不影响 §3 的三条结论（它们只依赖实测量），
   但说明这 167 ms 的内部结构还没看清。
4. **单次运行**，每档一遍。可重复性证据是 `group_size=8` 跑了两遍
   （Δ准入 186 / 189 ms，差 1.6%），以及 L2 臂六轮的 `queue/T` 全落在 0.55–0.57。
5. **绝对 TTFT 仍偏高（2.7 秒）**：沿用单请求 bench 的配置
   （`--chunked-prefill-size -1`、triton、无 CUDA graph）。两臂同配，相对可信。
6. **背景负载是"短输入 + 长生成"**，不是"不断到达的新请求"。后者 churn 缓存更厉害，
   未测。
7. **`forward_duration` 是准入 → 完成**，`max_new_tokens=1` 时它≈prefill，但严格说
   还含一步采样和输出流转。两臂同配，差值可信。

---

## 8. 怎么复现

```bash
cd /home/lpl/sglang/benchmark/hicache

# 流水线，逐个层组（当前默认）
/home/lpl/sglangtest/.venv/bin/python bench_steady_state_l3_probe.py \
  --server l3_fused --io-threads 8 --run-id split-fused-t8-tl \
  --page-size 512 --hit-tokens 16384 --suffix-tokens 5120 --probes 8 \
  --max-total-tokens 100000 --hicache-ratio 6.0 --port 31967 \
  --store-root /zion0/kv-aio-bench/l3-split-tl

# 阻塞读对照：只改 --server
#   --server l3_nopipe

# group_size 扫描：--group-size 20 / 40 / 80
```

拆分脚本（把 `queue_duration` / `forward_duration` 和流式读时间线join起来）不在
仓库里，逻辑是：从服务端日志取 `input_len=21504` 的 `ReqTimeStats` 行，按出现
顺序前 8 条是 l2 臂、后 8 条是 l3 臂；再按 `txn == rid` 配上
`layerwise_stream read:` 行的 `start` / `end`。

原始数据：`benchmark/hicache/results/l2l3_fusion/split-{nopipe-t8,fused-t8,fused-t8-tl,g20,g40,g80}/`
