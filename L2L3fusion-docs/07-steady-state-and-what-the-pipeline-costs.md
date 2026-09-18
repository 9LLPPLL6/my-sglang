# 阶段 07：稳态负载下的流水线，以及它到底贵在哪

| | |
|---|---|
| 阶段 | 07 |
| 日期 | 2026-09-18 |
| 前一阶段 | [06-concurrent-l2-vs-l3-on-gpfs.md](06-concurrent-l2-vs-l3-on-gpfs.md) |
| 状态 | 五组实测；**结论未收敛**，最后一节列出两个竞争假设和判别实验 |
| 后续 | **§4、§6、§7.4 已被 [阶段 08](08-where-the-180ms-actually-goes.md) 推翻**，见各节开头的标注 |
| 模型 | Llama-3.3-70B-Instruct-FP8，TP=2 |
| 存储 | GPFS `/zion0` |
| 脚本 | `benchmark/hicache/bench_steady_state_l3_probe.py` |
| 分支 | `L2-L3fusion-continuous-read` |

> 阶段 06 的负载是**突发**：8 个请求同时打出去，`max_new_tokens=1`。
> 那意味着系统里**从不存在"正在推理"的请求**，预取没有可以重叠的对象，
> GPU 在读存储时干等——这不是流水线该被检验的场景。
>
> 这一阶段换成稳态：背景请求持续 decode，探针逐个到达。
> 结果是**流水线在 GPFS 上净亏 85 ms**，而且给它补上多线程之后只回收 5 ms。

---

## 1. 一句话结果

同一稳态场景（page 512，H=16384，S=5120，8 个探针逐个到达，6 个背景请求持续 decode）：

| 配置 | L2 探针 p50 | L3 探针 p50 | **暴露** | 背景 tok/s |
|---|---|---|---|---|
| 流水线，第0组=1层，单线程 | 2651 | 2832 | **+181 ms** | 34.8 / 34.6 |
| 流水线，第0组=8层（均分），单线程 | 2661 | 2841 | **+180 ms** | 34.7 / 34.6 |
| 流水线，8 线程 | 2729 | 2904 | **+175 ms** | 25.1 / 25.4 |
| 流水线，8 线程 + 读取埋点 | 2720 | 2884 | **+164 ms** | 25.6 / 26.0 |
| **关流水线，8 线程整段读** | 2640 | 2735 | **+95 ms** | 35.0 / 35.3 |

四个发现：

1. **关掉流水线反而快 85 ms。**
2. **第 0 组的大小无关紧要**：1 层 → 8 层，暴露动了 **1 ms**。
3. **给流水线补上多线程只回收 5 ms**（180 → 175）。
4. **流式路径的实测读带宽只有 7.5 GiB/s/rank，而阻塞路径是 44.7 —— 差 6 倍。**

第 4 条是这一阶段最重要的数字，而它在本阶段之前**根本无法观测**（见 §4）。

---

## 2. 为什么要换负载

阶段 06 的驱动一次性发 8 个请求、`max_new_tokens=1`：

| | 阶段 06（突发） | 本阶段（稳态） |
|---|---|---|
| 请求到达 | 8 个同时 | 背景常驻 + 探针逐个 |
| `max_new_tokens` | **1**，prefill 完就结束 | 背景数千，持续 decode |
| 预取时 GPU 在干什么 | **空转**（8 个都在等各自的预取） | **为背景做 decode** |
| 有没有 overlap 对象 | 没有 | 有 |
| 探针 TTFT 口径 | 20 秒（8 个 prefill 挤在一起） | **2.7 秒**（一次只有一个） |

`max_new_tokens=1` 是最致命的一处——系统里永远没有"正在推理"的请求，
所以**根本不存在可以被掩盖的对象**。

新驱动（`bench_steady_state_l3_probe.py`）：

```
① 预热 8 个独立探针前缀（写穿到 GPFS）
② 按臂准备（l2: 只驱逐显存 / l3: 全清 + 冷存储）
③ 起 6 个背景请求：512 输入 + 长生成 → 调度线程持续跑 decode
④ 等 8 秒进入稳态
⑤ 逐个发探针（16384 前缀 + 5120 后缀，max_new_tokens=1），测 TTFT
⑥ 显式 abort 背景，等引擎排空
```

### 两处必须做对的工程细节

**背景必须显式 abort。** 第一次运行时 L3 臂直接挂了：
`RuntimeError: flush_cache never succeeded`。关闭客户端连接**不足以**让服务端停止生成，
请求会继续跑完它的 `max_new_tokens`，而下一臂的 `flush_cache` 拒绝在有请求在飞时执行。
现在 `shutdown()` 调 `/abort_request {"abort_all": true}`，再用 `flush_cache` 自身的重试当排空屏障。

**每个探针一个独立前缀。** 共享前缀会让第一个探针把它拉进 L2，后面全变成 L2 命中。

---

## 3. 数据是干净的

| 证据 | L2 臂 | L3 臂 |
|---|---|---|
| 8 个探针的层级归属 | host 16384 / storage 0 | host 16384 / storage 0 |
| 真实层级 | **L2 命中** | **L3，8/8 全部走流式** |
| 进程读盘 | **0.00 GiB** | **20.00 GiB = 预期 100%** |
| 归属断言异常 | 0 | 0 |
| 探针 TTFT 抖动 | 2640–2695（跨度 55 ms） | 2818–2894 |

驱动里加了**运行时断言**：L2 臂任一探针出现 `storage>0`（前缀被背景负载挤出 L2）
或 L3 臂出现 `device>0`，当场报出来。**这是这个实验最容易被静默破坏的地方**——
背景请求持续写穿到 host，完全可能把探针前缀 LRU 掉。八次运行全部 0 异常。

容量按 54% 水位配的：`host 600k token`，需求 `131k(前缀) + 41k(后缀) + 100k(evict填充) + 51k(背景) = 323k`。

---

## 4. 埋点：之前根本看不见流式路径读了多少

那条带宽日志只存在于 **whole-prefix** 后端：

```
storage/layerwise/hicache_layerwise_file.py:208
    "layerwise_file read: pages=%d bytes=%d ms=%.2f open_ms=%.2f io_ms=%.2f GiB/s=..."
```

流式走的是另一个模块 `layerwise_storage/file_backend.py`，**一行都不打**。
所以前四组实验里，`reads` 字段全是 `{}`，流式路径的 IO 时间只能**拿阻塞路径的带宽去推**
——而那正是两者比较时争论的那个数。

本阶段给它补了同格式的报告（`layerwise_stream read:`），外加每分片的字节数。
补上之后第一次看到真相：

```
layerwise_stream read: pages=32 extents=640 bytes=1.25GiB ms=166.04 open_ms=2.37
                       GiB/s=7.53 threads=8
                       shard_bytes=0:160MiB,1:160MiB,...,7:160MiB
```

| | 单 rank 带宽 | 单探针读取 | extent 数 | extent 大小 |
|---|---|---|---|---|
| 阻塞读（8 线程） | **44.7 GiB/s** | ~30 ms | 64 | 20 MiB |
| **流式（8 线程）** | **7.5 GiB/s** | **166 ms** | **640** | **2 MiB** |

三条读法：

- **分片完全均衡**（8 个各 160 MiB，一字不差）→ 不是负载倾斜，8 个线程都在工作
- `open_ms=2.2` → page 512 之后 open 已经可以忽略（阶段 04 那堵墙矮了）
- **同样的数据被切成 10 倍多的小 IO，分 10 批下发**，GPFS 在这个粒度上喂不饱

> **阶段 08 修正**：上面那张表的 166 ms 是对的，但把它当成"落在关键路径上的代价"
> 是错的。同配置下把读压到 35 ms（`group_size=80`），TTFT 只改善 27 ms——
> 这 166 ms 大部分被准入的会合窗口吸收了。见
> [阶段 08 §3③](08-where-the-180ms-actually-goes.md)。
>
> 表里"extent 粒度是带宽只有 1/6 的直接原因"这个归因也是错的，见 §7.4 的标注。

### 这推翻了本阶段中途的一个结论

看到"多线程只回收 5 ms"时，我写过"说明瓶颈不在提交，在分组的串行往返（~80 ms）"，
并据此估算"IO 只占 30 ms"。**那 30 ms 是拿阻塞读的带宽推的，错了。**
流式路径的读**本身就要 166 ms**。没有埋点之前，这个错误无法被发现。

---

## 5. 本阶段的代码改动

| commit | 改了什么 |
|---|---|
| `1ac69318a3` | **删掉 `first_group_layers`**。第 0 组不再单独定尺寸，和其它组一样均分——实测它的大小只值 1 ms |
| `ad670cab7c` | 层门支持多消费者 + 槽位改为找空位，`--hicache-storage-max-concurrent-streams`（默认 1） |
| `11f4b7410e` | **流式路径分片提交**：每线程一个 AIO context，按页分配，`io_threads=1` 时行为与改前逐字节相同 |
| `ecf73ad364` | 流式读取埋点 |

### 关于分片提交

阶段 04 已经证明瓶颈是**提交者**不是队列深度（fio：单线程深队列 20.3 GiB/s 封顶，
多线程 134.5）。而流式路径此前被 `server_args` **明确拒绝**使用 `--hicache-storage-io-threads`
（"the layerwise streaming pipeline submits through its own single-context controller"）。

也就是说，此前所有"流水线 vs 阻塞读"的比较，**流水线都带着单 context 的劣势**。
本阶段消除了这个不公平，结果是：**劣势消除后流水线仍然亏 80 ms**。

### 关于生产者环的坑（已在实现中规避）

层门的生产者槽位原来是 `% num_counters` 盲目轮转 + assert 兜底。并发流式下第 N+1 条流
会绕回到还在飞的槽位；`python -O` 会把 assert 去掉，那时不是崩溃而是**把别人的进度
计数器清零重用**——正在等第 20 层的请求被告知"已完成"，去读没搬到的 KV。
现在改成**跳过被占用的槽位、全满显式报错**，上限由 `busy()` 按配置拒绝。

---

## 6. 结论未收敛：两个竞争假设

> **阶段 08 结论：两个假设都不对。**
> - 假设 A 方向对、机制错：我以为"第 0 组早已落地、只是调度线程没来看"，
>   实际上读**还没开始**——它也要等调度主线程跑到
>   `_drain_and_alloc_storage_hit` 才被提交。漏掉的这一段比我说的那一段更大。
> - 假设 B 直接否掉：forward 侧暴露是 −2 到 −18 ms（六组一致），且读快 4.9 倍
>   只换回 27 ms。
>
> 下面的判别实验**没有做，也不必做**：把每个请求的
> `queue_duration` / `forward_duration` 拆开，一次就同时排除了两个假设。
> 见 [阶段 08](08-where-the-180ms-actually-goes.md)。

暴露 164 ms。有两个解释，**现有数据分不开**：

### 假设 A：暴露的是准入延迟，不是读

背景 6 个请求、25.6 tok/s，一轮出 6 个 token：

```
25.6 ÷ 6 = 4.27 轮/s  →  调度心跳 234 ms
```

`check_progress`（推进流水线、检查第 0 组、发起跨 rank 确认）**只在这个心跳上被调用一次**。
读（166 ms）跑在工作线程上、确实和背景 decode 重叠了，但第 0 组落地后，
调度线程要到**下一个心跳**才注意到。暴露 164 ms **< 一个心跳 234 ms**，量级吻合。

### 假设 B：暴露的就是读

forward 被层门完全卡住，10 组串行，读多久暴露多久。
读 166 ms vs 暴露 164 ms，**几乎相等**。

### 判别实验

**改变调度心跳，看暴露跟不跟着变。** 背景请求从 6 个降到 2 个 → decode batch 变小 →
每轮更快（预计 80–100 ms）：

| 结果 | 结论 | 修法 |
|---|---|---|
| 暴露降到 ~80 ms | **假设 A** | 第 0 组落地时主动唤醒调度，而不是等下一个心跳 |
| 暴露仍 ~165 ms | **假设 B** | 粒度问题：调大 `group_size`，或让组间确认不串行 |

**在这个实验做完之前，不要根据本阶段的数字去改流水线。**

> **阶段 08 把这句话说得更强**：按本阶段的数字去改流水线（调大 `group_size`、改 extent 粒度）最多值 27 ms。杠杆不在读，在准入前那两个调度会合窗口。

---

## 7. 已经可以下的结论

无论 A 还是 B，下面几条都成立：

1. **GPFS 上目前的最优配置是关掉流水线**：
   `--hicache-storage-load-mode full_wait --hicache-storage-io-threads 8 --page-size 512`，
   此时 L3 相对 L2 只差 **95 ms（1.04×）**，其中约 30 ms 是不可避免的 IO。
2. **`--hicache-storage-max-concurrent-streams` 应保持默认 1。**
   阶段 06 两个负载下并发流式都是负优化（+291 ms / +975 ms，尾部 max 涨 6.2 秒）。
3. **page 512 是无条件的赚**：阻塞路径聚合带宽 43 → 90 GiB/s，`open` 占比从 1/4 降到 6%。
4. ~~**流式路径的 extent 粒度（2 MiB × 640 个）是它带宽只有阻塞路径 1/6 的直接原因**，
   无论这 166 ms 是否落在关键路径上，它本身就该改。~~
   **阶段 08 推翻：粒度不是原因。** `group_size=80` 用同样的 2 MiB × 640 个 extent
   跑出 35.8 GiB/s，和阻塞路径持平。原因是 `_advance_read_ahead` 一组一组提交，
   `group_size=8` 时一次只喂 64 个 extent，GPFS 队列太浅。而且按 §3③ 这值 27 ms。

---

## 8. 口径局限

1. **绝对 TTFT 仍偏高（2.7 秒）**：服务端沿用单请求 bench 的配置
   （`--chunked-prefill-size -1`、triton backend、无 CUDA graph）。两臂同配，相对比较可信。
2. ~~**GPFS pagepool 无法证伪**（阶段 03 §6）；本阶段 `--churn-gib` 用默认 0。~~
   **阶段 08 后续已实测**：本节点 pagepool 为 32 GiB，`--churn-gib 40` 把它完整挤掉
   之后，带宽 50.4 → 51.0 GiB/s、读 span 24.8 → 24.5 ms，**无可测差异**。
   这正是 O_DIRECT 真正绕过 pagepool 时该有的结果。
3. **`io_threads=8` 在 page 512 下不一定是最优**，阶段 04 §6 说真机只扫过 1 和 16 两个点，
   harness 里最优落在 4–8。本阶段未扫。
4. **背景负载是"短输入 + 长生成"**，不是"不断到达的新请求"。后者会 churn 缓存更厉害、
   对 L2 压力更大，更接近生产，未测。
5. **单次运行**，每档只跑了一遍。L2 臂跨运行的一致性（2640–2729，跨度 3.4%）
   是目前唯一的可重复性证据。

---

## 9. 怎么复现

```bash
# 流水线 + 多线程 + 埋点
python benchmark/hicache/bench_steady_state_l3_probe.py \
  --server l3_fused --io-threads 8 \
  --model /home/lpl/models/Llama-3.3-70B-Instruct-FP8 \
  --store-root /zion0/kv-aio-bench/l3-instr \
  --store-max-size 300Gi --store-min-free 300Gi \
  --page-size 512 --max-total-tokens 100000 --hicache-ratio 6 \
  --hit-tokens 16384 --suffix-tokens 5120 --probes 8 \
  --group-size 8 --max-concurrent-streams 1 \
  --background-requests 6 --background-output-tokens 20000 \
  --port 31965 --run-id steady-instr-t8

# 关流水线的对照：只改 --server
  --server l3_nopipe --io-threads 8
```

结果目录（`benchmark/hicache/results/l2l3_fusion/`）：

| run-id | 配置 |
|---|---|
| `steady-p512-s1-v2` | 流水线，第0组=1层，单线程 |
| `steady-even8` | 流水线，均分，单线程 |
| `steady-nopipe-t8` | **关流水线，8 线程**（当前最优） |
| `steady-fused-t8b` | 流水线，8 线程 |
| `steady-instr-t8` | 流水线，8 线程 + 埋点 |

流式路径的带宽在服务端日志里：`grep "layerwise_stream read:" <run>/l3_fused.log`。
