# 把 L3 的准入延迟压向 L2：探索报告

2026-09-24 · 分支 `L2-L3fusion-admission-latency` · Llama-3.3-70B-Instruct-FP8，TP=2 · GPFS `/zion0`

---

## 一句话结论

| 基线 | 起点 | 最终 | 改善 |
|---|---|---|---|
| **A：无背景、单请求** Δqueue | 39.04 ms | **25.63 ms** | **−13.41 ms（−34%）** |
| **B：有背景、并发 8** Δqueue | 520.7 ms | **483.3 ms** | −37.4 ms（−7.2%，见 §7.2） |

四项优化落地，全部在 `python/sglang/srt/mem_cache/` 内，**198 插入 / 81 删除，8 个相关单测全绿**。

**但"和 L2 一样快"没有达成，也不可能在当前架构下达成**：基线 A 剩下的 25.63 ms 里，**15.09 ms 是第 0 组的磁盘读**，这是 GPFS 的固定延迟，不是可以省掉的软件开销。

---

## 1. 先把钱花在哪查清楚

之前只知道"没有单一大头"。这次给准入路径每一步加了计时（临时脚手架，最后剥离），拿到基线 A 的 39.04 ms 的实测构成：

```
0   → 2.9    阶段 A（存储存在性查询）+ host 内存分配
2.9 → 11.6   阶段 B：build_plan 4.0 + 提交 640 个 extent 4.7 = 8.7 ms   ← 全在调度线程
11.6 → 35.9  等第 0 组落地                                  24.3 ms   ← 最大项
35.9 → 40.8  init_load_back + 扫描
```

`init_load_back` 的分项里 **`tree_insert` 2.9 ms 和 `alloc_device` 0.4 ms 是 L2 命中也要做的**，不计入 Δ；L3 独有的只有 `build_key` 0.4 + `match_prefix` 0.03 + `attach_device` 0.2 ≈ 0.6 ms。

### 先排除四个假设

动手前用微基准和已有埋点排除了四个看似合理的怀疑对象：

| 假设 | 实测 | 结论 |
|---|---|---|
| 640 个 extent 的 msgspec 校验贵 | 造 640 个 = 0.55 ms（去校验 0.23） | 排除 |
| 计划构建整体贵 | 微基准 1.7 ms | 部分成立，见优化 2 |
| 跨 rank 共识贵 | `g0_global_ready` 与 `g0_local_done` 相差 **0.01 ms** | 排除 |
| 轮询粒度粗 | 17 ms 内轮询 12 次，**每 1.4 ms 一次** | 排除 |

---

## 2. 优化 1：一个层组作为一个整体入队

**问题**：一个层组展开成"每页每个 KV 部分一个 extent"，32 页前缀就是 640 个。每个 extent 各自：重算一次页路径字符串、取 inflight 锁、取统计锁、取仲裁器队列锁。

**改法**：`_enqueue_group` 替代 `_enqueue_extent`。每页只解析一次 fd 和路径（**32 次而非 640 次**），整组记录一次 `dict.update`，按分片批量入队（新增 `IoArbiter.enqueue_many`）。

**效果**：`submit_ms` 4.68 → 2.87 ms（−39%）；`open_ms` 2.23 → 0.94 ms（−58%）。

---

## 3. 优化 2：一个读计划只验一次它的暂存区

**问题**：`build_read_plan` 对每个层组调一次 `get_layer_group_buffer_meta`，而后者每次都重新验证**同一份** `host_indices` 并重新推导同一个页列表——**10 次**。末尾还有三次额外遍历（`all` / `sum` / `any`）每组各 64 个 extent。

**改法**：新增 `host_pool.page_indices_for()`，`build_read_plan` 验一次把结果传进去；三个聚合量折进构建循环。

**效果**：`plan_meta_ms` 3.41 → 1.81 ms（−47%）。

**优化 1+2 合计：`stageB_begin` 8.65 → 5.71 ms（−34%）。**

---

## 4. 优化 3：只有还在等第 0 组的请求才轮询

**问题**：`controller.poll()` **一次推进全部事务**，而扫描循环对每个等待请求各调一次。8 个请求做了 64 次推进，其中 **56 次是重复的**。实测每次 `poll()` 要 **3.9 ms**（它不只是轮询，还包含把就绪组交给 H2D 流）。

**改法**：只有 `not (admission_ready or aborted)` 的事务才触发 poll。已决的判定不会因轮询改变；已在流式传输的事务由层门的 `_pump` 推进（`_pump` 里也调 `poll()`，所以不会停摆）。

**效果**：并发 8 时每请求 `poll` 中位 3.93 → **0.00 ms**；Δqueue 507.1 → 491.5。

**对单请求无效**——只有一个请求时本来就没有冗余。

---

## 5. 优化 4：准入读在飞时不提交 read-ahead

这是收益最大也最反直觉的一项。

### 判别实验

第 0 组只有 128 MiB/卡，按整条读的 40 GiB/s 算该是 3 ms，实测 **18.85 ms**（折合 7.4 GiB/s）。假设：第 0 组提交后 2.8 ms，另外 **576 个 read-ahead extent** 就涌进同样 8 个 IO 线程——**仲裁器的优先级只管自己的队列，内核 AIO 队列不认**。

用一个临时开关把第 1~9 组的提交推迟到第 0 组落地后：

| | 正常 | 推迟后续组 |
|---|---|---|
| 第 0 组落地 | 18.85 ms | **7.09 ms（−62%）** |
| 整条读 span | 25.18 | 32.16（+7，藏在 forward 后面） |
| Δqueue | 30.55 | **21.35（−30%）** |

**假设成立。**

### 为什么不能照搬实验版本

实验版靠 `advance()` 放行后续组，而 `advance()` 由调度线程驱动。空闲时每 1.4 ms 一次所以只差 2 ms；**有负载时调度线程 143 ms 才来一次，会重现阶段 08 记录的那个 132 ms 设备空转**（171 ms 的读里只有 39 ms 在真读）。

### 正确的位置：仲裁器

`IoArbiter.poll()` 末尾就有 `self.pump()`——**IO 工作线程排空完成后会自己重新填充队列**。所以规则做进仲裁器：

> 只要还有准入优先级的 I/O 在飞（或在排队），就不提交 READ_AHEAD。

第 0 组最后一个 extent 完成的瞬间，那个 IO 线程自己的 `poll() → pump()` 立刻放行后续组。**不经调度线程，因而没有空转窗口。** 加了 `defer_read_ahead=True` 构造参数可关。

### 效果与差距

仲裁器版把第 0 组从 18.85 降到 **15.09**，而实验版能到 7.09。**差的 8 ms 是 CPU 不是 I/O**：实验版连 576 个 extent 的**提交动作本身**（约 2.5 ms 的 Python，期间调度线程不轮询）都推迟了。这指向下一项优化（§8.1）。

**并发场景下这项没有收益**（Δ 491.5 → 501.0，第 0 组 94.6 → 102.5）：8 笔事务的准入 I/O 本身就是拥挤源，推迟 read-ahead 无济于事。单次运行噪声 ±10 ms，不能断言变差。

---

## 6. 被证伪的假设：重叠调度

探索开始时我的头号候选是"让 layerwise 支持重叠调度"，估值 142–1075 ms。**测下来是零。**

### 门禁不是不兼容，是没做

```
ValueError: --hicache-storage-load-mode=layerwise currently requires --disable-overlap-schedule.
```

`_validate_hicache_layerwise_compatibility` 的 docstring 写着 *"Fail closed for combinations the first layerwise implementation lacks"*，同一张清单上还有分块预填充、投机解码、PP、DP、CP、分离式部署。引入它的提交对重叠调度**没有任何针对性说明**——从没测过。

（我此前说"只有 MPS 和 Mamba no_buffer 强制关重叠调度"，**是错的**，grep 窗口太窄漏了这个校验函数。）

### 用不受门禁管的路径代测

门只管 `load_mode=layerwise`。`full_wait` 模式走**完全相同的阶段 A 和阶段 B**，不受门禁。于是：

| 场景 | 关重叠 | 开重叠 | 差 |
|---|---|---|---|
| 纯 L3 并发 8（主批次） | Δqueue +226.7 | +234.9 | 无 |
| 混合 4 L2 + 4 L3 | Δqueue **+1208.7** | **+1187.2** | 1.7%，噪声 |

### 为什么没用

重叠调度改变的是**规划工作在一个批次内的时相**，不是**调度循环的周期**：

```
iter k:  plan(k)【含阶段B+扫描】 → 异步发射(k) → 处理(k-1)结果【阻塞等 GPU】
```

循环仍然一个批次一圈，`check_hicache_events` 还是每批次跑一次。把时间轴对上就清楚了：

```
t=0      8 个探针入队，4 个 L2 匹配上
t≈6      L2 那批被准入并发射       ← plan(k)
t≈6+ε    plan(k+1) 跑 check_hicache_events  ← 阶段 B 唯一的机会，但存储查询还没回来
t≈6..1081  process(k) 阻塞等 L2 前向（1075 ms），期间一次都不跑
t≈1081   下一轮 plan()，阶段 B 才提交读
```

**重叠调度只多给了一次机会，而它落得太早。**

---

## 7. 两个基线的最终状态

### 7.1 基线 A：无背景、单请求

```
起点 39.04 ms
  ├ 阶段 A + host 分配     2.9        未优化
  ├ 阶段 B                 8.7  →  5.7   优化 1+2，−34%
  ├ 等第 0 组落地          24.3 → 15.1   优化 4，−38%
  └ init_load_back 独有部分 0.6        未优化
最终 25.63 ms
```

**剩下的 15.09 ms 是 GPFS 的固定延迟。** 128 MiB 跨 8 个 IO 线程，延迟受限而非带宽受限——所以缩小第 0 组没用（阶段 08 试过，只值 1 ms）。

**基线 A 的地板约为 17–20 ms**，除非改准入时机（§8.2）。

### 7.2 基线 B：有背景、并发 8

| 轮次 | Δqueue |
|---|---|
| 基线 `scanwork2-pc8` | 520.7 |
| 最终 第 1 轮 | 512.8 |
| 最终 第 2 轮 | 453.9 |
| 均值 | **483.3** |

**两轮跨度 59 ms > 改善量 37.4 ms，所以 −7.2% 只能看方向，不是确定值。**

**为什么改善这么少**：阶段 B 的串行时间确实从 152 ms 降到约 110 ms，但端到端几乎没动。把时间轴拆开（优化 1+2 阶段的对照）：

```
入队 → 提交第0组读   214.6 → 199.1   −15.5  ✓ 阶段B的优化在这里
第0组读               93.7 →  99.8   +6.1
读完 → 宣布准入        0.0 →   0.0    0
宣布准入 → 进批次    217.5 → 234.7   +17.2  ✗
```

**最后那段（扫描循环）是并发场景的最大单项**，文档 13 测过它的构成：`init_load_back` 160 ms + `check_prefetch_progress` 63 ms。**我这四项优化只碰到了后者的一部分。**

---

## 8. 还能做什么（按预期收益排序）

| # | 做什么 | 预期 | 受益基线 | 风险 |
|---|---|---|---|---|
| 1 | **连 read-ahead 的提交 CPU 也推迟**：先建并提交第 0 组的计划，其余组的计划构建和提交都推到第 0 组在飞之后 | 约 8 ms（15.1 → 7.1 的差） | A | 中：要拆 `build_read_plan` 的接口 |
| 2 | **削 `init_load_back` 的 20 ms/请求** | ≤160 ms | B | **先要埋点**拆成设备分配 / 树插入 / H2D 会话 |
| 3 | **合并 `_agree` 的 all-reduce**：现在每请求一次，8 次；实测 rank 偏斜使先到者干等 3.2 ms | 约 14 ms | B | 中：要求两个 rank 的 `_staged` 顺序一致 |
| 4 | **提前预取到 tokenizer 之后** | 消掉阶段 A 的 2.9 ms + 一次调度会合 | A、B | 大：tokenizer 没有基数树 |
| 5 | **提前准入**：读一提交就准入，让层门在 forward 里挡 | 单请求约 0（只是把等待从 queue 挪到 forward）；混合场景可能很大 | B | 大：改变失败语义 |

**不值得做的**：缩小第 0 组（阶段 08 测过只值 1 ms，因为是固定延迟不是传输量）；支持重叠调度（§6 已证伪）。

---

## 9. 方法论上栽的两个跟头

记下来因为比结论本身更容易重犯。

**埋点污染了被测路径。** 第一版在 `init_load_back` 末尾每请求打一行含 10 个键值对的格式化日志，8 次全在调度线程上，**正好落在我要测的那一段里**，把并发场景的 queue 抬高约 18 ms。关掉后数字才回到真值。教训：**埋点放在关键路径上，测量本身就是开销**，必须能开关。

**埋点弄挂了 13 个已注册单测。** 第一版给 `_FakeController` 和 `_FakeTransaction` 加了属性依赖。改成模块级旁路字典（不碰任何对象属性）后恢复。

---

## 10. 不确定的地方

1. **基线 B 的 −7.2% 不可靠**：两轮跨度 59 ms 大于改善量（§7.2）。要确定需要更多轮次。
2. **优化 4 在并发场景可能有害**：Δ 491.5 → 501.0，但单次运行噪声 ±10 ms，分不清。`defer_read_ahead` 有开关正是为此。
3. **基线 A 的每档只有 8 个探针、单次运行**，Δqueue 的轮间噪声约 ±3 ms，和单项优化同量级——所以主指标用了方差更小的 `stageB_begin` 和 `g0_local_done`。
4. **§5 里"差的 8 ms 是 CPU"是推断**，没有单独验证；§8.1 做了才能证实。
5. **没有测吞吐**。四项优化都在减少调度线程的工作，理论上对吞吐也有利，但没测。
6. **没有测非 GPFS 存储**。第 0 组的 15 ms 固定延迟是 GPFS 的性质，换 NVMe 结论可能不同。

---

## 11. 怎么复现

```bash
cd /home/lpl/sglang/benchmark/hicache
PY=/home/lpl/sglangtest/.venv/bin/python

# 基线 A：无背景、单请求
$PY bench_steady_state_l3_probe.py --server l3_fused --io-threads 8 \
  --probe-concurrency 1 --max-concurrent-streams 1 --page-size 512 \
  --hit-tokens 16384 --suffix-tokens 512 --probes 8 --background-requests 0 \
  --max-total-tokens 163840 --hicache-ratio 6.0 \
  --store-root /zion0/kv-aio-bench/<独立目录> --run-id <名字>

# 基线 B：有背景、并发 8
$PY bench_steady_state_l3_probe.py --server l3_fused --io-threads 8 --page-size 512 \
  --max-concurrent-streams 8 --hit-tokens 16384 --suffix-tokens 512 \
  --probes 8 --probe-concurrency 8 --background-requests 6 \
  --max-total-tokens 163840 --hicache-ratio 6.0 \
  --store-root /zion0/kv-aio-bench/<独立目录> --run-id <名字>
```

关掉优化 4 对照：在 `_Shard.__init__` 里传 `IoArbiter(context=..., defer_read_ahead=False)`。

原始数据：`results/l2l3_fusion/` 下的 `opt-base`（起点）、`opt-batch`、`opt-plan`、`opt-poll`、`opt-hold`（判别实验）、`opt-defer`、`fin-nobg`、`fin-c8-r{1,2}`（最终）；重叠调度的四轮在 `prox-{off,on}` 和 `mixov-{off,on}`。

相关文档：[13 并发准入](13-why-all-eight-admit-together.md)、[15 L3 是不是和 L2 一样快](15-is-l3-as-fast-as-l2.md)、[混合负载 TTFT](report-mixed-ttft-2026-09-24.md)
