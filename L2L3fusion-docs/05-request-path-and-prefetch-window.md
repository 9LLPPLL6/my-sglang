# 阶段 05：请求调度路径与 L3 预取的时间窗口

| | |
|---|---|
| 阶段 | 05 |
| 日期 | 2026-09-15 |
| 前一阶段 | [04-parallel-submission-and-the-open-cost.md](04-parallel-submission-and-the-open-cost.md) |
| 状态 | 调度路径已通过埋点实测；提速方案为**设计草案，尚未实现，尚未在真实负载验证** |
| 模型 | Qwen3-8B TP=1 / Llama-3.3-70B-Instruct-FP8 TP=2 |
| 埋点 | 临时补丁，已回滚；见 §4.2 |

> 前四个阶段都在优化"L3 读得多快"。
> 这一阶段换一个角度问：**预取到底有多少时间可以用？**
>
> 答案出乎意料——在低并发下只有 **0.2–14 毫秒**，
> 而在此之前有 **300–460 毫秒**被白白等掉。

---

## 1. 一句话结果

1. 预取从发起到被裁决，低并发下只有 **0.2–14 ms**。`best_effort` 策略在这个
   时间里完不成任何一次存储往返——**不是命中率低，是根本没给时间**。
2. 请求从分词完成到进入调度器，还要等 **300–460 ms**（长 prompt 场景）。
   这段时间调度线程正在跑 forward，不轮询 socket，**完全被浪费**。
3. 把预取提前到这段时间里，理论上能把预取可用时间放大 **20–27 倍**。
4. 提前的代价是要从 root 起重新计算哈希、多读一段冗余 KV。按 50 GB/s/卡估算，
   最坏情况 **18 ms**，相对 370 ms 的收益可以忽略。
5. **但以上全部基于合成负载测量。现有数据不足以支撑上线决策**，
   原因和所需验证见 §6。

---

## 2. 请求从 tokenizer 到被调度的完整路径

### 2.1 三个进程

```
HTTP server + TokenizerManager（主进程）
        │  ZMQ
        ▼
   Scheduler（子进程，每个 TP rank 一个）   ← 调度 + forward，拥有 GPU
        │  ZMQ
        ▼
 DetokenizerManager（子进程）
        │  ZMQ
        ▼
   TokenizerManager → SSE → 客户端
```

三者是流水线关系，不是函数调用。**TokenizerManager 发出请求后就不再参与调度**，
所有调度决策都在 Scheduler 进程内。

### 2.2 TokenizerManager 做的事

1. 归一化参数，建立一个等待结果的槽位（含 `asyncio.Event`）
2. 分词，长度 / 词表 / 多模态校验
3. 打时间戳、序列化、ZMQ 发给 Scheduler，**立即返回**
4. `await` 等待结果被推回

这一层没有任何排队或准入逻辑。

### 2.3 Scheduler 主循环

一个死循环，每轮四步：

```
① recv_requests()      非阻塞抽干 socket，TP rank0 收完 broadcast 给其它 rank
② get_next_batch_to_run()   决定这一轮算什么
③ run_batch()          丢给 GPU
④ process_batch_result()    出 token、发结果
```

**一轮循环 = 一次 forward = 一个 batch。** batch 成员每轮重组（continuous batching）。

生产模式是 overlap 调度：第 ③ 步 launch 完不等结果，先去处理**上一轮**的结果，
让 CPU 调度与 GPU 计算重叠。

### 2.4 请求怎么被录取

调度器手里两堆东西：`waiting_queue`（未开始 prefill）和 `running_batch`（正在 decode）。

每轮的决策顺序：

1. 上一轮的 prefill batch 并入 `running_batch`
2. **先尝试组新的 prefill batch**（prefill 优先于 decode）
3. 组不出来才做 decode

组 prefill batch 时，`PrefillAdder` 按序贪心吃 `waiting_queue`，每加一个检查预算：

| 预算 | 说明 |
|---|---|
| KV pool 剩余 + 可驱逐量 | 还要为已在跑的请求预留未来 decode 空间 |
| `chunked_prefill_size` | **整个 prefill batch 的总 token 预算**，默认 8192 |
| `max_prefill_tokens` | 仅在关闭 chunked prefill 时才生效 |
| `max_running_requests` | 并发上限 |

任一不过即 `break`。

> **容易误解的一点**：`chunked_prefill_size` 是 batch 级预算，不是单请求级。
> 4096-token 的 prompt，一个 prefill batch 只装得下 **2 个**（8192 / 4096），
> 不是 `max_prefill_tokens / 4096 = 4` 个。

### 2.5 一条请求的时间轴

```
客户端 POST
  │
  ├─ 分词完成 ────────────────────────────────── ① W1 起点
  ├─ ZMQ 发送
  │      （调度线程正在跑 forward，不轮询）        ← W1
  ├─ Scheduler recv_requests 取走
  ├─ 构造 Req 对象
  ├─ 走前缀树（L1/L2 命中）+ 发起 L3 预取 ──────── ② W1 终点 / W2 起点
  ├─ 进 waiting_queue
  │      （等待被录取，可能跨 0 到数十轮 forward）  ← W2
  ├─ 通过 PrefillAdder 预算检查 ───────────────── ③ W2 终点
  ├─ prefill forward
  └─ 首 token 返回
```

---

## 3. L3 预取逻辑

### 3.1 三级缓存的关系

**L1 和 L2 共用同一棵前缀树。** 树上每个节点要么持有 GPU 上的 KV（L1），
要么 GPU 那份已被驱逐、只剩 host 副本（L2）。**走一遍树同时得到两级命中**。

L3 不在树里，在外部存储上，只能靠 key 去问。

三级是严格串行、层层往下的关系：

```
走一遍前缀树 → L1 命中 + L2 命中 + "最深的已备份到 L3 的节点"
             ↓
未覆盖的那一段 = 整个 prompt − (L1 + L2)
             ↓
以已备份节点的哈希为锚点，往后逐页算哈希链，去 L3 问
```

**L3 查找的起点，就是 L1+L2 命中的终点。**

### 3.2 L3 的 key 是什么

一条**滚动哈希链**：第 N 页的 key = SHA256(第 N 页的 token, 第 N−1 页的 key)。

两个性质对后面的方案至关重要：

- **只依赖 token**，从 root 起算总是可行的。锚在半路只是省几次哈希，不是必需。
- **带 TP rank 后缀**（非 MLA 模型）：`hash_模型名_tpRank_tpSize`。
  所以**不存在"一个进程替所有 rank 查 L3"这回事**。

### 3.3 预取的三个阶段，跑在三个不同的线程上

| 阶段 | 做什么 | 线程 | 是否阻塞主循环 |
|---|---|---|---|
| A 查询 | `batch_exists`，只问"有没有"，不搬数据 | 预取线程 | 否 |
| B 分配 | 按命中数分配 host 内存 | **调度主线程** | 是 |
| C 读取 | `batch_get`，真正搬 KV 到 host | IO 线程 | 否 |

阶段 B 必须在主线程，而且必须在**同步点之后**——这是整个 HiCache 的设计基石：

> host 内存的分配和驱逐是每个 rank 各自做的，**没有任何跨 rank 通信**。
> 之所以不出错，是因为 host 池只在调度线程的同步点被改动，
> 且队列按**各 rank 取最小值**排空——每个 rank 消费同一序列的同一前缀，
> 输入一样、代码一样，结论必然一样。

违反这条纪律的后果不是数据错，是 **all_reduce 对不上导致挂死**。

### 3.4 预取的发起与裁决时间点

```
入队时（主线程）
  ├─ 走前缀树，算出未覆盖段
  └─ 发起预取 → 扔进队列，立即返回   ← W2 起点

后台（跨若干轮调度循环）
  ├─ 阶段 A：预取线程查 L3
  ├─ 阶段 B：主线程在 check_hicache_events 里分配 host 内存
  └─ 阶段 C：IO 线程读取

被遍历到时（主线程，在 waiting_queue 循环里）
  └─ check_prefetch_progress(rid)
       ├─ 未达终止条件 → continue（本轮跳过该请求，但不出队）
       └─ 终止 → 把已搬到 host 的页插入前缀树，成为 L2 命中  ← W2 终点
```

**关键：`check_prefetch_progress` 就在"被录取进 prefill"之前那一刻。
所以 W2 就是预取从发起到被裁决的全部时间。**

### 3.5 三种停止策略

| `--hicache-storage-prefetch-policy` | 行为 |
|---|---|
| `best_effort` | 第一次检查就立刻停，搬到多少算多少 |
| `wait_complete` | 必须全部搬完才放行（请求被 `continue` 留在队列里） |
| `timeout` | 按页数线性计算的超时 |

### 3.6 关键推论

`best_effort` 下，预取实际拥有的时间 **等于 W2**。
§4 的测量显示低并发下 W2 只有 **0.2–14 ms**——
**这解释了为什么低并发场景 L3 命中率上不去，且与后端快慢无关。**

---

## 4. W1 / W2 的定义与测量

### 4.1 定义

| | 定义 | 含义 |
|---|---|---|
| **W1** | 分词完成 → Scheduler 里 `match_prefix` 即将开始 | **能提前的量** |
| **W2** | `match_prefix` 开始 → 请求被录取进 prefill | **预取实际拥有的量** |

### 4.2 测量方法

- 在 `APIServerReqTimeStats` 里传播 `tokenize_finish_time`，
  跨进程用已有的 `diff_realtime_monotonic` 做时钟换算
- W1 终点打在进入 `_prefetch_kvcache` 之前
- W2 终点打在 `set_forward_entry_time` 处
- 额外记录 `iters`：请求入队后跨了几轮 forward 才被录取
- 埋点为临时补丁，测完已完整回滚

**踩过的坑**：`api_server_dispatch_finish_time` 是在 ZMQ 发送**之后**才打的，
序列化时仍为 0，必须改用发送前的 `api_server_dispatch_time`。

### 4.3 实验环境

- Qwen3-8B TP=1 / Llama-3.3-70B-FP8 TP=2，单机双卡 RTX PRO 5000 72GB
- attention backend = **triton**（本机 flashinfer 版本不匹配，无法用 flashinfer）
- 压测：`sglang.benchmark.serving`，`--dataset-name random-ids`（**随机 token，无前缀共享**）
- **未开启 HiCache**（W1/W2 的定义与 HiCache 无关；开启后 W2 会被预取策略本身影响，形成循环依赖）

### 4.4 数据

#### （a）W1 的三段分解（所有负载一致）

| 段 | 耗时 | 说明 |
|---|---|---|
| 分词完成 → ZMQ 发送前 | ≈ 0.004 ms | 可忽略 |
| **ZMQ 发送 → Scheduler 取走** | **W1 的 99.9%** | 等调度线程下一次轮询 |
| 构造 Req + `match_prefix` | 0.005–0.032 ms | 可忽略 |

**W1 几乎完全是"等调度线程回到循环顶部"。`match_prefix` 本身极便宜。**

#### （b）Llama-70B TP=2，变并发 + 变 prompt

| 负载 | n | W1 p50 | W1 p90 | W2 p50 | W2 p90 | W1 占比 | 提升倍数 |
|---|---|---|---|---|---|---|---|
| idle 1req/s 128in | 21 | 0.75 | 29.73 | 0.29 | 0.30 | 72.0% | 3.6× |
| conc=1 512in | 21 | 28.80 | 28.95 | 0.23 | 0.25 | 99.2% | 126.8× |
| conc=16 512in | 151 | 35.47 | 43.60 | 1.27 | 1.95 | 96.5% | 29.0× |
| conc=64 4096in | 81 | 604.47 | 788.63 | 23,204 | 59,430 | 2.5% | 1.0× |
| conc=128 8192in | 65 | 445.59 | 795.72 | 83,501 | 158,549 | 0.5% | 1.0× |

（单位 ms。TP1 rank 的数值比 TP0 只慢 0.1–11 ms，即 TP broadcast 的成本，可忽略。）

#### （c）Qwen3-8B，**固定 4096 输入**，只扫并发 ← 最有信息量的一组

| 并发 | n | **W1 p50** | W1 p90 | **W2 p50** | W2 p90 | 当场录取% | 跨轮 p50 | 跨轮 max | 提升 | TTFT p50 |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 121 | **356.15** | 423.55 | **13.74** | 1138.77 | 47.1% | 1 | 3 | **26.9×** | 2294 |
| 16 | 121 | **345.79** | 429.87 | **1251.50** | 3735.71 | 28.1% | 2 | 7 | 1.3× | 3728 |
| 24 | 121 | **309.11** | 618.80 | **2702.89** | 6750.09 | 17.4% | 4 | 11 | 1.1× | 5522 |
| 32 | 121 | **298.62** | 595.75 | **4148.23** | 9409.35 | 16.5% | 6 | 15 | 1.1× | 6906 |
| 48 | 121 | **402.32** | 678.98 | **5980.13** | 14691.63 | 13.2% | 8 | 23 | 1.1× | 8984 |
| 64 | 121 | **460.08** | 674.02 | **9555.03** | 19359.51 | 9.9% | 13 | 31 | 1.0× | 13221 |

### 4.5 结论

**① W1 由 prompt 长度决定，不是并发。**

固定 4096 输入、并发从 8 扫到 64，W1 只从 356 ms 变到 460 ms（+29%），中间还上下波动。
而 prompt 从 512 换成 4096，W1 从 ~7–10 ms 跳到 ~300–460 ms（40 倍）。

原因：W1 = 一次调度循环 = 一次 forward。短 prompt 时大部分轮次是 decode（十几毫秒），
长 prompt 时 prefill batch 顶到 `chunked_prefill_size`，一轮三四百毫秒。

> ⚠️ 表（b）看起来像"W1 随并发暴涨"，那是因为并发和 prompt 长度同时在变。
> 表（c）固定 prompt 后才看清真相。

**② W2 由并发决定，且变化幅度极大。**

同一段扫描里 W2 从 13.7 ms 涨到 9,555 ms，**700 倍**。
`iters` 字段直接印证：跨轮中位数 1 → 2 → 4 → 6 → 8 → 13。
**W2 就是 forward 的整数倍。**

**③ 收益窗口在低并发，不在高并发。**

| | 低并发 | 高并发 |
|---|---|---|
| W1（能提前多少） | 大（356 ms） | 大（460 ms） |
| W2（现在已有多少） | **极小（13.7 ms）** | 巨大（9.5 s） |
| 提前预取的价值 | **26.9×** | 1.0× |

低并发时请求当场就被录取（47% 一轮都不等），预取只有十几毫秒；
高并发时请求要排十几轮，预取有几秒可用，本来就不缺时间。

---

## 5. 提速思路：把预取提前

### 5.1 硬约束

W1 的本质是**调度主循环被 forward 占住、不轮询 socket 的那段时间**。

推论：**任何需要主循环参与的方案都吃不到 W1。** 提前发消息没用，因为没人收。
干活的必须是不在主循环里的线程。

### 5.2 为什么不能放在 TokenizerManager 进程

| | 能否查 L3 | 有 host pool | 有限流状态 | 能读前缀树 | 知道 DP 路由 |
|---|---|---|---|---|---|
| TokenizerManager | ❌ key 带 rank 后缀 | ❌ | ❌ | ❌ | ❌ |
| **Scheduler 的预取线程** | ✅ | ✅ | ✅ | ⚠️ 见下 | ✅ |

**前缀树为什么连预取线程也不能读**（三条，强度递增）：

1. `cache_controller` 结构上就没有树的引用——后台线程只在"哈希值 + host 下标"上工作
2. 树上没有锁，且驱逐会真的摘掉节点；加锁可以修，但会把 forward 路径的延迟和后台线程绑在一起
3. **最关键**：各 rank 的后台线程自由运行，在任意时刻读树会得到 rank 相关的快照。
   一旦这个快照影响了"去 L3 问哪个范围"，各 rank 就会问不同范围，
   紧接着的 `all_reduce(MIN)` 对不上 → **挂死**。加锁解决不了这一条。

### 5.3 方案

给已经存在、却一直吃不饱的预取线程开一条**绕过主循环的进料口**：

```
TokenizerManager
  分词完成、校验通过
    ├─【新】旁路 socket 发 PrefetchHint{rid, token 序列}
    └── 原路径不动 → 主 ZMQ

Scheduler rank i
  ├─ 主循环 ←──── 被 forward 占住（W1 就是这段）
  └─【新】监听线程
        ├─ 从旁路 socket 拉 hint
        ├─ **从 root 起**算完整哈希链（只依赖 token，各 rank 天然一致）
        ├─ 乐观分配 host 内存
        └─ 直接发起读取（合并 query+fetch，一次往返）

  主循环收到正式请求
        ├─ 走前缀树 → L1/L2 命中 K 页
        ├─ 对账：L3 给了 [0,M)，有用的是 [K,M)，[0,K) 释放
        └─ 进 prefill
```

**顺序反过来了**：L1/L2 查找从"定 L3 起点"变成"裁掉 L3 多读的部分"。

三条必须遵守的纪律：
- hint 必须由独立线程消费，不能经主循环转发
- **早期阶段不做任何 all_reduce**，MIN 同步推迟到主线程认领时做（否则丢一条 hint 就挂死）
- 孤儿预取（请求被中止 / DP 路由到别的 rank）需要 TTL 回收
- 阶段一先 gate 成 `dp_size == 1`；后续可把 hint 发出点移到 `data_parallel_controller`

### 5.4 从 root 起的代价：冗余读

冗余量 = `min(L1+L2 命中, L3 命中)`，上界是整条 prompt。

**基础单价**（Llama-70B fp8 KV = 160 KiB/token，50 GB/s/卡）：

| | TP=2 | TP=4 | TP=8 |
|---|---|---|---|
| 从 L3 读 1000 token | 1.64 ms | 0.82 ms | 0.41 ms |
| 重算 1000 token 的 prefill | 73 ms（实测外推） | — | — |

**读比算便宜 45 倍**——这是 L3 能成立的根本原因。

**冗余读的绝对代价**：

| 冗余 token | 数据量 | TP=2 | TP=4 | TP=8 |
|---|---|---|---|---|
| 1,000 | 0.15 GiB | 1.6 ms | 0.8 ms | 0.4 ms |
| 4,000 | 0.61 GiB | 6.6 ms | 3.3 ms | 1.6 ms |
| 8,000 | 1.22 GiB | 13.1 ms | 6.6 ms | 3.3 ms |
| 16,000 | 2.44 GiB | 26.2 ms | 13.1 ms | 6.6 ms |
| 32,000 | 4.88 GiB | 52.4 ms | 26.2 ms | 13.1 ms |
| 128,000 | 19.53 GiB | 209.7 ms | 104.9 ms | 52.4 ms |

对照 W1（300–460 ms）：**要吃光整个 W1，TP=2 需要 18 万个冗余 token，TP=8 需要 73 万个。**
远超任何真实 prompt 长度。

**浪费是自限的**：本地命中高 → 浪费大，但 L3 本来就没价值；
本地命中低 → 浪费小，而 L3 价值高。两者不会同时踩雷。

**真正该防的是带宽，不是延迟。** 按每请求浪费 1.75 GiB、TP=2 总带宽 100 GB/s 估算：
QPS=10 会吃掉 17.5% 的 L3 带宽，QPS=30 会超过一半。**延迟上没事，吞吐上会撞墙。**

建议的闸门：**早期预取跳过前 X 个 token**（X = 典型本地命中长度）。
哈希链仍从 root 算以保持各 rank 一致，只是不读前 X 页；
少读的部分由主线程走原路径补上，功能上完全安全。

---

## 6. 为什么现在还不能下结论

**§4 §5 的全部数字来自合成负载。它们足以说明机制，不足以支撑上线决策。**

### 6.1 现有数据的局限

| # | 局限 | 影响 |
|---|---|---|
| 1 | **`random-ids` 随机 token，请求之间零前缀共享** | L1/L2 命中恒为 0。而冗余读量 = `min(本地命中, L3命中)`，**这个方案最核心的代价在现有数据里根本没被测到** |
| 2 | **闭环压测，固定并发** | 真实流量是泊松到达 + 突发。W2 的分布（尤其是"当场录取"的比例）会不同 |
| 3 | **未开启 HiCache** | 开启后 `wait_complete` 会让 W2 包含预取等待时间，形成循环依赖；真实的 W2 分布需要在开启状态下重测 |
| 4 | **prompt 长度固定** | 真实负载是分布。W1 由 prompt 长度决定，所以 W1 也是分布，不是单值 |
| 5 | **triton backend，无 flashinfer** | 迭代时间偏长 → W1 偏大。调优过的部署上 W1 会更小，比例结论可能改变 |
| 6 | **单机 TP=2，`dp_size=1`** | 真实部署常是 TP=8 / 多 DP。DP 路由是方案的已知缺口 |
| 7 | **冗余读的带宽占用未在真实 QPS 下测** | §5.4 的吞吐风险是算出来的，不是测出来的 |

**其中第 1 条最致命**：整个方案的收益（提前 W1）已经测了，
但代价（冗余读）从头到尾是**估算**的，因为合成负载里没有前缀共享。

### 6.2 需要在真实负载上测什么

**必测（决定方案是否成立）**

1. **L1 / L2 / L3 三级命中长度的联合分布**
   → 直接给出冗余读量 `min(本地, L3)` 的真实分布，
     以及 §5.4 那张表该取哪一行
2. **W1 / W2 的真实分布**（不是单点）
   → 特别是 W2 落在"几毫秒"区间的请求占比，那才是这个方案的受益面
3. **prompt 长度分布**
   → W1 由它决定

**应测（决定参数和闸门）**

4. 真实 QPS 下冗余读占 L3 带宽的比例 → 决定要不要 §5.4 的"跳过前 X 页"闸门，X 取多少
5. 到达过程（泊松 / 突发）对"当场录取比例"的影响
6. TP=8 下的表现（W1 的 TP broadcast 成本随 TP 增长）

**验证方式**

7. **A/B**：开 / 关早期预取，对比 TTFT 分布、L3 命中率、L3 带宽占用
8. 三种 `prefetch-policy` 在开启早期预取前后的表现对比
   → 预期：`best_effort` 在低并发下从"结构性失效"变为可用

### 6.3 建议的落地顺序

| 阶段 | 内容 | 产出 |
|---|---|---|
| **0** | 把 W1 / W2 / `iters` 埋点做成 env 开关的常驻埋点，加上三级命中长度采集 | 线上可观测，**这是后面一切的前提** |
| **1** | 真实负载采集，得到 §6.2 的 1–3 项分布 | 判断方案是否值得做 |
| **2** | 实现旁路 hint + 早期查询（`dp_size==1`，早期阶段无 all_reduce） | A/B 验证 |
| **3** | 视结果决定是否让读取也提前（需要独立的 host staging arena，风险最高） | — |
| **4** | hint 发出点移到 `data_parallel_controller`，支持 DP | — |

**建议阶段 0 和 1 先做——它们成本低、无风险，而且没有它们，
阶段 2 之后的任何数字都无法解释。**

---

## 7. 与前几个阶段的关系

这个方案和 01–04 的优化是**正交**的，作用在流水线的不同环节：

```
现状：     [等主循环 W1]  查L3(往返①)  [等主线程分配]  读L3(往返②)   [被录取]
                                        ↑ 只有 W2 这么点时间可用

早期预取： 查+读 都挪到 W1 里做                          →           [被录取]
合并 q+f： [等主循环 W1]        读L3(唯一一次往返)                    [被录取]
两个都做： 读L3(一次往返) 在 W1 里就做完                              [被录取]
```

| | 早期预取 | 合并 query+fetch |
|---|---|---|
| 收益条件 | 长 prompt + 低并发（实测 conc ≤ 8） | **无条件** |
| 依赖负载 | 是 | 否 |
| 改动面 | 新 socket + 新线程 + 认领分支 + TP/DP 一致性 | 主要在 `cache_controller` 内部 |
| 主要风险 | all_reduce 不一致导致挂死 | host 内存过度占用 |

**如果只能做一件，合并 query+fetch 的性价比更高**——改动面更小、风险可退化
（内存紧张就退回两趟）、收益不挑工作点。两个都做时效果叠加。

---

## 附录 A：可复现的测量步骤

```bash
# 服务端
CUDA_VISIBLE_DEVICES=0,1 SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
PYTHONPATH=/home/lpl/sglang/python \
python -m sglang.launch_server \
  --model-path /home/lpl/models/Llama-3.3-70B-Instruct-FP8 \
  --tp 2 --attention-backend triton --disable-flashinfer-autotune \
  --port 31000 --max-running-requests 256

# 压测（固定 prompt 长度，只扫并发）
for C in 8 16 24 32 48 64; do
  python -m sglang.benchmark.serving --backend sglang --port 31000 \
    --dataset-name random-ids --random-range-ratio 1.0 \
    --random-input-len 4096 --random-output-len 32 \
    --num-prompts 120 --max-concurrency $C --disable-tqdm
done
```

埋点补丁（临时，已回滚）：
- `observability/req_time_stats.py`：`APIServerReqTimeStats.__getstate__` 传播
  `tokenize_finish_time` / `api_server_dispatch_time`；`SchedulerReqTimeStats` 加对应字段
- `managers/scheduler.py`：`_add_request_to_queue` 里在 `_prefetch_kvcache` 前打 W1 终点；
  `_get_new_batch_prefill_raw` 里在 `set_time_batch(..., "set_forward_entry_time")` 后打 W2 终点

**注意**：FP8 权重会触发 flashinfer autotune，本机版本不匹配会启动失败，
需加 `--disable-flashinfer-autotune`。

## 附录 B：术语

| 术语 | 含义 |
|---|---|
| L1 | GPU 显存里的 KV cache |
| L2 | host 内存里的 KV cache（与 L1 共用一棵前缀树） |
| L3 | 外部存储（文件系统 / GPFS / KV store）里的 KV cache |
| W1 | 分词完成 → Scheduler 里 `match_prefix` 即将开始 |
| W2 | `match_prefix` 开始 → 请求被录取进 prefill |
| 同步点 | 主循环里所有 rank 都必须参与的集合通信；host 池只在同步点之后改动 |
| 当场录取 | 请求在到达的同一轮调度循环里就被选进 prefill batch（`iters == 0`） |
