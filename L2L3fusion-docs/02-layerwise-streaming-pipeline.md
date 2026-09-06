# 阶段 02：分层流水线接入 UnifiedRadixCache

| | |
|---|---|
| 阶段 | 02 |
| 日期 | 2026-09-06 |
| 前一阶段 | [01-direct-io-l3-backend.md](01-direct-io-l3-backend.md) |
| 状态 | 单请求流式加载已在真机跑通并验证 KV 逐位正确；多请求并发和 admission 后失败重试未做 |
| 开关 | `--hicache-storage-load-mode layerwise` |

> 阶段 01 让 L3 **读得快**（924 → 444 ms）。
> 这一阶段让 L3 的读取**藏起来**：一次命中不再等整段前缀读完才开始算。

---

## 1. 一句话结果

在有计算可以掩盖的场景下，**存储读取被完全从 TTFT 里挤掉了**：

| 缓存前缀 8.8k token，后面接的新内容 | 不用流水线 | **用流水线** | 省了 |
|---|---|---|---|
| ~6 token（几乎没有计算窗口） | 462 ms | **390 ms** | 72 ms |
| ~2100 token | 731 ms | **460 ms** | **271 ms** |
| ~4200 token | 1058 ms | **751 ms** | **307 ms** |

（两边用的是同一个 `layerwise_file` 存储后端，唯一区别就是流水线开没开，
所以省下的时间纯粹是流水线的贡献。）

省下来的量稳定在 **~300 ms**，正好等于这段前缀的读盘时间。
换句话说：**只要后面有活干，读盘就不再出现在首字延迟里**。

---

## 2. 为什么之前不能这样做

模型是一层一层算的。算第 0 层的时候，它只需要第 0 层的 KV，
第 35 层的 KV 要到很久以后才用得上。

但原来的流程是"**全都读完才让请求进 GPU**"：

```
时间 ──────────────────────────────────────────────────►
存储读取  ████████████████████████████ 355ms
主机→显存                              ███ 23ms
GPU 计算                                  ████████ 后缀计算
                                                  ↑ 首字
```

流水线之后：

```
时间 ──────────────────────────────────────────────────►
存储读取  ██ ██ ██ ██ ██        ← 读第 g+1 组层
主机→显存    ██ ██ ██ ██ ██     ← 拷第 g 组层
GPU 计算        ████████████    ← 算第 g-1 组层（后缀）
                            ↑ 首字提前到这里
```

第 0 层是唯一藏不住的（前面没有任何计算），所以它被单独拎出来：
单独提交、最高 I/O 优先级、跨 rank 确认成功后才放请求进 GPU。

---

## 3. 这一阶段解决的核心难题：一个死锁

这是设计上最不显然的一点，值得单独讲。

模型的每一层都会调用 `wait_until(layer)`：**"第 L 层的 KV 拷进显存了吗？没有我就等。"**
而 SGLang 关掉 overlap 调度后，**模型 forward 跑在调度线程上**。

流水线也需要有人推进（收割存储完成、提交下一组的拷贝），
而它同样在调度线程上。于是：

```
调度线程 ──► 模型 forward ──► wait_until(第5层) ──► 阻塞等待
                                                    ↑
                              没有人能去推进流水线，因为唯一的线程正卡在这里
                                                    │
                              第5层的数据永远不会被提交 ────┘
```

**死锁。**

### 解法：等待的时候顺手干活

让这个"等待"本身去推进流水线：

```python
while 第L层还没提交:
    pump()          # 收割存储完成 → 提交下一组的 H2D
    短暂让出
```

语义上非常自然：**"GPU 正在等第 L 层，那就用 CPU 去把第 L 层取回来。"**

为什么不用后台线程？因为跨线程提交 CUDA 操作和分布式集合通信的风险高得多。
单线程"边等边泵"没有任何竞态，代价只是等待期间 CPU 在忙——反正它本来也没别的事做。

> 跨 rank 的一致性确认走 SGLang 缓存自己的 gloo 通信组，
> **不是模型的 NCCL TP 组**。在 forward 的两层之间插一个集合通信，
> 如果和模型自己的通信共用一个 communicator，NCCL 会按顺序错配，直接挂死。

---

## 4. 数据流：一次 L3 命中现在走哪些步骤

```
请求进入等待队列
   │
   ├─ (沿用原有机制，后台线程) 算页 hash、查 L3 存在性、跨 rank 对齐命中长度
   ├─ (沿用原有机制) 分配主机 staging 内存
   │
   ▼  ← 新增：从这里开始由流水线接管，不再交给阻塞 IO 线程
提交第 0 组（第 0 层），最高优先级
   │
   ▼
第 0 组所有 rank 确认成功
   │
   ▼  ← 请求获准进入 GPU（此时后面 35 层还在飞）
分配显存槽位、打开分层 H2D 会话、挂上 pump
   │
   ▼
模型 forward 开始
   ├─ wait_until(0) → 已就绪，直接算
   ├─ wait_until(1) → pump: 收存储完成 → 提交第 1 组 H2D → 算
   ├─ wait_until(9) → pump: ... → 算
   └─ ...
   │
   ▼
forward 结束 = 所有层都拷完了
   │
   ▼
主机 staging 发布成正常的 L2 节点（下次同样前缀直接 L2 命中）
```

### 一个关键的安全约束

主机 staging 在事务完成前**不挂进 radix tree**。
否则另一个请求会命中这段前缀，然后去读那些**后面几层还没到**的 KV，
产生静默的错误输出。它只在"所有组确认 + forward 消费完"之后才成为正常的 L2 条目。

---

## 5. 正确性怎么验的（以及我一开始验错了）

第一次验证我这样做：

- 冷启动跑一遍（全量重算）→ 记录输出
- 清缓存，再跑一遍（走 L3 流式）→ 记录输出
- 对比

结果 **3 次里 2 次不一样**，看起来像是 KV 被写坏了。

但这个对比是**错的**。做了对照实验才发现：

| 对比 | 结果 |
|---|---|
| 全量重算 vs L1 命中（KV 就在显存里，什么都没搬） | **有时也不一样** |
| L1 命中 vs L3 流式 | **3/3 完全一致** |

第一行说明：**"用缓存的前缀"和"全量 prefill"本来就可能有微小的浮点差异**
（不同的 kernel 路径 / tile 形状），在模型犹豫不决的位置会导致 argmax 翻转。
这是 SGLang 固有的特性，不是我引入的。

第二行才是真正的检验：两边都是在读缓存的 KV，
唯一区别是这些字节**怎么进的显存**。逐位一致 = 流水线搬运正确。

> 教训：验证的时候要控制变量到只剩下你改的那一件事。
> 拿"重算"当基准，测的是别人的问题。

另外还验了：服务器日志里 **0 次走旧路径、0 次降级**——
说明每一次 L3 命中都确实是流水线处理的，不是悄悄退回去了。

---

## 6. 改了哪些东西

### 新增

| 文件 | 作用 |
|---|---|
| `layerwise_storage/radix_bridge.py` | **桥接层**。所有 layerwise 专属逻辑都在这里：接管命中、驱动流水线、admission 消费、释放。缓存那边只留 6 个很小的 hook |

### 修改（都是加分支，关掉开关就走原路）

| 文件 | 改动 |
|---|---|
| `managers/cache_controller.py` | 层门支持 pump（解死锁）；流式会话吞并同批次的普通 load 队列 |
| `mem_cache/unified_radix_cache.py` | 6 个 hook：构造桥接层、路由命中、汇报 admission、消费 staging、认领 ack、汇报 producer |
| `managers/scheduler.py` | 把"私有 staging 要报成 host_hit_length"的条件从 buffer 模式放宽到通用（1 行） |
| `mem_cache/base_prefix_cache.py` | 新增 `holds_staged_prefetch` 属性 |
| `mem_cache/layerwise_storage/controller.py` | 复用存储后端的 writer/身份/对齐，保证读写 key 一致 |
| `server_args.py` | layerwise 模式强制要求 `layerwise_file` 后端 |

### 为什么"6 个 hook"很重要

`unified_radix_cache.py` 有 2900 行，是核心文件。
所有实质逻辑都放在新模块里，缓存那边只有 6 处形如
`if self.layerwise_bridge is not None: ...` 的分支。
关掉开关，这些分支一个都不会进，原路径**字节级不变**。

### fail-closed

第一版覆盖不到的情况一律**不启用流式，退回原来的阻塞读**，并打日志说明原因：

- 主机池不是 MHA（比如 MLA 模型）
- 有 side pool / sidecar pool（流式只搬 KV）
- 有多个缓存组件（SWA、Mamba）
- buffer_only 模式
- 后端不是共享页存储

这个守卫在真机上第一次启动就救了我一次：
`cache_controller.mem_pool_host` 其实是个 `HostPoolGroup` 包装，不是 KV 池本身。
守卫直接拒绝启用并说明原因，而不是拿着错误的指针去读盘。

---

## 7. 怎么用

```bash
python -m sglang.launch_server \
  --model-path <模型> \
  --enable-hierarchical-cache \
  --hicache-storage-backend layerwise_file \
  --hicache-storage-load-mode layerwise \
  --hicache-storage-first-group-layers 1 \
  --hicache-storage-group-size 8 \
  --hicache-storage-read-ahead-groups 2 \
  --hicache-io-backend direct \
  --hicache-mem-layout page_first_direct \
  --hicache-write-policy write_through \
  --hicache-host-memory-mode cache \
  --disable-overlap-schedule \
  --chunked-prefill-size -1 \
  --page-size 64
```

三个旋钮的含义：

- `first-group-layers=1`：第 0 组只放 1 层。它是唯一藏不住的，越小越好。
- `group-size=8`：后续每组 8 层。太小则每个 I/O 请求太碎（阶段 01 测过：
  2 层只能拿到裸带宽的 53%，8 层能到 94%）。
- `read-ahead-groups=2`：往前预读 2 组。在途组数上限是这个值 +1。

启动日志里应该看到：

```
Layerwise storage streaming enabled: first_group=1 layers, group=8 layers, read_ahead=2 groups
```

如果看到 `Layerwise storage streaming disabled: unsupported ...`，
说明落到了 fail-closed 的某一条，会自动退回原来的阻塞读。

---

## 8. 还没做的

| # | 事项 | 现状 |
|---|---|---|
| 1 | **多请求并发流式** | 一个时刻只允许一个。第二个请求自动退回阻塞读（不报错，只是没加速） |
| 2 | **admission 之后存储失败** | 直接抛异常。显存槽位已经发布在树里且只填了一半，继续跑会输出没读到的 KV。计划里的 poison/replay 还没做 |
| 3 | **TP > 1 验证** | 跨 rank 逐组确认的代码写了、单测过了，但只在 TP=1 上真机跑过 |
| 4 | **和正在跑的 decode 混合** | 没测 |
| 5 | **读写仲裁真机验证** | 仲裁器（第 0 层 > demand > 预读 > 写回）写了，没在真实混合负载下压过 |
| 6 | **MLA / SWA / Mamba** | fail-closed 挡住，不支持 |

第 1、2 项是 serving-ready 的硬门槛。

---

## 9. 附：性能怎么算出来的

对 8.8k token 前缀（Qwen3-8B TP=1，1.21 GiB KV）：

```
存储读取     355 ms   （3.4 GiB/s 的单块 NVMe）
主机→显存     23 ms   （PCIe 实测 52.8 GiB/s）
后缀计算    随后缀长度线性增长
固定开销      29 ms   （发一个 11 字符请求也要这么久）
```

不用流水线：`TTFT = 固定 + 读取 + H2D + 后缀计算`（全都串起来）
用流水线：  `TTFT = 固定 + 第0层读取 + max(读取, H2D, 后缀计算)`

代入 4200 token 后缀（计算 ≈ 589 ms）：

```
不用流水线： 29 + 355 + 23 + 589 + 杂项 ≈ 1058 ms   ← 实测 1058 ms
用流水线：   29 + 少量 + max(355, 589) + 杂项 ≈ 751 ms   ← 实测 751 ms
```

模型和实测吻合得很好，说明这个理解是对的：
**流水线把"相加"变成了"取最大值"，但它不能突破存储带宽本身。**

后缀短的时候（没有计算可以掩盖），`max()` 里最大的还是读取，
所以收益只有省下的那点 H2D 串行时间——这也解释了为什么 ~6 token 的场景只快了 72 ms。
