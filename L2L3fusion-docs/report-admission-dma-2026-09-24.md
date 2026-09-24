# 并发场景最大的一项：准入路径和流水线自己抢 DMA

2026-09-24 · 分支 `L2-L3fusion-admission-latency` · 续 [准入延迟探索](report-admission-latency-2026-09-24.md)

---

## 一句话结论

并发 8 时，`init_load_back` 里有 **17.1 ms/请求**（×8 = **137 ms**，占 533 ms 排队的 **26%**）
花在一行代码上：

```python
# cache_controller.py, move_indices()
elif self.mem_pool_host.layout == "page_first_direct":
    return host_indices, device_indices.cpu()
```

**128 KB 的 D2H 拷贝，排在流水线自己发起的 1.25 GiB H2D 后面等复制引擎。**

**空闲基准测不出来**：同一行在无背景、顺序单请求下是 **0.28 ms**。这解释了为什么上一轮
四项基于空闲分解做的优化在并发场景只换来 −7.2%（还在噪声里）。

**没有修掉**——修法需要动一族共享的分配器，超出这次能安全验证的范围。§6 给了三条路线。

---

## 1. 怎么找到的

上一轮结束时，并发场景最大的未解释项是扫描循环里的 `init_load_back`：
空闲 **4 ms/请求**，并发 **20 ms/请求**，**5 倍膨胀没有解释**。

把它切成六段，两个场景用同一套埋点测：

| 步骤 | 无背景、顺序 | 并发 8 | 倍数 |
|---|---|---|---|
| `build_key` | 0.73–1.12 | 0.25 | — |
| `match_prefix` | 0.03 | 0.03 | — |
| `alloc_device` | 0.50–0.59 | 0.21 | — |
| **`attach`** | **0.28–0.34** | **17.09–17.28** | **50×** |
| `insert` | 2.86–2.92 | 2.61 | — |

**膨胀全在 `attach` 一步上**，其余各步甚至略快（并发时 CPU 缓存更热）。

`attach` 里是 `controller.attach_device()` → `cache_controller.start_streaming_load()`。
给它加四段计时：

```
producer=0.03  move_indices=20.41  transfers=0.01  merge+session=0.10
```

**`attach` 的开销 100% 是 `move_indices`**，其余三段合计 0.12 ms。

---

## 2. 四次假设，四次被实验否掉

每一次都缩小了范围，所以都记下来。

### ① "`.cpu()` 在等计算流排空"

`device_indices` 是 `free_pages[:n]` 的切片视图，不是 GPU 计算结果；128 KB 拷贝本身约 10 µs。
所以第一反应是：它在等计算流上排队的背景解码。

**实验**：把拷贝放到独立 CUDA 流上。
**结果**：`move_indices` 仍 17.5–22.9 ms。

### ② "侧流没生效，所以 ① 还没被否掉"

**这个实验本身是无效的**：我用的是 `.to("cpu", non_blocking=True)`，而目标是**非锁页**内存——
PyTorch 在这种情况下会**静默退回同步拷贝**，侧流根本没起作用。

**用一个没有真正生效的干预去否定假设**，是这一轮犯的第一个方法论错误。

### ③ 直接把"等流"和"拷贝"分开计时

```
wait_stream=0.01  copy=21.72  n=16384 contig=True
wait_stream=0.02  copy= 0.03  n=16384 contig=True
```

显式 `current_stream().synchronize()` **只要 0.01–0.02 ms**——计算流是空的。
**慢的是拷贝本身**，而且在 21.7 和 0.03 之间跳变。

同一段代码、同样 128 KB、同样连续内存，**差 700 倍**（5.8 MB/s vs 4 GB/s）。
不是计算量，不是队列。

### ④ "是背景解码占住了 GPU"

**实验**：并发 8 但 `--background-requests 0`。
**结果**：`copy` 仍是 18.5–22.4 ms，`attach` 11.9–12.7 ms。**不是背景解码。**

### ⑤ "非锁页目标强制走同步路径"

**实验**：锁页目标 + 独立流，真正的异步 DMA。
**结果**：

```
pinned_copy=22.79 / 0.05 / 0.04 / 22.45 / 22.63 / 0.08 / 0.04 / 22.55 ...
```

**仍在 22 ms 和 0.05 ms 之间跳变。** 锁页不是答案。
（首次 `cudaHostAlloc` 确实要 22 ms，但每 rank 只有一次，不是这里的原因。）

---

## 3. 剩下的解释：和流水线自己的 H2D 抢复制引擎

排除了流等待、背景解码、锁页路径之后，剩下的是 **DMA 引擎被占满**。

量级对得上：流水线的 H2D 会话每事务要搬 **1.25 GiB/卡**，按实测的 ~50 GiB/s 算是 **25 ms**，
而慢的那些拷贝正好是 **22 ms**。

顺序发请求时每个会话搬完才轮到下一个请求，所以空闲基准测到 0.28 ms；
并发 8 时会话重叠，后面请求准入路径上的 128 KB D2H 就卡在一整次 1.25 GiB 传输后面。

> **这一条是推断不是直接测量。** 排除法加量级吻合，但我没有直接观测复制引擎的占用。
> 直接证据需要 nsys 之类的工具，没做。

---

## 4. 这次往返本身是多余的

`device_indices` 来自 `token_to_kv_pool_allocator.alloc()`，实现是：

```python
select_index = self.free_pages[:need_size]
self.free_pages = self.free_pages[need_size:]
return select_index
```

**纯 CPU 侧的切片**，值是分配器自己管理的整数。它们在 GPU 上，是因为 `kernel` io 后端的
索引核需要；而 `direct` + `page_first_direct` 这套配置需要它们在 CPU 上。

**于是索引走了一圈没有必要的往返**：CPU 管理的空闲列表 → GPU 张量 → DMA 拷回 CPU，
还要和流水线自己的数据搬运抢引擎。

而且**准入时并不使用这些 CPU 值**——`_l2_transfers` 只是把两个张量包进 `L2Transfer`，
真正用到是后面提交传输区间的时候（`_pump` 里，forward 期间）。

---

## 5. 为什么上一轮的优化没有传导

上一轮四项优化（整组入队、计划去重、跳过冗余轮询、延后 read-ahead）合计把
**空闲单请求** 的 Δqueue 从 39.04 压到 25.63（−34%），但**并发 8** 只从 520.7 到 483.3
（−7.2%，两轮跨度 59 ms > 改善量）。

现在知道原因了：**那四项都是基于空闲分解设计的，而空闲分解里 `attach` 只有 0.28 ms。**
并发场景真正的大头从来没进过我的视野。

**空闲基准会系统性地漏掉同步类和争用类瓶颈。** 这类问题只在有并发负载时显形，
而且现象（某一步突然慢几十倍）和 CPU 瓶颈完全不同。

同类的坑之前踩过一次：`torch.equal` 那个 GPU 同步也是只在准入路径、只在有负载时显形。
**两次都是"调度线程上的一次 CUDA 操作"**，但第一次是等同步，这次是抢 DMA——
现象相同，根因不同。

---

## 6. 三条修法路线

| 路线 | 做什么 | 预期 | 障碍 |
|---|---|---|---|
| **A. 消除往返** | 分配器维护一份 CPU 镜像，`alloc` 同时返回 GPU 视图和 CPU 值 | 137 ms/扫描 | `TokenToKVPoolAllocator` 有一族子类（Unified / SWA / NPUPaged…），`_alloc_device` 泛型调用；改动跨整族共享热路径 |
| **B. 推迟转换** | `move_indices` 推迟到第一次提交传输区间时（`_pump` 里，forward 期间） | 队列 −119 ms，forward +17 ms，**净 −100 ms** | `L2Transfer` 也被 `start_loading` 用，要确认惰性转换对两条路径都安全 |
| **C. 扫描分两趟** | 先把 8 个请求的设备槽位和索引转换全做完，再逐个开 H2D 会话 | 只有第一次拷贝慢 | `init_load_back` 在准入器深处被调用，重构侵入性大 |

**B 最可行**：改动局限在 `cache_controller` 和 `l2_transfer`，而且它把成本从
**所有请求共享的扫描循环**挪进**每个请求自己的 forward**——第 8 个请求原本要等前 7 个的
`attach`（7 × 17 = 119 ms）才能被准入。

**不值得做**：锁页缓冲（§2⑤ 已证无效）、侧流（§2① 已证无效）。

---

## 7. 数字汇总

```
并发 8 的 Δqueue = 533 ms
  ├ 入队 → 提交第0组读    ~199
  ├ 第0组读               ~100
  └ 宣布准入 → 进批次     ~235
        ├ init_load_back × 8   ~160
        │    └ attach × 8       137   ← 本报告定位的项，占总排队 26%
        │         └ move_indices 100%
        └ check_prefetch × 8    ~46
```

如果 `attach` 降到空闲时的 0.28 ms，Δqueue 应降到约 **395 ms（−26%）**。
这是个算术外推，不是实测。

---

## 8. 不确定的地方

1. **§3 的"抢 DMA"是推断**：排除法 + 量级吻合，没有直接观测复制引擎。需要 nsys 才能坐实。
2. **每档单次运行**，`attach` 的 17.1 ms 在多次运行里稳定（17.09 / 17.28 / 17.31 / 17.30 / 17.44 / 16.27），
   但 Δqueue 的轮间噪声达 59 ms。
3. **§7 的 395 ms 是外推**，没有实测。要验证需要先实现 §6 的某条路线。
4. **只测了 `page_first_direct` + `direct` io 后端**。其他布局走 `move_indices` 的别的分支，
   `kernel` 后端根本不做这次 D2H。
5. **没有测 TTFT 本身**，只测了 queue。forward 不受这一项影响，所以 TTFT 的改善量应与 queue 相同。

---

## 9. 怎么复现

埋点已从分支剥离，重做需要临时加回：

```python
# cache_controller.py, start_streaming_load 里分四段计时
# cache_controller.py, move_indices 的 page_first_direct 分支里拆 wait_stream / copy
# radix_bridge.py, init_load_back 里分六段计时，累加后每 8 次打一行
```

跑法（`--background-requests 0` 用来做 §2④ 的判别）：

```bash
cd /home/lpl/sglang/benchmark/hicache
/home/lpl/sglangtest/.venv/bin/python bench_steady_state_l3_probe.py --server l3_fused \
  --io-threads 8 --page-size 512 --max-concurrent-streams 8 \
  --hit-tokens 16384 --suffix-tokens 512 --probes 8 --probe-concurrency 8 \
  --background-requests 6 --max-total-tokens 163840 --hicache-ratio 6.0 \
  --store-root /zion0/kv-aio-bench/<独立目录> --run-id <名字>
```

原始数据：`results/l2l3_fusion/` 下的 `step-c8`、`step-nobg`（六段分解）、`att-c8`（四段分解）、
`side-c8`（无效的侧流实验）、`d2h-c8`（等流/拷贝拆分）、`nobg-c8`（去背景判别）、`pin-c8`（锁页实验）。
