# 阶段 13：并发 8 为什么是"一起准入"，而不是"一部分两轮一部分三轮"

| | |
|---|---|
| 日期 | 2026-09-22 |
| 模型 | Llama-3.3-70B-Instruct-FP8，TP=2 |
| 存储 | GPFS `/zion0`，O_DIRECT |
| 配置 | page 512、命中前缀 16384、后缀 512、8 探针并发 8、背景 6 路、`max_total_tokens` 163840、`max_concurrent_streams` 8 |
| 脚本 | `benchmark/hicache/bench_steady_state_l3_probe.py` |
| 前一阶段 | [10 并发预取](10-concurrent-prefetch.md)、[12 零 IO](12-zero-io-and-the-admission-sync.md) |

> **结论**：8 个请求一起准入，**不是因为它们在等读**。准入那一刻，全部 8 个的
> 整段读早已完成。525 ms 的排队时间里**没有一毫秒花在等 I/O 上**，全部是
> 调度线程上的**串行 CPU 工作**。
>
> **L3 的每请求准入开销是 L2 的 26 倍**（20.0 ms vs 0.76 ms）。
> 从入队到 prefill **只经历两个迭代**，但第二个迭代被这些串行工作撑成了 384 ms
> （正常一轮 143 ms）。
>
> 我事先列的三个假设（`batch_is_full` 短路、`NO_TOKEN` 容量拒绝、扫描提前返回）
> **全部被埋点否掉了**。

---

## 1. 问题

阶段 10 测到并发 8 时 L3 比 L2 多等约 3 轮调度。但按流水线的设计，
第 0 组读完就该准入；8 个请求的读是错开提交的，读完时刻也该错开。
**理想应该是一部分请求两轮、一部分三轮，而不是齐刷刷一起进同一个批次。**

三个候选解释：

1. `batch_is_full` 是粘性备忘，扫描被短路，请求没被及时发现；
2. `add_one_req` 返回 `NO_TOKEN`，容量不够，请求被拒后 `break`；
3. 调度器走了 `check_hicache_events` 后面的提前返回，压根没扫等待队列。

---

## 2. 埋点

在 `get_new_batch_prefill` 的等待队列循环上加计数器和计时，循环结束后打一行：

```
scan: queued=%d visited=%d skip_prefetch=%d added=%d stop=%s check_ms=%.1f add_ms=%.1f
```

- `visited` — 循环实际访问到的请求数
- `skip_prefetch` — `check_prefetch_progress` 返回 False 而被 `continue` 的数量
- `stop` — `ran-to-end` / `break:batch_is_full` / `break:add_one_req=<原因>`
- `check_ms` — 花在 `check_prefetch_progress` 上的总时间
- `add_ms` — 花在 `init_next_round_input` + `add_one_req` 上的总时间

---

## 3. 三个假设全错

L3 臂探针阶段只有两趟带 8 个请求的扫描：

```
scan: queued=8 visited=8 skip_prefetch=8 added=0 stop=ran-to-end check_ms=1.2  add_ms=0.0
scan: queued=8 visited=8 skip_prefetch=0 added=8 stop=ran-to-end check_ms=63.3 add_ms=160.3
```

- **`visited=8`**：每一趟都遍历了全部 8 个，没有漏看 → 假设 1 否
- **`stop=ran-to-end`**：从来没有 `break` → 假设 1、2 都否
- **两趟都打印了**：扫描确实跑了，没有被提前返回吃掉 → 假设 3 否

真相是：第一趟 8 个全部被 `check_prefetch_progress` 挡回（`skip_prefetch=8`），
第二趟 8 个全部通过（`added=8`）。**两趟之间翻转。**

---

## 4. 为什么两趟之间翻转

### 4.1 上游的门：`has_staged`

```python
# unified_radix_cache.py:1795
if self.layerwise_bridge is not None and self.layerwise_bridge.has_staged(req_id):
    return self.layerwise_bridge.check_progress(req_id)
if req_id not in self.ongoing_prefetch:
    return True
...  # 老路径：can_terminate_prefetch() → False
```

第一趟扫描时 8 个请求都还没 staged（阶段 B 尚未给它们建事务），走的是老路径直接返回 False。
所以那一趟只花了 1.2 ms——**它根本没碰流水线**。

### 4.2 `poll()` 推进的是全部事务，不是当前这一个

```python
# controller.py
def poll(self) -> None:
    """Advance every active transaction; called once per scheduler step."""
    for entry in tuple(self._active.values()):
        self.pipeline.advance(entry.transaction)
```

`check_progress` 第一句就是 `self.controller.poll()`。所以循环里**第 0 个请求**
那一次调用，就把全部 8 笔事务各推进一步。

但**不是"8 个同时就绪"**。一次 `poll()` 是顺序遍历，每笔事务的 `advance()` 要做
跨 rank 确认和第 0 组 H2D 提交，约 0.9 ms，所以 7 笔的 `admission_ready` 铺开在
6 ms 里（302 / 303 / 305 / 307 / 307 / 307 / 308）。

**第 8 笔（提交最晚的那个）在这次 `poll()` 里第 0 组还没好，没有转入就绪。**
它要等循环里下一个请求的 `check_progress` 触发第二次 `poll()`，才在 +328 ms 就绪。
这正是 `advances` 字段的含义——它记的是**该事务转入就绪时被推进过几次**，
7 笔是 1，最后一笔是 2。

就绪的判据只看第 0 组：

```python
# pipeline.py
def _group0_ready(self, transaction) -> bool:
    """Admission waits for agreement, not just for the local read."""
    return transaction.machine.group(0).state in (
        GroupState.GLOBAL_READY, GroupState.H2D_SUBMITTED,
        GroupState.DEVICE_READY, GroupState.RETIRED,
    )
```

### 4.3 `g0_local_done` 量的不是读，是"等调度线程回头看"

`g0_submit + g0_local_done` 全部收敛到同一瞬间（18 ms 之内）：

```
167.660 + 166.18ms = 167.826      167.743 +  86.50ms = 167.830
167.671 + 157.04ms = 167.828      167.773 +  56.82ms = 167.830
167.700 + 128.41ms = 167.828      167.794 +  36.94ms = 167.831
167.729 + 100.38ms = 167.829      167.814 +  30.10ms = 167.844
```

**先提交的 `g0_local_done` 反而最大（166 ms），最后提交的最小（30 ms）。**
这个字段是在 `_advance_consensus` 里盖的，而 `_advance_consensus` 只在调度线程
调用 `poll()` 时才跑。所以它量的是"提交完到调度线程回头看还要等多久"，
**不是读耗时**。

> 阶段 09–12 的表里我把这一列当作"读完成时刻"读过，是错的。

---

## 5. 完整时序图

单位是相对入队时刻的毫秒，全部来自埋点实测（`scanwork2-pc8`）。

```
时刻        调度线程                                   预取线程      IO 线程 x8
==============================================================================
 t1  +0ms   收到 8 个探针（并发发出）
            +- 8 个一起入队
            +- 查 L1/L2 -> 全部落空
            +- 发起 8 个存储查询 -------------------->  查 8x32 个 page
            +- 排空存储命中队列 -> 空的，查询还没回来
            +- 扫描等待队列：queued=8 visited=8
            |    check_prefetch_progress x8 -> 全 False   <- 读还没提交
            |    skip_prefetch=8 added=0  check_ms=1.2
            +- 建批次（只有 6 个背景请求）
      <== L2 探针就是在这一趟被准入的（queue 5.1ms，add_ms 6.1ms）==
            跑 forward
            |
            +--- +0 -> +142ms ---  实测（= t2 的阶段B 时刻减去入队）
                                   forward 占其中多少没有单独量
                                            查完，8 个结果入队
------------------------------------------------------------------------------
 t2 +142ms  排空存储命中队列（check_hicache_events）
            +- 请求0：分配 host 1.25GiB，提交读 ------------->  开始读 0
     +154ms +- 请求1：分配 host，提交读        ------------->  开始读 1
     +170ms +- 请求2 ...                                   ->  开始读 2
     +210ms +- 请求3 ...                                      整读0 完 +189
     +225ms +- 请求4 ...                                      整读1 完 +215
     +251ms +- 请求5 ...                                      整读2 完 +222
     +271ms +- 请求6 ...                                      整读3 完 +251
     +294ms +- 请求7：提交读                                   整读4 完 +269
            ^ 这 152ms 全在调度线程上串行，21ms/请求            整读5 完 +289

     +302ms 扫描等待队列：queued=8 visited=8
            +- 请求0 的 check_prefetch_progress -> poll() 遍历全部 8 笔事务
            |    +- 事务0：第0组已就绪 -> 宣布准入 +302
            |    +- 事务1：                       +303
            |    +- 事务2：                       +305
            |    +- 事务3/4/5：                   +307 +307 +307
            |    +- 事务6：                       +308   <- 第0组早好了，
            |    |                                          整条读到 +318 才完
            |    +- 事务7：第0组还没好 -> 不动
            |       （每笔约 0.9ms：跨 rank 确认 + 提交第0组 H2D）
            +- 请求1 的 check_prefetch_progress -> poll() 又遍历一遍
            |    +- 事务7：第0组好了 -> 宣布准入 +328   <- 所以只有它 advances=2
            +- 请求0..7：init_next_round_input + init_load_back  整读6 完 +318
            |    check_ms=63.3  add_ms=160.3 -> 共 224ms，20ms/请求
     +526ms +- 建批次 -> set_forward_entry_time                  整读7 完 +421
      <== L3 探针到这里才被准入 ==
            跑 forward（探针自己的 prefill，约 2.1 秒）
==============================================================================
```

### 图上三个容易读错的地方

**① "整读N 完"是整条前缀 10 个组读完，不是第 0 组。**
准入只查 `group(0)`（见 §4.2 的 `_group0_ready`），第 0 组只占 1/10（约 128 MiB），
而且以 `_ADMISSION_PRIORITY=0` 提交，其余 9 组是 `_READ_AHEAD_PRIORITY=2`。
事务 6 的第 0 组在 +308 前就好了，整条读到 +318 才完；这不矛盾。

**② 事务 7 是流水线按设计工作的唯一直接证据。**
它 +328 就被宣布准入，整条读 +421 才结束，**提前了 93 ms**。
如果准入真要等整条读完，它不可能在 328 就就绪。所以问题不在准入条件上。

**③ 从入队到 prefill 只经历了两个迭代。**
三条证据：`queued=8` 的扫描日志只有两条；早返回的条件是
`(batch_is_full or len(waiting_queue)==0) and chunked_req is None`，队列里有 8 个，
只可能被 `batch_is_full` 挡住，而两趟扫描都是 `stop=ran-to-end`，不会置上它；
最后是算术——t1 到 t2 相隔 **142 ms**，而一个背景解码迭代实测 **142.9 ms**，
中间再夹一轮就该是 286 ms。

> 142.9 ms 来自更早一轮实验（n=7）。本轮的 `Decode batch` 日志只有两条且都跨了
> 臂边界（算出 3125 / 1840 ms），没法自证迭代周期。

### "两个迭代"不等于"两轮的时间"

第二个迭代**不是一个正常的 143 ms 迭代**：

```
queue 526 ms  =  t1 剩余 142 ms          <- 正常的一轮背景解码
              +  t2 准入前的 384 ms      <- 被串行工作撑大的
                   +- 142->294  提交 8 个读        152 ms
                   +- 294->302  扫描开始             8 ms
                   +- 302->526  扫描循环           224 ms
```

**迭代数是 2，但时间是 3.7 轮。** 用"526 ÷ 143 = 折合几轮"来描述这件事是有害的：
它读起来像"等了 3 轮调度"，实际是"等了 2 轮，但第二轮被撑成了 2.7 倍"。

这个区别决定了改法：如果真是"多等一轮"，把那一轮省掉就解决了；
实际上第二轮里那 376 ms 的串行 CPU 工作跑不掉，除非把阶段 B 和
`init_load_back` 从调度线程上挪走。

### 请求 0 白等的 337 ms

它的数据 +189ms 就全部就位，+526ms 才被调度，中间**一毫秒都不是 I/O**：

```
189 -> 294    调度线程还在给另外 7 个请求分配 host、提交读
302 -> 526    扫描循环自己的 CPU 工作（check 63 + init_load_back 160）
```

**阶段 B 和扫描在同一根线程上，而且扫描排在阶段 B 后面。** 所以第一个读完的请求，
必须等它 7 个同伴都提交完读，扫描才轮得到它；轮到之后还要陪着这 8 个一起做完
224 ms 的准入工作，才一起进批次。

---

## 6. 525 ms 是怎么花掉的

按入队时刻对齐（均值，n=8，两次独立运行）：

| 段 | 第一次 | 第二次 | 内容 |
|---|---|---|---|
| 入队 → 第 0 组提交 | 209 | **215** | 阶段 B（host 分配 + 提交读）在调度线程上**串行**，8 个摊开 142→294 ms，约 **27 ms/请求** |
| 提交完 → 宣布准入 | 93 | **94** | 纯等下一趟扫描 |
| 宣布准入 → 实际准入 | 219 | **217** | 扫描循环本身：`check_ms` 63.3 + `add_ms` 160.3 |
| **合计** | 522 | **526** | 实测 L3 queue 525.8 |

`set_forward_entry_time` 在 `scheduler.py:3539`，**就在扫描循环之后几行**，
不是下一轮——所以第三段全部花在循环体内部，被 `check_ms`/`add_ms` 直接量到了。

### 同一份数据的另一种拆法：墙钟

上面那张表是**按请求取均值**（8 个请求各自那一段的平均），回答的是"平均每个
请求各段等了多久"。问"这 526 ms 里调度线程在干什么"要用墙钟，两者都等于 526，
但均值会把提交的错峰抹平——它把 152 ms 的串行提交报成 215 ms，其中 142 是在等。

```
  0 -> 142    142 ms   还没有任何读被提交。阶段 A 的存储查询在预取线程上
                       并行跑完，这段是在等调度线程下一趟（约一个背景迭代）
142 -> 294    152 ms   阶段 B 串行提交 8 个读
294 -> 302      8 ms   扫描开始
302 -> 526    224 ms   扫描循环遍历 8 笔事务（check 63.3 + add 160.3）
              ------
               526 ms
```

摊到每个请求：**阶段 B 19 ms/个，扫描循环 28 ms/个**
（其中 `init_load_back` 20 ms、`check_prefetch_progress` 7.9 ms）。

两个大头是**阶段 B 的串行提交**和**扫描循环里的 `init_load_back`**；
`check_prefetch_progress` 本身只占 63.3/526 = 12%，而且那 63 ms 里还包含
第一次调用的 `controller.poll()` 真正推进 8 笔事务（跨 rank 确认 + 提交第 0 组
H2D），不是轮询开销。两个 rank 的 `check_ms` 差很多（63.3 vs 38.7）是
`_agree()` 的跨 rank 等待，不是工作量差异——两边 `check_ms + add_ms` 合计
都是约 224 ms。

### 读完全不是瓶颈

同一轮里：

```
整段读 span           均值 59.2 ms，带宽 24.3 GiB/s
整段读结束时刻         均值 272 ms（最早 189，最晚 421）
准入扫描发生在         ~308 ms
实际准入              526 ms
```

**准入扫描发生时，8 个请求的整段读（不只是第 0 组）绝大多数已经结束。**
它们不是在等 I/O。

---

## 7. 和 L2 的对照：每请求准入开销 26 倍

同一份日志里 L2 臂的 8 个探针分两趟进（`added=3` 和 `added=5`）：

| | `check_ms` | `add_ms` | 每请求准入开销 |
|---|---|---|---|
| L2（8 个，分两趟） | 0.0 | **6.1** | **0.76 ms** |
| L3（8 个，一趟） | 63.3 | **160.3** | **20.0 ms** |

同一个调度器、同样的请求形状、同样的批大小。差的 19 ms/请求是
`init_load_back`（设备分配 + 树插入 + 开 H2D 会话）的代价，
和阶段 12 里单请求无背景测到的 Δqueue 19 ms **是同一笔钱**——
只是这里乘了 8 倍后变成了主项。

`check_ms` 的 63.3 ms 是 8 次 `poll()`（每次推进 8 个事务）加 8 次
`_agree()` 跨 rank all-reduce，次要项，但也不小。

---

## 8. 回答原问题

**"为什么不是一部分两轮一部分三轮"**，三层原因，从表层到根本：

1. **准入的粒度是"一趟扫描"，不是"一个请求"。** 调度线程一轮只扫一次等待队列，
   一趟扫描会把所有已就绪的请求一起带走。
2. **就绪判定确实是错开的，但错开的量级太小。** 一次 `poll()` 遍历全部事务，
   7 笔在 302–308 ms 内依次就绪，第 8 笔要等第二次 `poll()`，在 328 ms 就绪
   （`advances=2`）。**但这 26 ms 的错开全发生在同一趟扫描内部**，
   而批次是扫描结束时才建的，所以它们还是进同一个批次。
3. **最根本的：前提就不成立。** "先读完的先准入"假设读是瓶颈，但整条读在
   189–421 ms 就结束了，第 0 组更早，而请求卡在 526 ms。
   它们等的是调度线程把 384 ms 的串行 CPU 工作做完。

**从入队到 prefill 只经历了两个迭代**（证据见 §5③），不是三个。
但第二个迭代不是正常的 143 ms，而是 384 ms——152 ms 提交读 + 224 ms 准入循环，
本身就横跨约 2.7 个背景迭代。所以"折合 3 轮"这种说法是有害的：
**迭代数是 2，被撑大的是第二轮。**

---

## 9. 顺带发现：`max_concurrent_streams` 默认是 1

`hicache_storage_max_concurrent_streams` 默认值为 **1**，超出的请求走
`radix_bridge.start()` 里的 `if self.busy(): return False`，退回阻塞整段读。

有一轮实验漏设这个参数，意外得到了一组同批次内的 A/B（均值，n=8）：

| | queue |
|---|---|
| L2 | 9.0 ms |
| L3，7 个走**阻塞整段读** | **352.4 ms** |
| L3，1 个走**流水线** | **2069.9 ms** |

走流水线的那个在 7 个同伴被编进批次的那一趟扫描里 `skip_prefetch=1`——
第 0 组还没就绪，**错过了这一趟**。代价不是一轮，是 **1718 ms**：
批次一旦成形，容量被 7×16896 token 吃掉，掉队的要等到有请求腾出槽位。

**"错过一趟扫描"的惩罚是非线性的**，不是"再等一轮 ≈ 170 ms"。
这是单样本且和容量耦合，只当线索，没有单独验证。

---

## 10. 不确定的地方

1. **`add_ms` 160.3 ms 没有再往下拆。** `init_next_round_input`（树匹配）和
   `add_one_req`→`init_load_back`（设备分配 + 树插入 + H2D 会话）合在一起计时，
   分不开。阶段 12 的单请求数据指向 `init_load_back` 是大头，但没有直接埋点。
2. **`check_ms` 63.3 ms 里 `poll()` 和 `_agree()` 的比例未知。**
3. **"只有两个迭代"依赖一个跨运行的数字**：本轮无法自证背景迭代周期，
   142.9 ms 取自更早一轮（见 §5③）。要彻底坐实得加每轮迭代起止的埋点。
4. **§9 是单样本**，且流水线臂只有 1 个请求，和容量效应耦合。
5. **两次运行的三段拆分高度一致**（209/93/219 与 215/94/217），但都只有 n=8 一批探针。
6. **绝对 TTFT 不代表生产性能**：沿用单请求基准配置（关闭分块预填充、无 CUDA graph）。

---

## 11. 怎么复现

```bash
cd /home/lpl/sglang/benchmark/hicache
/home/lpl/sglangtest/.venv/bin/python bench_steady_state_l3_probe.py \
  --server l3_fused --io-threads 8 --page-size 512 --max-concurrent-streams 8 \
  --hit-tokens 16384 --suffix-tokens 512 --probes 8 --probe-concurrency 8 \
  --background-requests 6 --max-total-tokens 163840 --hicache-ratio 6.0 \
  --store-root /zion0/kv-aio-bench/<独立目录> --run-id <名字>
```

**`--max-concurrent-streams 8` 不能省**（默认 1，7 个请求会退回阻塞路径，见 §9）。

扫描埋点是加在 `get_new_batch_prefill` 上的临时代码，不在仓库里。

原始数据：`benchmark/hicache/results/l2l3_fusion/scanprobe-pc8/`（第一次）、
`scanwork2-pc8/`（第二次，带 `check_ms`/`add_ms`）、
`scanwork-pc8/`（漏设 `max_concurrent_streams` 的那一轮，即 §9）
