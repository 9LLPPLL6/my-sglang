# 阶段 06：中等并发下，GPFS 上的 L3 离 L2 还有多远

| | |
|---|---|
| 阶段 | 06 |
| 日期 | 2026-09-17 |
| 前一阶段 | [05-request-path-and-prefetch-window.md](05-request-path-and-prefetch-window.md) |
| 状态 | 已实测；两个臂的层级归属都用读盘字节数交叉验证 |
| 模型 | Llama-3.3-70B-Instruct-FP8，TP=2 |
| 存储 | GPFS `/zion0`（56T，实测聚合 43.36 GiB/s） |
| 脚本 | `benchmark/hicache/bench_concurrent_l2_vs_l3.py` |
| 分支 | `L2-L3fusion-continuous-read` |

> 01–04 都是单请求测量：一个请求、一个前缀、一次读。
> 这一阶段第一次在**并发**下问：L3 的 TTFT 离 L2 还有多远？
>
> 答案是 **+198 ms（1.06×）**——而且这还是在四分之三的请求
> **根本没走上流水线**的情况下测出来的。

---

## 1. 一句话结果

| 臂 | n | TTFT p50 | p90 | p99 | max | 总墙钟 |
|---|---|---|---|---|---|---|
| **L2** | 24 | **3426 ms** | 4092 | 4766 | 4766 | 11.36 s |
| **L3** | 24 | **3624 ms** | 4057 | 4062 | 4062 | 10.93 s |

- 中位数只差 **198 ms（1.06×）**。
- **p90 之后 L3 反而更快**（p99 4062 vs 4766），总墙钟也少 0.4 秒。
  L2 那两个 4766 ms 的长尾在 L3 里没出现。
- 但 L3 臂里只有 **6/24** 个请求走了流式路径，**18/24 退回了阻塞读**——
  这是 V1 的"同时只允许一个流式事务"限制，见 §4。

**结论：在 GPFS 这种高带宽存储上，中等并发下 L3 已经非常接近 L2；
而且现在的数字还留着一大块没兑现的余量。**

---

## 2. 实验设计

### 2.1 两个臂，一个服务端

复用 `bench_l2l3_fusion_ttft.py` 的 `l3_fused` 服务端配置，两个臂**只差前缀在哪一层**：

| 臂 | 准备动作 | 期望命中 |
|---|---|---|
| `l2` | 预热前缀（写穿到 L3）→ 只驱逐显存 | host |
| `l3` | 预热前缀 → `flush_cache` 清掉显存和 host → 让存储变冷 | storage |

同一个进程、同一份配置，所以两臂之差就是"这段 KV 从哪来"。

### 2.2 每个请求一个独立前缀

这是设计上最关键的一点。如果 24 个请求共享同一个前缀，**第一个到达的请求会把它拉进 L2，后面 23 个就变成 L2 命中了**，L3 臂被污染。

所以每个请求生成自己的 4096-token 随机前缀，24 个互不重叠。

### 2.3 参数

```
hit_tokens   = 4096    每请求独立前缀
suffix       = 1280    S/H = 0.31，正好在阶段 02 推导的掩盖拐点上
requests     = 24
concurrency  = 8       线程池
page_size    = 64
group_size   = 8       first_group_layers = 1
```

容量核对（全部有余量）：

```
GPU KV 需要   8 × (4096+1280) = 43,008  ≤ max_total_tokens 65,536
host 需要     24 × 4096 + 65,536 = 163,840  ≤ hicache_ratio 8 × 65,536 = 524,288
L3 每臂       24 × 4096 × 160 KiB = 15.0 GiB
```

### 2.4 一处必须打的补丁

`bench_l2l3_fusion_ttft.py` 的 `base_args` 里硬编码了 **`--max-running-requests 1`**。
那是单请求归因的正确选择，在这里完全错误——会把整波请求串行化。

驱动脚本在**运行时**把它覆盖成 `--max-running-requests <concurrency>`，
其余参数一律沿用被审计过的那套，以保证两臂同配：

```python
def patch_base_args(concurrency: int) -> None:
    original = B.base_args
    def patched(args, port):
        argv = original(args, port)
        i = argv.index("--max-running-requests")
        argv[i + 1] = str(max(concurrency, 1))
        return argv
    B.base_args = patched
```

---

## 3. 两个臂是干净的（证据）

| 证据 | L2 臂 | L3 臂 |
|---|---|---|
| 层级归属（每请求均值） | host 4096 / storage 0 | host 1024 / storage 3072 |
| 进程读盘字节 | **252 KiB**（≈0） | **15.00 GiB = 预期 KV 的 100%** |
| 写回校验 | store grew 15.00 GiB | store grew 15.00 GiB |
| 日志 declined / dropped / disabled / traceback | 0 | 0 |

**L3 臂整份 KV 确实是从 GPFS 读回来的**，不是 pagepool 糊弄过去的。

### 后端自报的读取，和 18/24 精确吻合

```json
{"ranks": 2, "batches_per_rank": 18, "l3_pages": 2304,
 "l3_bytes": 12079595520, "l3_ms": 349.97, "l3_open_ms": 90.14,
 "l3_io_ms": 259.46, "l3_gibps": 21.57, "l3_agg_gibps": 43.36}
```

算一下：

```
2304 页 = 18 请求 × 64 页/请求 × 2 rank              ✓ 精确
2304 页 × 5 MiB/页/rank = 11.25 GiB = 18/24 × 15.00  ✓ 精确
```

**这两条算术精确成立，等于独立证明了恰好 18 个请求走的是阻塞读路径**，
剩下 6 个走流式（流式的读由流水线驱动，不出现在这些日志行里）。

聚合 **43.36 GiB/s** 也和阶段 03 实测的 GPFS 41.8 GiB/s 对得上，是真实网络读。

---

## 4. 关键发现：四分之三的请求没走上流水线

```python
# radix_bridge.py
if self.busy() or self._staged:
    return False    # One streaming transaction at a time; the rest use the old path.
```

拆开两条路径看：

| 路径 | n | TTFT p50 | min | max |
|---|---|---|---|---|
| 流式（记为 host 命中） | 6 | **2102 ms** | 551 | 3672 |
| 退回阻塞读 | 18 | **3687 ms** | 1524 | 4062 |

**走上流式的请求，TTFT 中位数比阻塞读低 1585 ms。**

所以 §1 那个"+198 ms"是**四分之三请求走老路径**的结果。
如果把并发流式做出来，L3 在这个工作点上**有机会和 L2 打平甚至更快**。

### 它在哪里被触发

```
调度主循环（每轮）
 └─ _get_new_batch_prefill_raw()
     └─ tree_cache.check_hicache_events()
         └─ _drain_and_alloc_storage_hit()          按 TP-min 计数排空 prefetch_hit_queue
             └─ _try_alloc_storage_hit(operation)
                 ├─ mem_pool_host.alloc(...)        给这笔命中分配 staging
                 └─ if layerwise_bridge.start(operation):   ← unified_radix_cache.py:2231
                        return True                  流式接管
                    cc.prefetch_buffer.put(operation) # ← 拒绝后落到这里：老的阻塞读
```

`start()` 返回 `False` 时不报错、不重试，直接把 operation 塞进老路径的入口队列，
由 `prefetch_io_aux_func` 整段读完，**没有分层、没有提前准入、没有流水线**。

---

## 5. 为什么 GPFS 上差距这么小

```
读 15 GiB @ 43.36 GiB/s 聚合  ≈ 350 ms
8 并发下每请求 TTFT            ≈ 3400 ms
```

**存储读只占 TTFT 的 10%，而且分摊在 8 个并发请求里还能互相重叠。**

这印证了阶段 03 的结论：GPFS 带宽足够高时，读取本来就没多少可藏的。
反过来说——在本地单盘 NVMe（3.45 GiB/s）上同样的 15 GiB 要读 4.3 秒，
那时差距会完全是另一个量级。**这个结论只对高带宽存储成立。**

---

## 6. 口径局限

1. **绝对 TTFT 偏高（3.4 秒）**。服务端沿用单请求 bench 的配置：
   `--chunked-prefill-size -1`（关闭分块）、triton backend、无 CUDA graph。
   8 个请求的 prefill 全排在一起。两臂同配所以相对比较可信，**绝对值不代表生产**。
2. **GPFS pagepool 无法证伪**（阶段 03 §6 已记录：fadvise/mincore 在 GPFS 上是空操作）。
   本次 `--churn-gib` 用默认 0。读盘满额 15 GiB + 43 GiB/s 这两个数字说明读真发生了，
   但不能排除部分由 pagepool 供给。
3. **只测了一个工作点**（H=4096, S=1280, conc=8）。并发扫描、前缀长度扫描都没做。
4. **随机 token，请求间零前缀共享**。真实负载里请求之间常有共享前缀，
   那会改变 L1/L2 的命中结构（见阶段 05 §6.1 的同一条局限）。

---

## 7. 下一步：并发流式

### 7.1 真正的阻碍不在 bridge，在层门

```python
class LayerDoneCounter:
    num_counters = 3
    events = [LayerLoadingEvent(num_layers) for _ in range(3)]   # 生产者环
    consumer_index = -1        # ← 整个 batch 只等一个
    stream_pump = None         # ← 整个进程只有一个 pump

    def wait_until(self, threshold, timeout=None):
        if self.consumer_index < 0:
            return
        self.events[self.consumer_index].wait(threshold, ...)    # 只等这一个
```

**forward 算到第 k 层时只会等一个生产者到第 k 层。**
两个请求在同一个 batch 里各有各的流，现在的数据结构表达不了。

`start_streaming_load` 的注释承认了这点：

> A batch carries one consumer index, so an ordinary load-back queued by another
> request in the same batch **has to ride this session**; a second producer would
> leave that request's layers ungated.

现在能把别人排队的 load-back 合并进同一个 session，是因为那些请求的 host 数据
**已经完整**、只是搬运。第二个**流式**请求的数据还没到，合并不了。

### 7.2 好消息：跨 rank 一致性不是障碍

`TorchDistGroupConsensus` 已经按 `(transaction_id, group_id)` 分键，
而且用 `async_op=True` + 轮询，不阻塞调度循环：

```python
self._pending: dict[tuple[str, int], tuple] = {}
```

两笔事务的 collective 天然分开、互不干扰，只要各 rank 发起顺序一致——
而发起顺序由 `_advance_consensus` 按组序驱动，事务集合又来自同一个 broadcast 的请求流。

**这块不用改。这本来是最危险的部分。**

### 7.3 方案：把"等一个"改成"等所有在飞的流"

```
forward 算到第 k 层
  └─ wait_until(k)
       ├─ 等流 A 到第 k 层
       ├─ 等流 B 到第 k 层      ← 新增
       └─ 等普通 load-back 到第 k 层
```

batch 按最慢的那条流推进。听起来像退步，但对比现状——现在第二个请求
**整段退回阻塞读**，实测慢 1585 ms——等一条稍慢的流远好过走老路。

| # | 位置 | 改什么 |
|---|---|---|
| 1 | `LayerDoneCounter` | `consumer_index: int` → 一组 `(index, generation)`；`wait_until` 遍历 |
| 2 | `LayerDoneCounter.num_counters` | 3 → 至少 `最大并发流 + 2` |
| 3 | `LayerwiseRadixBridge` | `_streaming_req_id` → `dict[req_id, producer_index]`；`busy()` 改成容量判断 |
| 4 | `stream_pump` | 单个 → 推进所有在飞事务（`controller.poll()` 本来就遍历 `_active`，主要是错误归属） |

连带一处：`ready_to_load_host_cache()` 现在返回单个 int 给 `batch.hicache_consumer_index`。
**建议让 counter 内部维护"流式消费者集合"、batch 那个 index 只管普通 load-back**，
这样不用碰 `ScheduleBatch` 的字段。

### 7.4 最容易踩的坑

```python
def update_producer(self):
    self.producer_index = (self.producer_index + 1) % self.num_counters
    assert self.events[self.producer_index].finish_event.query(), \
        "Producer finish event should be ready before being reused."
```

生产者环按 `% num_counters` 轮转。并发流式下，第 N+1 个流会绕回到一个
**还在飞**的槽位 → **断言失败，进程挂掉**。

所以第 2 项不是"调大一点就行"：**必须有一个硬性的并发流上限，
且 `busy()` 要按这个上限拒绝**，不能让 producer 环绕回来。

### 7.5 分两步

- **第一步**：并发流上限从 1 提到 N（比如 4），结构照旧，batch 按最慢的流走。
  预计 4 个文件、200 行量级 + 单元测试。
- **第二步**（视结果再定）：per-request 层门，不同请求的同一层可以在不同时刻就绪。
  需要 attention 侧知道哪些行就绪，**深得多，先别碰**。

---

## 8. 怎么复现

```bash
python benchmark/hicache/bench_concurrent_l2_vs_l3.py \
  --model /home/lpl/models/Llama-3.3-70B-Instruct-FP8 \
  --store-root /zion0/kv-aio-bench/l3-conc \
  --store-max-size 200Gi --store-min-free 200Gi \
  --max-total-tokens 65536 --hicache-ratio 8 \
  --hit-tokens 4096 --suffix-tokens 1280 \
  --requests 24 --concurrency 8 \
  --port 31931 --run-id conc-gpfs-c8
```

本次结果：`benchmark/hicache/results/l2l3_fusion/conc-gpfs-c8/`
（`l2.json` / `l3.json` 含逐请求 TTFT 与层级归属，`summary.json` 是两臂合并）。

脚本自己的参数只有三个（`--concurrency` / `--requests` / `--suffix-tokens`），
其余全部透传给 `bench_l2l3_fusion_ttft.py` 的 parser，所以模型、存储、
分组大小这些旋钮的用法和单请求 bench 完全一致。

FP8 权重会触发 flashinfer autotune，本机版本不匹配，
`base_args` 里已有 `--disable-flashinfer-autotune`。
