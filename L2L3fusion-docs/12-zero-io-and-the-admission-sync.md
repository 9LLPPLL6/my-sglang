# 阶段 12：把 IO 置零之后，L3 还剩多少

| | |
|---|---|
| 日期 | 2026-09-22 |
| 模型 | Llama-3.3-70B-Instruct-FP8，TP=2 |
| 存储 | GPFS `/zion0`，O_DIRECT |
| 负载 | **无背景请求**，单请求逐个发，8 个探针 |
| 探针 | 命中前缀 16384 token + 新后缀 512 token，page 512 |
| 补丁 | `benchmark/hicache/fake_storage_io.patch`（不常驻仓库） |
| 前一阶段 | [11 page size 与前缀长度](11-page-size-and-prefix-length.md) |

> 前面十一个阶段一直在间接论证"L3 的开销不是存储"：换个办法让读变快，看 TTFT 不跟。
> 这一阶段**把磁盘直接去掉**，再把背景请求也去掉，于是剩下的只有 L3 的代码路径。
>
> 结果：**纯代码路径 46 ms，纯 IO 17 ms。** 而之前在真实负载下测到的 180–590 ms 里，
> **80–90% 是调度排队**。
>
> 更进一步，那 46 ms 里**一半以上是一行防御性检查**——它让调度线程同步等 GPU 23–45 ms。

---

## 1. 怎么做的

**零 IO**：短路点在最内层，`LinuxAioContext._submit` 不下发 `io_submit` 系统调用，直接把请求登记成已完成，`_get_events` 从伪队列取。**仲裁器、队列深度、分片、extent 状态机、共识、准入门控、层门、H2D 全部照常执行**——唯一不发生的是内核那一步。

目标缓冲区从未被写入，所以计算用的 KV 是 host 里的陈旧字节，**生成文本无意义，只有延迟数字可用**。硬证据是进程级读盘计数器：**0.00 GiB / 预期 20.00 GiB**。

**无背景请求**：调度线程空转，`check_prefetch_progress` 几乎连续在跑，于是阶段 08 那两个各约 160 ms 的会合窗口塌缩到接近零。剩下的 queue 时间就是预取流程自己的真实工作。

两个臂（L2 / L3）在同一个服务进程内先后跑，服务端参数完全相同。**注意 L2 臂也开着 L3 后端**，所以它也会为那 512 个新 token 发一次注定落空的存储查询——无背景时这次查询的代价≈0（撤销在同一圈内完成）。

---

## 2. 时刻表

全部是**距离请求入队的毫秒数**，8 个探针均值。每一列单调递增。

| 时刻(ms) | L2 | L3 零IO | L3 真IO |
|---|---|---|---|
| ① 第 0 组提交（流水线发起） | — | 6.7 | 6.9 |
| ② 读开始（首个 extent 进 AIO） | — | 7.0 | 7.0 |
| ③ 其余 9 组提交 | — | 14.3 | 11.8 |
| ④ **第 0 组落地** | — | **14.7** | **26.6** |
| ⑤ 流水线宣告可准入 | — | 14.7 | 26.6 |
| ⑥ 全部 10 组读完 | — | 17.7 | 41.9 |
| ⑦ **请求被准入** | **1.6** | **47.5** | **64.7** |
| ⑧ 请求完成 | 284.8 | 330.3 | 347.5 |

其他量：

| | L2 | 零IO | 真IO |
|---|---|---|---|
| TTFT 均值 | 291.5 | 336.5 | 353.8 |
| TTFT 标准差 | 5.0 | 5.5 | 8.5 |
| queue 均值 | 1.6 | 47.5 | 64.7 |
| **forward 均值** | **283.3** | **282.8** | **282.8** |
| 轮询次数 `advances` | — | 1.0 | 15.1 |
| 读盘 | 0.00 G | **0.00 G** | 20.00 G |

**forward 三档完全相同**，差异 100% 落在 queue 里。

---

## 3. 三条直接读出来的结论

### ① 早准入是兑现了的

⑤ 和 ④ 完全重合（14.7/14.7、26.6/26.6，共识零成本），而且**早于⑥全部读完**——真 IO 是 26.6 vs 41.9，早 15 ms。

之前在有背景时看不出这一点，是被调度窗口盖住了。设计意图（第 0 组落地就准入，其余组在 forward 后面补齐）在这里第一次被直接观察到。

### ② 磁盘只值 12–17 ms

第 0 组是整条前缀的 1/10：**128 MiB/卡、64 个 extent**（整条 1.25 GiB/卡、32 个 page、640 个 extent）。

```
零IO  第 0 组落地  14.7 ms
真IO  第 0 组落地  26.6 ms
                  ──────
第 0 组的磁盘时间  11.9 ms
```

按 queue 总账算是 17.2 ms（63.1 − 45.9），多出的约 5 ms 是准入之后其余 9 组的收尾，只漏出来一小部分。

### ③ 纯代码路径 46 ms

```
按 TTFT 均值:   代码路径 45.6 ms   IO 16.7 ms
按 queue 均值:  代码路径 45.9 ms   IO 17.2 ms
```

两条独立算法差 0.5 ms。

---

## 4. 那 46 ms 的大头是一行防御性检查

把⑤→⑦（流水线说"好了"到请求真正进批次）拆开。这一段是 `radix_bridge.init_load_back` 在做的事：

| 步骤 | 耗时 |
|---|---|
| build_key（构造 RadixKey 并 page 对齐） | 0.24 ms |
| match1（全量 `match_prefix`，校验跨度还能接上） | 0.03 ms |
| alloc_device（分配 16384 个设备槽位） | 0.25 ms |
| attach_insert（`attach_device` + 插基数树） | 2.93 ms |
| match2（第二次全量 `match_prefix`，所有权复核） | 0.14 ms |
| **verify（`torch.equal` 比对 16384 个索引）** | **22.6 – 35.0 ms** |
| 合计 | 26 – 38 ms |

两次全量 `match_prefix` 加起来只要 0.17 ms——基数树匹配很快，**我原本猜它是大头，猜错了**。

### 它不是在比较，是在等 GPU

在 verify 之前插一个显式 `torch.cuda.synchronize()` 分别计时：

| | 探针 A | 探针 B |
|---|---|---|
| `cuda_sync`（显式等 GPU 排空） | 23.45 | 22.61 |
| `verify`（`torch.equal` 本身） | 21.73 | **0.11** |

**`cuda_sync` 稳定在 22–23 ms。** 探针 B 的 verify 只剩 0.11 ms——**那才是真正比较 16384 个索引的成本**。探针 A 的 21.7 ms 是因为显式 sync 之后到 `torch.equal` 之间又有新的 GPU 工作排进来（H2D 拷贝线程在持续推送），它再等了一次。

那行代码：

```python
# radix_bridge.py  init_load_back 结尾
if len(match.device_indices) < span_end or not torch.equal(canonical, device_indices):
    raise LayerwiseStreamError(
        "... the insert freed or replaced slots the in-flight streaming H2D still targets"
    )
```

两个张量都在 GPU 上（`token_to_kv_pool_allocator.alloc()` 返回设备索引），而 `torch.equal` 要返回 Python `bool`，**必须同步 GPU**。

### 为什么这是个设计矛盾

流式流水线的全部意义是：**H2D 异步跟在 forward 后面，靠层门逐层门控**，调度线程不必等数据到齐。

而这行复核在 **forward 还没开始**时，就把调度线程钉在 GPU 上等了 23–45 ms。它把流水线想避免的同步又加了回来，只是加在了更早的位置。

而且它**只在流式路径上执行**（`init_load_back` 是 bridge 的方法），所以 100% 计入"L3 相对 L2 的额外开销"。阶段 10 的 trace 里 `aten::equal` 出现在"L3 独有帧"列表（8 次、23.4 ms），指的就是它。

---

## 5. 三个场景并排

| 场景 | Δ(L3 − L2) | 代码路径 | IO | 调度排队 |
|---|---|---|---|---|
| **无背景、单请求** | **62 ms** | 46 | 17 | ~0 |
| 有背景、单请求 | ~180 ms | 46 | 17 | **~117** |
| 有背景、并发 8 | ~587 ms | 46 | ~25 | **~516** |

优化天花板由此确定：

```
流水线内部（分组 / extent 粒度 / 并发流 / 多线程）   ≤ 46 ms
  └ 其中一半以上是那一行同步                        23–45 ms
存储与带宽                                          ≤ 17 ms
准入时机（预分配 / 提前预取 / 主动唤醒）             117 – 516 ms
```

前面几个阶段花在读上的功夫，**天花板就是那 17 ms**。

---

## 6. 不确定的地方

1. **`torch.equal` 等的到底是哪些 GPU 工作，没有查清。** 候选是这笔事务自己刚发起的 H2D 拷贝（第 0 组 128 MiB/卡）和插树产生的 GPU 操作，但我没有用 CUDA event 定位过。
2. **修法未验证。** 想到三条（在 CPU 侧比、记 event 在层门第一次等待时顺带校验、论证不变量已由锁引用计数保证），都没做。哪条可行取决于槽位所有权的具体保证。
3. **零 IO 下缓冲区是陈旧字节**，理论上随机比特可能解出 NaN/Inf 影响 kernel 时间。张量核矩阵乘是定长的，风险很低，但没有单独验证。
4. **L2 基线是"同一台开着 L3 的机器上的 L2 命中"**，不是"不开 L3"。开启 L3 对 L2 命中本身的代价（那次落空查询）另计——无背景时≈0，有背景时约一轮调度（阶段 09 测得 128 ms）。
5. **每档单次运行**，8 个探针。标准差 5.0 / 5.5 / 8.5 ms，相对 46 和 17 这两个差值足够小。
6. **绝对 TTFT（约 290 ms）不代表生产性能**：沿用单请求基准配置（关闭分块预填充、triton、无 CUDA graph）。三臂同配，相对差值可信。

---

## 7. 怎么复现

```bash
cd /home/lpl/sglang

# A: 真 IO
/home/lpl/sglangtest/.venv/bin/python benchmark/hicache/bench_steady_state_l3_probe.py \
  --server l3_fused --io-threads 8 --probe-concurrency 1 --max-concurrent-streams 1 \
  --page-size 512 --hit-tokens 16384 --suffix-tokens 512 --probes 8 \
  --background-requests 0 --max-total-tokens 163840 --hicache-ratio 6.0 \
  --store-root /zion0/kv-aio-bench/<目录A>

# B: 零 IO
git apply benchmark/hicache/fake_storage_io.patch
SGLANG_TEST_HICACHE_FAKE_STORAGE_IO=1 <同样的命令，换 --store-root>
git apply -R benchmark/hicache/fake_storage_io.patch
```

`--background-requests 0` 是关键：它让调度会合窗口塌缩，把"等"压到接近零，剩下的才是"做"。

准入内部的六段计时（§4 那张表）需要另一段临时埋点，加在 `radix_bridge.init_load_back` 里，跑完撤掉——本文档写作时用的就是这种方式，没有留在仓库里。

原始数据：`benchmark/hicache/results/l2l3_fusion/nobg-{realio,fakeio,admitwork,syncsplit}/`
