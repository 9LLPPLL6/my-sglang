# 阶段 01：Direct I/O 页文件 L3 后端 + 分层流水线地基

| | |
|---|---|
| 阶段 | 01 |
| 日期 | 2026-09-04 |
| 提交 | `[HiCache] Add a Direct I/O page-file L3 tier and the layerwise streaming foundation` |
| 状态 | `layerwise_file` 后端已在真机验证可用；分层流水线组件已建好并单测，尚未接入主流程 |
| 下一阶段 | 把流水线接进 `UnifiedRadixCache`（见本文第 6 节） |

> 面向对 SGLang 内部实现不熟悉的读者。读完你应该能回答三个问题：
> 我们改了什么、为什么新的 L3 比原来快、以及还差什么没做。

---

## 0. 一句话总结

原来的 L3（磁盘/并行存储层）**一页一页串行地读**，队列深度永远是 1，
所以一次 L3 命中和"干脆重算一遍"一样慢，等于白存。
新的 `layerwise_file` 后端**把一次命中的所有页一次性压进异步 I/O 队列**，
并用 `O_DIRECT` 直接落到目标内存，实测 TTFT 从 924 ms 降到 444 ms。

---

## 1. 背景：HiCache 的三层是什么

大模型推理时，已经算过的 token 会留下一份"KV cache"。同样的前缀再来一次，
直接把 KV 读回来就行，不用重算。SGLang 把这份缓存分三层存：

```
L1  GPU 显存      最快，最小       ——  几十 GB
L2  主机内存(DRAM) 中等             ——  几百 GB
L3  磁盘/并行存储  最慢，最大        ——  几 TB 甚至更多
```

一次请求进来，SGLang 按 L1 → L2 → L3 的顺序找有没有现成的 KV：

- **L1 命中**：什么都不用做，直接算。
- **L2 命中**：把 KV 从主机内存拷到显存（H2D），很快。
- **L3 命中**：先从磁盘读到主机内存，再从主机内存拷到显存。

问题出在最后一条。

### 这个项目的目标

让 L3 快到**上层感觉不到它的存在**——对调度器和模型来说，
就好像 L2 的容量凭空扩大了一个 L3 那么多。
原来是 `L1(a) + L2(b) + L3(c)`，目标是 `L1(a) + L2(b+c)`。

注意这是**语义上的融合，不是数据搬家**：L2 和 L3 物理上还是两层，
只是 L3 快到不值得单独感知了。

---

## 2. 问题有多严重：实测数据

测试环境：Qwen3-8B，单卡（TP=1），page size 64，8832 token 的缓存前缀。

> 测之前必须先清 page cache（`posix_fadvise(DONTNEED)`）。
> 不清的话，老后端读的其实是内存里的副本，测出来的是内存速度不是磁盘速度，
> 数字会漂亮但完全没有意义。这一步很容易被忽略。

| 路径 | 首字延迟 (TTFT) |
|---|---|
| 不用缓存，直接重算这段前缀 | 931 ms |
| L3 命中，**原来的 `file` 后端** | **924 ms** |
| L3 命中，**新的 `layerwise_file` 后端** | **444 ms** |
| L2 命中（数据已在主机内存） | 51 ms |

看第二行：**原来的 L3 命中（924 ms）和重算（931 ms）一样慢**。
辛辛苦苦把 KV 存到磁盘上，一分钱便宜都没占到。这就是"L3 太慢"的具体形态。

新后端快了 2.08 倍，终于比重算划算了。
但离 L2 的 51 ms 还有距离——那是下一阶段（分层流水线）要解决的。

---

## 3. 为什么原来的 L3 慢

### 3.1 关键：队列深度是 1

原来的 `file` 后端（`hicache_storage.py` 里的 `HiCacheFile`）是这样读的：

```python
def batch_get(self, keys, target_locations, ...):
    return [self.get(key, loc) for key, loc in zip(keys, target_locations)]
    #      ^^^^^^^^^^^^^^^^^^^^^^ 一个列表推导，一页读完才读下一页
```

`self.get()` 内部是普通的 `open()` + `readinto()`。也就是说：

```
读第 1 页 → 等它读完 → 读第 2 页 → 等它读完 → ... → 读第 138 页
```

NVMe SSD 的性能来自**并行**：它内部有几十个通道，你同时给它 100 个请求，
它能同时处理；你一次只给一个，它大部分时间在空转。
一次 8832 token 的命中要读 138 页 × 9 MiB = 1.2 GiB，
串行读就是把一块能跑 3.4 GiB/s 的盘当成 1.9 GiB/s 的盘用。

### 3.2 隔离实验：把这一条单独量出来

抛开 SGLang，直接对同样的 138 个 9 MiB 文件做两种读法：

| 读法 | 耗时 | 带宽 |
|---|---|---|
| A. 串行 buffered 读（老后端的做法，队列深度=1） | 630 ms | 1.93 GiB/s |
| B. `O_DIRECT` + Linux AIO，全部同时在飞（队列深度=512） | 355 ms | 3.41 GiB/s |

**1.77 倍**，纯粹来自队列深度。

### 3.3 剩下的差距：每页一次往返的开销

端到端是 924 ms → 444 ms（2.08 倍），比隔离实验的 1.77 倍还多一点。
多出来的部分是**每页一次往返**的框架开销：

- 老路径：138 次独立的 `open` / `readinto` / `close`，每次都要过 Python 解释器、
  抢 GIL、查元数据缓存；这些开销**串在读盘的关键路径上**，一个都藏不住。
- 新路径：1 次 `io_submit` 就把几百个请求交给内核，然后一轮 `io_getevents` 收割。
  Python 只被调用两次。

```
924 ms (老)  ≈  630 ms 读盘  +  ~290 ms 每页往返开销
444 ms (新)  ≈  355 ms 读盘  +   ~90 ms 批量提交/收割开销
```

顺带澄清一个容易想当然的点：老路径确实多了一次"临时 buffer → KV 池"的内存拷贝，
但实测这块很便宜——9 MiB 一页，分配 0.03 ms、拷贝 0.03 ms，
138 页加起来也就 8 ms 左右。**拷贝不是瓶颈，串行才是**。

### 3.4 `O_DIRECT` 的附带好处

`O_DIRECT` 让磁盘 DMA **直接写进主机 KV 池的目标地址**，跳过内核 page cache。
除了省掉上面那次拷贝，更重要的是**不污染 page cache**：
L3 的容量按设计就远大于内存，page cache 根本装不下，
命中率很低却会把别人的热数据挤出去。

顺带说一句：在这台 251 GB 内存的机器上，如果不清 page cache，
老后端的读会被内存全部接住，测出来反而"更快"。
这正是 3.2 节要用隔离实验、以及测试要主动清缓存的原因。

---

## 4. 具体改了哪些东西

改动分三组。**第一组现在就能用**，第二组是给下一阶段（分层流水线）打的地基，
第三组是测试和工具。

### 第一组：能立刻用的快速 L3 后端

| 文件 | 作用 |
|---|---|
| `python/sglang/srt/mem_cache/storage/layerwise/hicache_layerwise_file.py` | **新后端本体**。用 `--hicache-storage-backend layerwise_file` 开启。一批页拆成 K/V 两组 extent，一次全部提交进 AIO 队列，直落主机 KV 池 |
| `python/sglang/srt/mem_cache/layerwise_storage/aio_engine.py` | Direct I/O 引擎：对齐探测、对齐 buffer、裸 Linux AIO 系统调用（`io_setup` / `io_submit` / `io_getevents`），不依赖 libaio |
| `python/sglang/srt/mem_cache/layerwise_storage/page_format.py` | 页文件格式：头部 + K 区 + V 区 |
| `python/sglang/srt/mem_cache/layerwise_storage/page_writer.py` | 写入端。先写临时文件再原子 `rename`，读者永远看不到写了一半的页 |
| `python/sglang/srt/mem_cache/layerwise_storage/page_store_evictor.py` | **磁盘容量管理**（见下方警告） |
| `python/sglang/srt/mem_cache/storage/backend_factory.py`（改） | 注册 `layerwise_file` |
| `python/sglang/srt/managers/cache_controller.py`（改） | 把 `layerwise_file` 加进"零拷贝后端"名单，走 `batch_get_v1` 而不是带临时 buffer 的通用路径 |
| `python/sglang/srt/server_args.py`（改） | 新增后端选项和 layerwise 相关开关 |
| `python/sglang/srt/environ.py`（改） | 新增 3 个环境变量（存储根目录、容量上限、剩余空间水位线） |

> ⚠️ **为什么要有容量管理**：开发过程中我没加上限，写满了机器的磁盘，
> scheduler 进程被 OOM killer 杀掉，整个 server 挂了。
> 现在默认保留 8 GiB 剩余空间（`SGLANG_HICACHE_LAYERWISE_MIN_FREE_SPACE`），
> 写不下就直接拒绝这次写入。理由很简单：**丢一个缓存页只是多算一次，
> 把盘写满是整个服务下线**。

#### 页文件长什么样

一个逻辑页 = 一个文件。文件内容和主机内存里 `page_first_direct` 布局的
"扁平页"**逐字节一致**，所以两边都不需要重排 KV：

```
[对齐的头部]                 ← 模型指纹、TP size/rank、dtype、几何、offset、校验和
K[第 0 层 .. 第 N-1 层]       每层 = page_size × 本 rank 的 KV head 数 × head_dim
[补零到对齐边界]
V[第 0 层 .. 第 N-1 层]
[补零到对齐边界]
```

两个设计决定：

- **补的零只存在于磁盘上**，永远不会被当成主机内存的容量。
- **身份信息放在头部，不放在文件名里**。文件名只负责"是哪个页"，
  这样以后改格式不用重命名文件。读的时候对不上（换了模型 / 换了 TP 数 /
  换了 dtype）就**直接判定失败去重算**，绝不把认不出来的字节塞进 KV cache。

目录结构按"必须分区的维度"来分：

```
<root>/format-v1/<模型指纹>/tp-<size>/rank-<rank>/<hash 桶>/<页 hash>.kv
```

不同 TP rank 存各自的 KV 分片，互不干扰。用 hash 桶分 256 个子目录，
避免一个目录塞进几十万个文件（并行文件系统在这种情况下会明显变慢）。

### 第二组：分层流水线的地基（已建好并单测，尚未接进主流程）

这一组是为了下一阶段准备的。目标是把 L3 读取、主机→显存拷贝、GPU 计算
三件事**重叠起来**（详见第 6 节）。

| 文件 | 作用 |
|---|---|
| `layerwise_storage/types.py` / `backend.py` | 值类型 + 异步分层读取接口（`begin_read` / `submit_group` / `poll` / `request_cancel`） |
| `layerwise_storage/state_machine.py` | 事务和层组的状态机、私有 buffer 的所有权（保证"恰好释放一次"） |
| `layerwise_storage/plan_builder.py` | 把"主机内存里的位置"和"文件里的位置"拼成一份按层分组的读取计划 |
| `layerwise_storage/io_arbiter.py` | I/O 仲裁：第 0 层 > 马上要用的组 > 预读 > 写回；写回在有读需求时最多占 25% 队列，且有防饿死机制 |
| `layerwise_storage/consensus.py` | 跨 TP rank 的**非阻塞**逐组一致性确认 |
| `layerwise_storage/pipeline.py` | 流水线驱动器（核心逻辑，见下） |
| `layerwise_storage/controller.py` | 对调度器暴露的统一入口 |
| `layerwise_storage/file_backend.py` | 按层组范围读的存储后端 |
| `mem_cache/pool_host/mha.py`（改） | 新增 `get_layer_group_buffer_meta()`：算出某几层在主机内存里的精确指针、文件偏移、是否对齐 |
| `mem_cache/l2_transfer.py`（改） | 主机→显存传输支持**分批提交**（原来只能一次提交所有层），老接口保持不变 |
| `managers/cache_controller.py`（改） | 新增流式 H2D 会话；层完成事件加上"代号"，防止模型等到上一批遗留的 CUDA event |
| `mem_cache/base_prefix_cache.py` + `managers/schedule_policy.py`（改） | 区分显存槽位是"树拥有"还是"请求私有"，失败时才能精确回收 |

流水线里有三条**不能放松**的顺序约束，单测把它们钉死了：

1. 存储的完成回调**可以乱序**，但跨 rank 的一致性确认**必须严格按组序发起**
   ——否则两个 rank 对"下一个该确认哪组"的看法不一致，集合通信会死锁。
2. 一组数据必须**所有 rank 都确认成功**才能进入主机→显存传输
   ——否则丢了一页的那个 rank 会把半对的 KV 发布出去。
3. 第 0 层单独提交、单独等待，它是唯一没有计算可以掩盖的一组。

还有一条内存安全约束：**读失败时不能立刻释放 staging 内存**，
必须等存储后端确认这次 I/O 已经彻底终止（内核不会再写这块内存了）才能放。
否则会出现"内存已经还给别人了，磁盘 DMA 还在往里写"的灾难。

### 第三组：测试和工具

| 文件 | 内容 |
|---|---|
| `test/registered/unit/mem_cache/test_layerwise_file_backend.py` | 真实写盘 + 按层范围读回，验证字节完全一致；对齐直落 / 非对齐走 bounce / 缺页失败降级 |
| `test/registered/unit/mem_cache/test_layerwise_storage_pipeline.py` | 流水线的顺序和故障语义（上面那三条约束 + 所有权） |
| `test/registered/unit/mem_cache/test_layerwise_storage_controller.py` | 端到端：写盘 → 发现 → 分层读回 → 按序交给 H2D → 精确释放 |
| `test/registered/unit/mem_cache/test_layerwise_page_store_evictor.py` | 容量上限与剩余空间水位线（就是上面那次把盘写满的回归测试） |
| `test/registered/unit/mem_cache/test_layerwise_storage_state_machine.py` | 状态机 |
| `test/registered/unit/mem_cache/test_mha_host_layer_group_meta.py` | 主机侧层组几何计算 |
| `benchmark/hicache/bench_layerwise_storage.py` | 存储微基准：裸 Direct AIO 上限 vs 分层范围读，报告比值和第 0 层延迟 |

---

## 5. 怎么用

```bash
python -m sglang.launch_server \
  --model-path <模型> \
  --enable-hierarchical-cache \
  --hicache-storage-backend layerwise_file \
  --hicache-io-backend direct \
  --hicache-mem-layout page_first_direct \
  --hicache-write-policy write_through \
  --hicache-host-memory-mode cache \
  --page-size 64
```

环境变量：

```bash
SGLANG_HICACHE_LAYERWISE_ROOT=/mnt/parallel-fs/sglang-hicache   # 存储根目录
SGLANG_HICACHE_LAYERWISE_MAX_SIZE=2Ti                          # 字节上限，空或 0 = 不限
SGLANG_HICACHE_LAYERWISE_MIN_FREE_SPACE=8Gi                    # 剩余空间水位线，0 = 关闭
```

后端启动时会**探测**文件系统的 Direct I/O 对齐要求，不写死 4096
（这台机器的 ext4 实测是 512）。如果目录所在的文件系统根本不支持 `O_DIRECT`
（比如 tmpfs），会**直接报错而不是悄悄退回 buffered**——
因为悄悄退回会让后面所有的带宽测量失去意义。

### 调参：`group-size` 是决定性旋钮

`benchmark/hicache/bench_layerwise_storage.py` 在单块 NVMe 上扫出来：

| 每组层数 | 达到裸 Direct AIO 上限的比例 |
|---|---|
| 2 | 53% |
| 4 | 79% |
| **8** | **94%** |
| 16 | 94% |

分得太细，每个 I/O 请求就太小，磁盘的 IOPS 成了瓶颈。
换到你的并行存储上要重新扫一遍——不同存储的最佳点不一样。

---

## 6. 还没做完的：分层流水线

现在的 `layerwise_file` 是"**把整段前缀读得尽可能快**"。
下一阶段是"**边读边算，把读的时间藏起来**"。

### 目标是什么

现在（读完才能开始）：

```
时间 ────────────────────────────────────────────────►
存储读取  ████████████████████████████
主机→显存                              ██████
GPU 计算                                      ████
                                              ↑ 首字在这里出来
```

流水线之后（三件事重叠）：

```
时间 ────────────────────────────────────────────────►
存储读取  ██ ██ ██ ██ ██ ██        ← 读第 g+1 组层
主机→显存    ██ ██ ██ ██ ██ ██     ← 拷第 g 组层
GPU 计算        ██ ██ ██ ██ ██ ██  ← 算第 g-1 组层
                ↑ 首字提前到这里
```

关键在于：模型是**一层一层**算的，算第 0 层的时候根本用不到第 30 层的 KV。
所以没必要等全部 36 层都读完才开始，读完第 0 层就可以放行了。

第 0 层是唯一藏不住的（前面没有任何计算可以掩盖它），
所以它被单独拿出来：单独提交、给最高 I/O 优先级、
所有 rank 确认成功后才允许请求进入 GPU。

### 还差的四步

配置开关、异步接口、状态机、流水线驱动器、H2D 分批会话、跨 rank 确认
——这些都已经写好并且单测通过了。
缺的是**接进 `UnifiedRadixCache`**（那是个 2900 行的核心文件）：

1. 存储命中时路由到 `LayerwiseStorageController`，admission 只等第 0 组。
2. 在所有组都确认、且 forward 消费完之前，**不要**把主机 staging 挂进 radix tree
   ——否则别的请求会命中一个后面几层还没读完的半成品。
3. 失败时按"请求私有"的所有权精确释放显存槽位，主机 staging 等 I/O 彻底终止再放。
4. 多请求并发流式加载、和正在跑的 decode 混合、admission 之后失败的重试路径。

这一步会改动 prefetch 的状态机，风险比前面高，需要边接边在真机上验证。

---

## 7. 怎么复现上面的数字

这台开发机没有 editable 安装，用 `PYTHONPATH` 覆盖即可：

```bash
export PATH=$HOME/.local/bin:$PATH        # JIT kernel 需要 ninja
SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
PYTHONPATH=/home/lpl/sglang/python \
/home/lpl/sglangtest/.venv/bin/python -m sglang.launch_server \
  --model-path /home/lpl/models/Qwen-8B --attention-backend triton \
  --cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled \
  ...上面第 5 节的 hicache 参数...
```

（装的 flashinfer 是 0.6.14，代码要求 0.6.17，所以用 triton attention backend
并设置那个 skip 环境变量绕开版本检查。）

跑单测：

```bash
PYTHONPATH=/home/lpl/sglang/python \
/home/lpl/sglangtest/.venv/bin/python \
  test/registered/unit/mem_cache/test_layerwise_storage_controller.py
```

跑存储微基准：

```bash
PYTHONPATH=/home/lpl/sglang/python \
/home/lpl/sglangtest/.venv/bin/python \
  benchmark/hicache/bench_layerwise_storage.py --root /path/to/store --group-size 8
```

> ⚠️ 这台机器的磁盘已经用到 98%（剩约 20 GB）。
> 一次 17k token 的前缀就要写 2.5 GB，跑几轮就能把盘写满。
> 务必加上容量上限，跑完删掉存储目录。

---

## 8. 附：改动一览

```
新增
  L2L3fusion-docs/01-direct-io-l3-backend.md                 本文档
  python/sglang/srt/mem_cache/layerwise_storage/             13 个模块 + README
  python/sglang/srt/mem_cache/storage/layerwise/             layerwise_file 后端
  test/registered/unit/mem_cache/test_layerwise_*.py         5 个测试文件
  test/registered/unit/mem_cache/test_mha_host_layer_group_meta.py
  benchmark/hicache/bench_layerwise_storage.py

修改
  python/sglang/srt/server_args.py                           新后端选项 + layerwise 开关与校验
  python/sglang/srt/environ.py                               3 个环境变量
  python/sglang/srt/managers/cache_controller.py             流式 H2D 会话 + 代号化层事件 + 零拷贝名单
  python/sglang/srt/managers/schedule_policy.py              区分 load-back 槽位所有权
  python/sglang/srt/mem_cache/base_prefix_cache.py           InitLoadBackResult / LoadBackOwnership
  python/sglang/srt/mem_cache/l2_transfer.py                 H2D 分批提交（老接口不变）
  python/sglang/srt/mem_cache/pool_host/mha.py               get_layer_group_buffer_meta()
  python/sglang/srt/mem_cache/storage/backend_factory.py     注册 layerwise_file
  test/registered/unit/...                                   配套测试更新
```

**默认行为完全不变**：不加 `--hicache-storage-backend layerwise_file`
就走原来的路径，所有新开关默认关闭。
