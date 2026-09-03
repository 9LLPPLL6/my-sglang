---
title: "沐曦 GPU 的 HiCache L1↔L2 适配接口清单"
description: "SGLang HiCache 在 MetaX GPU 显存与主机内存之间搬运 KV cache 时必须修改的代码和接口。"
---

本文面向 SGLang 运行在沐曦（MetaX）GPU 上、通过 HiCache 在 GPU 显存（L1）与主机内存（L2）之间双向搬运 KV cache 的场景。

本文中的“需要适配”特指：SGLang 当前实现包含 NVIDIA 专用代码或缺少 MetaX 分支，必须修改源码、编译配置或扩展包才能在沐曦上工作。mcPyTorch/MXMACA 已公开支持且 SGLang 无需修改的通用接口不列入清单。

本文基于 SGLang 提交 `b294bd4bc7`，检查日期为 2026-08-27。

## 范围

```text
L1：MetaX GPU 显存中的 KV cache
        │
        │ HiCache D2H / H2D transfer
        ▼
L2：CPU host memory 中的 KV cache
```

适配必须覆盖两个方向：

- D2H：将 KV cache 从 MetaX 显存备份到 L2 主机内存；
- H2D：L2 命中后将 KV cache 加载回 MetaX 显存。

本文不涉及 L2 主机内存与并行存储之间的接口。

## 结论

HiCache 在沐曦上的最小可用目标建议限定为：

```text
--hicache-io-backend direct
--hicache-mem-layout layer_first
```

该组合的核心数据操作是 ATen `Tensor.copy_(non_blocking=True)`，不需要为 mcPyTorch 重写数据拷贝逻辑；但 SGLang 仍通过 `sgl_kernel.kvcacheio` 扩展暴露 `transfer_kv_direct`，因此必须完成 MetaX 版扩展的构建、算子注册和发布。

需要修改的内容分为两组：

1. 跑通上述最小目标必须完成：MetaX 能力分支、host pool 的 operator 选择、L2 pinned host memory 分配分支、MetaX 版 `sgl-kernel.kvcacheio` 构建和 direct operator 注册。
2. 只有要求启用 `--hicache-io-backend kernel` 时才需要完成：AOT/JIT 搬运 kernel 中的 NVIDIA PTX 和 warp 假设改写。

## 必须修改的接口

### P0-1：增加 MetaX HiCache 能力和 operator 选择分支

涉及位置：

```text
python/sglang/srt/platforms/interface.py
python/sglang/srt/platforms/__init__.py
MetaX SRT platform plugin
python/sglang/srt/mem_cache/memory_pool_host.py
python/sglang/srt/mem_cache/pool_host/mha.py
python/sglang/srt/mem_cache/pool_host/mla.py
```

MetaX 使用 CUDA DispatchKey，SGLang 将其识别为 CUDA 路径本身没有问题。当前 host pool 代码的问题是：只要 `is_cuda()` 为真，就会导入整组 AOT operator，并尝试启用 JIT HiCache kernel；即使配置为 `direct`，也不能只安装 `transfer_kv_direct`。

需要通过 MetaX SRT platform plugin 或等价实现提供独立 capability，至少区分：

```text
supports_hicache_direct
supports_hicache_aot_kernel
supports_hicache_jit_kernel
```

行为要求：

- 保留 MetaX 的 CUDA DispatchKey 和 `cuda` device type；
- direct 模式只导入和调用该 layout 实际需要的 direct operator；
- `supports_hicache_aot_kernel=false` 时不导入 AOT kernel operator；
- `supports_hicache_jit_kernel=false` 时不调用 `can_use_hicache_jit_kernel()`；
- 完成某一 kernel 适配后，再通过对应 capability 启用。

### P0-2：为 MetaX 选择 PyTorch pinned-memory allocator

涉及位置：

```text
python/sglang/srt/mem_cache/pool_host/common.py
```

当前 `ALLOC_MEMORY_FUNCS` 对 `npu`、`musa` 使用：

```python
torch.empty(..., pin_memory=True)
```

其他设备默认进入：

```python
torch.cuda.cudart().cudaHostRegister(...)
```

当前 host pool 按 `device_pool.device` 查找 allocator。MetaX 的 device type 仍为 `cuda`，因此会进入后者。allocator 选择需要改为读取 MetaX platform identity/capability，并使用 mcPyTorch 的 pinned-memory allocator，不再直接依赖 NVIDIA `cudart()` 入口：

```python
if is_metax:
    alloc_func = alloc_with_pin_memory
```

MetaX 官方 vLLM platform 将 pin memory 能力声明为可用，因此这里采用 PyTorch pinned-memory 路径，不再把 raw `cudaHostRegister` 作为沐曦适配接口。

该修改应覆盖 MHA、MLA 以及实际使用的其他 host pool，因为这些 pool 最终都通过 `ALLOC_MEMORY_FUNCS` 分配 L2 buffer。

### P0-3：构建并发布 MetaX 版 `sgl_kernel.kvcacheio`

涉及位置：

```text
python/sglang/kernels/aot/csrc/kvcacheio/transfer.cu
sgl-kernel 的构建和 wheel 发布配置
```

需要使用 mcPyTorch、cu-bridge 和 cucc 重新编译扩展，并保持 CUDA DispatchKey 下的 operator 注册。最小目标必须提供：

```text
sgl_kernel.kvcacheio.transfer_kv_direct
```

如果需要 `page_first_direct + direct`，还必须提供：

```text
transfer_kv_per_layer_direct_pf_lf
transfer_kv_all_layer_direct_lf_pf
```

`transfer.cu` 同时包含 direct 实现和 NVIDIA 专用 kernel。为了先交付 direct 模式，需要增加 MetaX 编译分支，将尚未适配的 PTX/kernel 代码排除或替换，保证 translation unit 能由 cucc 编译并正确加载。

## 仅 kernel 模式需要修改的接口

以下内容不属于 `direct + layer_first` 最小目标。只有项目要求启用 `--hicache-io-backend kernel` 时才需要适配。

### P1-1：改写 AOT 搬运 kernel

涉及位置：

```text
python/sglang/kernels/aot/csrc/kvcacheio/transfer.cu
```

必须修改：

| 当前实现 | MetaX 修改要求 |
| --- | --- |
| 非 ROCm/MUSA 分支固定 `WARP_SIZE 32` | 使用 MACA 的编译期 warp 定义或设备查询结果 |
| `ld.global.nc.b64` | 改为 cucc 支持的 C++/MACA load 实现 |
| `st.global.cg.b64` | 改为 cucc 支持的 C++/MACA store 实现 |
| lane、warp、block 数量按固定 32 计算 | 全部改为使用 MetaX warp size |

按客户实际模型和 layout 提供对应 operator：

| 场景 | 需要提供的 operator |
| --- | --- |
| MHA，`layer_first/page_first + kernel` | `transfer_kv_per_layer`、`transfer_kv_per_layer_pf_lf`、`transfer_kv_all_layer`、`transfer_kv_all_layer_lf_pf` |
| MLA，`layer_first/page_first + kernel` | `transfer_kv_per_layer_mla`、`transfer_kv_per_layer_mla_pf_lf`、`transfer_kv_all_layer_mla`、`transfer_kv_all_layer_mla_lf_pf` |

### P1-2：改写 JIT HiCache kernel

涉及位置：

```text
python/sglang/kernels/ops/kvcache/hicache.py
python/sglang/kernels/jit/csrc/kvcacheio/hicache.cuh
python/sglang/kernels/jit/csrc/kvcacheio/hisparse.cuh
python/sglang/kernels/jit/csrc/kvcacheio/hisparse_spec.cuh
```

必须修改：

- 为 JIT loader 增加 MetaX/cuCC 编译分支；
- 将 `ld/st.global.L1::no_allocate`、`ld.global.nc`、`st.global.cg` 等 NVIDIA PTX 改为 cucc 支持的实现；
- 将 warp、lane、shuffle 和 block 划分切换为 MetaX warp 定义；
- 仅在 MetaX 版 JIT kernel 成功构建和注册后启用 `supports_hicache_jit_kernel`。

## 修改结果清单

| 编号 | 交付物 | 适用范围 |
| --- | --- | --- |
| M1 | MetaX HiCache capability gate 和 host pool operator 选择 | 必须 |
| M2 | MetaX pinned host-memory allocator 分支 | 必须 |
| M3 | 包含 `transfer_kv_direct` 的 MetaX `sgl-kernel` wheel | 必须 |
| M4 | `page_first_direct` direct operator | 使用该 layout 时必须 |
| M5 | 去除 NVIDIA PTX/固定 warp 假设的 AOT operator | 使用 `kernel` backend 时必须 |
| M6 | MetaX JIT loader 和 JIT HiCache kernel | 启用 JIT kernel 时必须 |

## 最小正确性验证

验证只围绕实际修改项展开。

### 1. 扩展加载

在 MetaX 环境中确认：

```python
from sgl_kernel import kvcacheio

assert hasattr(kvcacheio, "transfer_kv_direct")
```

若启用其他模式，再检查该模式对应的 operator。

### 2. KV 搬运 round trip

对客户实际使用的模型类型、dtype 和 layout 执行：

```text
MetaX GPU KV → L2 host buffer → MetaX GPU KV
```

至少覆盖连续页和非连续页索引，并逐元素比较搬运前后的 KV tensor。

### 3. HiCache L2 命中

启动最小配置：

```bash
python3 -m sglang.launch_server \
  --model-path MODEL_PATH \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-io-backend direct \
  --hicache-mem-layout layer_first
```

触发 KV 从 L1 备份到 L2，再访问相同前缀，确认发生 L2 hit、KV 被加载回 L1，并且模型输出与关闭 HiCache 时一致。

## 不在本文范围

- L2 主机内存与并行存储之间的 backend 接口；
- 存储系统直接访问 GPU 显存的数据路径；
- PD disaggregation、LMCache、FlexKV 等其他 KV cache 传输实现。

## 参考资料

- [SGLang HiCache design](../docs/docs/advanced_features/hicache_design.mdx)
- [SGLang HiCache best practices](../docs/docs/advanced_features/hicache_best_practices.mdx)
- [沐曦 mcPyTorch 用户指南](https://developer.metax-tech.com/api/client/document/file/212/preview/?file_type=pdf)
- [MetaX vLLM platform：CUDA device type 与 pin-memory capability](https://github.com/MetaX-MACA/vLLM-metax/blob/master/vllm_metax/platform.py)
- [MetaX vLLM plugin design：CUDA DispatchKey、扩展重新编译和 PTX 改写](https://github.com/MetaX-MACA/vLLM-metax/issues/4)
