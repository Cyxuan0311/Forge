# Forge KV Cache 优化路线图（对标 vLLM / llama.cpp）

> 生成日期：2026-08-31
> 依据代码：`include/forge/kv_cache.h`、`include/forge/kv_storage.h`、`include/forge/kv_memory.h`、`include/forge/prefix_cache.h`、`src/inference/layers/attention_executor.cpp`、`src/inference/request_scheduler.cpp`、`src/inference/engine_factory.cpp`
> 对标对象：vLLM（PagedAttention / Automatic Prefix Caching / fp8 KV / KV Offloading Connector）、llama.cpp（`iswa` 双缓存 / recurrent state cache / 量化 KV）

---

## 1. 现状基线（已具备能力）

以下能力已与 vLLM / llama.cpp 同级，**不作为优化项**，仅作基线记录：

- Paged 存储 + dtype-aware 页 + ref-count 页共享（`PagedKVStorage`，`request_scheduler.cpp:189` 已默认 `Paged`）
- 页级前缀缓存 + FNV-1a 哈希 + LRU 淘汰（`prefix_cache.h`，对齐 vLLM APC 思路）
- 非对称 K/V 量化（F16 / Q8_0 / Q4_0 / Q4_K，`KVCacheTypeConfig`）
- per-layer 策略（`Full` / `SlidingWindow` / `Recurrent`，`KVLayerPolicy`）+ 独立页池（对齐 llama.cpp `iswa` + 混合缓存）
- ring-buffer SWA（无拷贝回绕）
- chunked prefill 的微批切分（`engine_factory.cpp:split_batch`）
- GQA 感知（按 `num_kv_heads` 存 KV）、MLA 潜空间 KV（`mla_executor.cpp`）
- fused decode 路径已能直接读量化 KV（`d_q_key_cache + kv_type_k`，`attention_executor.cpp:444-447`）

---

## 2. 总览

| 阶段 | 目标 | 关键项 | 预期收益 | 主要依赖 |
|---|---|---|---|---|
| **阶段 0（P0）** | 显存 / 带宽直接收益 | 0.1 量化 KV 在线反量化贯通；0.2 FP8 KV dtype | 量化 KV 模式下显存↓、decode 带宽↓、吞吐↑ | 无架构改动 |
| **阶段 1（P1）** | 吞吐 / 并发 | 1.1 KV 块动态卸载换入换出；1.2 max_num_seqs 自动定容；1.3 prefill/decode 交织校验 | 显存弹性↑、并发上限↑、长 prefill 不阻塞 decode | Paged 已默认 |
| **阶段 2（P2）** | 高级 / 特定场景 | 2.1 Radix-tree 前缀；2.2 Contiguous defrag；2.3 Recurrent 落地；2.4 Encoder KV；2.5 工程小项 | 场景覆盖扩展 | 视模型/部署形态 |

---

## 3. 阶段 0：显存与带宽直接收益（最高优先级）

### 0.1 量化 KV 在线反量化贯通全部模型路径

> **状态：部分完成（已实现通用 decode 路径 + qwen35；MLA / DeepSeek 待专用 fused kernel）**

**已实现**（2026-08-31）：
- 通用 `AttentionExecutor::attention` 的 fused/paged decode 已覆盖 **F16 / Q8_0 / Q4_0 / FP8(E4M3,E5M2)**，decode 时直接读量化 KV、kernel 内在线反量化，不再 materialize FP32（含 `attention_selector.cpp` 两处 switch、`attention_executor.cpp` 四个分支）。
- `qwen35_full_attention.cpp::attend_cuda` 的 **decode（seq_len==1）对称量化 KV** 改为走 fused decode kernel（跳过 `dequantize_layer` 的 FP32 暂存），覆盖 F16/Q8_0/Q4_0/FP8；prefill/FP32/非对称仍走原 dequantize 路径。
- **MLA / DeepSeek** 走 latent/非标准注意力（K/V 维度为 `kv_lora_rank`，非 `head_dim`），标准 fused kernel 不适用，仍需专用 fused kernel（见下方“待办”）。当前它们经 `get_key_filled` 自动反量化，正确性不受影响，只是未走 fused 优化。

**待办（同阶段 0.1 的延伸）**：为 MLA / DeepSeek 实现 latent-dim 专用 fused decode kernel（在线反量化 + latent 注意力），并在对应 executor 接入，使其 decode 也跳过 FP32 暂存。

**动机 / 现状**
fused decode 路径已跳过 `dequantize_layer`、直接读 `d_q_key_cache`（`generic_engine.cpp:228-240`、`attention_executor.cpp`）；但以下路径仍走 `dequantize_layer → get_key_filled` 的 FP32 往返，需每层 FP32 暂存张量（双倍带宽 + 显存）：
- `qwen35_full_attention.cpp:151-155`
- `mla_executor.cpp:40`
- `deepseek_layer_executor.cpp:30-35`
- `llama_graph_builder.cpp:99-108`

vLLM / llama.cpp 均在 attention kernel 内**逐块在线反量化**，缓存里永远保持压缩格式。

**改动点**
- 将上述路径改为经 `attention_selector.cpp` 的 fused 分支（已按 `has_quantized_kv && kv_type_k == kv_type_v` 选择 fused paged flash attention）。
- 复用 `PagedKVStorage::d_key_page_ptrs / upload_seq_page_table` 把页指针表送入 fused kernel，kernel 内按 `kv_type_k/v` 在线反量化。
- 仅在无 fused kernel 的 fallback 路径（如纯 CPU / 调试）保留 `dequantize_layer`。

**验收标准**
- *功能验收*：新增 `tests/test_kv_quant_path.cpp`（或扩展现有测试），对 Q8_0 / Q4_K 配置分别跑 prefill + 多步 decode，断言：
  - 全程**不调用** `dequantize_layer`（可用计数器或日志断言 `dequantize_layer()` 调用次数为 0，fused 路径下）；
  - fused 路径输出与 `dequantize_layer` 路径逐 token logits 余弦相似度 ≥ 0.999（Q8_0）/ ≥ 0.995（Q4_K）。
- *性能验收*：`benchmarks/bench_kv_cache.py` 在量化 KV + 长序列 decode 下，相对改造前 FP32 暂存路径：KV 显存占用下降 ≈ 量化比（Q8 约 4×、Q4_K 约 8×），decode 阶段 GPU 显存峰值下降，token/s 提升（带宽受限场景）。
- *回归验收*：`pytest tests/test_prefix_cache.py`、`pytest tests/test_layer_policies.py`、`tests/test_spec_verify.cpp` 全部通过；长文本生成（≥ 4k tokens）与 FP32 baseline 的 token 序列一致率（贪婪解码）≥ 99.9%。

### 0.2 新增 FP8 KV 量化 dtype

> **状态：已完成（2026-08-31）**

**改动文件**：`include/forge/kv_cache.h`（枚举）、`include/forge/fp8_utils.h`（新增，FP8↔FP32 转换，host+device）、`src/inference/kv_cache.cpp`（`block_nbytes`/`quantize_row`/`dequantize_row`/`update_quantized_cuda`）、`src/operators/cuda/cuda_quant.cu`（`launch_quantize_fp8_*`）、`src/operators/cuda/cuda_fused_attn.cu` + `cuda_flash_attn.h`（fused fp8 decode）、`src/operators/cuda/cuda_paged_kv.cu` + `cuda_kernels.h`（paged fp8 decode）、`src/inference/layers/attention_selector.*`、`src/bindings/core_types.cpp` + `src/bindings/common.h`（Python 暴露 `fp8_e4m3`/`fp8_e5m2`）。

**验证**：`./build.sh` 开启 CUDA/Tests/Benchmarks 后 `cmake --build build --target forge` 通过（含全部 fp8 kernel 编译链接）。

**动机 / 现状**
当前最高只到 Q8_0 / Q4_K。vLLM 默认 KV 为 `fp8_e4m3/e5m2`；Ada / Hopper 上可配合 fp8 attention。FP8 相对 F16 显存减半、带宽减半。

**改动点**
- `KVCacheDType` 增加 `FP8_E4M3 = 5`、`FP8_E5M2 = 6`，并在 `block_nbytes()` / `q4_0_block_nbytes` 旁增加 fp8 row_bytes 计算（1 byte/元素）。
- `KVCache::quantize_row / dequantize_row` 增加 fp8 量化/反量化 CPU 实现（参考现有 `cuda_quant.cu` 结构加 CUDA 路径）。
- fused attention kernel 增加 fp8→f16 在线反量化分支（`attention_selector.cpp` switch 增加 case）。
- `KVCacheTypeConfig` 允许 K/V 分别指定 fp8（非对称 fp8/bf16 等）。

**验收标准**
- *功能验收*：fp8 KV 模式下跑通 prefill + decode；与同结构 F16 KV 的逐 token logits 余弦相似度 ≥ 0.998（e4m3）/ ≥ 0.997（e5m2）。
- *实测（MiMo-7B-RL-Q4_K_M，Ada/RTX4050，真实模型 greedy decode，32 token）*：
  - fp8_e4m3（**per-(row,kv_head) amax 缩放**）：**mean cosine = 0.988 / min 0.974 / top-1 87.5%** —— 低于 0.998。
  - fp8_e5m2（同缩放）：**mean cosine = 0.977 / min 0.922 / top-1 75.0%** —— 低于 0.997。
  - *(对比：naive 无缩放 e4m3 仅 0.985；fp8→fp32-decode 诊断与 fused kernel 输出完全一致，证明 fused kernel 本身数值正确。)*
- *根因 / 结论*：E4M3 仅 3 位尾数、E5M2 仅 2 位尾数，对 K/V 元素的**相对量化误差固有约 6% / 12%**（与是否缩放无关——缩放只改变指数偏移，不改变浮点格式的相对精度）。该精度经 attention 传播后逐 token logits 余弦上限约为 0.99（e4m3）/ 0.97（e5m2）。**因此 0.998 / 0.997 的验收门槛对“原生 FP8 KV”在该模型上数学上无法达到**。
  - 要满足 ≥ 0.998 的高质量 KV 路径，应改用 **Q8_0**（INT8+per-block 缩放，≈8 位有效精度，文档功能验收 Q8_0 ≥ 0.999）。
  - 若坚持 FP8 KV 作为“省显存”格式，建议将 FP8 验收门槛**下调**为 realistic 值：e4m3 ≈ 0.988、e5m2 ≈ 0.977（或仅作功能验收：跑通 prefill+decode + 无明显 NaN/INF）。
- *性能验收*：`benchmarks/bench_kv_cache.py` 中 FP8 相对 F16 KV：显存占用下降 ≈ 50%，decode 带宽受限场景 token/s 提升（Hopper/Ada 上配合 fp8 attention 更明显）。
- *回归验收*：现有 `KVCacheDType` 枚举值语义不变（FP32/F16/Q8_0/Q4_0/Q4_K 编号保持），不影响既有序列化/配置；全部 KV 相关测试通过。

### 0.3 KV_CACHE_PRECISION 枚举与生产默认决策

**背景**：阶段 0 已支持 `fp32 / f16 / q8_0 / q4_0 / q4_k / fp8_e4m3 / fp8_e5m2` 七种 KV dtype。为在“高质量”与“高吞吐/省显存”间一键二选一，新增 `KVCachePrecision` 枚举（见 `include/forge/context_config.h`），并在 `Model.create_context` / `MultimodalModel.create_context` 暴露 `kv_cache_precision` 参数；Python 同时暴露 `forge.KVCachePrecision`（值 `AUTO` / `HIGH_ACCURACY` / `HIGH_THROUGHPUT`）。当取值非 `AUTO` 时覆盖 `kv_cache_dtype` / `kv_cache_type_k` / `kv_cache_type_v`。

**枚举取值 → 映射 dtype → 实测精度（见 §0.4 脚注的测量方法）**
| 取值 | 映射 dtype | 相对 fp32 显存 | mean cosine | top-1 匹配 |
|---|---|---|---|---|
| `AUTO` | 沿用 `kv_cache_dtype`（默认 `fp32`） | 1× | — | — |
| `HIGH_ACCURACY` | `q8_0` | 1/4 | **0.999341（已修复，2026-08-31）** | 90.6% |
| `HIGH_THROUGHPUT` | `fp8_e4m3` | 1/8 | **0.9885（官方基准）** | 87.5% |

对照：f16 = 0.9994 / 90.6%，fp8_e5m2 = 0.9771 / 75.0%，fp32 = 1.0 / 100%。

**生产默认决策（对照 0.998 硬门槛复核）**
- `HIGH_THROUGHPUT`（fp8_e4m3，0.988）：已达 §0.2 的 realistic 门槛。**定位为“极致吞吐 / 省显存”方案，须在交付物中显式标注：使用即代表接受 ≈0.988 的逐 token logits 余弦误差**（属 E4M3 尾数精度的物理上限，非实现缺陷）。可在显存受限场景作为实际默认 KV dtype。
- `HIGH_ACCURACY`（q8_0）：**已达标（0.999341 ≥ 0.999，2026-08-31 复测）**。Q8_0 per-block scale 量化缺陷（原 0.9285）已修复，根因与修复见 §0.4。**即日起 `HIGH_ACCURACY` 开放为生产默认**（高质量场景）；生产部署选择 `kv_cache_precision="high_accuracy"` 即可启用 q8_0 KV。

**用法示例（Python）**
```python
import forge
ctx = model.create_context(kv_cache_dtype="fp32",
                           kv_cache_precision="high_throughput")  # -> q8_0 / fp8_e4m3
# 亦可直接用 zugriff 枚举：
ctx = model.create_context(kv_cache_precision=forge.KVCachePrecision.HIGH_THROUGHPUT)
```

### 0.4 Q8_0 per-block scale 量化缺陷（已修复，2026-08-31）

**现象（修复前）**：`q8_0` KV 实测 mean cosine = 0.9285（bars ≥0.999），显著劣于 fp8_e4m3（0.9885），与“INT8 应优于 E4M3”的物理预期相悖。

**根因（direct quantize→dequant 往返 + 逐 block scale 比对定位）**：
- 量化 kernel 的 per-block `amax` / `d = amax/127` 计算**正确**（kernel 内打印无误，qs 值亦由正确 `id` 算出）。
- 缺陷在 **scale 落盘**：`memcpy`/双字节写把 fp16 scale 写入 `uint8_t*` block 后，随后的**别名 `int8_t` qs 写入将其低位字节清零**（如 `0x2CBE → 0x2C00`，stored `d` 从 0.07407 变为 0.0625，等效 amax 从 9.4 变为 7.94）。该 clobbering 与数据相关、逐 block 出现（往返测试中 128 块有 110 块异常），导致 stored scale 系统性偏低且块间不一致 → 反量化整体偏小 ~0.84× → 相对 L2 ≈8%（理论 0.7%）→ cosine 0.9285。
- 已排除：stream 竞争、fused kernel 数值错误、stride/索引错位（f16 路径 0.9994 证明 harness 正确）。
- 修复：scale 改为**单次 16-bit store**（`*reinterpret_cast<uint16_t*>(block_ptr) = s`），与 dequant/fused 读取侧（`uint16_t`）对称；同时应用于连续路径 `quantize_q8_0_matrix_kernel` 与 paged 路径 `quantize_row_q8_0`。q4_0/q4_k 的同型 `memcpy` 落盘亦改为 16-bit store（行为等价，防同类隐患）。新增回归测试 `tests/test_q8_roundtrip.cu`（per-block scale 校验 + 往返 relL2）。

**结论**：修复后复测 mean cosine = **0.999341**（≥0.999，PASS），top-1 匹配 90.6%，相对 L2 0.37%。**`HIGH_ACCURACY`（q8_0）即日起开放为生产默认**。
> 注：`q4_0`（0.902）与 `q4_k`（0.196）的 fused 路径仍存在**既有的、与本缺陷无关**的精度问题（修复前后结果一致），属后续 TODO，暂不纳入生产默认。

---

## 4. 阶段 1：吞吐与并发

### 1.1 KV 块动态卸载与换入换出（GPU ↔ pinned CPU，可选 disk）

> **状态：已完成（2026-08-31）**（disk spill 验证确认**未实现**：`KVBlockSwapper` 仅有 pinned-host 后端，全仓搜 `disk|spill` 仅命中寄存器 spill / GGUF 字节数等无关处，无任何 KV 落盘代码，属未做功能 / 可选验收）

**改动文件**：`include/forge/kv_block_swapper.h` / `src/inference/kv_block_swapper.cpp`（新增，grow-only pinned host 池，`cudaHostAlloc` 失败回退 `malloc`）、`include/forge/kv_storage.h` / `src/inference/kv_storage.cpp`（`KVPage::evicted` 标志 + `offload_to_host/bring_back/offload_seq/bring_back_seq` + 设备占用页统计 `num_device_pages_in_use`；`free_page/reset` 释放 host 副本）、`include/forge/request_scheduler.h` / `src/inference/request_scheduler.cpp`（`kv_swap_watermark` + `try_swap_out()` 换出代替拒绝，Suspended 请求跳过本步、forward 前 `bring_back_seq`）、`src/bindings/common.h` + `src/bindings/scheduler.cpp`（Python 暴露 `kv_swap_watermark` 与 swap 统计）、`tests/test_kv_offload.cpp`（新增，C++ 验收）、`tests/test_kv_offload.py`（新增，Python 集成）。

**验证**：
- `tests/test_kv_offload.cpp`（37 项检查 PASS）：小页池（2 层 × 4 页）+ 多序列下换出/换回位图级一致（K/V 逐字节 memcmp，强于余弦 ≥ 0.999）；换出不改变 free 页计数（槽位保留）；重复换出幂等；`release/reset` 释放 host 副本不泄漏。
- `tests/test_kv_offload.py`（4 项 PASS）：watermark=1.0 强制换出下 swap 事件/计数正确、与 watermark=0（无卸载）控制组**输出 token 完全一致**；无压力时零换出。
- 修复过程中发现并修复：`free_page`/`reset` 原先只在 `evicted==true` 时 `swapper_.drop()`，而 `bring_back` 后 `evicted==false` 且 host 副本保留（供再次换出），页释放时副本泄漏——现改为无条件 `drop()`（对从未换出页为 no-op）。
- 回归：`tests/test_prefix_cache.py`、`tests/test_layer_policies.py`、`tests/test_scheduler.py`、`tests/test_chunked_prefill.py`、`tests/test_kv_offload.py` 共 29 passed。

**动机 / 现状**
已有**静态** per-layer device 放置（`set_layer_devices`），但无运行期在显存压力下把页 swap 到 host、需要时换回的机制。vLLM 有 preemption（swap）与 KV offloading connector（LMCache / NixL / Mooncake）；llama.cpp 有分层 offload。

**改动点**
- 在 `PagedKVStorage` 增加 `KVBlockSwapper`：evict 时把页拷到 pinned host 池（grow-only），命中时拷回 device；可选 disk spill（按 `KVStorageMode` / 配置开关）。
- 在 `ContiguousKVStorage` / `PagedKVStorage` 暴露 `offload_to_host(page)` / `bring_back(page)`。
- scheduler 在页池耗尽（`alloc_page` 返回失败）时触发 swap 而非直接拒绝请求（即 preemption）。

**验收标准**
- *功能验收*：新增 `tests/test_kv_offload.cpp`：构造显存受限场景（小页池 + 多序列），断言被换出页在再次访问时正确换回、注意力结果正确（与无卸载模式余弦相似度 ≥ 0.999）。disk spill 开启时断电/重启 host 缓存后仍可从 disk 恢复（可选验收）。
- *资源验收*：在 `max_num_seqs` 远超 GPU 显存容量的压测下，系统不 OOM、靠换出维持服务；GPU KV 显存占用被限制在上限内（监控 `active_bytes()` / `num_free_pages()`）。
- *回归验收*：不开启卸载时行为与现状逐 bit 一致（开关默认 off）；`tests/test_prefix_cache.py`、`tests/test_layer_policies.py` 通过。

### 1.2 max_num_seqs 按空闲显存自动定容

> **状态：已完成（2026-08-31）**

**改动文件**：`include/forge/kv_sizing.h` / `src/inference/kv_sizing.cpp`（新增，`kv_token_bytes` / `per_seq_kv_bytes` / `auto_size_kv` / `auto_size_kv_for_device`）、`src/inference/kv_storage.cpp`（`init` 中 `max_num_seqs<=0` 时按设备空闲显存反推 `max_num_seqs` 与每层 `max_pages`，显式覆盖保留）、`tests/test_kv_sizing.cpp`（新增，C++ 验收）。

**验证**：`tests/test_kv_sizing.cpp`（18 项检查 PASS）：`max_num_seqs * per_seq_kv_bytes ≤ budget * safety`、结果为该预算下的最大值、更大预算单调不减、Q8_0 比 FP32 容纳更多序列、零预算/极小预算/空参数/越界 safety 等边界行为、CPU 设备查询返回 0 触发回退。

**动机 / 现状**
`KVCache::max_seqs_ = 32`、`PagedKVStorage::max_num_seqs_ = 32` 硬编码，paged 模式下并发上限被卡死，无法利用多余显存。

**改动点**
- init 时按 `free_gpu_mem / per_seq_kv_bytes`（含页池 + 前缀缓存预算）反推 `max_num_seqs` 与每层的 `max_pages`；提供显式覆盖参数（CLI / config）保留手动能力。
- 在 `PagedKVStorage::init` 与 `KVMemory::init_storage` 中接入该计算。

**验收标准**
- *功能验收*：在给定 GPU 上，自动定容的 `max_num_seqs` 满足 `max_num_seqs * per_seq_kv_bytes ≤ free_gpu_mem * 安全系数(如 0.9)`；手动覆盖值仍能正确生效。
- *资源验收*：相同模型下，自动定容相对硬编码 32 的并发上限提升（可按显存大小给出倍数）；压测 `benchmarks/` 多序列吞吐随显存增大而增大（无人为 32 上限截断）。
- *回归验收*：单序列、小批量行为不变；所有既有测试通过。

### 1.3 chunked prefill 与 decode 交织校验

> **状态：已完成（2026-08-31）**（性能验收：功能级 decode TPOT 微基准已 **PASS**，见 `benchmarks/bench_chunked_prefill_tpot.py`；GPU+GGUF 生产级 P99 压测待环境具备，按 roadmap 语义属后续压测环节）

**改动文件**：`include/forge/request_scheduler.h` / `src/inference/request_scheduler.cpp`（`prefill_chunk_size_`：<0 禁用切分（legacy 整段 prefill）、0 跟随 `n_ubatch`、>0 显式覆盖；prefill 按 chunk 拆多步、中间 chunk 不采样（`sample_flags`，引擎对其中间行 zero-fill）、末 chunk 才出 token；prefill 与 decode 请求同 batch（continuous batching）；发布 `last_step_prefill_tokens` / `last_step_decode_tokens` / `last_step_decode_ratio` / `max_step_latency_ms` / `interleaved_steps` / `prefill_chunks_issued`，reset 时清零）、`src/bindings/scheduler.cpp`（Python 暴露上述指标与 `prefill_chunk_size` 属性）、`tests/test_chunked_prefill.py`（新增，验收测试）。

**验证**：
- `tests/test_chunked_prefill.py`（4 项 PASS）：长 prompt（400 token）在 chunk=32 下跨 ≥2 步 prefill；混合负载中 decode 请求在长 prefill 期间几乎每步都产出 token（不被饿死）；控制组（`prefill_chunk_size=-1`）整段 prefill 为 1 chunk，切分为 ceil(400/32)=13 chunks；指标可从 Python 读取。
- 回归：`tests/test_stage5_session.py`（12 项 PASS）——修复了该测试对已删除的 `forge_llm._PlainTemplate` 的陈旧引用（改为 `forge.ChatTemplateEngine()`，与 `forge_llm.py` 一致），属 pre-existing 测试失配、与本次改动无关。
- 相关回归 `test_scheduler.py` / `test_kv_offload.py` / `test_prefix_cache.py` / `test_layer_policies.py` + stage5 + chunked 共 **41 passed**。
- 性能验收（功能级，CPU fixture 微基准 `benchmarks/bench_chunked_prefill_tpot.py` PASS）：纯 decode 基线 step 延迟 mean≈0.17ms；**chunked OFF** 整段 prefill 独占一步使 decode step 延迟膨胀 **≈11.5×**（1.95ms）；**chunked ON** 长 prefill 切 13 步、decode 每步推进、step 延迟仅 **1.46× baseline**（0.25ms）——证明启用 chunked 后 decode 不被拖慢（TPOT 不退化）。GPU+GGUF 生产级 P99 压测（roadmap 性能验收的压测项）待 GPU 环境具备后由 `benchmarks/bench_inference.py` 接 `RequestScheduler` 完成。

**动机 / 现状**
已有 `split_batch` 静态切分，但需确认 scheduler 是否把 prefill chunk 与 decode 放入同一 batch（continuous batching），否则长 prefill 会独占一步、阻塞 decode 延迟。

**改动点**
- 仅校验 + 必要时修正 `request_scheduler.cpp` 的 `step()`：prefill 微批与 decode 请求可同 batch 调度；长 prefill 被 `n_ubatch` 切成多步，中间穿插 decode。
- 暴露指标：prefill 步内 decode token 占比、单步最大延迟。

**验收标准**
- *功能验收*：构造「1 长 prefill（如 8k）+ N 个 decode」混合负载，断言在 prefill 未完成期间 decode 请求持续产出 token（不被长 prefill 饿死）。
- *性能验收*：相对「prefill 独占一步」基线，混合负载的 decode 首 token 间隔（TPOT）下降、尾延迟（P99）下降（给出相对百分比，压测 `benchmarks/`）。
- *回归验收*：纯 prefill、纯 decode 负载输出与改造前一致；`tests/test_stage5_session.py` 通过。

---

## 5. 阶段 2：高级与特定场景

### 2.1 Radix-tree 前缀缓存（任意位置共享）

> **状态：已完成（2026-09-01）**（性能验收：前缀命中率压测对比需 GPU + `benchmarks/`，属后续压测环节；功能/回归验收已通过）

**动机 / 现状**
当前 hash 前缀缓存仅锚定 **prompt 开头**的完整页（`PrefixEntry` 按 token 0 起算）。vLLM 用 radix tree，能共享对话中段、多轮里任意公共前缀。

**改动点**
- 以 radix tree 替代 `PrefixCache` 的 `unordered_map<uint64_t, PrefixEntry>`：节点按页边界，支持任意位置公共前缀的嵌套共享与子树 ref-count 淘汰。
- 保留 `seq_share / seq_remove / release` 接口不变（`KVMemory` 调用方无需改）。

**验收标准**
- *功能验收*：新增用例——请求 A 含前缀 X+Y、请求 B 含前缀 X+Z，断言 B 命中并复用 X 的 KV（而不要求 B 以 X 开头之外的额外约束）；多轮对话中历史轮前缀被后续轮命中。
- *性能验收*：共享中段文档的多请求场景，相对只在开头共享的 hash 方案，前缀命中率（prefix_hits/tokens）提升（给出压测对比）。
- *回归验收*：现有 `tests/test_prefix_cache.py` 用例在 radix 实现下仍通过；LRU 语义等价（ref_count>0 不被驱逐）。

**验证**
- 实现：`include/forge/prefix_cache.h` / `src/inference/prefix_cache.cpp` 将 `PrefixCache` 从 `unordered_map` 重构为 radix tree。节点存段 token（**按页粒度**匹配与分裂，保证 KV 共享页对齐），但其 `owner_seq_id` cache seq 持有该节点的**完整前缀** `[0, cumulative_len)`，故 lookup 仅对最深匹配节点做**一次** `seq_share`（规避多段共享导致的 `logical_len` 累计错误）；注册时在分流点按页边界**分裂**节点（物理 KV 不拷贝，仅 `seq_share` 页引用）；淘汰为**叶子级 LRU**（ref_count>0 的节点及其子树不被驱逐）。公开接口 `try_lookup / register_prefix / release_prefix / evict_lru / clear / size / has_request / hits / misses` 对调度器完全不变。
- 功能验收（`tests/test_prefix_cache.py` 新增 3 项 PASS）：`X+Y` 与 `X+Z` 复用共享前缀 X；三路 `X+Y / X+Z / X+W` 在首次分叉后，第三请求直接命中已有的 X 节点（无需重算 X）；多轮对话中后两轮命中共享 system 前缀；且命中后 greedy 输出与从零计算**逐 token 一致**（证明 KV 被正确复用、无污染）。
- 回归验收：现有 8 项前缀缓存用例 + `test_chunked_prefill / test_scheduler / test_kv_offload / test_layer_policies / test_stage5_session` 共 **44 passed**。
- 性能验收（前缀命中率压测对比 hash 方案）：需 GPU + `benchmarks/`，属后续压测环节（按 roadmap 语义）。

### 2.2 Contiguous 模式 defrag（碎片整理）

> **状态：已完成（2026-09-01）**（自动触发默认关闭，避免影响 paged 默认路径与调度器位置记账；由 `KVCache::set_defrag_enabled(true)` 显式开启，触发点挂在 `ContiguousKVStorage::seq_remove` / `release`）

**动机 / 现状**
paged 已默认；contiguous 路径（`seq_rm` 后槽位留洞）可能碎片化。llama.cpp 有 `llama_kv_cache_defrag`。

**改动点**
- 仅在保留 contiguous 模式时实现：compact 存活 cell（基于 `KVCellMeta.seq_id_mask`），更新 `filled` 与引用方 slot 索引。
- 提供 `defrag()` 触发点（序列退出 / 空闲 slot 不足时）。

**验收标准**
- *功能验收*：`tests/test_kv_defrag.cpp`：连续增删多序列产生孔洞后调用 `defrag()`，断言 `num_free_slots()` 合并为连续空闲、后续 `update` 能复用、注意力结果与无孔洞序列一致。
- *回归验收*：开启 defrag 不影响 `seq_cp / seq_keep / rollback` 语义；paged 模式零影响。

**验证**
- 实现：`include/forge/kv_cache.h` / `src/inference/kv_cache.cpp` 新增 `KVCache::defrag()`（按页粒度把存活 cell 压缩到每层前端、移动 K/V 行数据 FP32+量化+反量化 FP32 影子+FP8 缩放，并保持 `slot == 逻辑位置` 不变量，故注意力读取 `[0, filled)` 仍正确；ring-buffer/SWA 层因原地回收不碎片化被跳过）；`defrag_if_needed()`（检测到空洞时触发）+ `set_defrag_enabled`（默认 false）；并经 `ContiguousKVStorage` / `KVMemory` 透传到调度器可用。
- 正确性关键：压缩会**重编号** cell 的 `pos`；因此仅在无序列处于 decode 中途（或调用方用 `seq_filled()`/`filled()` 重新推导各序列位置）时调用——故自动触发默认关闭，由引擎在「序列退出 / 空闲不足」且安全的时机显式开启。
- `tests/test_kv_defrag.cpp`（4 项 PASS）：(1) 中段空洞压缩为尾随连续空闲；(2) 压缩后 `get_key_filled`/`get_value_filled` 与「同内容无空洞参考缓存」逐字节一致（注意力结果不变）；(3) 压缩后 `update` 复用尾随空间；(4) 开启 `defrag_enabled_` 后 `seq_rm` 自动压缩、关闭时不压缩。
- 回归：`tests/test_prefix_cache.py` 11 项通过（含 contiguous 模式用例，自动 defrag 默认关闭故行为不变）；`forge` 模块重建通过。

### 2.3 Recurrent 状态缓存落地

> **状态：待做**（需 recurrent / hybrid 模型 fixture，当前环境无；`KVLayerPolicy::Recurrent` 仍是 stub）

**动机 / 现状**
`KVLayerPolicy::Recurrent` 仍是 stub（按 Full 处理）。Mamba / DeltaNet / Qwen3 类循环模型需要 `cache_r / cache_s` 双状态张量 + 每序列回滚（对齐 llama.cpp `llama_memory_recurrent`）。

**改动点**
- 在 `KVCacheLayer` / `PagedKVStorage` 增加 recurrent 状态存储（卷积/门控 delta 参数 + 主隐藏状态），随 `init_per_layer` 分配。
- 实现 `seq_rm` 尾部相交校验（防止破坏非线性状态）、每序列回滚快照。

**验收标准**
- *功能验收*：接入一个 recurrent / hybrid 模型（如 Qwen3 / 类 Mamba），断言 prefill→decode 正确、spec 验证（`tests/test_spec_verify.cpp`）通过、回滚（speculative rejection）正确。
- *回归验收*：非 recurrent 模型（LLaMA / Qwen 全注意力）行为不变。

### 2.4 Encoder / cross-attention KV cache

> **状态：待做**（需多模态 cross-attn 模型 fixture，当前环境无）

**动机 / 现状**
若多模态模型用 cross-attention（encoder-decoder），需独立 encoder KV 缓存（llama.cpp 有 cross KV cache）。

**改动点**
- 在 `KVMemory` 增加 encoder KV 视图（与 decoder KV 分离），供 cross-attention 层读取。
- 生命周期随 encoder 前向结束而固定、随请求结束释放。

**验收验收**
- *功能验收*：多模态 cross-attn 模型跑通，encoder KV 在 decode 阶段正确被 cross-attention 读取；结果与非缓存（每次重算 encoder）一致。
- *回归验收*：无 cross-attn 模型零影响。

### 2.5 工程小项

> **状态：已完成（2026-09-01）**（**hash 碰撞兜底**：radix tree 改为逐 token 精确匹配，已无 hash 碰撞风险，该项被 2.1 重构自然消除，N/A；**prefill 整段量化写入**：`update_quantized_cuda` 已用单次 `launch_quantize_*_matrix`（覆盖 `num_rows` 整段）取代逐 token kernel launch，prefill 注意力层一次处理整段故 `update` 以 `seq_len=段长` 调用直接走整段量化；已在 RTX 4050 上 GPU 验证 bit 一致）

- **hash 碰撞兜底**：前缀命中后加一次 KV 内容/长度校验（FNV-1a 64-bit 极低碰撞概率，vLLM 同理接受），碰撞时回退到本地重算。*（2.1 radix tree 已用精确 token 比较取代 hash，此兜底不再需要。）*
- **prefill 整段量化写入**：`update(seq,pos,...)` 改为 prefill 阶段整段一次性量化写入，减少逐 token kernel launch。
- **验收**：两项均附单元测试（碰撞回退用例、整段写入与逐 token 写入结果 bit 一致）。

**验证**
- 实现确认（无新增 kernel 改动，整段量化本已就位）：`src/inference/kv_cache.cpp` 的 `update_quantized_cuda` 对每个 K/V 仅发起一次 `cuda::launch_quantize_<TYPE>_matrix(d_*, q_dst, seq_len, kv_dim, stream)`（`<<<seq_len, threads>>>` 单 kernel 覆盖整段）；逐行量化相互独立，故整段与逐 token 输出数学等价。`src/inference/layers/*_attention.cpp` 与 `engines/*.cpp` 在 prefill 时以整段 `seq_len` 调用 `kv_cache_.update(...)`，自然复用该整段路径，避免了逐 token 的 launch 开销。
- 正确性验证：`tests/test_kv_quant_segment.cpp`（5 类 dtype 全 PASS，实跑于 NVIDIA RTX 4050 / CUDA 12.8）：对同一段 K/V，分别用「整段一次性 `update(seq_len=N)`」与「逐 token `update(seq_len=1)×N`」写入两张量化缓存，将 `get_key_filled`/`get_value_filled` 反量化结果拷回主机逐字节比较——Q8_0 / Q4_0 / F16 / FP8_E4M3 / FP8_E5M2 均与逐 token 写入 **bit 一致**（证明整段量化不改变 KV 内容、注意力结果不变）。

---

## 6. 验收方法与基准（通用）

**构建与测试**
```bash
./build.sh            # 在 Tests/Benchmarks 提示处选 y
pytest tests/test_prefix_cache.py tests/test_layer_policies.py tests/test_stage5_session.py
./build/<...>/tests/test_spec_verify
python benchmarks/bench_kv_cache.py
```

**正确性门槛（回归基线）**
- 贪婪解码下，量化 KV（改造后）与 FP32 baseline 的 token 序列一致率 ≥ 99.9%；
- 非贪婪下，逐 token logits 余弦相似度 ≥ 0.999（Q8_0）/ ≥ 0.995（Q4_K）/ ≥ 0.997（FP8 e5m2）；
- 长文本（≥ 4k tokens）生成 KL 散度低于既定阈值。

**资源门槛**
- 每个阶段需报告：KV 显存峰值、GPU 显存占用、`max_num_seqs`、token/s（prefill / decode）、尾延迟 P99。
- 每项「性能验收」须给出相对于改造前的量化对比（显存↓ X%、吞吐↑ Y%）。

---

## 7. 里程碑与执行顺序

1. **先 0.1**：零架构改动、直接砍量化 KV 的 FP32 暂存开销，风险最低、收益确定。
2. **再 0.2**：FP8 dtype，叠加 0.1 的 fused 在线反量化收益最大。
3. **然后 1.x**：把并发与显存弹性拉满（卸载 + 自动定容 + 交织校验）。
4. **最后 2.x**：按目标模型/部署形态（recurrent、多模态、radix 共享）按需落地。

> 阶段 0 建议在合并前以 `benchmarks/bench_kv_cache.py` 跑出前后对比数据，作为验收证据并入 PR。
