# Forge CUDA Execution Backend 优化计划

> 本文件由 `plan.md`（v2 重新评估报告）推导而来，所有引用的代码位置均已对当前
> `master`（commit `2c959ae`，278 commits）逐条核验。
>
> 目标：**不动 runtime 架构，只把 `ExecutionPlan → CUDA kernel` 这一段现代化。**
> `ExecutionPlan / ComputeGraph / MemoryPlanner / PrefixCache / PagedKV / CudaGraphRunner`
> 全部保持复用。

---

## 0. 基线事实（核验结论）

### 0.1 已具备、本计划不再重复实现

| 能力 | 证据 |
|---|---|
| ExecutionPlan 架构规划 | `src/inference/execution_plan.cpp:134-153`（SSM→Qwen35Hybrid / MLA→DeepSeekMLA / Gemma4→PerLayerKV / MoE→Phimoe），`:179-198` 创建期一致性校验 |
| Q8_1 + dp4a GEMV | `src/operators/cuda/cuda_gemv_instances.cu:38-60`，16 种量化类型中 15 种已接入 |
| MMQ tiled 量化 GEMM | `src/operators/cuda/cuda_mmq.cu`（1679 行），`matmul_cuda.cpp:187-189` 在 M>1 优先走 MMQ |
| Attention 路径选择 | `src/inference/layers/attention_selector.cpp:7-65` |
| FP8 KV（E4M3/E5M2） | `include/forge/kv_cache.h:17-18` + `include/forge/fp8_utils.h` |
| FP16 Tensor Core cuBLAS GEMM | `src/operators/cuda/cuda_quant.cu:1107-1109`（`CUBLAS_GEMM_DEFAULT_TENSOR_OP`） |
| Paged KV / Prefix Cache / CUDA Graph / Speculative | 均已就位，见 `plan.md` 第 24 节 |

### 0.2 实测规模（用于评估重构成本）

| 文件 | 行数 |
|---|---:|
| `src/operators/cuda/cuda_gemv.cu` | 3584 |
| `src/operators/cuda/cuda_fused.cu` | 2925 |
| `src/operators/cuda/cuda_quant.cu` | 1991 |
| `src/operators/cuda/cuda_mmq.cu` | 1679 |
| `src/operators/cuda/cuda_fused_attn.cu` | 1058 |
| `src/operators/cuda/cuda_paged_kv.cu` | 916 |
| `src/operators/cuda/cuda_flash_attn.cu` | 401 |
| `src/operators/cuda/cuda_moe.cu` | 292 |

### 0.3 本计划依赖的构建/验证入口

```bash
# 构建（默认 USE_CUBLAS=ON，见 CMakeLists.txt:87）
cmake -B build -DCMAKE_BUILD_TYPE=Release -DFORGE_USE_CUDA=ON \
      -DUSE_CUBLAS=ON -DFORGE_BUILD_TESTS=ON -DFORGE_BUILD_BENCHMARKS=ON
cmake --build build -j$(nproc)

# 现有 micro-benchmark（tests/，由 FORGE_BUILD_BENCHMARKS=ON 开启）
# 注意二进制名为 forge-bench-*，且位于 build/tests/ 下
build/tests/forge-bench-prefill_gemm        # GEMM/MMQ prefill
build/tests/forge-bench-output_proj_q6k     # Q6_K GEMV
build/tests/forge-bench-ffn_q2k             # Q2_K FFN
build/tests/forge-bench-moe_real            # MoE 真实权重
build/tests/forge-bench-moe_multigemv       # MoE 多 GEMV

# 正确性回归
build/tests/test_mmq_reference        # MMQ 参考实现比对
build/tests/test_q8_roundtrip         # Q8 量化往返
build/tests/test_iq2xs_compare / test_iq3s_compare
python -m pytest tests/test_quant_validation.py tests/test_cross_validation.py

# Python 层 benchmark
python benchmarks/bench_inference.py
python benchmarks/bench_kv_cache.py
python benchmarks/verify_fp8_kv_accuracy.py
```

---

## 1. 分期总览

```text
Phase 0  基线工程      ── 清理 + 建基线，零性能风险
   ↓
Phase 1  补课型优化     ── 低风险、确定性收益（MoE/dual/split-K）
   ↓
Phase 2  Attention 后端化 ── 中等风险，prefill 是最大收益点
   ↓
Phase 3  Tensor Core 量化 GEMM ── 高风险高收益，仅限 M>16（Q4_0/Q4_K/Q8_0 ✅）
   ↓
Phase 3.5 Q8_0 MMQ 基线修复 ── 补齐真实 M>1 基线
   ↓
Phase 4  KernelSelector 2.0 + Autotuner ── 架构性收益（4-A/4-B ✅，4-C ⏸）
   ↓
Phase 5  Quantized Tensor-Core Coverage ── Q6_K/Q5_K 并入，✅ COMPLETE（**永久冻结**）
   ↓
Phase 6  Native 低精度执行模型 ── FP16→BF16→FP8 W8A8→NVFP4（**当前**）
   ↓
Phase 7  Advanced GPU Backends / Autotuning ── 长期（arch 专用 kernel、persistent tuning）
```

| 阶段 | 主题 | 风险 | 预期收益面 |
|---|---|---|---|
| Phase 0 | 基线工程 | 极低 | 无直接收益，解锁后续 |
| Phase 1 | 补课型优化 | 低 | MoE decode、QKV/FFN 拼接、output proj |
| Phase 2 | Attention 后端化 | 中 | prefill（含量化 KV）、长上下文 decode |
| Phase 3 | Tensor Core 量化 GEMM | 高 | prefill / 大 batch |
| Phase 3.5 | Q8_0 MMQ 基线修复 | 低 | 正确基线 |
| Phase 4 | Selector + Autotuner | 中 | 全链路（消除人工调参） |
| Phase 5 | Quantized TC Coverage（Q6_K/Q5_K）✅ COMPLETE | 中 | prefill / MoE 覆盖面 |
| Phase 6 | Native LP：FP16 → BF16 → FP8 W8A8 → NVFP4（进行中） | 高 | 全链路（结构性） |
| Phase 7 | Advanced Backends / Tuning | 高 | 长期 |

**执行原则**：每个阶段独立可回滚；只有上一阶段的验收项全绿才进入下一阶段。

---

## Phase 0 — 基线工程

**目标**：消除会持续制造返工的工程债，并建立可比对的性能与正确性基线。

### 0-A. 把 `plan.md` 纳入版本管理

`plan.md` 当前未被 git 跟踪（`git ls-files` 查不到），已迭代两版且含大量判断依据。

```bash
git add plan.md docs/optimization_plan_zh.md && git commit -m "docs: add source-verified optimization plan"
```

### 0-B. 修复 stale generator（**本阶段最高优先级**）

**问题**：dispatch 表存在两份互相矛盾的 source of truth。

```38:46:src/operators/cuda/cuda_gemv_instances.cu
const GemvFn gemv_dispatch[21] = {
    /* FP32=0  */ nullptr,
    /* FP16=1  */ nullptr,
    /* Q4_0=2  */ launch_gemv_q4_0_q8_1,                          // Q8_1+dp4a
    /* Q4_1=3  */ launch_gemv_q4_1_q8_1,                            // Q8_1+dp4a (Phase 5)
    /* Q4_K=4  */ launch_gemv_q4_k_q8_1,                            // special Q8_1+dp4a
```
对比生成器模板（**仍然是旧的**）：
```63:63:tools/generate_gemv_instances.py
    /* Q4_0=2  */ launch_gemv_q4_0_transB,                          // special smem/splitK
```

一旦有人重新生成 instances，会把已下线的 scalar 路径重新带回 dispatch。

**改动**：
1. 建立 `src/operators/cuda/kernel_manifest.yaml`，作为唯一 source of truth：
   ```yaml
   q4_0:
     gemv_decode: launch_gemv_q4_0_q8_1
     gemv_batch:  launch_gemv_q4_0_q8_1_batch
     mmq:         launch_mmq_q4_0
   # ... 其余 15 种
   ```
2. `tools/generate_gemv_instances.py` 改为读 manifest 生成 `cuda_gemv_instances.cu`、
   tests、benchmark 三处。
3. 加 CI 检查：生成结果与实际文件 `diff` 非空即失败。

**验收**：
- `python tools/generate_gemv_instances.py --check` 退出码 0
- `git diff --exit-code src/operators/cuda/cuda_gemv_instances.cu` 在重新生成后为空

### 0-C. 死代码清理

已确认下线但仍参与编译的符号（dispatch 表中已无引用）：

| 符号 | 位置 |
|---|---|
| `launch_gemv_q4_0_transB` | `cuda_gemv.cu:237` |
| `gemv_q4_0_transB_smem_kernel`（含 scalar dot `:167-168`） | `cuda_gemv.cu:~150-235` |
| `gemv_q4_0_splitK_kernel` | `cuda_gemv.cu:~180-235` |
| `launch_gemv_q4_0_transB_dual` | `cuda_gemv.cu:329` |
| `gemm_tiled_kernel` / `launch_gemm_tiled` | `cuda_gemm.cu:7,49`（仅 `FORGE_USE_CUBLAS=0` 时用到） |

**注意**：`launch_gemm_tiled` 不能删——它是无 cuBLAS 环境的 fallback（`cuda_quant.cu:1075-1077`）。
改为在 `cuda_gemm.cu` 顶部加注释说明其定位，避免后人误判为主路径。

**验收**：编译无 warning；`nm build/libforge* | grep q4_0_transB` 无输出（确认真正移除）。

### 0-D. 建立基线数据

在**任何**性能改动之前，跑一遍并归档：

```bash
# 先开启 benchmark 开关并构建（见 tests/CMakeLists.txt:21,216）
cmake -B build -DFORGE_BUILD_BENCHMARKS=ON
cmake --build build -j$(nproc) --target forge-bench-prefill_gemm \
    forge-bench-output_proj_q6k forge-bench-ffn_q2k \
    forge-bench-moe_real forge-bench-moe_multigemv

TAG=$(git rev-parse --short HEAD)-$(git diff --quiet -- src include && echo clean || echo dirty)-phase0
mkdir -p baseline/$TAG
# 记录硬件与构建配置（见基线目录 META.txt）
for b in prefill_gemm output_proj_q6k ffn_q2k moe_real moe_multigemv; do
    ./build/tests/forge-bench-$b > baseline/$TAG/$b.txt 2>&1
done
# Python 层 benchmark 需要真实模型权重，无权重时跳过：
#   python benchmarks/bench_inference.py > baseline/$TAG/inference.txt
```

同时记录硬件信息（GPU 型号、SM 数、driver、CUDA 版本），后续所有收益都必须
对照同一台机器。

### 0-E. 文档措辞修正（顺手）

`docs/dependencies.md:14` 写 `opt-in via USE_CUBLAS=ON`（暗示默认关闭），
与同文件 `:94` 的 `GPU + cuBLAS (default) ... ON` 不一致。改为 `enabled by default`。

---

## Phase 1 — 补课型优化（低风险，确定性收益）

**目标**：把已经落后于主路径的部件拉齐到现有技术水平。这一阶段**不引入新架构**，
收益可预测。

### 1-A. 【最高优先级】MoE expert GEMV 迁移到 Q8_1 + dp4a

**这是本计划中最明确、风险最低的收益点，`plan.md` 未覆盖。**

**问题**：主 GEMV 路径早已全面 dp4a 化，但 MoE expert GEMV 仍停在 scalar 时代。

```cpp
// src/operators/cuda/cuda_moe.cu:171, 207
using Traits = CudaQuantTraits<DT>;
...
sum += Traits::dot_block(block_ptr, x_token, base, K);
```

而 `CudaQuantTraits<DT>::dot_block()` 是逐元素 scalar 实现：

```cuda
// src/operators/cuda/cuda_gemv_tmpl.cuh:38-41
for (int j = 0; j < block_elements; ++j) {
    if (base + j >= K) break;
    sum += x[base + j] * (static_cast<float>(q4_unpack(qs, j)) * scale);
}
```

对照主路径 dispatch（`cuda_gemv_instances.cu:41`）：Q4_0 → `launch_gemv_q4_0_q8_1`
（Q8_1 + dp4a）。也就是说 **MoE 正在用被主路径淘汰的实现**。

调用点确认在热路径：`src/inference/layers/gemma4_moe.cpp:69-81`，
覆盖 Q4_0 / Q4_K / Q6_K / Q8_0 四种。

**改动**：
1. 在 `cuda_moe.cu` 中，MoE kernel 启动前先把 `x` 量化成 Q8_1 到 `scratch_pool()`
   （复用 `quantize_q8_1_kernel`，已在 `cuda_gemv.cu` 中大量使用）。
2. 为每个 expert block 复用该 Q8_1 buffer（多个 expert 共享同一 activation，
   正好契合 MoE 结构，量化一次、所有 expert 复用）。
3. 添加 `CudaQuantTraits<DT>::dot_block_q8_1()` 或直接复用 `cuda_mmq.cuh` 中
   已有的 dp4a dot（`cuda_mmq.cu:170` 等处的 `forge_dp4a`）。
4. MoE 目前只实例化 4 种类型（`:242-251`），工作量可控。

**风险**：`scratch_pool()`（定义于 `cuda_common.h:28-39`）是**全局单例**，
同一 kernel 内多次 `ensure()` 会返回同一块或不同偏移，需确认在
`n_expert_used` 个 block 并发读同一个 Q8_1 buffer 时没有写冲突（只读，应无冲突）。
另外 CUDA Graph capture 期间需保证 scratch 地址稳定。

**验收**：
- `./build/tests/forge-bench-moe_real` 与 `forge-bench-moe_multigemv` 对比 Phase 0 基线，
  记录 decode（batch=1）与 prefill（batch=32）两档加速比
- `python -m pytest tests/test_quant_validation.py` 全绿
- 新增 `tests/test_moe_dp4a_reference.cpp`：与 FP32 参考实现的余弦相似度 ≥ 现有路径

**预期**：decode 路径量化后每 expert 不再重复 dequant，收益主要来自消除
逐元素 unpack；prefill 收益更明显。

---

### 1-B. 消除 `cuda_matmul_transB_dual` 的 D2D 拷贝

```cpp
// src/operators/cuda/matmul_cuda.cpp:236-252
void cuda_matmul_transB_dual(...) {
    (void)K;
    auto out1 = ops::matmul_transB(a, b1);
    auto out2 = ops::matmul_transB(a, b2);
    for (int m = 0; m < M; ++m) {
        cudaMemcpyAsync(o_data + m * N, o1_data + m * N1, N1 * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(o_data + m * N + N1, o2_data + m * N2, N2 * sizeof(float), cudaMemcpyDeviceToDevice);
    }
}
```

问题：两次中间分配 + M 轮 D2D memcpy + 两趟 weight 读取。
`（void）K;` 表明这个函数的签名也没用干净。

**改动**：写 `gemv_dual_q8_1` 单 kernel——量化 x 一次，一个 kernel 内完成两个
weight 矩阵、直接写入最终 `out` 的对应区间。

参考先例：`cuda_gemv.cu:266` 已有 `gemv_q4_0_transB_dual_kernel`，
以及 `cuda_fused.cu` 中的 `launch_ffn_up_fused_q4_0_q8_1`
（`matmul_cuda.cpp:266` 调用）——证明"共享量化 x + 双输出"这个模式在本仓库已跑通。

**验收**：`forge-bench-ffn_q2k` + 新增 dual-path 计时；确认无中间分配
（`FORGE_PROFILING=ON` 下观察 allocator 调用次数下降）。

**状态（已完成，2026-09-04）**：按收窄方案实现——`cuda_matmul_transB_dual` 在 `M==1`
时通过新增的 `gemv_into_slice`（镜像 `cuda_matmul_transB` 的 M==1 派发）把 b1/b2
**直接写入 `out` 的对应切片**（首 N1 列 / 末 N2 列），去除中间张量分配与 M 轮
D2D 交错拷贝；`M>1` 保留原 allocate+interleave 回退以保证交错布局正确。
数值由 `tests/test_phase1_bc_reference.cpp`（`dual(M=1)` vs FP32 参考，cos≈0.99999、
maxabs 在容差内）验证。
> 偏离原计划处：未采用 `gemv_dual_q8_1` 单 kernel（量化 x 一次的融合双输出），
> 因其对泛型 dtype 侵入大且与本阶段收窄范围不符；本方案已达成"去中间分配 + 去 D2D"目标。

---

### 1-C. split-K 的 memset + atomicAdd → 协作式归约

现存两处：

| 位置 | kernel |
|---|---|
| `cuda_gemv.cu:258` + `:233` | `launch_gemv_q4_0_transB` 的 split-K 分支 |
| `cuda_gemv.cu:3578` + `:3552` | `launch_output_proj_q4_0_q8_1` |

后者是**活跃路径**（`llama_graph_builder.cpp:319` 调用），优先级更高。

**问题**：`cudaMemsetAsync(out, 0, N*4)` + 每 warp `atomicAdd(&out[row], sum)`。
在 M=1、N 大（output_proj 场景）时，memset 本身是一次全量写，atomic 又引入
不确定的累加顺序（也影响结果可复现性）。

**改动**：改为 CTA 内两级归约——warp 内 shuffle reduce，warp 间经 shared memory
归约，最后**单一线程**写一次 `out[row]`。

**风险**：输出顺序变化会导致数值结果与原先有细微差异（浮点加法不满足结合律）。
需在 `test_q8_roundtrip` 与 `test_operators.py` 的容差内。

**验收**：
- `forge-bench-output_proj_q6k` 对比基线
- 输出与 FP32 参考的余弦相似度不低于原实现
- 连续两次运行结果 bit-exact（可复现性提升）

**状态（已完成，2026-09-04）**：`output_proj_q4_0_q8_1_kernel` 改为**每 row 一个 CTA**——
warp 内 shuffle 归约 → shared memory 跨 warp 归约 → 单线程写一次 `out[row]`；
移除 `cudaMemsetAsync` 与 `atomicAdd`。数值由 `tests/test_phase1_bc_reference.cpp`
验证：cos≈0.99999、maxabs 在容差内、且与 FP32 参考一致；**两次运行 bit-exact**
（可复现性达成，原 atomicAdd 的累加顺序不确定性消失）。

---

### Phase 1 出口条件

- [x] `kernel_manifest.yaml` 成为唯一 source of truth，CI 有检查
      - manifest 文件 + `tools/generate_gemv_instances.py` 生成器早已就位；
      - 新增 `.github/workflows/build.yml` 的 `kernel-manifest-check` job，
        运行 `python tools/generate_gemv_instances.py --check`（生成文件漂移即失败）。
- [x] MoE dp4a 版本数值正确且 `bench_moe_real` 优于基线（见 Phase 1-A）
- [x] dual matmul 无中间分配（M==1 直接切片写入；M>1 保留回退）
- [x] output_proj 无 memset/atomic，且结果可复现
- [x] 全部 `tests/` 下 C++ 与 Python 测试通过
      - `ctest` 22/22 通过（含交叉测试：neon-arm64 / vsx-ppc64 在 QEMU 下运行成功）；
      - Python 无模型子集 `pytest` 407 passed（6 个 gpu/tinyllama 用例跳过）。

> 已知遗留（不在 Phase 1 范围，已隔离）：opt-in 的 `forge-bench-cpu_decode`
> benchmark 用全路径 `#include "src/operators/cpu/arch/x86/fused.h"`，但其目标的
> include 路径不含仓库根，导致构建失败。该 benchmark 为 `FORGE_BUILD_BENCHMARKS`
> 门控（默认 OFF），与交叉测试/Phase 1 无关；本次构建以
> `-DFORGE_BUILD_BENCHMARKS=OFF` 隔离。交叉测试本身已修复并跑通。

---

## Phase 2 — Attention 后端化

**目标**：把 attention 从"thread-centric 单实现"升级为"多后端可选"。
这是 `plan.md` 第 4-6 节的核心建议。

### 2-A. 现状确认

```cuda
// src/operators/cuda/cuda_flash_attn.cu:44-63  (prefill)
for (int j = tid; j < kv_len; j += BLOCK_SIZE) {
    float dot = 0.0f;
    for (int d = 0; d < HEAD_DIM; ++d) dot += q_row[d] * k_row[d];   // Q 从 global 反复读
    ...
    float local_acc[HEAD_DIM];                                        // 寄存器压力
    for (int d = 0; d < HEAD_DIM; ++d)
        local_acc[d] = local_acc[d] * correction + weight * v_row[d];
}
```

优点（不要误判为"错误实现"）：single-pass online softmax、不重复计算 QK、
不用 atomicAdd、易验证。

真正的问题：**thread-centric 而非 tile-centric**，因此无法接入 Tensor Core，
且 `local_acc[HEAD_DIM]` 在 HEAD_DIM=256/512 时严重限制 occupancy。

> 注：decode 路径**已经**把 Q 放进 shared memory（`:223-224`、`:314-315`），
> 本阶段只针对 prefill。

### 2-B. 【关键缺口】prefill 目前完全没有量化/分页 KV 支持

```cpp
// src/inference/layers/attention_selector.cpp:63-64
    // GQA prefill (seq_len > 1)
    return AttentionPath::CUDA_FP32_PREFILL;
```

**无论 KV 是什么 dtype，prefill 一律走 FP32 路径。**
这意味着 Q4_0 / Q8_0 / FP8 KV cache 目前**只在 decode 阶段省显存**，
prefill 时仍需物化成 FP32。长 prompt 场景下这是双重开销（显存 + 带宽）。

**改动**：新增 prefill 路径枚举并接入 selector：

```cpp
CUDA_FUSED_Q4_0_PREFILL, CUDA_FUSED_Q8_0_PREFILL,
CUDA_FUSED_FP8_E4M3_PREFILL, CUDA_FUSED_FP8_E5M2_PREFILL,
CUDA_PAGED_Q4_0_PREFILL, CUDA_PAGED_FP8_E4M3_PREFILL, ...
```

优先级：先做 **FP8 prefill**（KV 已量化、dequant 便宜、收益最直接），
再做 Q8_0 / Q4_0。

**状态（已完成，2026-09-05）**：全部落地——`CUDA_FUSED_{Q4_0,F16,Q8_0,FP8_E4M3,FP8_E5M2}_PREFILL`
与 `CUDA_PAGED_{FP32,Q4_0,F16,Q8_0,FP8_E4M3,FP8_E5M2}_PREFILL` 共 11 个 prefill 路径（枚举见
`attention_selector.h:45-67`），selector 按 dtype×paged 全分派，executor 11 个 case 全接入
（`attention_executor.cpp:608-631`），内核在 `cuda_tiled_prefill.cu`，paged 通过 device 端
`page_ids` 直接遍历页表、全程无 FP32 物化。落地顺序与本段一致：先 FP8（提交 `da8c4fd`→`cdd65ac`），
再 F16/Q8_0/Q4_0 与 paged（`a43f7ee`→`a83a743`）。Q4_K 无 fused policy：256 元素 super-block 与
HEAD_DIM<256 的 per-head 边界不对齐，保持 FP32 materialize 回退（见 `attention_selector.h` 注释）。
验收：`attn-fp8-prefill`、`tiled-attn-formats` 两个 ctest 全绿（总 25/25）。

### 2-C. 引入 AttentionBackend 分层

```cpp
enum class AttentionBackend {
    Scalar,       // 现有 local_acc[HEAD_DIM] 实现，保留为参考/回退
    Vectorized,   // float4 向量化 + shared memory Q tile
    TensorCore,   // MMA tiled，Phase 3 落地
};
```

在 `attention_selector.cpp` 中增加 backend 维度，先只实现 `Vectorized`：
Q tile 进 shared memory、K/V 用 float4 加载、warp 级归约，
**消除 `local_acc[HEAD_DIM]`**。

**验收**：
- 新增 `tests/test_attention_backend_parity.cpp`：三种 backend 输出与 FP32
  参考的余弦相似度 ≥ 0.999
- `python -m pytest tests/test_forward.py tests/test_cross_validation.py` 全绿
- `bench_inference.py` 记录 prefill tok/s 变化

**状态（已完成 + 一处核心假设修正，2026-09-05）**：enum `AttentionBackend{Scalar,Vectorized,
TensorCore}` 与 Vectorized kernel（Q tile 进 smem + float4 K/V，`cuda_flash_attn.cu:194`）就位，
FP32 prefill 默认路由 Vectorized；parity 验收 `attn-backend-parity` 全绿。

**实测推翻本节核心假设**（RTX 4050 Laptop, sm_89，`tests/bench_tiled_attn.cpp`）：
真正消除 `local_acc[HEAD_DIM]` 的 KV-tiled kernel（`cuda_tiled_prefill.cu` 的
`fused_attn_tiled_prefill_kernel`）在本机全面劣化：

| 路径 | tiled/legacy 加速比 |
|---|--:|
| decode（FP32/F16/Q8_0, kv=2k–32k） | 0.22–0.56× |
| FP32 prefill（q=512, kv=4k–64k） | 0.44–0.49× |
| Q8_0 prefill（q=512，对比物化+vectorized） | ~0.60× |

即「register pressure → occupancy → 性能」链条在本机不成立：legacy/vectorized 已无 spilling、
访存延迟被充分隐藏，tiled 的 shared memory traffic / 同步 / 寻址成本超过了寄存器节约。
**结论：`local_acc[HEAD_DIM]` 是实现细节而非性能指标；优化目标不是"消灭某个数组"，而是
workload-aware 选择实测最快的执行策略。** 因此 Vectorized kernel 仍保留 `local_acc[HEAD_DIM]`
（源码注释已自认 tiling deferred），`choose_attention_backend` 也被移除、后端选择内联进 executor。

**决策**：`cuda_tiled_prefill.cu` 冻结性能调优、保留为 **experimental candidate**（不删除）——
它是量化/分页 prefill 免物化的唯一路径，继续承担该职责并保证正确性；本机所有新 kernel 的
采用一律以实测为准，不以资源占用指标为准。TensorCore 后端与数据驱动 dispatch 移交 Phase 3/4。

**资源占用证据（`cuobjdump --dump-resource-usage` + `ptxas -v`，sm_89；本机无 NVGPU 计数器权限，
无法跑 Nsight Compute，此项为可获取的最高证据等级）**——直接回答「spill / occupancy 是否瓶颈」：

| kernel | REG | spill（ptxas -v） | 理论 CTA/SM | 实测 |
|---|---|---:|---|---:|
| legacy decode `<128,4>`（快路径） | 255（顶格） | **628B st / 388B ld** | ~2（~12%） | 有 spill 仍最快 |
| vectorized `<128,128>`（快路径） | 186 | 0 | ~2（~12%） | FP32 prefill 2×快于 tiled |
| tiled `fused_attn_tiled_prefill_kernel`（慢） | 33–48 | 0 | ~11（~70%） | 0.25–0.60× |

结论比"无 spill + 高 occupancy 才快"更极端：**快路径 legacy decode 本身有真实 spill 且
occupancy 仅 ~12%，依然最快**；tiled 把 occupancy 提升约 5× 反而让性能减半。三条路径共同
指向：本 workload 是 latency/指令效率主导，occupancy 与 spill 都不是第一阶性能变量，寄存器节约
换来的 tiling/staging/同步成本是净损失。

### 2-D. GPU 常驻 paged metadata

按 `plan.md` 第 17 节，把序列页表 metadata 常驻 GPU，消除 decode 热路径中
每次全量 H2D 的上传，并避免共享 staging buffer 的多序列相互覆盖。

**状态（✅ 已完成，2026-09-05）**：实现为 per-(seq, layer) 的常驻 device mirror。

- `SequencePageTable` 新增 4 个字段（`d_mirror / d_cap / d_synced / d_dirty`），每个
  (seq, layer) 拥有一块独立、持续存活的 device 页表，**不再有跨序列共享的 staging buffer**，
  并发序列互不覆盖。
- `upload_seq_page_table_unlocked` 热路径只做**增量追加**：decode 每次只长 1 页，故每 token
  仅同步新增页（O(1)），不再 O(total pages) 线性增长；capacity 用 grow-only 翻倍（amortized
  O(1)），超限时整体 realloc。
- 前缀变化（`seq_remove` 中间删页 / `seq_share`）将对应镜像标 `d_dirty`，下一次上传整表重写，
  保证 device 端与 host 权威表一致。
- 生命周期闭环：`release()` / `reset()` / 析构统一 `cudaFree` 镜像，CPU 后端不受影响。

**验收**：新增 `tests/test_kv_paged_mirror.cpp`（ctest `kv-paged-mirror`），以与 allocator 同规则
的 host 仿真逐字节比对 device 镜像，覆盖①跨页增长、②容量 realloc、③双序列独立指针互不污染、
④`seq_remove` 脏标记 + 释放页回收重用、⑤release/reset 清理；与 `kv-offload` / `kv-defrag` /
`attn-fp8-prefill` / `tiled-attn-formats` 一并全绿。

> **Phase 2 出口核对（2026-09-05）**：2-B ✅、2-C ✅（目标修正为 workload-aware dispatch；
> TensorCore/autotune 移交 Phase 3/4）、2-D ✅（已实现并验收）。Phase 2 全部子项收口。

---

### Phase 2 正式收敛（2026-09-05）

**命名规范（代码/文档统一）**：实测最快的 decode 路径不再叫 "legacy"（语义上易被误读为
"待删除的旧实现"），统一定名为 **streaming decode**（kernel 真名 `flash_attn_gqa_decode_kernel`，
bench 输出列保留历史标签 `legacy_ms` 指同一 kernel）。`cuda_tiled_prefill.cu` 全量 tiling
定名为 **experimental**：性能调优冻结，只保留 paged/量化免物化职责。

**措辞严格限定（防止结论越界）**：以下经验值仅对
`consumer GPU（RTX 4050 sm_89）/ 当前 Forge attention kernel / 当前单请求 workload` 成立：

1. `register pressure ↓ → occupancy ↑ → performance ↑` 在此不成立；REG 255 + 真实 spill +
   ~12% occupancy 的 streaming decode 仍最快（3.1 ms @ 32K），而 REG 33–48 + ~70% occupancy
   的 tiled 只有 0.25–0.60×。
2. 由此导出 **Decode Hypothesis**：decode attention 是 **latency/instruction limited**，而非
   occupancy 或 DRAM-bandwidth limited（实测仅 ~44% DRAM peak）。推论：(a) 保持直接 KV
   streaming；(b) 避免 shared-memory staging；(c) 避免跨 warp 同步；(d) 只在**不引入额外
   staging/sync** 的前提下压低寄存器；(e) 优化对象是 per-warp 工作量粒度与指令吞吐。
3. **不扩展到其他条件**：B>8、长 prefill、多并发请求、数据中心 GPU（H100/B100）或大 M
   上，tiled / 高 occupancy / TensorCore 可能反转；届时以实测为准。
4. **Selector/Autotuner 铁律**：不得以 occupancy 或寄存器数作为 kernel 排序的 proxy，必须以
   目标 workload 的实测 latency/throughput 为最终标准。

**路线拆成两条互不污染 track**（不再叫笼统的 "attention optimization"）：
- **Decode track**：Streaming → register sweet spot → per-warp work 粒度 → latency 优化。
- **Prefill track**：Vectorized baseline（当前 2.1× tiled，已足够好）→ Q/K/V tile + Tensor Core
  → 大 M 场景，**不再试图把 decode 取向的 tiled kernel 做快**。

**Phase 4 搜索空间（killing `block_size` 单一维度）**：
`{warps/CTA, KV chunk per warp, vector width, unroll factor, register budget (min CTA/SM), seq tile, cooperative mode}`。
- **第一批实验 A–F**（针对 streaming decode）：A baseline；B reg −10%；C reg −20%；D unroll×2；
  E unroll×4；F warp work ×2。观测量：latency、registers、spill、DRAM GB/s —— 只看**有无免费收益**
  （255→~200 且不引入 sync/staging/spill）。
- **第二批**：REG 255→~168 使 2 CTA→3 CTA 时，验证对象是"**第 3 个 CTA 带来的 latency hiding 是否
  抵消压寄存器付出的指令成本**"——2 CTA vs 3 CTA 对比，不是 255 vs 168 regs 对比。
- **DB 记录维度**：GPU arch / dtype / head_dim / q_len / kv_len / batch / paged + **objective**
  `{MinLatency, MaxThroughput, MinP99}`（batch=1 → latency，B≥16 → throughput，serving → P99）。

---

## Phase 3 — Tensor Core 量化 GEMM（**限定 M > 16**）

**状态：已封版（2026-09-05）**

```
✓ cuBLAS FP16 TensorCore backend（sm80+，运行时检测）
✓ Q4_0 / Q4_K dispatch（auto: M>=128 → TC，env override: FORGE_MATMUL_PATH /
  FORGE_FORCE_TC / FORGE_FORGE_MMQ）
✓ M-threshold sweep + fine sweep（M∈{8,26,32,64,96,112,128,144,160,200}）
✓ crossover 验证：Q4_0/Q4_K MMQ→TensorCore ≈ M≈96-110（RTX 4050, K=4096, N=128256）
✓ 保守默认阈值 M>=128（不取边缘交点）
✓ M=1 decode 隔离（1.84 ms / 161 GB/s，无回退）
✓ forced-path / auto-path parity（bench 三 pass，auto 与强制路径逐档一致）
✓ 回归套件：mmq-ref 45/45、rows-vs-decode 105 处 OK
✓ 文档（本段）
```

**核心结论（已作为 runtime policy）**：RTX 4050 / Llama-3 tested shape 下，
Q4_0/Q4_K 的 MMQ→TensorCore crossover 约在 **M≈96–110**，最终选择 **M≥128**
作为默认阈值。Q8_0 因 M>1 dp4a 基线缺失被排除在 auto 规则外——该问题已由
**Phase 3.5（下节）**修复并重测。

**目标**：为量化 GEMM 增加第二种 execution backend。

### 3-A. 范围限定（重要）

`plan.md` 第 24 节把此项列为 P0 首位，但表述过宽。必须限定范围，否则容易
投入大量工作却在 decode 上测不出收益：

| M | 瓶颈 | 建议路径 |
|---|---|---|
| M = 1（decode） | memory-bound，weight 读取 | **保持 dp4a**，Tensor Core 无收益 |
| M = 2~16 | 过渡区 | MMQ vs TC 需实测（交给 Phase 4 autotuner） |
| M ≥ 32（prefill） | compute-bound | **Tensor Core** |

理由：dp4a 本身就是 INT8 的 SIMT 实现；MMA 的优势在算力密度，只在
compute-bound 区间兑现。llama.cpp 也是按 batch size 在 MMQ 与
FP16 cuBLAS Tensor Core 之间 dispatch。

### 3-B. 改动

1. 引入路径枚举：
   ```cpp
   enum class QuantComputePath { DP4A, TensorCore };
   ```
2. 在 `matmul_cuda.cpp:176-205` 的 `M > 1` 分支中，按 `M` 与 GPU arch
   在 MMQ 与 Tensor Core 之间 dispatch（当前是无条件优先 MMQ）。
3. Tensor Core 路线：在 shared memory 中把量化权重 dequant 成 FP16，
   再喂 `mma.sync.aligned.m16n8k16`（Ampere+）；Blackwell 上评估 FP8/FP4 路径。
4. 优先支持 Q4_0 / Q4_K / Q8_0 三种（`test_mmq_reference.cpp` 已覆盖），
   其余沿用 MMQ。

**风险**：最高的一项。需要 `sm_80+` 特性检测与运行时降级；
dequant-to-FP16 会额外消耗 shared memory，需重新做 occupancy 调优。

**验收**：
- `tests/test_mmq_reference.cpp` 扩展覆盖 TC 路径（45/45 绿）
- `forge-bench-prefill_gemm` 在 M=32/64/128 档位对比 MMQ，记录加速比（已完成并
  加密 M∈{96,112,144,160,200}，见下方"实测收敛"）
- M=1 档位性能**不得回退**（实测 1.84 ms / 161 GB/s 保持，见下）

---

### Phase 3 实测收敛（2026-09-05）

**实现（代码路径）**：
- `matmul_cuda.cpp` M>1 分支增加 dispatch 维度，DP4A（MMQ）/ TensorCore 两条
  路径。env 覆盖（按优先级）：`FORGE_MATMUL_PATH={mmq|dp4a|tc}`，以及 benchmark
  别名 `FORGE_FORCE_TC=1`（=tc）与 `FORGE_FORCE_MMQ=1`（=mmq/dp4a）；
  其余情况落入 **auto 规则（当前默认 landing 状态）**：
  `Q4_0/Q4_K 且 M>=128 → TensorCore；其余 dtype/M → DP4A`。
- Tensor Core 路线实现为**整矩阵 dequant→FP16 + cuBLAS FP16 GEMM**
  （`launch_cublas_gemm_quant_fp16`，N/256 长度行 stride、m8n8k16 块内 noz/z 布局、
  FP16 静态 scratch 缓存），dequant 与 host 端 CPU 反量化共享同一套 block 公式。
  覆盖 Q4_0 / Q4_K / Q8_0 三种（arch ≥ sm_80，运行时检测），其余类型仍走 dp4a。
- 正确性验证基准：`tests/bench_prefill_gemm.cpp` 重建为 CUDA 三 pass sweep
  （K=4096 / N=128256 / M∈{8,26,32,64,96,112,128,144,160,200}）：
  强制 dp4a、强制 TC、以及 **默认 auto**（回归检查 auto 在 M≥128 落在 TC、
  M<128 落在 dp4a 的时间与结果一致性）。验证口径与 `test_mmq_reference`
  完全一致：**host 侧 fp64 golden，按采样网格（M 全行 × 2069 列）的 global-max 归一，
  tol 5e-2**；权重生成器保证有限 fp16 scale（避免随机字节的 NaN/Inf 假信号）。

**正确性（bench 实测，best-of-10）**：

| dtype | fp64 ref 最大 rel | rows-vs-decode 最大 rel |
|---|---:|---:|
| Q4_K dp4a/TC | 4.1e-3 / 4.5e-4 | 1.2e-4 / 4.1e-3 |
| Q4_0 dp4a/TC | 4.5e-3 / 2.5e-4 | 2.8e-3 / 3.6e-3 |
| Q8_0 dp4a/TC | 3.5e-3 / 2.8e-4 | 0.0 / 3.5e-3 |

扩展后的 `test_mmq_reference`（MMQ + TensorCore 45 例）保持全绿。

**性能（llama-3 K=4096, N=128256，RTX 4050 Laptop sm_89，单位 ms，best-of-N）**：

先粗扫（M∈{8,26,32,64,128,200}）定位交点，再在 M∈{96,112,128,144,160}
加密确认安全区（下表为加密后的整组，best-of-3/4）：

| dtype | M=8 | M=26 | M=32 | M=64 | M=96 | M=112 |
|---|---:|---:|---:|---:|---:|---:|
| Q4_K  dp4a / TC | 5.9 / 24.7 | 10.5 / 28.0 | 9.7 / 24.8 | 21.8 / 25.9 | 37.3 / 25.8 | 39.0 / 28.2 |
| Q4_0  dp4a / TC | 3.1 / 21.0 | 7.2 / 21.0 | 6.7 / 17.8 | 19.5 / 20.9 | 23.7 / 20.5 | 29.7 / 22.2 |
| Q8_0  dp4a / TC | 24.0 / 20.6 | 86.8 / 20.7 | 104 / 17.4 | 218 / 21.1 | 329 / 20.8 | 385 / 22.5 |

| dtype | M=128 | M=144 | M=160 | M=200 |
|---|---:|---:|---:|---:|
| Q4_K  dp4a / TC | 43.4 / 28.3 | 45.4 / 29.4 | 53.0 / 29.5 | 70.2 / 34.3 |
| Q4_0  dp4a / TC | 32.0 / 22.4 | 35.2 / 24.1 | 43.1 / 23.5 | 54.6 / 26.9 |
| Q8_0  dp4a / TC | 441 / 22.7 | 435 / 24.4 | 481 / 23.7 | 601 / 28.0 |

TC 加速比（dp4a/TC）：
- **Q4_K：0.24x@8 → 0.38x@26 → 0.84x@64 → 1.44x@96 → 1.38x@112 →
  1.52x@128 → 1.54x@144 → 1.80x@160 → 2.12x@200**（best-of-3）。
- **Q4_0：0.15x@8 → 0.34x@26 → 0.93x@64 → 1.15x@96 → 1.34x@112 →
  1.80x@128 → 1.46x@144 → 1.84x@160 → 2.03x@200**。
  M=112~160 单档有 ±20% 散热噪声（TC 侧 20~34 ms 抖动），但 **M≥128 的全部
  档位稳定 ≥1.4x，M≈96-110 为 crossover**。
- **Q8_0：M≥26 后 4.2x~22.7x 单调放大**（该"收益"是基线病的产物，见结论 2）。

auto 回归列与强制路径在每档 M 完全对齐（tc 档与 auto 档在 M≥128 同 kernel，
时间一致；M<128 auto 落在 dp4a；Q8_0 全档 dp4a），且 rows-vs-decode 105 处
校验全 OK、M=1 fp64/解码行 1.84 ms 无回退。

**large-M 平台延伸（2026-09-05，同机/同形状，best-of-3；`large` 模式
M=128..1024，rows-vs-decode 每档 OK、auto 与强制路径一致）**：

| dtype | M=128 | M=256 | M=384 | M=512 | M=768 | M=1024 |
|---|---:|---:|---:|---:|---:|---:|
| Q4_K  dp4a / TC | 47.4 / 25.4 | 88.0 / 34.7 | 135 / 39.5 | 188 / 47.1 | 287 / 71.6 | 386 / 89.8 |
| Q4_K  speedup | **1.87x** | **2.53x** | **3.42x** | **3.99x** | **4.01x** | **4.30x** |
| Q4_0  dp4a / TC | 29.4 / 19.8 | 66.5 / 28.1 | 97.6 / 35.0 | 133 / 45.2 | 237 / 60.1 | 321 / 84.3 |
| Q4_0  speedup | **1.48x** | **2.37x** | **2.78x** | **2.96x** | **3.94x** | **3.81x** |

结论：TC 收益**不**在 128-256 进入平台，而是随 M 单调放大、到 M≈512-768 才
稳定在 ~4x（Q4_K 上限 GFLOPS ≈ 12.5 TFLOPS）。TC 因此应定位为**明确的
large-M quantized prefill backend**，M>=128 的 auto 规则在更大 M 区间只会
越用越赚，无反转风险。

**结论 / 设计修正**：
1. **3-A 表"M ≥ 32（prefill）compute-bound → Tensor Core"在 llama-3 N=128k
   尺度上被实测推翻**。TC 每次调用需把权重整矩阵重物化为 FP16
   （Q4 全量 1.05 GB B_fp16 写 + GEMM 读一遍），固定开销 ~6-10 ms，
   在 M=32/64 时吃掉全部 MMA 优势，直到 M≈96-110 才反超 dp4a。
2. **默认 dispatch 落地（最终）**：
   `Q4_0/Q4_K 且 M >= 128 → TensorCore`，其余 DP4A。实验依据：
   RTX 4050 上 llama-3 形状的 crossover 在 **M≈96-110**，M=128 是保守默认
   阈值——宁可放弃 M=96~127 的 <1.5x 潜在收益，也不把边缘阈值硬编码进默认
   路径（真实模型还有 shape / 对齐 / workspace / graph capture 抖动）。
   M=1 decode 结构上不走该选择器，解码延迟不受此规则影响。
3. **Q8_0 明确不参与 auto 阈值**：其 4.2x~22.7x 的 TC "收益"不可解释为
   "TC 比 DP4A 快"，而是当前 `M>1` dp4a 基线是**逐行 decode 回退**
   （Q8_0 无 MMQ/batched 实例；M=128 需 ~111 次 weight 读取 ≈440 ms）。
   在真正 MMQ/batched 路径落地并重测（M∈{16,32,64,128,200,256,512}）之前，
   Q8_0 的 M>1 保持 dp4a——不把 baseline bug 伪装成性能收益。
   （修订注脚：本条措施已被 **Phase 3.5（上节）封版取代**——Q8_0 M>1 真 MMQ
   已落地（M=128 从 ≈440ms 降至 46ms），crossover 重测 ≈M48-64，随后并入
   `M>=128 → TC` auto 规则。本条保留作 Phase 3 封版时刻的历史依据。）
4. 留档句：**"Q4_0/Q4_K Tensor Core crossover experimentally established
   around M≈96-110 on RTX 4050 for the tested Llama-3 shape (K=4096,
   N=128256). M=128 is selected as the conservative default threshold.
   Q8_0 results are excluded from threshold inference until its M>1 MMQ
   baseline is corrected."**
5. 后续把硬编码 `M>=128` 迁移进 Phase 4 KernelSelector/autotuner，按
   （arch, dtype, M, N, K）实测选择——auto 规则的 env 别名（`FORGE_FORCE_TC`/
   `FORGE_FORCE_MMQ`）保留作回归开关。
6. 验证口径顺带硬化：早期 per-column `max(1,|acc|)` 口径会在近零列上把
   int8/FP16 噪声放大成"假 MISMATCH"（曾误判 MMQ 大 N 出错）；切换到
   suite 同款 global-max 归一后全部 OK，确认 45 例参考测试未见的大 N 区间
   亦无 kernel 缺陷。

---

## Phase 3.5 — Q8_0 M>1 真 MMQ（消除虚假基线）

**状态：已封版（2026-09-05）**

```
✓ Q8_0 MMQ 特化 kernel（mul_mat_q_kernel 新增分支）
✓ mmq_dispatch[Q8_0] 注册（此前为 nullptr → M>1 落入 warp-per-(m,n) batch GEMV）
✓ 验证：rows-vs-decode max rel 4.15e-07（MMQ 与 decode 位级一致）；mmq-ref 45/45 保持
✓ Q8_0 并入 auto（M>=128 → TC），bench_prefill_gemm 判定同步更新
✓ 回归：105 处 rows-vs-decode/fp64 全 OK，summary 0 mismatch
```

**起因**：Phase 3 收敛时 Q8_0 的 TC 加速比达 4.2x~22.7x，但那是假象——其
`M>1` 的 dp4a 基线是 `launch_gemv_q8_0_q8_1_batch`（warp-per-(m,n)、无 weight
tile 复用，每个 (m,n) 独立重读整行 → M 次 weight 读取：M=128 ≈ 71 GB ≈ 441 ms）。
TC 相对它快 19x 只说明基线病态，不代表 TC 优于真 MMQ。

**实现**（`src/operators/cuda/cuda_mmq.{cu,cuh}`）：照 Q4_0 的 32-elem-block tile
结构为 Q8_0 增加特化。与 Q4_0 的差异仅在两处：Q8_0 block = fp16 scale + **32 个
raw int8**（34 B、qs 从 block 起 2 B 对齐），故 tile 用 8-int 直读、dot 为 8 次
纯 dp4a、**无 nibble 展开、无 bias 项**（`vec_dot_q8_0_q8_1_mmq` = `d·d8·Σdp4a`，
与 decode 路径逐位同构）。`mmq_txs_q8_0`：`qs=I*65`（256 窗口 = 64 ints + pad）、
`dm=8I+I/2`。J=32 行 tile 复用，weight 读取从 `M×N×row` 降到 `(M/32)×N×row`。

**重测 crossover（RTX 4050 / K=4096 / N=128256，dp4a=新 MMQ，best-of-2）**：

| M | 8 | 26 | 32 | 64 | 96 | 112 | 128 | 144 | 160 | 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q8_0 dp4a/TC(ms) | 4.1/21.2 | 11.2/21.2 | 12.2/19.2 | 27.3/20.6 | 37.3/21.3 | 44.0/22.5 | 46.0/20.6 | 55.2/24.0 | 51.3/24.1 | 76.5/26.1 |
| TC/dp4a | 0.19x | 0.53x | 0.64x | 1.32x | 1.75x | 1.96x | 2.23x | 2.30x | 2.13x | 2.93x |

**结论 / 设计修正**：
1. **基线病根治证据**：Q8_0 M>1 默认（dp4a）路径 M=128 从 441 ms → **46 ms
   （9.6x）**，M=200 从 601 ms → 76 ms（7.9x）。新 TC/dp4a 曲线与 Q4_0/Q4_K
   同量级（~2-3x），"TC 比 DP4A 快 19 倍" 正式证伪。
2. **Q8_0 的真实 MMQ→TC crossover ≈ M48-64**（低于 Q4_* 的 M≈96-110：Q8_0
   dp4a 计算更重、TC 的 dequant 固定开销占比相对更小），且 M≥128 后稳定进入
   ≥2.2x 安全区、M=200 升至 2.93x（收益仍随 M 放大）。
3. **auto 决策（已拍板并入，2026-09-05）**：Q8_0 并入 auto，与 Q4 共用
   `kTensorCoreMinM = 128` 保守阈值（128 ≫ Q8_0 的 48-64 crossover），规则收敛为
   `dtype∈{Q4_0,Q4_K,Q8_0} && M>=128 → TC` 单一形式，小 prefill（M<128）保持
   DP4A/MMQ、M=1 原 decode 不受影响。bench 回归佐证：并入后 Q8_0 M≥128 的
   auto 列与 forced TC 一致（M=128: 22.0 vs 22.0ms；M=200: 27.3 vs 26.1ms），
   0 mismatch；105 处 rows-vs-decode/fp64 校验全 OK。阈值以具名
   `constexpr kTensorCoreMinM`（matmul_cuda.cpp）承载，注释标明是
   RTX 4050/profile 的默认 heuristic，留给 Phase 4 按 (arch, dtype, M, N, K)
   实测替换。
4. **Phase 3.5 收益叙事修正**：Q8_0 的收益应表达为两段独立的、
   已解耦的事实——① M>1 默认路径从 batched-GEMV 回退修复为真 MMQ
   （M=128: 441→46ms，9.6×，属"修基线"，非用户可感知的推理快进）；
   ② 修复后 MMQ vs TC 的真实收益 2.2-2.9×（M≥128 区间，与 Q4 同量级）。
   二者不可混写为"Q8_0 TC 提速 19×"。
5. Phase 4 的 KernelSelector 收敛面扩大为三 dtype × M（×K、N、GPU 后续）。

---

## Phase 4 — KernelSelector 2.0 + Autotuner

**目标**：把 `attention_selector` 的模式推广到 GEMV / MMQ / MoE，并从
heuristic 升级为 autotuned。

> **状态（2026-09-05）：4-A / 4-B / 4-B.1 收敛验证已落地（本轮提交），4-C decode
> 变体实验仍暂缓。**
> 工作范围按当轮决策：matmul 族 + MoE（attention_selector 收编另行）。4-A 是行为不变的
> 纯重构（回归全绿 + bench 三 pass parity），4-B 是默认关闭的实测驱动 dispatch（opt-in），
> 4-B.1 用真实 end-to-end 式 workload 量化了 autotune 的开销与稳态收益。

### 4-A. 统一 Selector ✅（2026-09-05）

落地：

- `include/forge/kernel_selector.h`：`KernelDomain`（MatMul / MoE）、`KernelProblem`
  （domain / dtype / M / N / K / batch / sm_arch）、`KernelVariant`
  （`GEMV_Q8_1`、`GEMV_BATCH`、`GEMV_DEQUANT_TRANSB`、`GEMV_FP32_TRANSB`、
  `GEMV_FP32_CUBLAS`、`MMQ`、`MMQ_TENSORCORE`、`MATMUL_BATCH_GEMV`、
  `MATMUL_DEQUANT_CUBLAS`、`MATMUL_CUBLAS`、`MOE_EXPERT_GEMV`、`UNSUPPORTED`）、
  `select_kernel()` + 名字函数。
- `src/operators/cuda/kernel_selector.cpp`：策略集中——decode 的 Q8_1 → batched →
  dequant fallback 链、FP32 大 N（>4096）阈值、prefill 的 M>=128→TC 规则与
  `FORGE_MATMUL_PATH` / `FORGE_FORCE_TC` / `FORGE_FORCE_MMQ` override、MoE 支持集
  （Q4_0/Q4_K/Q6_K/Q8_0）、arch 探测。
- `src/operators/cuda/matmul_dispatch.h`：把 matmul_cuda.cpp 的本地查表统一为
  `internal::lookup_*`，selector 与 launcher 共用同一能力事实源。
- `matmul_cuda.cpp` / `gemma4_moe.cpp` 收编为 select+switch 薄壳。

行为不变验收：`mmq-ref / moe-dp4a-ref / phase1-bc-ref / attn-backend-parity /
attn-fp8-prefill / tiled-attn-formats / q8-roundtrip / iq2xs-compare / iq23s-compare /
spec-verify / dflash / kv-* / generic-kernels` 全绿；`forge-bench-prefill_gemm` 三 pass
（dp4a / TC / auto）parity 保持。

### 4-B. Benchmark-driven dispatch ✅（2026-09-05）

落地（默认关闭，opt-in 实测；硬阈值 → 按签名实测）：

- `KernelSignature`（domain / dtype / M / N / K / batch / sm_arch）——复用了
  `ExecutionPlan::plan_id()` / `CudaGraphKey` 已有的 shape-signature 思路。
- 进程内决策缓存（std::map + mutex）：`select_kernel_cached()` /
  `kernel_cache_record()` / `kernel_cache_stats()` / `kernel_cache_reset()`。只作用于
  可测 slice（量化 prefill、sm80+、TC-eligible dtype）；env 强制路径（mmq / dp4a / tc）
  绕过缓存、保持 benchmark 权威（`FORGE_MATMUL_PATH=auto` 不算强制）。
- `FORGE_AUTOTUNE=1` 时，matmul dispatch 在签名首次出现时对 MMQ vs TensorCore 用
  **真实张量**实测（各 3 次取 min，TC 计时含其整权 FP16 dequant 代价），胜者写入
  缓存——按签名替换 `kTensorCoreMinM = 128` 默认。
- 验收 `tests/test_kernel_selector.cpp`（ctest `kernel-selector`）通过：政策规则
  （decode / prefill 边界 / sm_arch 维度 / MoE 支持集）、env override 权威性、相同
  problem 两次选择一致、缓存命中统计可观测、measured 覆盖 rule、reset 清空。

实测证据（RTX 4050 sm_89，llama-3 K=4096 / N=128256，`FORGE_AUTOTUNE=1`）：

- M≥128：auto ≈ tc（与规则一致，无偏移）；
- M=64~112 低于旧阈值处 auto 已按实测翻 TC——Q8_0 crossover ≈64、Q4_0 ≈64-96、
  Q4_K ≈96-112，与 Phase 3/3.5 测得的 MMQ→TC crossover 吻合；
- M=8/26/32：仍守 dp4a（TC 实测更慢）；
- 默认（无 `FORGE_AUTOTUNE`）行为与 4-A 完全一致。

### 4-B.1 收敛验证（2026-09-05，本轮）

> **`KernelSignature` 定位 = 当前 slice 的“最小充分签名”，不是永久完备的 kernel 行为模型**
> （代码注释同）。它只覆盖 `(domain, dtype, M, N, K, batch, sm_arch=0 运行时探测)`，
> 且永远叠加在 launch 端 probe 语义之上（`MMQ_TENSORCORE` 仍先在 launch probe 失败再落
> dp4a）。未来需要更细 key 的 slice **应扩展签名而非复用字段**，已知缺口：transB / 内存布局
> （目前固定 transB）；量化格式与布局（Q4_0 / Q4_K / Q8_0 block 尺寸与布局互不相同，任何
> dtype 桶都不得把多个量化格式折叠成同一个「dtype」key）；激活 dtype / 权重 dtype / compute
> dtype（现在是 FP32 激活入、int8 / FP16 计算，FP16/BF16 激活需要独立 key）；workspace /
> accumulator 模式（atomic vs 非 atomic、workspace 分配策略，随 batched/paged 变体落地）。

**bench 命令**：`forge-bench-prefill_gemm 14 3 conv`（CUDA `conv` 模式）。workload = 真实
重复 prefill（3 dtype × 2 条 200-token 序列 × 每序列 10 个逐层投影形状 = 每 pass 60 个
matmul、30 个唯一签名），同一进程内测 4 场景并输出 kernel cache 统计（新增
`misses` / `tc_selected` / `dp4a_selected` 计数，接入 report）。下表为本 snapshot
（3 dtype）；conv 自 Phase 5-A 起含 Q6_K（4 dtype / 40 签名），重跑数据见 §5-A：

| 场景 | 总耗时(ms) | vs 规则 |
|---|---|---|
| 1. autotune OFF（默认规则 M>=128→TC，冷缓存） | 1387.9 | 1.00× |
| 2. autotune ON · 首次运行（冷缓存，30 签名逐个实测） | 6967.4 | 5.02× |
| 3. autotune ON · 第二次运行（热缓存，纯 hit） | 1161.2 | 0.84× |
| 4. 强制 DP4A / 强制 TC（参考，绕过缓存） | 1875.6 / 1441.4 | 1.35× / 1.04× |

结论（RTX 4050 sm_89，llama-3 K=4096 / N=128256）：

- **first-run overhead ≈ 5806 ms / 30 实测签名（≈194 ms/签名）**：首遇签名一次性付出的
  6 次候选 probe（3×MMQ + 3×TC，TC 计时含整权 FP16 dequant）。opt-in、仅 miss 时发生，
  稳态后不再出现；overhead 已量化、按设计可接受。
- **cache hit rate = 87.5%**（240 lookups / 210 hits / 30 misses；measured = TC 19 + DP4A 11）。
- **稳态收益：1161 ms vs 规则 1388 ms ≈ +19.5%**，且优于单一强制 TC（1441 ms）与单一
  强制 DP4A（1876 ms）——按签名实测确实胜过任何单一路径。break-even ≈ 26 次重复 pass
  （每 pass 稳态省 ~227 ms；对固定形状 serving 很快回本）。
- **warm-cache 回读 measured 覆盖 rule**：Q4_0 / Q4_K M=96~112、Q8_0 M=64~112 实测翻 TC
  （下方 fp64 复验全 OK，max rel ≤ 4.1e-3 ≪ 5e-2）；M=8/26/32 与 M≥128 与规则一致。
- 回归：`kernel-selector` 单测全绿（含新增 miss / TC / DP4A 计数断言）。
- 编号已按拍板重排：**Quantized Tensor-Core Coverage 定为 Phase 5**（见下），原
  「Native 低精度执行模型」顺延为 Phase 6（4-C 维持暂缓）。

### 4-C. Decode track 前置实验路线（P4-1 ~ P4-5）

> **状态（2026-09-05）：计划已固化，暂缓执行**——只立项、不进行任何代码变更，开工以本次
> 决策为准。与「Phase 2 正式收敛」节配套阅读（streaming decode = 快路径、Decode Hypothesis、
> 双 track、首批 A–F 清单），该节是本路线的依据。

Phase 4 先于统一 Selector 落地 attention decode 变体矩阵，顺序与验收：

| 步骤 | 内容 | 验收 |
|---|---|---|
| P4-1 | streaming decode 变体矩阵 bench（首批 A–F 的最小可执行版） | 得到 reg 甜点曲线 |
| P4-2 | 第二批 2 vs 3 CTA 判定实验 | 回答「第 3 个 CTA 是否值得」 |
| P4-3 | 结果落 JSON DB（arch/dtype/seq/objective 键） | DB 可查询 |
| P4-4 | `AttentionBackend` 纳入 StreamingDecode 变体名 + dispatch | 变体可路由 |
| P4-5 | Prefill TensorCore track（独立，Phase 3 事件） | — |

**P4-1 最小侵入方案（不破坏现有任何二进制）**：

1. `flash_attn_gqa_decode_kernel` 增加第三模板参数 `int MIN_CTA = 0`，配合
   `__launch_bounds__(NUM_WARPS * 32, MIN_CTA)`（0 = 无约束，现编译结果 **bit-identical**）。
2. 现有 launcher 仅将 `<HD, NW>` 改为 `<HD, NW, 0>`，默认路径零变化。
3. bench 新增 streaming 段：`MIN_CTA ∈ {2, 3, 4, 6}` × `kv_len ∈ {2048, 8192, 32768}`，
   重跑 decode 直通路径；每个实例用 `ptxas -v` 记录 reg/spill。
4. 判定只看两件事：**免费收益**（255 → ~200 且不新增 spill / sync / staging）与
   **3 CTA 的 latency hiding 是否抵过压寄存器代价**。

**P4-2 第二批：2 vs 3 CTA**——验证对象是「第 3 个 CTA 带来的 latency hiding 是否抵消压
寄存器付出的指令成本」，比较 **2 CTA vs 3 CTA**（REG ~255 → ~168），而不是 255 vs 168 regs。

**P4-3 JSON DB（先极简，再扩展）**：

```json
{ "sm89:fp32:decode:hd128": { "kv_2k": "streaming", "kv_32k": "streaming_lb3" } }
```

扩展键：GPU arch / dtype / head_dim / q_len / kv_len / batch / paged + **objective**
（decode batch=1 → MinLatency；B≥16 → MaxThroughput；serving → MinP99）。

**P4-4 变体名与 dispatch**——`AttentionBackend` 纳入 streaming 变体命名（如
`StreamingDecode` / `StreamingDecodeLB3`）；selector 只按实测 DB 路由，不以
occupancy/register 排序（收敛节铁律）。

**P4-5 独立 track**——Prefill TensorCore 归 Phase 3 事件，不与 decode tuning 在同一个
kernel 上混做。

---

## Phase 5 — Quantized Tensor-Core Coverage

> **状态（2026-09-05）：Phase 5 ✅ COMPLETE —— 5-A Q6_K / 5-B Q5_K / 5-C per-dtype
> conservative policy / 5-D 最终验收（Correctness + Dispatch + 强制路径 + Performance/
> Regression 四件套）全部完成并封存；Phase 6 Native LP 正式开启（见下）；4-C 维持暂缓。**

**目标**：不是「多支持两个量化格式」，而是验证核心假设——现有
`DP4A/MMQ ↔ FP16-dequant Tensor Core` 双后端模型能否从 Q4/Q8 扩展到 Q6_K/Q5_K，并沉淀
`dtype × M → crossover → KernelSelector` 的决策链路。

**Phase 5 完成标准（不用「支持某格式」当完成线，每种格式要求五项闭环）**：
DP4A/MMQ backend + Tensor Core backend（FP16 整矩阵 dequant）+ kernel 级正确性 +
crossover 实测 + selector 集成——5-C 起 selector 集成为 **per-dtype 保守默认规则 + 4-B
autotune 实测覆盖**（层级：静态规则 → `FORGE_AUTOTUNE=1` 实测 override → force env 最高优先）。
所有 crossover 都是 RTX 4050 sm_89 / K=4096 / N=128256 观测值，**不作普适阈值**
（架构/形状相关）。

```text
Phase 5 — coverage × 五项闭环            (crossover 均观测值, 非普适)
         MMQ   TC    正确性   crossover      selector
Q4_0      ✅   ✅     ✅      ~96-110        ✅ (保守 128)
Q4_K      ✅   ✅     ✅      ~96-110        ✅ (保守 128)
Q8_0      ✅   ✅     ✅      ~48-64         ✅ (保守 96)
Q6_K      ✅   ✅     ✅      ~80            ✅ (保守 96)
Q5_K      ✅   ✅     ✅      ~64-80         ✅ (保守 96)
```

**Phase 5 收口链路（5-A~5-C 全 ✓）**：每个 dtype 均具备
`Correctness ✓ → crossover ✓ → selector ✓ → autotune ✓`（4-B autotune 可对任意
measured 签名突破静态阈值，见 §5-C）。

### 5-A. Q6_K Tensor-Core Coverage ✅

**核心验证结论：双后端模型对 Q6_K 成立。** Q6_K 的基础设施（dp4a MMQ 实例、FP16 整矩阵
dequant + cuBLAS FP16）此前已存在，但被 dispatch 白名单挡在 TC 之外——auto 默认永远走
dp4a。本轮把 Q6_K 折入 Phase 4-B 可测 slice 并完成全链路验证：

- **放行**：`tensorcore_eligible()` + auto 规则白名单加入 Q6_K（注释同步），4-B
  cache/autotune slice 与 `select_kernel_cached` 自动覆盖 Q6_K（M>1、sm80+、cacheable）。
- **正确性**（kernel 级，fp64 golden，5 shapes）：`test_mmq_reference` TC 套件新增
  `Q6_K/T` 行——**50/50 cases passed，worst rel 2.97e-4**（Q6_K MMQ 路径此前已覆盖）。
- **crossover 测量**：bench 新增 `forge-bench-prefill_gemm 14 N q6k`（按 M 交错 dp4a/TC，
  消除 pass 间 GPU 时钟漂移——早期整列式扫描噪声 2.7x，交错后曲线干净单调）：

  | M | dp4a(ms) | TC(ms) | TC/DP4A |
  |---|---|---|---|
  | 8 | 6.28 | 26.66 | 0.24× |
  | 16 | 6.54 | 22.97 | 0.28× |
  | 32 | 10.56 | 23.73 | 0.45× |
  | 48 | 18.45 | 25.16 | 0.73× |
  | 64 | 22.42 | 23.25 | 0.96× |
  | 80 | 29.33 | 26.92 | 1.09× |
  | 96 | 34.80 | 27.14 | 1.28× |
  | 112 | 42.92 | 27.69 | 1.55× |
  | 128 | 45.85 | 28.66 | 1.60× |
  | 144 | 65.84 | 38.00 | 1.73× |
  | 160 | 68.20 | 38.25 | 1.78× |
  | 200 | 87.00 | 40.41 | 2.15× |

  (RTX 4050 sm_89、K=4096 / N=128256、best-of-5；本机单点 ±20% 抖动)
  **MMQ→TC crossover ≈ M=80**；≤64 dp4a 明确占优（0.24-0.96×），≥96 TC 稳定 1.28×+。

> **Caveat（workload / 架构相关）**：The crossover is workload- and
> architecture-dependent; M≈80 is observed on RTX 4050 sm_89 for K=4096/N=128256
> and is not treated as a universal threshold（换 RTX 4090/5090/H100 或不同 shape 会变）。
- **dispatch 规则**：与 Q4/Q8 共用保守 `M>=128`（一条规则，5-C 再做按 dtype 阈值拆分）；
  4-B autotune 在真实 workload 上把 Q6_K 96-112 段实测翻 TC（见下）。
- **回归扩展**：`conv` 模式与默认 parity sweep 纳入 Q6_K（4 dtype / 40 签名）。

**真实 workload 证据（`conv`，4 dtype / 40 签名，passes=3）**：
规则 1986.9 ms → autotune 稳态 1647.9 ms（**+20.6%**）；forced DP4A 2592.0 / forced TC
1998.0；first-run overhead 8269 ms over 40 sigs（206.7 ms/sig）；cache hit 87.5%（320
lookups/280 hits）；measured TC 26 / DP4A 14。Q6_K 实测在 M=96/112 翻 TC，fp64 12/12 全 OK
（Q6_K worst 3.51e-3 ≪ 5e-2）。**4-B.1 的 30-sig 数字保留为 Phase 4 快照**（conv 自本轮起
含 Q6_K，重跑为 40-sig 数据；Phase 5-B 后 conv 为 5 dtype / 50 sig，最新数字见 §5-B）。

### 5-B. Q5_K Tensor-Core Coverage ✅

Q5_K 是 K 系最后一个缺 TC backend 的 dtype。盘点确认：dp4a MMQ 实例 `launch_mmq_q5_k`
与 decode/GEMV 通路 ✅；但 **FP16 整矩阵 dequant 缺失**——`launch_cublas_gemm_quant_fp16`
的 guard/switch 只收 Q4_0/Q4_K/Q6_K/Q8_0，`tensorcore_eligible` 也不含 Q5_K（MoE 无
Q5_K 专家，维持 UNSUPPORTED）。Q5_K block = 176 B：`d + dmin + sc[12] + qh[32] + ql[128]`，
与 Q4_K/Q6_K 同属 K-super-block 家族，dequant 数学与既有 fp32 matrix kernel 一致。

本轮完成（工作流完全复刻 5-A，不重新设计）：
- **补齐 backend**：新增 `launch_dequant_q5_k_matrix_fp16`（与既有 fp32 Q5_K matrix
  dequant 同布局同数学，逐 256-super-block 展开 4×32 高/低 nibble + qh 高位移位），注册进
  header 与 `launch_cublas_gemm_quant_fp16` guard/switch → Q5_K 进入 TC 可测 slice。
- **正确性**（kernel 级，fp64 golden）：TC 套件加 `Q5_K/T` 行——**55/55 cases passed**，
  Q5_K/T 5/5、worst rel 4.10e-4（≪ TC tol 0.01）；q5k 扫描每 M 另有 fp64 + rows-vs-decode
  双校验全过。
- **放行**：`tensorcore_eligible`/auto 白名单加 Q5_K；**auto 阈值当时维持共享 `M>=128` 未动**
  （per-dtype 拆分留给 5-C）；Q5_K 同步进入 4-B cache/autotune slice 与 kernel-selector
  单测（512/128→TC、32→MMQ；5-C 起 Q5_K 保守阈值为 96，见 §5-C）。
- **interleaved M 扫描**（`forge-bench-prefill_gemm 14 N q5k`，标准交错法；best-of-3）：

  | M | dp4a(ms) | TC(ms) | TC/DP4A |
  |---|---|---|---|
  | 8 | 6.66 | 26.85 | 0.25× |
  | 16 | 8.63 | 26.90 | 0.32× |
  | 32 | 11.76 | 24.22 | 0.49× |
  | 48 | 18.44 | 24.08 | 0.77× |
  | 64 | 25.41 | 24.06 | 1.06× |
  | 80 | 33.16 | 26.06 | 1.27× |
  | 96 | 36.43 | 28.42 | 1.28× |
  | 112 | 45.40 | 28.26 | 1.61× |
  | 128 | 52.93 | 28.29 | 1.87× |
  | 144 | 65.69 | 30.16 | 2.18× |
  | 160 | 66.94 | 30.53 | 2.19× |
  | 200 | 98.59 | 43.40 | 2.27× |

  (RTX 4050 sm_89、K=4096/N=128256；crossover ≈ **M≈64–80**——best-of-5 独立扫描报
  80、best-of-3 报 64，落在 ±20% 抖动带内；**workload/架构相关，非普适阈值**)
- **parity 回归**（5 dtype sweep，当时规则为共享 M>=128）：Q5_K 全 M 行 auto==rule
  （<128→dp4a、≥128→TC；M=64 的 `auto mismatch` 与 Q6_K/Q4_0 同类——跨 pass 时钟抖动的
  分类伪影，规则实际执行 MMQ）。
- **conv 全链路**（5 dtype / 50 签名，passes=3，**共享-128 规则下的快照**；5-C 已按
  per-dtype 规则重跑，最新数字见 §5-C）：rule 2590.5 ms → autotune 稳态 2184.8 ms
  （**+18.6%**）；forced DP4A 3371.3 / forced TC 2606.7；first-run overhead 10687 ms over
  50 sigs（213.7 ms/sig）；cache hit 87.5%（400 lookups / 350 hits）；measured TC 33 /
  DP4A 17；**Q5_K 实测提前翻 TC 于 M=64/96/112**，fp64 15/15 全 OK（Q5_K worst 4.41e-4）。
  §5-A 的 conv 数字为其当时 4-dtype 快照。

**跨 dtype 观察**：五个格式 crossover 区间呈
`Q8_0≈48–64 < Q5_K≈64–80 ≈ Q6_K≈80 < Q4_K/Q4_0≈96–112`——随 bit-width/每元素算量大体单调：
dp4a 在更低 bit 上单点成本更便宜、TC 侧也更快，故 crossover 后移（Q5/Q6 很接近）。
这与「双后端 + M 处 crossover 是普遍规律」的假设一致。但**该有序趋势不构成 bit-width
定律**：格式性能由 dtype features（block size、scale layout、dequant/MMQ tile 结构、
memory traffic、int 指令数）共同决定，Q5_K 与 Q6_K 在 ~16 M 内非常接近即是反例——
未来 cost model 的特征应是 dtype features 而非 bits。5-C 据此把 auto 层从单一共享
`M>=128` 升级为 per-dtype 保守阈值（见 §5-C），统一成本模型留待后续（5-E 候选）。

### 5-C. Per-Dtype Conservative Dispatch Policy ✅

**核心决策（本轮）**：不把实测 crossover 原样硬编码成固定阈值（诱惑版本：
`Q8_0>=64 / Q5_K>=80 / Q6_K>=80 / Q4_*>=128`——否决）。原因正是 5-A/5-B 已证明的：crossover
是**区间**而非整数（best-of-3/5 抖动、时钟/温度漂移、M=96/112 边缘噪声），且真正最优点依赖
N/K/arch/workload。因此 5-C 的目标是**把实测结果转成保守的 per-dtype heuristic，并明确
fallback 到 4-B autotune**。

**默认策略（landing state）**——static heuristic → 保守 threshold →（`FORGE_AUTOTUNE=1`）实测
override：

```text
                Selector
                   │
        ┌──────────┴──────────┐
   default mode          autotune mode
   (无 benchmark 开销)    (FORGE_AUTOTUNE=1)
        │                     │
 conservative              measure
 per-dtype threshold      candidates
        │                     │
        └──────────┬──────────┘
                   ↓
               best kernel
```

每个阈值都**刻意留安全 margin**（kick in 于 TC 领地深处、吸收带边缘 +20-40% 抖动），不是
crossover 点本身：

```text
dtype    实测 crossover 带        保守阈值(M>=)   margin    说明
Q8_0     ~48–64                   96             +32..48   bits 最少 / dp4a 最省
Q5_K     ~64–80                   96             +16..32
Q6_K     ~80                      96             +16
Q4_K     ~96–110                  128            +18..32   bits 最多 / dp4a 最贵
Q4_0     ~96–110                  128            +18..32
```

- **实现形态**（拒绝 if-else 蔓延）：`kernel_selector.cpp` 的 `kTcMinM[]` 表
  `{dtype → min_m}` + `tc_min_m(dt)`（未列 dtype 回落 `kTcMinMDefault=128`，最保守级）。
  共享 `kTensorCoreMinM=128` 常量退役。bench 持镜像 `tc_min_m()`（注释强制同步）。
- **分层优先级（自上而下最高优先，5-C 未动）**：
  `FORGE_MATMUL_PATH=mmq|dp4a|tc` / `FORGE_FORCE_TC|MMQ` 最高 → `FORGE_AUTOTUNE=1`
  measured 覆盖任意签名（含**向下**突破：Q8_0/Q5_K/Q6_K 的 64 实测翻 TC 会被记录）→
  静态 per-dtype threshold 兜底。
- **单测**：`test_kernel_selector.cpp` 每个 dtype 在阈值两侧建期望（Q8_0/Q5_K/Q6_K：
  96→TC、64→MMQ；Q4_0/Q4_K：128→TC、96→MMQ；sm75 恒 MMQ；cache 测试含「measured 记录
  把 Q8_0@64 向下翻 TC」）——全绿。
- **kernel-selector 注释同步**：header 与 `kernel_selector.cpp` 顶部写明三层策略与
  crossover 带/margin 依据。

**回归证据（5-C 规则下重跑）**：

- **conv 全链路**（5 dtype / 50 签名，passes=3，多轮复跑一致；计时取 min(best-of-N)/
  median/max 口径，单次 latency 不作证据——见 §5-D）：rule **2367.8/2491.4/2528.8 ms**
  （共享-128 下同构跑为 2590.5，规则本身快 ~6-9%——Q8_0/Q5_K/Q6_K 的 96/112 行提前进
  TC）→ autotune 稳态 **2139.6/2163.4/2218.2 ms（+10.7%，另一轮复跑 +13.7%）**；forced
  DP4A 3358.4 / forced TC 2597.4（均绕过缓存）；first-run overhead ~10.5-12.6 s over 50
  sigs；cache hit 87.5%（400/350）；measured **TC 32 / DP4A 18**。回读决策表确认
  Q8_0/Q5_K/Q6_K 在 M=96/112 与所有 ≥threshold 行 `static=TC、effective=TC`（规则实际
  派发 TC）；残留 measured 提前翻 TC 只在**阈值之下**：Q4_0/Q4_K@96/112、Q8_0@64、
  Q5_K@64（Q6_K@64 是否翻 TC 因运行而异——交叉带本就不是确定整数）——恰是 autotune 应
  接管的地带；measured 派发 fp64 re-verify 15/15 OK。
- **parity sweep**（5 dtype，best-of-3 + min/median/max 每 cell）：fp64 全 0 MISMATCH；
  M=96 边界行 auto≈TC 干净；残余 `auto mismatch` 伪影只出现在热漂移样本（Q4_0@64、
  Q6_K@128，同 cell max/min 达 ~2.5x）——bench 按「接近哪个 forced pass」分类，短核
  抖动下不可靠；conv 回读（直接读缓存签名）才是 rule 权威判定。force env 路径全程不受
  影响。
- **实测边界观察**：真实 workload 里低阈值 dtype 从 M=64 起 TC 可能实测胜出（运行相关）
  ——默认规则保守留 96/128，`FORGE_AUTOTUNE=1` 用户把 64–96 带自动回收；默认用户零
  benchmark 开销拿到良好默认。这正是 4-B 分层设计要表达的分工。

**记录口径（防误读）**：每个 dtype 的 crossover 一律记作**带/区间**，伴随一句固定备注：
`Observed crossover band: M≈X–Y; conservative static threshold is M≥Z (deliberate margin);
autotune is responsible for fine-grained selection inside the crossover region.`
（Q5_K 的 best-of-5≈80 与 best-of-3≈64 不是矛盾，是同一带 ±20% 抖动内的两次抽样。）

**后续（5-E 候选，不在本轮）**：把分层再推进一步成经验 cost model——估
`T_MMQ(M,N,K,dtype)` / `T_TC(M,N,K,dtype)` 并比较，autotune 记录即校准数据；届时
`kTcMinM` 表退化为模型初值。特征用 dtype features，不用 bits。

### 5-D. Final Acceptance ✅（已完成）

5-C landing state 下执行四件套验收（不新增任何功能），计时一律 **best-of-N(min)+median+
max**（单次 latency 不作证据）：

**① Correctness**（5 dtype × M 网格 {1, 32, 64, 96, 112, 128, 200, 512}，DP4A/TC/AUTO/
AUTOTUNE）——全部通过：
- `forge-mmq-ref-test` **55/55**（kernel 级 fp64 golden，MMQ/TC）。
- 默认 parity sweep：每 M 做 rows-vs-decode、每 dtype×pass 在 M=32/128 做 fp64 anchor
  （5-C 起把 M=128 锚点扩进来了）、M=1 decode 做 fp64 —— **MISMATCH 0**。
- large-M plateau（M=128-1024）：**MISMATCH 0**；M=512 行 auto==TC（输出逐位一致，
  max rel ~3.5e-3 ≪ 5e-2）。
- conv measured 派发 fp64 re-verify **15/15 OK**（worst ~4.4e-4）。
- backend 一致性结论：≥threshold 行 AUTO==TC 逐位一致；MMQ 与 TC 各行均在 tol 内，
  证明「数学与 M 无关、规则派发正确」——`max_rel_error` 跨 backend 一致。

**② Dispatch**（5-D 核心，三层分层都在 `forge-kernel-selector-test` 固定）：
- 静态规则：Q8_0/Q5_K/Q6_K `M=64→MMQ、M=96→TC`；Q4_K/Q4_0 `M=96→MMQ、M=128→TC`；
  M=200/512 全部 TC；sm75 恒 MMQ（本轮单测补 M=200/512 行）。
- `FORGE_AUTOTUNE=1` 突破静态阈值：conv 决策表（static/effective/source 分列）实测翻
  TC 6 行、全部在阈值之下——Q4_0/Q4_K@96/112、Q8_0@64、Q5_K@64（source=autotune-cache，
  tag=`** overrides static`）。Q6_K@64 是否翻 TC 因运行而异 ⇒ 交叉带逐次漂移，进一步
  证明不能把单次实测硬编码成固定阈值。

**③ 强制路径（highest-priority invariant，最终证明）**：
- conv 的 forced 行在 warm cache 存有 measured 记录时仍**绕过缓存**执行：forced DP4A
  min/med/max = 3358.4/3364.3/3380.9 ms、forced TC = 2597.4/2604.8/2640.0 ms，与 cache
  内容无关、行为与 Phase 3 一致。优先级
  `FORCE env → FORGE_AUTOTUNE measured record → per-dtype conservative static policy`
  由构造保证（force 分支在缓存咨询前短路）。

**④ Performance / Regression**（conv，5 dtype/50 sig、passes=3，每 pass 3 次整 workload 采样）：
```
1 rule (autotune OFF)   min 2367.8  median 2491.4  max 2528.8 ms   1.00x   (per-dtype static policy)
2 autotune ON, first run    12599.1 ms (one cold pass: 每签名实测一次)     5.32x
3 autotune ON, steady   min 2139.6  median 2163.4  max 2218.2 ms   0.90x   (+10.7% vs rule)
4 forced DP4A           min 3358.4  median 3364.3  max 3380.9 ms   1.42x   (绕过缓存)
5 forced TC             min 2597.4  median 2604.8  max 2640.0 ms   1.10x   (绕过缓存)
cache: 400 lookups / 350 hits (87.5%) / 50 measured;  TC selected 32 / DP4A 18
```
- 与 5-C 复跑互证（rule 2436.4 / steady 2142.6 = +13.7%）：pass 漂移 ±3-7%，故取
  min+median 口径。median 尺度的结论稳定：**per-dtype static rule 相对共享-128 快 ~6-9%
  （不开 autotune 的普通用户也受益）；autotune 稳态再 +11-14%**。
- 默认 parity sweep：fp64 0 MISMATCH；`auto mismatch` 伪影仅 2 条（Q4_0@64、Q6_K@128，
  后段热漂移，同 cell max/min ~2.5x）——规则派发本身由 conv 回读逐行确认。
- 回归：`forge-kernel-selector-test` / `forge-mmq-ref-test` / 默认 parity / large plateau /
  conv 全绿；跨 dtype 无规则倒退。

**验收矩阵（达成）**：

```text
                 DP4A/MMQ      TensorCore      static   autotune
Q4_0                 ✓              ✓            128      96/112 翻 TC
Q4_K                 ✓              ✓            128      96/112 翻 TC
Q5_K                 ✓              ✓             96       64 起翻 TC(视运行)
Q6_K                 ✓              ✓             96       64/96 翻 TC(视运行)
Q8_0                 ✓              ✓             96       64 起翻 TC
                     Correctness ✓ / crossover ✓ / selector ✓ / autotune ✓
```

```text
Phase 5 — Quantized Tensor-Core Coverage       Status: COMPLETE (2026-09-05)

Covered formats: Q4_0 / Q4_K / Q5_K / Q6_K / Q8_0
Each format provides:
  ✓ DP4A/MMQ backend            ✓ TensorCore backend
  ✓ correctness validation      ✓ crossover characterization
  ✓ selector integration        ✓ autotune compatibility

Dispatch layering (highest first): FORCE env
                                   → FORGE_AUTOTUNE measured record
                                   → per-dtype conservative static policy
```

**冻结声明（永久 freeze，2026-09-05）**：Phase 5 至此**永久冻结**——验收矩阵、代码、
bench 口径（min/median/max、`static/effective/source` 决策表）全部封存为 Phase 5 的最终
形态，**不再为覆盖 Q2/Q3/IQ 系列扩大范围**。技术路线
`Q4/Q5/Q6/Q8 → DP4A/MMQ ↕ TensorCore → workload-aware selector → autotune` 已闭环；
后续演进全部并入 Phase 6：per-dtype 阈值 → 5-E cost model 候选改为在 Phase 6 dtype
policy 中发生；bench 的 min/median/max 分布口径与可解释决策表作为 Forge 性能数据与
dispatch 的**默认记录格式**长期保留。

---

## Phase 6 — Native 低精度执行模型

> **状态（2026-09-05）：Phase 5 永久冻结；Phase 6 按新路线推进。已完成：6-A 立项 +
> 「FP32 提升点」源码审计（6-A-0）+ **6-A-1 dtype contract hardening** + **6-A-2 native
> FP16 RMSNorm**（首个真实 `__half` kernel：FP16 存储/算术 + FP32 reduction；正确性测试
> `rms-norm-fp16` 已过，hidden=4096/8192 × 多 batch 的 max_abs_err ≈ 5e-4，即仅 FP16 输出
> 存储舍入；微基准 hidden=8192 时 FP16 0.085 ms / 198 GB/s vs FP32 0.228 ms / 147 GB/s，约
> 2.7x）。**6-A-2 已正式封版**——验证了 Phase 6-A 三条核心原则（activation / norm_compute
> 解耦、operator boundary 显式暴露 host/device/dtype 错误、基准须分 launch-bound /
> bandwidth-bound 两 regime；详见 6-A-2「封版原则」）。**6-A-3 native FP16 SiLU 已交付**
> （max_abs_err ≈ 9.7e-4）。**6-A-4 native FP16 activation family 已交付**——主链激活
> GELU / GELU-tanh（unary）+ silu_multiply / gelu_multiply（gated：LLaMA/Qwen/DeepSeek 用
> SiLU-gated、Gemma4 用 GeGLU）全部原生 FP16（max_abs_err ~1e-3 unary / ~3.9e-3 gated，仅
> FP16 存储舍入）；gated 变体复用 SILU / GELU 的契约闸门，无需新增枚举。
> 推进顺序已按 7 步重编号：RMSNorm ✅ → SiLU ✅ → 激活函数 ✅ → matmul ✅（6-A-5，本轮）→
> graph/binding → FP16 attention → 端到端验证。**
>
> **硬原则：Phase 6 不再沿用「每种 dtype 加一个 if」。** 一切 native 执行统一由
> `ComputeDTypePolicy` 描述；没有 policy 就没有新路径。**FP16/BF16 到达尚无 native
> kernel 的 operator = 显式报错，永不静默按 float 读（dtype_contract.h）。**

**依赖**：Phase 3 / Phase 5（FP16 与低精度路径在量化 GEMM backend 与 selector 之上演进）。

### 6-0. ComputeDTypePolicy（贯穿 6-A..6-D 的统一抽象）

`plan.md` 第 13 节的建议方向正确，Phase 6 采纳为总纲：

```cpp
struct ComputeDTypePolicy {
    DataType weight_dtype;
    DataType activation_dtype;
    DataType accumulator_dtype;
    DataType kv_dtype;
};
```

现状 `DataType` 单一枚举决定全部（`include/forge/types.h:7-29`），且 `BF16 = 15`
注释明确写着 `converted to FP32 at load time`——**FP16/BF16 目前都不是执行 dtype**。
Policy 落位后，selector（Phase 5 的 `static/effective/source` 解释面）自然延伸到
`weight_dtype/activation_dtype` 组合，例如：

```text
Q4_K + FP16 compute          FP8 W8A8
    W  = Q4_K                    W  = FP8
    A  = FP16                    A  = FP8
    Acc = FP32                   Acc = FP32
    KV  = FP8                    KV  = FP8
```

### 6-A. Native FP16 Execution（当前阶段）

目标不是「支持 FP16 tensor」——已有 FP16 TC GEMM；目标是
**让 FP16 从 weight/input dtype 变成贯穿 execution graph 的 native compute dtype**：

```text
FP16 weights + FP16 activation → FP16 Tensor Core → FP32 accumulation
        （而非）FP16 weight → convert → FP32 activation → FP32 path
```

**第一步 = 源码审计（本轮已执行，只读，零代码改动）**：结果见下。

#### 6-A-0 FP32 提升点审计记录（2026-09-05，只读审计）

**总体判断**：当前 CUDA 链是一个 **全 FP32 域** 架构——activation、norm 输入/输出、attention
Q/K/V、所有非量化权重、GEMM 输入输出，全部以 `const float*` / `float*` 进入 kernel。FP16 只
出现在三处"局部低精度"：量化权重的 fp16 scale、KV cache 的低精度存储、TensorCore B 侧的
fp16 反量化缓冲。**不存在 FP16 native 端到端路径**；且存在两个"伪 FP16 helper"——名字带
`_fp16`，实际把 `void*` reinterpret 成 `float*` 再调 FP32 kernel（传真 FP16 数据会按 4B/元素
读，静默错位）。好消息：`Tensor::data()` 是 `void*`、nbytes 走 `dtype_size` 表、CUDA Graph
捕获/重放路径**不含任何 float/dtype 假设**——承载层已 dtype 无关，FP32 假设集中在
**算子接口 / 图构建 / 绑定入口** 三处，改起来有清晰收敛点。

| # | 路径归属 | FP32 提升机制（一句话） | 关键锚点 | 分类 |
|---|---|---|---|---|
| 1 | graph（中间张量） | 图构建所有中间张量硬编码 `DataType::FP32` | `llama_graph_builder.cpp:101,111,125,244,316`（deepseek/qwen35 同构） | true-blocker |
| 2 | graph（自定义 op） | lambda 内直接 `static_cast<const float*>(inputs[i]->data())` | `llama_graph_builder.cpp:249-251,319-321` | true-blocker |
| 3 | binding（入口） | Python 输入张量一律建 `DataType::FP32` | `src/bindings/common.h:100,329` | true-blocker |
| 4 | tensor / storage | `Tensor::data()` 为 `void*`，`dtype_` 默认 FP32，字节数走 traits 表 | `include/forge/tensor.h:50-51,107`；`quant_traits.h:136-137` | 中性（无需改，可承载 FP16） |
| 5 | CUDA Graph | 捕获/重放纯指针+字节，**无 float/dtype 假设**（假设在被捕获的 kernel 里） | `src/inference/graph/graph_cuda_runner.cpp`（`float/dtype/DataType` 零命中）、`include/forge/cuda_graph_runner.h` | 中性（无需改） |
| 6 | matmul（激活入口） | `cuda_matmul_transB` 把激活一律 cast 成 `const float*` | `matmul_cuda.cpp:321-322,334-335` | true-blocker |
| 7 | matmul（出口） | 输出固定 `DataType::FP32`，无 FP16 出口；dual interleave 按 float 行拷 | `matmul.cpp:600`；`matmul_cuda.cpp:349-359` | true-blocker |
| 8 | matmul（dispatch） | 非量化权重统一落 `MATMUL_CUBLAS` / `GEMV_FP32*`，FP16=1 槽位为 nullptr → 静默按 4B 读 | `kernel_selector.cpp:207-211,261-262`；`cuda_gemv_instances.cu:64,88`；`cuda_mmq.cu:1854` | true-blocker（含误读风险） |
| 9 | kernel 签名 | `GemvFn/MmqFn/GemvBatchFn` 激活签名写死 `const float*` | `cuda_kernels.h:106-107,115`；`cuda_mmq.cu:1850` | true-blocker |
| 10 | 伪 FP16 helper | `launch_rms_norm_fp16` / `launch_silu_fp16` 实为 `float*` 别名再调 FP32 kernel | `cuda_rms_norm.cu:62-66`、`cuda_rms_norm.h:12-13`；`cuda_activation.cu:22-24`、`cuda_activation.h:10` | true-blocker（命名承诺未兑现） |
| 11 | norm / activation 算子 | 无条件分配 FP32 输出并强制 `const float*` 输入 | `norm.cpp:29,34-42`；`activation.cpp:16-23,37-43` | true-blocker |
| 12 | cuBLAS | `launch_cublas_sgemm` 全 FP32（`CUDA_R_32F`×2）；`launch_cublas_gemm_fp16_fp32` 的 A 仍是 `const float*`，computeType 固定 `CUBLAS_COMPUTE_32F` | `cuda_quant.cu:1133,1143-1145,1155-1156,1178-1180` | 记录 + 提升点（native 需 `CUDA_R_16F` / `CUBLAS_COMPUTE_16F` 选项） |
| 13 | TC 量化路径（A 侧） | 量化 B 反量化到 FP16 scratch 走 TC，但 FP32 激活需额外 `launch_quantize_f16_matrix` 转一次 | `cuda_quant.cu:1162-1179,1200-1249` | dequant-by-design + A 侧提升点 |
| 14 | Q4 fallback | `dequant_b_on_device` 反量化到 **FP32** device scratch 再 sgemm（无 fp16 版 dequant） | `matmul_cuda.cpp:70-77,263-264` | dequant-by-design |
| 15 | 量化 GEMV / MMQ / MoE | FP32 激活 → Q8_1 量化 → dp4a int32 块累加 → fp16 scale 上浮 float | `cuda_gemv.cu:88-108,190`；`cuda_mmq.cu:163-326,935`；`cuda_moe.cu:170-180` | dequant-by-design（FP16 激活需先升 FP32→int8） |
| 16 | KV（权威存储） | KV 存储以 FP32 tensor 为准（`fp32_data()/fp32_row()`），低精度 KV 由 FP32 源量化而来（含 F16↔FP32 SIMD 转换） | `kv_cache.cpp:87-144,548,565,664-666,729-767,1471-1476` | 提升点（native 应让低精度为权威） |
| 17 | KV（设备侧合法路径） | F16 / Q8_0 / FP8 KV 在打分处 on-the-fly upcast 成 float 点积累加 | `cuda_paged_kv.cu:444-500(F16),545-599(Q8_0),750-910(FP8)` | 合法（记录） |
| 18 | KV（非 fused shadow） | 非 fused 解码遇低精度 KV 整层 `dequantize_layer` 成 FP32 shadow 再喂 FP32 attention | `generic_engine.cpp:238-241`；`phimoe_engine.cpp:589-593` | 提升点 |
| 19 | tiled prefill | `DqF16/DqQ8_0/DqFp8` 的 `deq()` 一律返回 `float` 参与计算 | `cuda_tiled_prefill.cu:71-147` | 合法（记录） |
| 20 | dtype 枚举 / 转换工具 | `BF16 = 15 // converted to FP32 at load time`；`convert.py` 把 GGML **F16 权重在入口即预转 FP32** | `include/forge/types.h:25`；`tools/convert.py:91-94` | 提升点（FP16 权重在进入 runtime 前就被消灭） |

统计：20 条，其中 **true-blocker 9 条**；其余为 dequant-by-design（Q4/Q8 既定设计，非 Phase 6
引入）、合法低精度路径、结构性提升点与两处无需改动的中性层。

**动手顺序（下一轮开工依据，先 policy 后 kernel）**：
1. 落地 `ComputeDTypePolicy`（`weight/activation/accumulator/kv` 四字段）——否则 6-A 会退化成又一组 per-dtype if；
2. 修 #10 两个伪 FP16 helper（补真实 `__half` 实现，或改名消除误导）——唯一"命名即债"项；
3. 收敛 activation 入口（#6 #7 #9）：`cuda_matmul_transB` + `GemvFn/MmqFn` 签名增 FP16 维度；
4. #8 dispatch 空槽：FP16 权重要么补真 kernel，要么由 selector 显式 reject，**不能**静默按 FP32 读；
5. #1 #2 #3 图与绑定去 FP32 硬编码（中间张量 dtype 由 policy 决定）；
6. #12 #13 cuBLAS 入口 `CUDA_R_16F` + `CUBLAS_COMPUTE_16F` 选项，A 已是 FP16 时省掉转置/转码；
7. #16 #18 #20 让低精度成为权威（KV 直接低精度、非 fused 去 shadow 物化、`convert.py` 保留 FP16 权重）。

**边界声明**：MMQ/DP4A 的 int32 块内累加 + fp16 scale 上浮是 Q4/Q8 的既定 dequant 设计，
**不属于** Phase 6 的 FP32 强制；6-A 只改 activation / 输出 tensor / 非量化权重 / 伪 FP16 helper
四处。

<!-- @@P6A-AUDIT@@ -->

#### 6-A-1 dtype contract hardening（2026-09-05，本轮交付 ✅）

> 按立项口径，6-A-1 一次交付三件事：建立 `ComputeDTypePolicy`；修复 #10 两个伪 FP16
> helper（审计确认零调用点 → **删除**，不补 fake 实现也不留改名占位）；给所有 FP16 未
> 覆盖的 CUDA operator 边界加**显式 reject**（throw / 返回 Unsupported，绝不静默按 float 读）。
> 对应原动手顺序的 1 + 2 + 4-reject 半边。真实 native kernel 一律不在本轮，按重编号
> 顺序进 6-A-2..6-A-7。

**新增文件**

| 文件 | 作用 |
|---|---|
| `include/forge/compute_dtype_policy.h` | `ComputeDTypePolicy{weight/activation/accumulator/norm_compute/kv}`，默认全 FP32；`fp32_policy()` 是当前唯一完全受支持配置。scalar/norm 维度中 **`norm_compute` 已在 6-A-2 引入**（RMSNorm 归约 dtype，native FP16 RMSNorm 强制 FP32 归约，与 `activation=FP16` 解耦）；其余 scalar / 残差归约等仍待真正需要的 op 出现再扩，不做投机抽象 |
| `include/forge/dtype_contract.h` | 契约：FP16/BF16 到达 operator 边界 = 显式 `throw`（消息含 op / dtype / Phase 6-A 指引），不静默 reinterpret。`op_supports_lowp()` 是 6-A-2..6-A-8 **唯一扩展点** |

**修复 #10（伪 FP16 helper → 删除）**：`launch_rms_norm_fp16`（`cuda_rms_norm.{cu,h}`）
与 `launch_silu_fp16`（`cuda_activation.{cu,h}`）全部移除，`cuda_kernels.h` 同步（真实
`launch_silu` 声明保留）。命名即债，占位不配存活。

**显式 reject 落点**（全部经 `require_dtype_supported`，位于各自 `USE_CUDA` 分支）：
`ops::rms_norm`（x 与 weight）、`ops::silu`、`core/op_kernels.cpp` 的 add/mul/silu/rms_norm
四个 dst-injection kernel、`cuda_matmul` / `cuda_matmul_transB` / `cuda_matmul_transB_dual`
（a/b/out 三元组）——其中 matmul 的 b 侧堵住 #8 的 FP16 权重静默落 CUBLAS 按 4B 读。

**测试**：`tests/test_dtype_contract.cpp`（host-only、不启 kernel，CTest 名 `dtype-contract`）。
断言：policy 默认值与非默认可表示性；FP32 + Q4_0/Q4_K/Q5_K/Q6_K/Q8_0 是合法执行 dtype；
FP16/BF16 对 audited op 集全部抛错且消息含 op 名与 "has no native kernel"；`op_supports_lowp`
当前为空集并带一致性不变式（一旦白名单某 op，契约必须接受 FP16/BF16）。

**验收对照**（Level 1 满足即"永远不会再静默出错误数字"）：

| Level | 定义 | 状态 |
|---|---|---|
| L1 | 伪 FP16 API 从源码消失，命名与行为一致 | ✅ |
| L2 | FP32-only CUDA operator 对 FP16 显式 reject | ✅（test 锁定覆盖面） |
| L3 | native `__half` RMSNorm（max_abs_err ≈ 5e-4，仅 FP16 存储舍入） | ✅（6-A-2 交付） |
| L4 | native `__half` SiLU（max_abs_err ≈ 9.7e-4，仅 FP16 存储舍入） | ✅（6-A-3 交付） |
| L5 | native FP16 matmul（FP16 存储 + FP32 累积，M=1..512 直连 cuBLAS） | ✅（6-A-5 交付） |

**重编号路线**（按 6-A-1 之后用户确认的 7 步推进；6-A-2 为首个 native kernel，现已交付）：
- 6-A-2 native FP16 RMSNorm ✅（本轮）—— FP16 存储/算术 + FP32 reduction，`norm_compute=FP32`；
- 6-A-3 native FP16 SiLU ✅（本轮）—— 仿 6-A-2 落地 `__half` kernel（FP16 storage + FP32 sigmoid）；
- 6-A-4 native FP16 activation family ✅（本轮）—— GELU / GELU-tanh（unary）+ silu_multiply / gelu_multiply（gated，主链真实入口）；与 SiLU 同构；
- 6-A-5 native FP16 matmul ✅（本轮）—— FP16 权重直连 cuBLAS FP16 GEMM（computeType=`CUBLAS_COMPUTE_32F`，FP16 存储 + FP32 累积），`ops::matmul_transB` 按 dtype 路由 FP16 输出；Phase 5 量化 / FP32 路径零改动；
- 6-A-6 graph / binding 集成 ⏳（本轮部分交付）—— dst 注入去 FP32 硬编码 + planner 保留 FP16 dtype
  + 真实 FP16 图链逐位一致 + dtype trace；CUDA Graph capture/replay 需先补流贯通；
- 6-A-7 FP16 attention —— FlashAttention / scaled_dot_product 的数值路径 FP16 化（仅改数值 dtype，不改 attention 选择架构）；
- 6-A-8 端到端 FP16 验证 —— 全链路跑通 + 精度/带宽 regress。

**Phase 6-A 进度（截至 6-A-5 封版）**

| 步骤 | 内容 | 状态 |
|---|---|---|
| 6-A-0 | FP32 提升点审计（只读） | ✅（2026-09-05） |
| 6-A-1 | dtype contract hardening | ✅（2026-09-05） |
| 6-A-2 | Native FP16 RMSNorm | ✅（2026-09-05） |
| 6-A-3 | Native FP16 SiLU | ✅（2026-09-05） |
| 6-A-4 | Native FP16 activation family | ✅（2026-09-05） |
| 6-A-5 | Native FP16 Matmul | ✅（2026-09-06，封版） |
| 6-A-6 | Graph / Binding 集成 | ⏳ 部分（6-A-6.1/2/3/5/6 ✅；6-A-6.4 受限于流贯通） |
| 6-A-7 | Native FP16 Attention | ⏸ |
| 6-A-8 | End-to-end FP16 验证 | ⏸ |

**边界（6-A 边界重申，依重编号路线更新）**：对标 6-A-0 中性层——QK/RoPE 维持 FP32（既定设计）
直到 6-A-7 明确覆盖（6-A-4 仅覆盖 activation family：GELU / GELU-tanh / gated SiLU·GeGLU，
不动 attention）；Attention 的数值路径 FP16 化进入 6-A-7（仅改数值 dtype，不改 attention
选择架构与 scaled_dot_product 的 kernel 选择逻辑）。

<!-- @@P6A-1-DELIVERED@@ -->

<!-- @@P6A-2-DELIVERED@@ -->

#### 6-A-2 native FP16 RMSNorm（2026-09-05，本轮交付 ✅）

> 按重编号路线第 1 步。首个真实 native `__half` kernel。计算模型严格遵循立项口径：
> **FP16 存储/算术 + FP32 reduction/accumulator**（不是"全 half"），数值稳定性不牺牲。

**新增 / 修改**

| 文件 | 改动 |
|---|---|
| `include/forge/dtype_contract.h` | `op_supports_lowp()` 扩展点首用：RMS_NORM 进入白名单（6-A-2 唯一，其余仍 false）。BF16 仍无 kernel，过契约闸门后由 CUDA 路由显式报错 |
| `include/forge/compute_dtype_policy.h` | 新增 `norm_compute`（默认 FP32）。native FP16 RMSNorm 强制 FP32 归约，与 `activation=FP16` 解耦——`accumulator` / `norm_compute` 独立于 `activation` |
| `src/operators/cuda/cuda_rms_norm.{cu,h}` | 新增 `rms_norm_kernel_fp16` + `launch_rms_norm_fp16`（加权）与 `rms_norm_unweighted_kernel_fp16` + `launch_rms_norm_unweighted_fp16`（无权重）。FP16 in/out，reduction 在 float |
| `include/forge/cuda_kernels.h` | 声明 `launch_rms_norm_fp16` / `launch_rms_norm_unweighted_fp16`（含 `<cuda_fp16.h>`） |
| `src/operators/cpu/norm.cpp` | `ops::rms_norm` 按 dtype 路由：FP16(+FP16 weight)→native fp16；FP32→原 fp32 kernel；无权重走 `launch_rms_norm_unweighted*`（**修复**原代码把**主机** `ones` 向量当 device 指针传入 CUDA kernel 的潜在 bug）；BF16 / dtype 混合 → 显式 throw。输出 dtype 跟随输入 |
| `tests/test_dtype_contract.cpp` | 拆分「已拒绝 / 已接受」两类断言；RMS_NORM 现进白名单（`lowp_accepted`）；补 `norm_compute` 默认 FP32 断言 |
| `tests/test_rms_norm_fp16.cpp`（新） | CTest `rms-norm-fp16`：FP32 reference 对照（hidden=4096/8192 × rows∈{1,32,128,512} + 无权重）+ 微基准（FP16 vs FP32 延迟/带宽），无 GPU 优雅 SKIP |

**正确性（FP32 reference 对照：输入/权重先编码为 FP16 再喂 kernel，reference 用同一解码值，故残差仅来自 FP16 输出存储舍入）**

| 配置 | max_abs_err | max_rel_err |
|---|---|---|
| hidden=4096, rows=1..512 | ~4.9e-4 | ~4.9e-4 |
| hidden=8192, rows=1..512 | ~4.9e-4 | ~4.9e-4 |
| hidden=4096, 无权重 | ~4.9e-4 | — |

误差量级 = FP16 输出存储舍入（满足 1e-3 量级约束）；结构类错误（错轴 / 错权重 / 漏归约）会给出 O(1) 垃圾并被阈值（0.02 / 0.05）拦截。无权重路径初版曾因主机 `ones` 缓冲误当 device 指针读出乱值（max_abs_err=3.6），修复后归位。

**微基准（kernel-only，hidden=8192 / rows=512 / iters=50）**

| 路径 | 延迟 | 带宽 |
|---|---|---|
| FP16 | 0.085 ms/iter | 198 GB/s |
| FP32 | 0.228 ms/iter | 147 GB/s |

结论：memory-bound 的大 hidden 下 native FP16 约 **2.7x** 延迟收益（流量减半）；小 hidden 两者均 launch-bound，差异可忽略。数值路径与 FP32 一致（FP32 reduction）。

**验收**：L3 ✅；CTest `dtype-contract` + `rms-norm-fp16` 均过。

**封版原则（Phase 6-A 核心，本轮 6-A-2 验证成立，后续 6-A-3..6-A-8 沿用）**

1. **`activation=FP16` 与 `norm_compute=FP32` 解耦是正确的。** 输入/输出存储真正进入
   FP16，但 RMSNorm 的 reduction 保持 FP32，兼顾数值稳定性。这意味着其他算子可以采用
   **不同的 compute policy** 而不被一个全局 `activation dtype` 绑死——`storage precision
   ≠ reduction / transcendental compute precision` 是贯穿 6-A 的统一纪律。
2. **无权重路径的 host/device 误用 bug 必须保留在此记录中（且原就存在于 FP32 路径）。**
   这次 dtype migration 不只是"新增 FP16"，同时暴露了以前被掩盖的 runtime contract 问题。
   以后 6-A/6-B 坚持：**任何 host/device、dtype、layout 错误都必须在 operator boundary 处
   显式暴露**，绝不让新 dtype 路径把这些问题重新隐藏。
3. **`8192 / 512` 的 2.7x 很有意义，但不是 FP16 的"通用收益"。** 它准确解释是
   `memory traffic↓ + FP16 storage + better memory efficiency`，而非"FP16 RMSNorm 天然
   2.7x 快"。基准必须分两个 regime：**small hidden → launch-bound**；**large hidden →
   bandwidth / memory efficiency**。后续 Phase 6 的 benchmark 一律保留这种 workload 分类。
4. **每个 native dtype 都应保留「native lowp ⇄ FP32 reference」的链式验证，而非仅单 kernel
   correctness。** 6-A-5 的真实链（`RMSNorm → Matmul → SiLU`）对 FP32 参考对照时，直接暴露了
   AVX2 `silu_mul_f32_vec` 的 Cephes `exp` 缺 clamp bug（`|gate|>~61` 出错误 SiLU）——单 kernel
   测试不会触发。后续 6-B/BF16、6-C/FP8、6-A-7 attention 一律沿用这种 chain validation，并配合
   dtype trace（见 6-A-6）核对中间张量不出现 `FP16 → FP32 → FP16` 物化。

<!-- @@P6A-3-DELIVERED@@ -->

#### 6-A-3 native FP16 SiLU（2026-09-05，本轮交付 ✅）

> 按重编号路线第 2 步。第二个 native `__half` operator，沿用 6-A-2 的"低精度 storage +
> 必要的 FP32 compute"原则，不重新设计整体架构。计算模型：
> **FP16 in → FP32 widen → sigmoid(x)=1/(1+exp(-x)) → x·sigmoid → FP16 out**。
> SiLU 是逐元素算子，无归约，故无需新增 policy 维度（compute 精度在 kernel 内部，与
> RMSNorm 的 `norm_compute` 解耦逻辑一致）。

**新增 / 修改**

| 文件 | 改动 |
|---|---|
| `include/forge/dtype_contract.h` | `op_supports_lowp()` 扩展点：SILU 进入白名单（现为 RMS_NORM + SILU）；`OpType::SILU` 对 FP16/BF16 过契约闸门，BF16 由 CUDA 路由显式报错 |
| `include/forge/cuda_kernels.h` | 声明 `launch_silu_fp16` |
| `src/operators/cuda/cuda_activation.cu` | 新增 `silu_kernel_fp16` + `launch_silu_fp16`（FP16 in/out，sigmoid/exp 在 float）；补 `<cuda_fp16.h>`。原伪 `launch_silu_fp16`（reinterpret 为 float）已在 6-A-1 删除，本轮以真实 `__half` kernel 补回 |
| `src/operators/cpu/activation.cpp` | `ops::silu` 按 dtype 路由：FP16→`launch_silu_fp16`、FP32→原 kernel、其余（BF16 等）显式 throw；输出 dtype 跟随输入；CPU 路径保持 FP32-only（native FP16 仅 CUDA，与 RMSNorm 一致） |
| `tests/test_dtype_contract.cpp` | SILU 从「已拒绝」集移入「已接受」白名单集；6-A-3 一致性不变式同步 |
| `tests/test_silu_fp16.cpp`（新） | CTest `silu-fp16`：FP32 reference 对照（n∈{4096,8192,512·4096,512·8192}）+ 微基准（两 regime），无 GPU 优雅 SKIP |

**正确性（FP32 reference 对照：输入先编码为 FP16 再喂 kernel，reference 用同一解码值，残差仅来自 FP16 输出存储舍入）**

| 配置 | max_abs_err | max_rel_err |
|---|---|---|
| n=4096 / 8192 / 512·4096 / 512·8192 | ~9.7e-4 | ~4.8e-4 |

误差量级 = FP16 输出存储舍入（满足 1e-3 量级约束），与 RMSNorm 同构；结构错误（错轴/漏算）会被阈值（0.02 / 0.05）拦截。

**微基准（kernel-only，两 regime）**

| 配置 | 类别 | FP16 | FP32 |
|---|---|---|---|
| n=4096 / 8192 | launch-bound | ~0.0065 ms / 2-5 GB/s | ~0.0066 ms / 5-10 GB/s |
| n=512·4096 | bandwidth-bound（工作集入 L2） | 0.036 ms / 232 GB/s | 0.037 ms / 454 GB/s |
| n=512·8192 | bandwidth-bound | 0.064 ms / 260 GB/s | 0.211 ms / 159 GB/s |

结论：**small n 两者均 launch-bound，FP16 与 FP32 几乎无差——FP16 并非"天然更快"**；
large n 进入 memory-efficiency 区间，FP16 流量减半带来收益，但**倍率随工作集（L2 vs HBM）
浮动，不是固定 2.7x**。这与 6-A-2 封版原则 #3 完全一致——Phase 6 后续 benchmark 一律保留
launch-bound / bandwidth-bound 两档分类。

**验收**：L4 ✅；CTest `dtype-contract` + `silu-fp16` 均过（RMSNorm 测试无回归）。

<!-- @@P6A-4-DELIVERED@@ -->

#### 6-A-4 native FP16 activation family（2026-09-05，本轮交付 ✅）

> 按重编号路线第 3 步。把"主链 activation family"做成 FP16 闭环，让 6-A-5 matmul 能跑通
> `matmul → native activation` 链。严格限定在真实主链入口，**不为了枚举完整去支持无调用路径
> 的 activation**（RELU 等未碰）。计算模型与 6-A-2/6-A-3 完全一致：FP16 存储 + FP32 超越函数
> 计算（erf / tanh / sigmoid）+ FP16 输出。

**覆盖（主链真实入口）**

| 算子 | 形态 | 主链使用 |
|---|---|---|
| `gelu` | unary | 通用激活 |
| `gelu_tanh` | unary | ViT projector 等 |
| `silu_multiply` | gated（SiLU-gated） | LLaMA / Qwen / DeepSeek FFN |
| `gelu_multiply` | gated（GeGLU） | Gemma4 FFN |

**新增 / 修改**

| 文件 | 改动 |
|---|---|
| `include/forge/dtype_contract.h` | 白名单新增 `GELU` + `GELU_TANH`；gated 变体 `silu_multiply` / `gelu_multiply` 复用 `SILU` / `GELU` 闸门（不新增枚举） |
| `include/forge/cuda_kernels.h` | 声明 `launch_gelu_fp16` / `launch_gelu_tanh_fp16` / `launch_silu_multiply_fp16` / `launch_gelu_multiply_fp16` |
| `src/operators/cuda/cuda_activation.cu` | 新增 `gelu_kernel_fp16` + `launch_gelu_fp16`、`gelu_tanh_kernel_fp16` + `launch_gelu_tanh_fp16` |
| `src/operators/cuda/cuda_elementwise.cu` | 新增 `silu_multiply_kernel_fp16` + `launch_silu_multiply_fp16`、`gelu_multiply_kernel_fp16` + `launch_gelu_multiply_fp16`（补 `<cuda_fp16.h>`） |
| `src/operators/cpu/activation.cpp` | `ops::gelu` / `ops::gelu_tanh` 按 dtype 路由：FP16→native、FP32→legacy、其余显式 throw；输出 dtype 跟随输入 |
| `src/operators/cpu/elementwise.cpp` | `ops::silu_multiply` / `ops::gelu_multiply` 按 dtype 路由（gated 复用 SILU/GELU 闸门）；输出 dtype 跟随 gate；CPU 路径保持 FP32-only |
| `tests/test_dtype_contract.cpp` | `GELU` 从「已拒绝」集移入「已接受」白名单集；`GELU_TANH` 进白名单；一致性不变式同步 |
| `tests/test_activation_fp16.cpp`（新） | CTest `activation-fp16`：FP32 reference 对照（unary + gated，n∈{8192,512·4096}）+ 微基准（两 regime），无 GPU 优雅 SKIP |

**正确性（FP32 reference 对照：输入先编码为 FP16 再喂 kernel，reference 用同一解码值，残差仅来自 FP16 输出存储舍入）**

| 配置 | max_abs_err | max_rel_err |
|---|---|---|
| gelu / gelu_tanh（unary） | ~9.7e-4 | ~4.8e-4 |
| silu_multiply / gelu_multiply（gated） | ~3.9e-3 | ~4.9e-4 |

gated 略高：两个 FP16 操作数相乘再存 FP16，乘积舍入比 unary 多一档，但仍远低于 0.02 阈值。

**微基准（kernel-only，两 regime）**

| 配置 | 类别 | FP16 | FP32 |
|---|---|---|---|
| gelu n=4096 | launch-bound | 0.0065 ms / 2.5 GB/s | 0.0065 ms / 5.1 GB/s |
| gelu n=4194304 | bandwidth-bound | 0.070 ms / 240 GB/s | 0.213 ms / 158 GB/s |
| silu_multiply n=4096 | launch-bound | 0.0156 ms / 1.6 GB/s | 0.0155 ms / 3.2 GB/s |
| silu_multiply n=4194304 | bandwidth-bound | 0.101 ms / 250 GB/s | 0.323 ms / 156 GB/s |

结论与 6-A-2/6-A-3 一致：small n 均 launch-bound（FP16 不更快），large n 进 memory-efficiency 区间（FP16 流量减半，~2-3x）。

**验收**：CTest `dtype-contract` + `activation-fp16` 均过（RMSNorm / SiLU 测试无回归）。

<!-- @@P6A-5-DELIVERED@@ -->

#### 6-A-5 native FP16 matmul（2026-09-06，本轮交付 ✅）

> 按重编号路线第 4 步，**只解决 Native FP16 Matmul 的计算链**。范围严格锁定：
> 不顺手做 graph / attention / binding / BF16。目标链
> `FP16 input → native FP16 RMSNorm → FP16 Matmul → native FP16 activation → FP16 output`
> 已端到端跑通（见 6-A-5.3）。

**核心决策**
- 不新增独立 FP16 算子签名，把 native 分支并入 `cuda_matmul_transB`（与 6-A-2/3/4 的
  "dtype 路由"模式一致），只白名单 `MUL_MAT_TRANSB`；non-transB `MUL_MAT` 保持审计态
  （无 native FP16 row-major-B 路径，继续显式 reject），避免静默走 float。
- native 路径：`__half` A/B/C 直送 cuBLAS，`computeType = CUBLAS_COMPUTE_32F`
  （**FP16 存储 + FP32 累积 + FP16 输出**）。对 M=1（解码 GEMV）与 M>1（prefill）统一用
  cuBLAS，分别吃到 Tensor Core 的 FP16 吞吐。路径内**无 FP32 中间张量物化**（No FP32
  intermediate tensor materialization）；FP32 累加（accumulation）是 `computeType=CUBLAS_COMPUTE_32F`
  故意要求的、由 compute policy 决定，**并非"整个 matmul 没有 FP32 运算"**。区别必须写清：
  - *No FP32 tensor materialization*：没有 FP16→FP32→FP16 的中间张量（输出 `out` 结构性为 FP16，
    经 6-A-5.3 全程 dtype 不变量验证）；
  - *FP32 compute 是有的*：仅 Tensor Core 内部累加与 cuBLAS 标量 `alpha/beta` 进入 FP32，这是
    compute policy 要求的数值稳定性，不是隐式降级。
- Phase 5 量化路径（Q4_0/Q8_0…）与 FP32 路径**零改动**：FP16 分支在 `require_dtype_supported`
  闸门之后、float 转型之前提前 return，FP32/量化走原分支。

**新增 / 修改**

| 文件 | 改动 |
|---|---|
| `include/forge/dtype_contract.h` | 白名单新增 `MUL_MAT_TRANSB`（契约 `op_supports_lowp` 唯一扩展点） |
| `include/forge/cuda_kernels.h` | 声明 `launch_cublas_gemm_fp16`（`__half` A/B/C，FP16 存储 + FP32 累积） |
| `src/operators/cuda/cuda_quant.cu` | 新增 `launch_cublas_gemm_fp16`（与 `launch_cublas_gemm_fp16_fp32` 同构，仅 C 为 `CUDA_R_16F`） |
| `src/operators/cpu/matmul.cpp` | `ops::matmul_transB`：A/B 均为 FP16 时分配 **FP16** 输出并走 native 分支（bias 暂不支持，显式 reject） |
| `src/operators/cuda/matmul_cuda.cpp` | `cuda_matmul_transB`：A/B 均为 FP16 时直接 `launch_cublas_gemm_fp16`（transB=true），其余路径不变 |
| `src/operators/cpu/arch/x86/elementwise_kernels.h` | **附带修复**：AVX2 `silu_mul_f32_vec` 的 Cephes `exp` 缺 FP32 范围钳制（±88.376），`|gate|>~61` 时指数位溢出产生错误 silu；补 clamp 后 CPU 参考路径对大 activation 正确（被本阶段链测试暴露） |
| `tests/test_dtype_contract.cpp` | `MUL_MAT_TRANSB` 从「已拒绝」集移入「已接受」白名单集；扩展点测试同步 |
| `tests/test_matmul_fp16.cpp`（新） | CTest `matmul-fp16`：6-A-5.1 直接 launch + 6-A-5.2 `ops::matmul_transB` 路由，对照 FP32 reference；M 扫描微基准 |
| `tests/test_matmul_fp16_chain.cpp`（新） | CTest `matmul-fp16-chain`：6-A-5.3 RMSNorm→Matmul→SiLU 真实链，dtype 不变量 + 正确性 |

**正确性（FP32 reference 对照：输入先编码 FP16 再喂 kernel，reference 用同一解码值；残差仅来自 FP16 输出存储舍入）**

| 配置 | max_abs | max_rel |
|---|---|---|
| 6-A-5.1 native launch（M=1/8/64, K=N=4096） | ~2.9e-2 | ~5e-4 .. 1.9e-3 |
| 6-A-5.2 `ops::matmul_transB` 路由（M=1/8/64） | ~3.1e-2 | ~5e-4 .. 2.1e-3 |
| 6-A-5.3 真实链（M=1/8/32, K=4096, N=11008） | ~0.08 | ~4e-2 .. 1.1e-1 |

链测试相对误差略高：激活（RMSNorm 输出 ~1.9）以 FP16 存储，其 ~1e-3 舍入在 K=4096 累加中传播，对小幅值输出元素表现为数 % 相对误差（matmul 输出幅度 ~147，FP16 ULP ~0.125）；单算子测试已证明 matmul 本身精确到 FP16 舍入。

**微基准（native FP16 cuBLAS vs FP32 cuBLAS，M 扫描；K=N=4096，FP16 流量减半）**

| M | FP16 | FP32 |
|---|---|---|
| 1 | 0.209 ms / 0.1 GB/s | 0.426 ms / 0.1 GB/s |
| 8 | 0.186 ms / 1.1 GB/s | 0.408 ms / 1.0 GB/s |
| 32 | 0.191 ms / 4.1 GB/s | 0.404 ms / 3.9 GB/s |
| 128 | 0.215 ms / 14.6 GB/s | 0.637 ms / 9.9 GB/s |
| 512 | 0.762 ms / 16.5 GB/s | 2.728 ms / 9.2 GB/s |

解码（M=1）受 launch 开销主导，FP16 仍 ~2x；prefill（M≥128）进带宽区间，FP16 流量减半带来
~1.7-2.9x 提速，与 Phase 3 量化 TC 的带宽收益同构（两条路径并存：量化权重走 TC 反量化、FP16
权重走本 native 路径）。

**验收**：CTest `dtype-contract` + `matmul-fp16` + `matmul-fp16-chain` 均过；`rms-norm-fp16` /
`activation-fp16` 无回归（silu clamp 修复不影响小输入）。

<!-- @@P6A-6-DELIVERED@@ -->

#### 6-A-6 Native FP16 Graph / Binding 集成（2026-09-06，本轮交付 ⏳ 部分）

> 按重编号路线第 5 步。**不再写新 CUDA kernel**——目标是证明 6-A-5 那条已经正确的算子链
> （RMSNorm → Matmul → SiLU）在 `ComputeGraph` / 内存规划 / dst 注入 / dispatch 里**仍然保持 FP16**。
> 交付 6-A-6.1 / 6-A-6.2 / 6-A-6.3 / 6-A-6.5 / 6-A-6.6；6-A-6.4（CUDA Graph capture/replay）
> 部分交付，受阻原因见下。

**审计结果：FP16 提升点不在 graph builder，而在两处"静默 FP32"**

| 位置 | 问题 |
|---|---|
| `src/core/compute_graph.cpp`（dst 注入） | 目标张量**硬编码 `DataType::FP32`**：`planned_ptr` 上的 dst view 一律按 FP32 建，与算子真实输出 dtype 无关 |
| `src/core/op_kernels.cpp`（dst 内核） | 全部 dst 内核（add / mul / silu / rms_norm / rope）是 **FP32-only**：`static_cast<const float*>` + 调 `launch_rms_norm` 等 FP32 核。FP16 链进来会被**当 float 误读再跑 FP32 核** |
| `src/core/memory_planner.cpp` | `MUL_MAT_TRANSB` 尺寸硬编码 `sizeof(float)`；且 `estimate_output_size` **只解析"图输入"作首输入**，链中下游节点（首输入是上游节点输出）返回 size=0 → 永不分配缓冲 → dst 注入被跳过、退化 legacy |

即：真正的"FP16 → FP32"不是 graph builder 干的（`GraphBuilder` 按原样持有张量、不碰 dtype），
而是 **dst 注入 + 规划器** 两处把 FP32 当成唯一可能。

**核心改动（对应用户给的 5 点）**

| 点 | 改动 |
|---|---|
| 6-A-6.1 图构建不提升 | 无需改：dtype 由输入张量决定；真正提升点已按下两点修掉 |
| 6-A-6.2 binding 保留 FP16 | `MemoryPlanner::plan()` 沿图**传播 shape/dtype**（拓扑序），使下游节点也拿到缓冲；`MUL_MAT_TRANSB` 尺寸改用**首输入元素宽度**（`dtype_size(input dtype)`），不再 `sizeof(float)` |
| 6-A-6.3 dst 注入不 materialize FP32 | dst 张量改用算子**真实输出 dtype**（storage-preserving：跟随首输入 dtype，不再强制 FP32）；`rms_norm_kernel_dst` / `silu_kernel_dst` 新增 **FP16 分支**（`launch_rms_norm_fp16` / `launch_silu_fp16` 直写 FP16 dst）；FP16 输入且该 op 的 dst 内核不支持 FP16 时**不注入**（避免误转型），改走 legacy |
| 6-A-6.5 真实图跑通 | 新增 CTest `graph-fp16` |
| 6-A-6.6 dtype trace | 新增 `FORGE_TRACE_DTYPE=1` + `ComputeGraph::dtype_trace()`：记录每个节点的输入 dtype / 输出 dtype / 是否 dst 注入 |

**新增 / 修改**

| 文件 | 改动 |
|---|---|
| `include/forge/compute_graph.h` | 新增 `DTypeTraceRecord`（name / input_dtypes / output_dtype / dst_injected / device）、`dtype_trace()`、`trace_dtype_enabled()` |
| `src/core/compute_graph.cpp` | dst 输出 dtype 按算子推断（去 FP32 硬编码）；dst 注入加 FP16 门槛；execute() 记录 dtype trace；`FORGE_TRACE_DTYPE` 环境开关 |
| `src/core/op_kernels.cpp` | `rms_norm_kernel_dst` / `silu_kernel_dst` 新增 FP16 分支（CUDA 直调 FP16 launch；CPU 无 native FP16 核，按既定"CPU 路径 FP32-only"上投后存 FP16） |
| `src/core/memory_planner.cpp` | plan() 增加 shape/dtype 传播；`MUL_MAT_TRANSB` 尺寸按输入 dtype |
| `tests/test_graph_fp16.cpp`（新） | CTest `graph-fp16`：真实 FP16 图链 + dtype trace 断言 + FP16 缓冲地址/尺寸稳定性 |
| `tests/CMakeLists.txt` | 注册 `graph-fp16` |

**dtype trace（实测输出，`FORGE_TRACE_DTYPE=1`）**

```text
Node                  Input     Output    dst_injected
-------------------------------------------------------
rms_norm              fp16      fp16      1
matmul                fp16      fp16      0
silu                  fp16      fp16      1
```

全程 `fp16 → fp16`，**没有出现 `FP16 → FP32 → FP16`**。`matmul` 的 `dst_injected=0` 是预期的：
`MUL_MAT_TRANSB` 目前只有 legacy 内核（`ops::matmul_transB`，已是 native FP16），没有注册 dst 内核；
它的输出仍是 FP16 并被拷入图缓冲。把它也做成 dst 注入会影响主线 engine 的既有路径，
留待专门一步。

**正确性（同一批算子 + 同一批张量，只多一层 ComputeGraph）**

| 配置（K=4096, N=11008） | 与独立算子链对比 |
|---|---|
| M=1 / 8 / 32 | `max_abs = 0.000e+00`（**逐位一致**） |

即：graph 层（规划 / dst 注入 / dispatch）**没有引入任何数值差异**——这正是 6-A-6 要证明的
"same operators + same tensors + ComputeGraph ⇒ 结果一致"。

**6-A-6.4 Stream plumbing + CUDA Graph capture/replay：已交付（2026-09-06）**

> Runtime prerequisite（非 FP16 优化）：修的是 Graph/Dispatch 的基础能力，
> FP16 只是第一个把问题暴露出来的 workload。FP32 图同样受惠（普通路径零行为变化）。

此前状态：实际流捕获**失败**（`cudaStreamBeginCapture failed: operation not permitted
when stream is capturing`），根因是**流未贯通**：所有 launch 打在 legacy 默认流，
而 `OpDispatch` 签名里没有流参数。本步将其贯通（单执行流，不碰 multi-stream /
event / stream 池 / async allocator；MUL_MAT_TRANSB 只做 stream 贯通、不做 dst 注入）。

改动一览：

| 文件 | 改动 |
|---|---|
| `include/forge/op_dispatch.h` | `OpKernelFn/Dst` + `execute/execute_dst` 加 `cudaStream_t stream = 0`；CPU 构建 fallback 类型守卫 |
| `src/core/op_dispatch.cpp` | 透传 `stream`；legacy fallback 的 `copy_from` 保持 legacy 语义（capture 走 graph 路径的 Async 拷贝） |
| `src/core/op_kernels.cpp` | 5 dst 内核前转 `stream`；`mul_mat_transb` 传 `stream`；`flash_attn_gqa` 仅改签名（stream 留待 6-A-7） |
| `src/operators/cpu/{norm,elementwise,activation}.cpp` | legacy 注册 lambda 加参（忽略 `stream`，行为不变） |
| `include/forge/operator_matmul.h` + `cpu/matmul.cpp` + `cuda/matmul_cuda.{h,cpp}` | `matmul_transB/cuda_matmul_transB` 加 `stream = 0`；FP16 分支 `launch_cublas_gemm_fp16(..., stream)`（原硬编码 `0`）；FP32/quant 分支保持 legacy（FP32 capture = 已知限制） |
| `include/forge/compute_graph.h` + `src/core/compute_graph.cpp` | `execute/execute_node(stream = 0)`；D2D 拷贝 `cudaMemcpyAsync(..., stream)`；内部零同步（plumbing ≠ 同步策略） |
| `graph_runtime.{h,cpp}` | `GraphRuntime` 持有 `capture_stream_`（懒创建、析构才销毁；capture/instantiate/replay 同柄）；capture 闭包 `graph_->execute(cs)` |
| `tests/test_graph_fp16.cpp` | capture 强断言 A/B/C/D + alloc 探针（见下） |

Matmul allocation 硬门槛（实测通过，无需 allocator 改动）：capture 窗口内 pool
`allocs +1 / hits +1`——matmul temp 命中 freelist，**零 `cudaMalloc`**（A/B 两次
eager 执行已 warm 池）。备选的 node-scratch / async allocator 均未启用。

实测（`graph-fp16`，M=8, K=1024, N=2048）：

```text
alloc probe : allocs 26->27, hits 3->4   # 1 次池分配、全命中
capture+replay OK, buffer=0x... size=32768 B = 8*2048*2 (FP16)
A == B == C == D  (逐位一致，max_abs = 0)
```

口径（6-A-6.4 实际兑现的是 FP16 capture contract，不是全图 stream-safe）：

```text
FP16:
    full capture/replay supported
FP32:
    ordinary execution supported
    full capture not yet guaranteed
    because legacy GEMV/MMQ paths still contain hardcoded stream=0 launch behavior
```

**验收（回归）**：CTest **34/34 全过**（6-A-7 后为 35/35，新增 `attn-fp16`），其中 `spec-verify` /
`dflash` 是走完整 engine + planner 的端到端测试——证明 `MemoryPlanner` 的全局改动
（shape/dtype 传播、matmul/attention 尺寸按 dtype）无回归。

#### 6-A-7 Native FP16 Attention（2026-09-06，Phase 1 correctness 已交付）

> 不重写 FlashAttention。第一阶段只追求 `FP16 Q/K/V → FP32 reduction/softmax → FP16 output`
> 的正确性；性能（`__half2` / TensorCore）留待 Phase 2，且 Phase 2 入口锁死为 correctness 全绿、
> 只实测 `FP32 Vectorized` vs `FP16 baseline`（延续 Phase 2 铁律：不用 occupancy/register/理论吞吐排名）。

dtype 语义（唯一正确性标准）：`Q/K/V FP16 存储 → QK^T·rsqrt(hd) → FP32 online-softmax →
FP32 PV → FP16 输出`。scale 约定与 FP32 核一致（`cuda_flash_attn.cu:357`：`dot *= 1/sqrt(HD)`，
恰一次）；softmax 保持 fused online，无独立 kernel（`SOFT_MAX` 不进 contract）。

| 点 | 改动 |
|---|---|
| 7.0 contract | `op_supports_lowp` 加 `FLASH_ATTN_GQA`（sole extension point）；`MemoryPlanner` attention 尺寸按首输入元素宽度（6-A-6.2 同口径）；`dtype-contract` 测试加 accepted |
| 7.1 decode | `flash_attn_gqa_decode_kernel_fp16`（同 TU，1 block/head，Q→smem，KV streaming，16B 加载=8 halves，`local_acc` 保留；HD 16/32/64/96/128，更大 host 侧 loud 拒绝） |
| 7.2 prefill | `flash_attn_gqa_kernel_vectorized_fp16`（Vectorized 镜像，`local_acc[HD]` 保留；仅替换 dtype boundary） |
| selector | 新增 `CUDA_F16_DECODE / CUDA_F16_PREFILL` + `AttentionProblem.q_dtype`；`q FP16 → F16`，`q FP32 → 原路径`（`CUDA_FP32_*` 内无 dtype-if，path 名即语义） |
| executor | 新增两 `case`（FP16 输出，K/V 非 FP16 / mask 非 FP16 / HD 越界一律 loud）；`attend()` 加 `stream = 0`（FP16 核第一天即 honor） |
| dispatch | `flash_attn_gqa_kernel` FP16 分支（decode/prefill，honor stream；MHA/混合 dtype/越界 HD loud） |
| 测试 | 新 CTest `attn-fp16`：dtype 硬断言（先 `O==FP16` 再比数值）+ CPU-FP32-ref parity + FP32 自洽 + A/B 流逐位 + graph capture/replay 逐位 + selector 单测 + HD256 loud |

实测（`attn-fp16`）：decode `max_abs≈5e-05`（HD 16/32/64/128，kv 128/512）、prefill
`≈1e-04`——均为 FP16 舍入量级（核内数学为 FP32）；A==B、C==D 逐位一致。

非目标（明确未做）：TensorCore/MMA、`__half2`、paged FP16、MLA FP16、FP16 KV-cache 集成
（executor 遇到非 FP16 KV 时 loud 失败，见代码注释）、CPU FP16 SIMD、第三套 dispatch
（attention 仍走 `AttentionSelector`，matmul 仍走 `KernelSelector`）。

### 6-B. Native BF16 Execution

紧随 6-A 的第二步。BF16 枚举已存在但实际是 `BF16 → load → FP32`（`types.h:15` 注释即
自认）：真正要做的是

```text
BF16 weights + BF16 activation → BF16 Tensor Core → FP32 accumulation
```

完成时 policy 拥有 FP16/BF16 双实列，dtype abstraction 才算成熟。

### 6-C. FP8 W8A8（真正的 low-precision compute backend）

```text
FP8 weight + FP8 activation → Tensor Core → FP16/FP32 accumulation
```

与已完成的 **FP8 KV（KV8）是两回事**：KV8 只压 cache 路径；W8A8 才进入计算路径。

### 6-D. NVFP4（Blackwell，压轴）

```text
NVFP4 weight + low-precision activation → Blackwell Tensor Core
```

**共同顺序规则（每层）**：先把 activation 路径打通，再动 weight——否则会变成
"FP16 GEMM 夹在 FP32 首尾之间"，白白增加转换开销。

---

## 2. 跨阶段约束与风险登记

### 2.1 `scratch_pool()` 是全局单例

`src/operators/cuda/cuda_common.h:28-39`（另有独立的 `mmq_scratch()` 供 MMQ 使用）。
实测共用情况：

| 文件 | `scratch_pool()` 调用处 |
|---|---:|
| `cuda_gemv.cu` | 40 |
| `cuda_fused.cu` | 7 |
| `cuda_sampler.cu` | 1 |
| `cuda_mmq.cu` | 0（改用专属 `mmq_scratch()`） |

影响：
- Phase 1-A（MoE dp4a）必须在同一 kernel 内只读复用，不能跨 kernel 持有
- CUDA Graph capture 期间地址必须稳定（现有 `CudaGraphRunner` 要求）
- 多流并发时有别名风险

**建议**：Phase 0 阶段先给 `scratch_pool()` 增加"当前有哪些 buffer 存活"的调试
断言，避免后续优化引入隐式冲突。

### 2.2 数值回归容忍度

任何改变浮点累加顺序的改动（1-C 协作归约、3-B MMA、6-A FP16 accumulator）
都会导致与 FP32 参考存在细微差异。必须**先**确定容忍度标准：

| 场景 | 建议指标 |
|---|---|
| GEMV / GEMM | 与 FP32 参考余弦相似度 ≥ 0.999 |
| Attention | 输出余弦相似度 ≥ 0.999 |
| 端到端 | `test_cross_validation.py` 现有标准 |

### 2.3 CUDA Graph 兼容性

`CudaGraphRunner::is_compatible()`（`cuda_graph_runner.h:66`）要求
graph 路径内**零 cudaMalloc**。Phase 1-A / 1-B 引入的新 kernel 若使用
`scratch_pool().ensure()`，需确认在 capture 前已预分配。

### 2.4 每阶段回滚点

每阶段开始前打 tag：

```bash
git tag -a phase0-start -m "Phase 0 baseline"
# 出问题： git checkout phase0-start
```

---

## 3. 建议的提交粒度

不要一个 PR 做完整阶段。建议按以下粒度提交，每个都可独立 review 与回滚：

```text
Phase 0
  ├─ docs: 纳入 plan.md + optimization_plan_zh.md
  ├─ refactor(cuda): 引入 kernel_manifest.yaml，重生成 gemv instances
  ├─ refactor(cuda): 移除已下线的 q4_0_transB 系列 kernel
  └─ docs: 修正 dependencies.md 中 USE_CUBLAS 默认措辞

Phase 1
  ├─ perf(moe): expert GEMV 迁移到 Q8_1 + dp4a          ← 单项最大收益
  ├─ perf(matmul): dual matmul 单 kernel 化，去掉 D2D copy
  ├─ perf(gemv): output_proj 用协作归约替换 memset+atomic
  └─ test(cuda): 新增 moe_dp4a_reference 与 attention parity 测试

Phase 2
  ├─ feat(attn): 新增量化/分页 prefill 路径并接入 selector
  ├─ feat(attn): AttentionBackend 分层，实现 Vectorized backend
  └─ perf(attn): paged metadata 常驻 GPU

Phase 3
  ├─ feat(gemm): QuantComputePath 枚举与 M 分档 dispatch
  └─ perf(gemm): Q4_0/Q4_K/Q8_0 的 FP16 MMA 路径

Phase 4
  ├─ refactor(core): 统一 KernelSelector
  └─ feat(core): benchmark-driven dispatch 与 kernel cache
```

---

## 4. 与 plan.md 的偏差说明

本计划对 `plan.md` v2 做了三处调整，均基于源码核验：

| plan.md 的表述 | 本计划的调整 | 原因 |
|---|---|---|
| P0「Tensor-Core quantized GEMM」 | 限定为 **M ≥ 32**，M=1 明确保持 dp4a | dp4a 即 INT8 SIMT；decode 是 memory-bound，MMA 无收益。plan 第 3 节的 M 分档图其实已含此意，但优先级表压扁了 |
| 优先级表 NVFP4 标 P2 / 价值 ★★★★★ | 移入 Phase 6 末位，与价值解耦 | 优先级须受依赖约束：NVFP4 需要 Phase 3 的 TC backend 与 Phase 6 的 FP16 activation 先行 |
| 未提及 MoE 实现落后 | 新增 **Phase 1-A**，列为本计划单项最大收益 | `cuda_moe.cu:207` 仍用 `CudaQuantTraits::dot_block()`，而该函数是 scalar（`cuda_gemv_tmpl.cuh:38-41`）；主 GEMV 路径早已 dp4a 化。这是"补课"而非"创新"，风险低、收益确定 |

另：`plan.md` 末尾「README 与 CMakeLists 关于 `USE_CUBLAS` 默认值冲突」一条
经核验**不成立**（README 未声明默认值，`docs/dependencies.md:94` 与
`CMakeLists.txt:87` 一致为 ON），本计划仅保留 `docs/dependencies.md:14`
的措辞修正（0-E）。
