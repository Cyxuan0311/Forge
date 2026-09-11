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
- 6-A-8 端到端 FP16 验证 ✅（本轮）—— 全链路跑通 + 三项验收（dtype trace / parity / CUDA Graph）。

**Phase 6-A 进度（截至 6-A-8 封版，Phase 6-A correctness 全部交付）**

| 步骤 | 内容 | 状态 |
|---|---|---|
| 6-A-0 | FP32 提升点审计（只读） | ✅（2026-09-05） |
| 6-A-1 | dtype contract hardening | ✅（2026-09-05） |
| 6-A-2 | Native FP16 RMSNorm | ✅（2026-09-05） |
| 6-A-3 | Native FP16 SiLU | ✅（2026-09-05） |
| 6-A-4 | Native FP16 activation family | ✅（2026-09-05） |
| 6-A-5 | Native FP16 Matmul | ✅（2026-09-06，封版） |
| 6-A-6 | Graph / Binding 集成 | ✅（2026-09-06，6-A-6.1..6.6 全部交付，流贯通由 6-A-7 补齐） |
| 6-A-7 | Native FP16 Attention | ✅ **Phase 1 COMPLETE / FROZEN**（2026-09-06） |
| 6-A-8 | End-to-end FP16 验证 | ✅ **COMPLETE / FROZEN**（2026-09-06，correctness baseline 冻结，Phase 2 解锁） |

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

**封版记录（2026-09-06）：6-A-7 Phase 1 = COMPLETE / FROZEN**

```text
6-A-7 Phase 1 -- Native FP16 Attention
STATUS: COMPLETE / FROZEN

├── Decode              ✅  (flash_attn_gqa_decode_kernel_fp16)
├── Prefill             ✅  (flash_attn_gqa_kernel_vectorized_fp16)
├── FP32 accumulation   ✅  (QK^T / online-softmax / PV 全 FP32)
├── FP16 output         ✅  (dtype 硬断言，非仅数值)
├── Selector            ✅  (CUDA_F16_DECODE / CUDA_F16_PREFILL, q_dtype 驱动)
├── Stream plumbing     ✅  (FP16 核第一天即 honor stream, A/B 流逐位)
├── CUDA Graph          ✅  (节点级 capture/replay 逐位一致)
├── Contract            ✅  (FLASH_ATTN_GQA 白名单, 唯一扩展点)
├── Planner             ✅  (attention 尺寸按首输入元素宽度)
└── CTest 35/35         ✅  (新增 attn-fp16)
```

Deferred（Phase 2 性能项，未混入本轮 correctness）：
- FP16 KV-cache、paged FP16 attention、`__half2`、TensorCore/MMA、performance tuning。

**下一拍 = 6-A-8 端到端 FP16 validation（已立项）**：6-A-8 全绿前 Phase 2（性能）不开工；
Phase 2 顺序锁死 FP16 baseline → `__half2` → TensorCore，Decode / Prefill 分开测，
第一轮只回答 "FP16 baseline vs FP32 path 增益"，第二轮才比 half2/TensorCore vs baseline。

#### 6-A-8 End-to-end FP16 验证（2026-09-06，本轮交付 ✅ COMPLETE / FROZEN）

> 目标：证明完整 inference 闭链（RMSNorm → QKV → Attention → residual → gated FFN →
> next layer → logits）在真实 ComputeGraph 执行下不发生 `FP16→FP32→FP16` 隐式回退或
> dtype contract 断裂。最小闭环 = 2 层 llama 式图（D=256, nh=8, nkv=2, hd=64, FF=512,
> V=128），prefill（q_len=16）与 decode（q_len=1, K/V 直接消费 KV-cache 输入）两种 dispatch
> 模式，FP16 与 FP32 跑**同一拓扑**（同一组 float 模型值驱动两种 dtype 物化，差异只来自
> FP16 存储/计算）。

**新增 / 修改**

| 文件 | 改动 |
|---|---|
| `tests/test_e2e_fp16.cpp`（新） | CTest `e2e-fp16`：8.0 = 新 elementwise FP16 核的 native ⇄ FP32 reference 链式验证；8.1 = 2 层闭环三项验收（下） |
| `src/operators/cuda/cuda_elementwise.cu` | 6-A-8.0：`add_fp16_kernel` / `multiply_fp16_kernel` + `launch_add_fp16` / `launch_multiply_fp16`（FP16 存储、逐元素 FP32 计算、FP16 输出、honor stream） |
| `include/forge/cuda_kernels.h` | 声明上述两个 launch |
| `src/core/op_kernels.cpp` | `add_kernel_dst` / `mul_kernel_dst` 加 FP16 分支（CUDA 走 native 核，CPU 本地 upcast→FP32 运算→FP16 存储）；contract 断言 `ADD`/`MUL` |
| `include/forge/dtype_contract.h` | 白名单加 `ADD` / `MUL`（residual 与 gated FFN 的主链入口；6-A-8.0 唯一扩展） |
| `src/core/memory_planner.cpp` | **修复** `verify_plan()` 假阳性：overlap 检查只对 lifetime 重叠（`max(begin) <= min(end)`）的分配生效——e2e 图的 buffer 复用触发了大量误报 |
| `src/operators/cuda/cuda_common.h` | **修复** cuBLAS 死流 segfault：`get_cublas_handle()` 每次调用都 `cublasSetStream`（含 stream==0）——旧 `if (stream)` 守卫让 handle 绑死在已销毁的 capture 流上，后续 decode matmul 在死流上 launch（`cuLaunchKernel` 内 segfault） |
| `tests/test_dtype_contract.cpp` / `tests/test_matmul_fp16.cpp` | `ADD`/`MUL` accepted 断言同步 |
| `tests/CMakeLists.txt` | 注册 `forge-e2e-fp16-test` → CTest `e2e-fp16` |

**三项验收（实测，固定阈值已写死进测试）**

1. **dtype trace（`FORGE_TRACE_DTYPE=1`）**：prefill **28/28 节点** FP16 in/out（无隐式回退），
   decode **24/24**；dst 注入 13/28、13/24（全部 rms/residual/silu/gate_up 走 planner buffer，
   非 legacy allocate+copy 绕路）。
2. **FP16 vs FP32 parity（同一拓扑、同一模型值）**，阈值固定
   `max_abs<0.05 / max_rel<0.05(guard |ref|>=0.15) / cosine>=0.999 / top8 agree>=7`：

   | 模式 | max_abs | max_rel | cosine | top8 |
   |---|---|---|---|---|
   | prefill | 2.275e-04 | 1.194e-03 | 0.9999998 | 8/8 |
   | decode  | 1.498e-04 | 7.701e-04 | 0.9999998 | 8/8 |

   guard 非空转有断言（`max|ref|` 0.207 / 0.187 ≥ 0.15）。实测较阈值有 ~2 个数量级余量。
3. **CUDA Graph 真链路**：ordinary → capture → replay×3，两模式输出逐位一致，
   logits buffer 地址/尺寸稳定（prefill 4096 B / decode 256 B，均为 FP16 宽度）。

8.0 链式验证：`add_fp16` / `mul_fp16` max_abs≈1e-3（FP16 输出舍入量级），A/B 流逐位一致。

decode 拓扑说明：q 来自当前 token（`wq` 投影），K/V 直接以 `k_cache`/`v_cache` 图输入
（kv_len=16）进入 attention——真实 decode 的 cache 消费语义，不从单 token 重算 K/V；
FP32/FP16 decode 核均 attend 全部 `total_len`（causal 语义一致）。

**验收（回归）**：CTest **36/36 全过**（6-A-7 后为 35/35，新增 `e2e-fp16`）。

非目标（明确未做）：真实模型加载（GGUF/sampler/argmax）、RoPE（维持 FP32，既定设计）、
FP16 KV-cache 集成（cache 以现成 FP16 张量喂入，不测 cache 写入路径）、>2 层 / 真实
模型尺寸、任何性能测量（Phase 2 开工后按铁律做）。

**封版记录（2026-09-06）：6-A-8 = COMPLETE / FROZEN —— Phase 6-A correctness baseline 冻结，
Phase 2（性能）解锁。** Phase 2 顺序不变：FP16 baseline → `__half2` → TensorCore，
Decode / Prefill 分开测，第一轮只回答 FP16 baseline vs FP32 path 增益。

#### Phase 2 第一轮（2026-09-06，FP16 baseline vs FP32 path ✅ 实测完成）

> 回答唯一问题：**Native FP16 baseline 到底比现有 FP32 path 快多少？**
> 方法铁律：只看真实 kernel 计时（cudaEvent 包络 back-to-back launch），不用
> occupancy/register/理论吞吐排名；Decode / Prefill 分开测；每个 op 按 workload
> 分类（small=launch-bound / large=bandwidth-bound）。RMSNorm / SiLU / matmul 的
> 对照已在各自 6-A 交付中实测记录，本轮不重复。

**新增**

| 文件 | 改动 |
|---|---|
| `tests/bench_fp16_baseline.cpp`（新） | Phase 2 第一轮基准：attention decode / prefill（FP16 核 vs FP32 核，kv/q sweep）+ 6-A-8.0 elementwise ADD/MUL（两 regime）。可执行 `forge-bench-fp16-baseline`，不进 CTest（沿用 tiled-attn 约定） |
| `tests/CMakeLists.txt` | 注册 `forge-bench-fp16-baseline`（构建但手动运行） |

**实测（RTX 4050 Laptop，cudaEvent，warmup 后多 iter 平均）**

attention decode（q_len=1，GQA nh=8 nkv=2 hd=64，FP16 decode 核 vs FP32 decode 核）：

| kv_len | FP16 ms/tok | FP32 ms/tok | speedup |
|---|---|---|---|
| 128（launch-bound） | 0.0132 | 0.0131 | **0.99x** |
| 512 | 0.0148 | 0.0194 | **1.31x** |
| 2048（bandwidth-bound） | 0.0249 | 0.0439 | **1.76x** |

attention prefill（q=kv causal，FP16 vectorized vs FP32 vectorized）：

| q_len | FP16 ms | FP32 ms | speedup |
|---|---|---|---|
| 64（小） | 0.0685 | 0.0744 | **1.09x** |
| 256 | 0.3639 | 0.4545 | **1.25x** |
| 1024（大） | 2.8612 | 4.0039 | **1.40x** |

elementwise（6-A-8.0，FP16 vs FP32，traffic = 2 读 + 1 写）：

| op | n | FP16 | FP32 | speedup |
|---|---|---|---|---|
| add / mul | 4096（launch-bound） | ~0.0066 ms | ~0.0064 ms | **~0.97x** |
| add | 16M（bandwidth-bound） | 0.5503 ms / 182.9 GB/s | 1.1672 ms / 172.5 GB/s | **2.12x** |
| mul | 16M（bandwidth-bound） | 0.5579 ms / 180.4 GB/s | 1.1692 ms / 172.2 GB/s | **2.10x** |

**第一轮结论**：

1. **增益来自带宽，不来自算力**：workload 越接近 bandwidth-bound，FP16 增益越大
   （decode 长上下文 1.76x、elementwise 2.1x≈字节数减半的理论比 2.0x 一致）；
   launch-bound 区（kv=128、n=4096、q=64）FP16 ≈ 持平（0.97–1.09x），不亏。
2. **decode 增益随 kv 增长**（0.99x → 1.31x → 1.76x）：KV-cache streaming 是 decode
   主导成本，FP16 KV 减半读取字节直接兑现；512 点处于 launch/bandwidth 过渡带，
   运行间波动（1.31–1.68x）属正常。
3. **prefill 增益温和**（1.09x → 1.40x，随 q 增大）：vectorized FP32 核本身效率较高，
   FP16 的收益主要在 K/V 读取减半；QK^T/PV 仍是 FP32 计算（既定数值策略）。
4. **基线已冻结**：以上数字即 `__half2` / TensorCore 候选（第二轮）必须击败的基线；
   launch-bound 区无收益 → 候选优化应聚焦 bandwidth-bound 区（half2 向量化 K/V 加载、
   TensorCore MMA），预期增益上限 = 进一步减少的访存 + 计算单元切换收益。

**验收（回归）**：CTest **36/36 全过**（本轮未改任何库代码，仅新增 bench）。

#### Phase 2 第二轮（2026-09-06，Decode `__half2` candidate ✅ 实测完成 —— 负结果，归档）

> 回答唯一问题：**`__half2` candidate 是否真的比冻结 baseline 快？**
> 答案：**否。全部 regime ≈1.00x，candidate 未胜出；production path 维持 6-A-7 baseline 不变。**
> TensorCore/MMA 候选未开工。

**candidate 设计（契约内，不破 FP32 math）**

| 文件 | 改动 |
|---|---|
| `src/operators/cuda/cuda_flash_attn.cu`（+`flash_attn_gqa_decode_kernel_fp16_h2`） | 候选核：`__half22float2` 成对转换（替代 8× 标量 cvt）+ V 行 16B 加载前置（与 QK 计算重叠访存延迟）。**明确不用 HFMA2**——half2 乘积会引入 FP16 舍入，破坏 6-A-7 锁定的「QK^T/softmax/PV 全 FP32」契约 |
| 同上（+`launch_flash_attention_gqa_decode_fp16_h2`） | bench/test 专用入口；**不接 selector/dispatch**，production path 不动 |
| `include/forge/cuda_kernels.h` | 候选声明（注明 bench-only 与不逐位原因） |
| `tests/test_attention_fp16.cpp`（+`check_decode_h2_candidate`） | 三闸门：① candidate legacy/capture stream A/B **逐位**；② vs baseline `max_abs<1e-3`（≤1 FP16 输出 ULP）；③ vs CPU FP32 ref parity。模板覆盖 hd∈{16,32,64,128}、kv∈{128,512,2048} |
| `tests/bench_fp16_baseline.cpp`（+`bench_decode_h2`） | A/B 段：同配置 candidate vs baseline，**best-of-5 交替测量**（内层口径不变：warmup+cudaEvent+mean） |

**实测（RTX 4050 Laptop，best-of-5 交替）**

| kv_len | baseline ms/tok | h2 candidate ms/tok | h2 vs baseline |
|---|---|---|---|
| 128（launch-bound） | 0.0088 | 0.0089 | **0.991x** |
| 512（transition） | 0.0110 | 0.0112 | **0.988x** |
| 2048（bandwidth-bound） | 0.0250 | 0.0250 | **1.001x** |

复跑第二次：0.993x / 0.988x / 0.998x —— 稳定中性。

**两个实测发现（比数字更有价值）**：

1. **顺序测量的时钟态伪影**：最早一次 A/B（顺序各测一次）给出 kv=512 = 1.394x 的「大胜」，
   复跑掉到 1.08x；改 best-of-5 **交替**测量后稳定在 0.988x。原因：顺序测量时两个核落在
   不同 clock/thermal 态。**教训固化：凡 A/B 必须交替 + best-of-N，单次顺序测量不作为结论依据**
   （round 1 的 kv=512 = 1.31x 同样受此影响偏乐观，结论方向不变但幅度应以本轮协议为准）。
2. **逐位一致假设被证伪**：candidate 与 baseline 求和顺序文本相同，但 nvcc 的 FMA 收缩决策
   依赖操作数来源（baseline 直读 global，candidate 读 hoisted `v8[]` 寄存器数组），两次编译
   收缩选择不同（如 2×FMUL+FADD vs FFMA），单元素出现 1 FP16 ULP 差（实测最大 4.8e-7）。
   因此正确性闸门定为「≤1 ULP（1e-3）+ stream A/B 逐位」，而非跨编译逐位。

**第二轮结论**：

1. 契约内（FP32 math）的 `__half2` 指令级杠杆已在 decode 上穷尽：cvt 成对化与加载前置均无
   稳定增益——decode 在大 kv 处带宽饱和（round 1 已证）、小 kv 处 launch 主导、过渡区也无
   稳定赢面。**decode FP16 路径视为到达带宽地板，收口。**
2. 剩余理论杠杆只有两个，都不针对 decode：HFMA2（需契约变更，需用户拍板）与 TensorCore/MMA
   （本质是算力换 bandwidth 的 prefill 杠杆）。decode 不因它们受益（字节已到 FP16 下限）。
3. candidate 归档为 measured dead end：代码保留（bench/test 专用入口），不进 production。

**验收（回归）**：CTest **36/36 全过**（attn-fp16 含新增 candidate 闸门）。

**下一拍 = Phase 2 第三轮**：TensorCore/MMA candidate，目标 prefill（compute 密集侧）；
decode 不再投入。若 prefill 实测同样接近带宽地板，则 Phase 2 可以提前收口——FP16 的收益
已由存储流量减半全部兑现。

#### Phase 2 第三轮（2026-09-06，Prefill TensorCore/wmma candidate ✅ 实测完成 —— 稳定正收益）

> 回答唯一问题：**wmma candidate 是否真的比冻结 baseline 快？**
> 答案：**是。全 regime 稳定正收益（q=64 ≈2.5–2.8x、q=256 ≈1.45x、q=1024 ≈1.09x），
> 两次独立运行可重复；production path 是否切换待用户拍板（见轮末决策点）。**

**candidate 设计（含一项契约 amendment）**

| 文件 | 改动 |
|---|---|
| `src/operators/cuda/cuda_flash_attn.cu`（+`flash_attn_gqa_kernel_mma_fp16`） | wmma 候选核：`16x16x16 f16` 操作数 + `f32` accumulator；Br=32（2 warp × 16 q 行）/ Bc=32 KV tile；Q fragment 跨 KV 循环驻留寄存器；在线 softmax 全 FP32（max/exp/sum/重缩放），P 以 f16 存 smem 供 PV wmma 使用 |
| 同上（+`launch_flash_attention_gqa_prefill_mma_fp16`） | bench/test 专用入口；**不接 selector/dispatch**，production path 不动；仅 HEAD_DIM=64 |
| `include/forge/cuda_kernels.h` | 候选声明（注明 P-f16 契约 amendment 与 bench-only） |
| `tests/test_attention_fp16.cpp`（+`check_prefill_mma_candidate`） | 三闸门：① candidate stream A/B **逐位**；② vs baseline `max_abs<5e-3`；③ vs CPU FP32 ref `max_abs<1e-2`。覆盖 q∈{16,64,48,64}×kv∈{64,64,80,128}（含 q/kv 尾部非对齐、kv>q） |
| `tests/bench_fp16_baseline.cpp`（+`bench_prefill_mma`） | candidate vs baseline，**best-of-5 交替测量**（沿用第二轮固化的协议） |

**契约 amendment（本轮唯一新增舍入）**：TensorCore 要求 PV 的 P 操作数为 f16（baseline
中 P 一直是 FP32 计算后直接参与加权）。QK^T 乘积仍精确（11+11 mantissa < 24），softmax
全 FP32，PV accumulate 全 FP32；唯一新增舍入是 P→f16 一次量化，量级 ≤2^-11 相对误差，
实测被 vs-baseline 闸门（5e-3）覆盖。**该 amendment 只存在于 candidate；production
baseline 契约不变，切换与否随 promote 决策一并生效。**

**实测（RTX 4050 Laptop，best-of-5 交替）**

| q=kv（causal，nh=8 nkv=2 hd=64） | baseline ms | mma candidate ms | mma vs baseline |
|---|---|---|---|
| 64（launch/occupancy 主导） | 0.0508 | 0.0184 | **2.766x** |
| 256（transition） | 0.2544 | 0.1744 | **1.459x** |
| 1024（大 workload） | 2.1603 | 1.9801 | **1.091x** |

复跑第二次：2.494x / 1.450x / 1.089x —— 两次独立运行同一水平，**确认非时钟态伪影**
（对比第二轮 1.394x→0.99x 的塌缩模式，本轮没有出现）。

**实测解读**：
1. 小 q 处收益最大的主因不是 MMA 吞吐本身，而是**grid 形态改变**：baseline「1 block/q 行」
   在 q=64 时只有 64×8 个 block、每 block 串行扫全 KV；candidate 改为 q-tile（Br=32）后
   并行度与 KV 内并行都更均衡，加上 wmma 替换标量 QK/PV 点积。
2. q=1024 处收益收敛到 ~1.09x：candidate 配置（Br=32/Bc=32/2 warp/64 thread）偏小，
   smem 与 wmma 利用率未饱和；有明确续作空间（更大 tile、4 warp、hd 分片），但那属于
   「promote 之后再调」的性能迭代，不在本轮。
3. correctness 侧：max_abs vs baseline 最大 4.883e-04（闸门 5e-3）、vs FP32 ref 最大
   4.898e-04（闸门 1e-2），stream A/B 逐位一致 4/4（含尾部非对齐）。

**验收（回归）**：CTest **36/36 全过**（attn-fp16 含新增 candidate 闸门）。

**决策点（待用户拍板）**：
- 方案 A（promote）：把 wmma prefill 接入 selector（如 `CUDA_F16_PREFILL` 增加 mma 路径
  或 workload-aware 切换点），按 6-A-7 相同验收链（selector parity + graph capture/replay
  + e2e）重跑封版。
- 方案 B（先续作性能迭代）：promote 前先把 Br/Bc/warp 数调优（预期把 q=1024 从 1.09x
  拉开），一次性以更强配置进 production。
- 两案共同前提：P-f16 amendment 随 promote 生效；decode path 不受影响（维持 6-A-7 baseline）。

> **用户拍板（2026-09-07）：方案 B。** Round 4 = MMA Prefill tile/config autotuning；
> Round 4 结束后再决定 promote。固定 hd=64/wmma/FP16 QKV/FP32 acc/P→FP16 amendment；
> 只扫 Br∈{32,64} × Bc∈{16,32,64} × warps∈{2,4}；不做 ldmatrix / cp.async / softmax
> redesign / FP16 KV / decode MMA / selector promotion。promote 门槛：最佳配置
> q=64>1.10x、q=256>1.10x、q=1024>1.05x 且 correctness/graph 全绿，否则归档。

#### Phase 2 第四轮（2026-09-07，MMA Prefill tile/config scan ✅ 实测完成 —— 门槛全过，promote 成立）

> 回答 Round 4 唯一问题：**q=1024 的 1.09x 是 TensorCore 的上限，还是 tile 配置的上限？**
> 答案：**是配置的上限（叠加一个 dynamic-smem 实现伪影），不是 TensorCore 的上限。**
> 最佳单一配置 `Br=64 / Bc=16 / 4 warps` 全档 2.44x / 2.35x / **1.45x**，两次扫描复现；
> 用户设定的 promote 三门槛全过 → **promote 成立**（接入 selector 为下一拍，待启动确认）。

**改动表**

| 文件 | 改动 |
|---|---|
| `src/operators/cuda/cuda_flash_attn.cu` | ① kernel 主体参数化为 `mma_prefill_body<HD,BR,BC,NW>`（`__device__ __forceinline__`，warp 拥有 BR/NW 行、M_FRAGS 个 16 行 fragment；softmax lane 覆盖 BC/2 列；staging 泛化为 TPR_Q/TPR_K）；② 两个 wrapper：`flash_attn_gqa_kernel_mma_fp16`（static smem，≤48KB）与 `_dyn`（extern carve，>48KB 才用）；③ `launch_flash_attention_gqa_prefill_mma_cfg` 显式 config 入口（9 合法组合，非法组合 loud 失败）；④ 原 round-3 入口保持 (32,32,2) 语义不变（实测数值逐位一致） |
| `include/forge/cuda_kernels.h` | cfg 入口声明（注明扫描集与 Br=32&4w 排除原因：wmma M=16 要求每 warp ≥16 行） |
| `tests/test_attention_fp16.cpp` | `check_prefill_mma_cfg_scan`：round-3 三闸门（stream A/B 逐位 + vs baseline 5e-3 + vs ref 1e-2）× 9 config × 4 shape（含 q/kv 尾部非对齐、kv>q），每 config 输出聚合最差值 |
| `tests/bench_fp16_baseline.cpp` | `bench_prefill_mma_scan`：9 config × q∈{64,256,1024}，每 config 内部 baseline/candidate **严格交替 best-of-5**（协议不变），输出全表 + 每 q 最佳 |

**Round 4 关键发现（比数字更重要）：dynamic smem 实现税 ≈20-25%**

第一版重构把所有 config 统一改走 dynamic smem（为支持 (64,64) 的 66-68KB），结果
`(32,32,2)` 在 q=1024 只测得 0.898x——与 Round 3 的 1.091x 矛盾（baseline 仅漂移 ~3%，
纯时钟态解释不了 25% 的 candidate 端劣化）。修复：主体抽成 device body，≤48KB 的 config
全部走 static smem（编译期已知共享地址），仅 (64,64) 两个 config 付 dynamic 税。修复后
`(32,32,2)` 立即回到 1.092x（R3 = 1.091x，完全吻合）。

**教训固化：同一 kernel 的 static vs dynamic smem 不是零成本别名，bench kernel 禁止
为了实现便利统一改走 dynamic smem；需要 >48KB 时必须把 static/dynamic 作为被测变量
分开呈现。**

**扫描实测（RTX 4050 Laptop，best-of-5 交替，两次独立运行）**

q=64（launch/occupancy 主导，最佳 = 小 tile 多 block）：

| config | run1 | run2 |
|---|---|---|
| **Br=32 Bc=16 w=2** | **2.880x** | **2.898x** |
| Br=32 Bc=32 w=2（R3 shape） | 2.733x | 2.724x |
| Br=64 Bc=16 w=4 | 2.428x | 2.440x |

q=256（最佳 = 4 warp 小 Bc）：

| config | run1 | run2 |
|---|---|---|
| **Br=64 Bc=16 w=4** | **2.343x** | **2.348x** |
| Br=64 Bc=32 w=2 | 2.073x | 2.101x |
| Br=32 Bc=32 w=2（R3 shape） | 1.403x | 1.428x |

q=1024（最佳同为 Br=64 Bc=16 w=4）：

| config | run1 | run2 |
|---|---|---|
| **Br=64 Bc=16 w=4** | **1.451x** | **1.449x** |
| Br=32 Bc=16 w=2 | 1.269x | 1.226x |
| Br=64 Bc=32 w=2 | 1.203x | 1.185x |
| Br=32 Bc=32 w=2（R3 shape） | 1.092x | 1.095x |
| Br=64 Bc=64 w=2（dynamic） | 0.677x | 0.658x |

解读：
1. **单一配置 `Br=64/Bc=16/4w` 即可通过全部 promote 门槛**：2.43/2.44x（>1.10）、
   2.34/2.35x（>1.10）、1.45/1.45x（>1.05）。无需 workload-aware 多配置切换即可 promote；
   若 selector 想做 workload-aware，小 q 段（q≤64）`(32,16,2)` 略优（2.88 vs 2.44），
   但差异不值得双配置复杂度，第一版 promote 用单一 config。
2. **Bc=16 出人意料地强**：Bc 大（32/64）反而伤——更大的 smem tile 换来的 MMA 复用
   抵不过在线 softmax 的 serial 依赖链变长（每 tile 的 max/exp/P-store 批量更大）与
   occupancy 下降。(64,64,2) 全档最差（0.66-0.69x），smem 66KB → 1 block/SM。
3. **4 warp 轴只在大 q 有意义**：q=64 时 (64,16,4) 2.44x < (32,16,2) 2.88x（block 数
   减半伤并行度）；q≥256 后 4 warp 全面胜出。
4. (64,64,4) 的 correctness 误差 2.5e-3 仍是全组最大（闸门 5e-3/1e-2 内），源于 Bc=64
   的更长在线 softmax 链，非 bug。

**验收（回归）**：CTest **36/36 全过**；9 config correctness 三闸门全绿
（最差 vs ref 2.544e-03 / vs baseline 2.533e-03，均来自 (64,64,4)；stream A/B 逐位 9/9）。

**promote 判定（按用户门槛）**：`Br=64/Bc=16/4w` = 2.44x / 2.35x / 1.45x，
q=64>1.10 ✓、q=256>1.10 ✓、q=1024>1.05 ✓、correctness/graph 全绿 ✓ → **promote 成立**。
P-f16 amendment 随 promote 生效。

**非目标（明确未做）**：ldmatrix 手写化、cp.async、fused softmax redesign、FP16 KV-cache、
decode MMA、selector promotion（= 下一拍 Round 5）、workload-aware 多配置 selector。

**下一拍 = Phase 2 第五轮（Round 5）：promote `Br=64/Bc=16/4w` 到 production path**——
接入 selector（prefill FP16 路径），按 6-A-7 验收链重跑（selector parity + dtype trace +
CUDA Graph capture/replay + e2e-fp16），P-f16 amendment 写入 dtype contract 文档后封版。

#### Phase 2 第五轮（2026-09-07，MMA Prefill promote ✅ 实测完成 —— COMPLETE / FROZEN）

**目标**：把 Round 4 扫描胜者 `Br=64/Bc=16/4w` 接入 production path，重跑 6-A-7 验收链，
写入 P-f16 amendment，封版。

**改动表**：

| 文件 | 改动 |
|---|---|
| `src/operators/cuda/cuda_flash_attn.cu` | 原向量化 FP16 prefill 抽为显式入口 `launch_flash_attention_gqa_prefill_fp16_vectorized`（冻结基线，可 A/B 可回退）；生产入口 `launch_flash_attention_gqa_prefill_fp16` 改为条件分发：`hd==64 && causal && mask==nullptr` → `launch_mma_prefill_cfg_impl<64,64,16,4>`，其余 scope 一律 fallback 向量化 kernel（行为与 6-A-7.2 完全一致） |
| `include/forge/cuda_kernels.h`、`src/operators/cuda/cuda_flash_attn.h` | 新增 `_vectorized` 显式入口声明；更新 round-3/4 入口注释（"NOT wired" 已过时） |
| `include/forge/dtype_contract.h` | P-f16 amendment 写入 6-A-7 注释块：PV 的 P 操作数存 f16（≤2^-11 相对误差/元素），QK^T f16*f16 在 f32 精确、softmax/PV 累加仍 f32；仅限 hd=64 causal 无 mask scope |
| `tests/test_attention_fp16.cpp` | round-3/4 基准臂改用显式 `_vectorized` 入口（保持 A/B 语义纯净）；新增 `check_prefill_production_routing`：生产入口 hd=64 输出与显式 cfg 入口 (64,16,4) **逐位一致**，hd=128 输出与显式向量化入口**逐位一致**（含 q/kv 尾部 shape 48/80） |
| `tests/bench_fp16_baseline.cpp` | 全部基线臂改用显式 `_vectorized` 入口 |

**接入面说明**：selector 决策不变（`CUDA_F16_PREFILL`），executor 与 `op_kernels.cpp`
graph 路径都调用同一生产入口，因此 eager、CUDA Graph capture/replay、e2e 三链自动走
MMA kernel，无需改动 dispatch 层；mask/非 causal/其他 head_dim 自动回落冻结基线。

**实测数据（Round 5 时点复跑，best-of-5 交替 A/B，同一协议）**：

| q=kv | (64,16,4) vs 向量化基线 | 门槛 | 判定 |
|---|---|---|---|
| 64 | 1.964x | >1.10x | ✓ |
| 256 | 2.312x | >1.10x | ✓ |
| 1024 | 1.476x | >1.05x | ✓ |

复跑备注：q=64 档本轮 best 为 (32,16,2)=2.879x（Round 4 为 2.44x），小 q 档对时钟态敏感、
波动可见；(64,16,4) 全档 ≥1.4x 稳过门槛，维持单一 production 配置不做 workload-aware
多配置（非目标）。

**验收结果（全部实测）**：
- `attn-fp16`：PASS。selector 单测 OK；decode A==B 全绿；round-3/4 gates 全绿
  （(64,16,4)：vs ref 4.883e-04 / vs baseline 4.883e-04 / streamA==B=1）；
  production routing 两条逐位断言通过；`check_prefill`（现走生产入口→MMA）
  max_abs=9.228e-05 < 1e-2；`check_graph_node_prefill` eager/capture/replay 逐位 OK
  （capture/replay 现经生产入口 → wmma kernel）。
- `e2e-fp16`：PASS。dtype trace 主链无隐式回退；FP16 vs FP32 logits parity 固定阈值内；
  全图 CUDA Graph ordinary→capture→replay 逐位稳定（prefill/decode 两模式）。
- CTest：**36/36 通过**（含 attn-fp16、e2e-fp16、dtype-contract、graph-fp16、matmul-fp16 全家）。

**非目标（明确未做）**：ldmatrix 手写化、cp.async、fused softmax redesign、FP16 KV-cache、
decode MMA、workload-aware 多配置 selector、Br=32/Bc=16 按档分派。

**封版声明**：Phase 2（Round 1–5）COMPLETE / FROZEN。
- Round 1（FP16 baseline vs FP32）：promote，6-A-7。
- Round 2（decode `__half2`）：负结果归档，维持 6-A-7 baseline。
- Round 3（wmma prefill Br=32/Bc=32/2w）：正结果，未接 production（被 Round 4/5 取代）。
- Round 4（tile 扫描）：胜者 (64,16,4)，promote 判定成立。
- Round 5（promote）：生产入口接入 (64,16,4) + P-f16 amendment 入档，验收链全绿。

**Deferred（后续候选，未排期）**：q=64 档 (32,16,2) 的 workload-aware 分派、更大 tile/
hd 分片续作、FP16 KV-cache、paged FP16 attention、half2 decode 收口结论沿用 Round 2。

### 6-B. Native BF16 Execution

紧随 6-A 的第二步。BF16 枚举已存在但实际是 `BF16 → load → FP32`（`types.h:15` 注释即
自认）：真正要做的是

#### Phase 6-B-0（2026-09-07，BF16 提升点源码审计 ✅ 只审计未改码）

**方法**：`DataType::BF16` / `GgmlDType::BF16` 全量出现点（include+src 共 40 处）逐一定性；
辅以 `CUDA_R_16BF` / `CUBLAS_COMPUTE_32F` 基建查证与 `static_cast<const float*>` 误读面扫描。
**结论先行：与 6-A-0 时期形态一致——"表面支持 BF16"，唯一正常入口（GGUF 加载）在
load time 就转成 FP32，运行时链路从不真正见到 BF16；且存在 3 组「BF16 可达 → 当 FP32
静默读」的 P0 漏洞，与 6-A-0 #8/#10 同型。**

**链路审计表**（用户指定覆盖面）：

| # | 层 / 文件 | BF16 现状（实测） | 定性 |
|---|---|---|---|
| 1 | `types.h:25` | `BF16 = 15`，注释自认 "converted to FP32 at load time" | 事实声明 |
| 2 | `quant_traits.cpp:52`（DataTypeTraits） | name="bf16"、size=2 ✓（Tensor nbytes 分配正确）、dequant=nullptr | 基础无害 |
| 3 | `compute_dtype_policy.h` | 仅 `fp32_policy()` 工厂；无 BF16 policy 组合 | 6-B-1 缺口 |
| 4 | `dtype_contract.h:31-33,58-79` | `is_native_execution_dtype` 排除 BF16 ✓；但 `op_supports_lowp` 白名单 8 算子（RMS_NORM/SILU/GELU×2/MUL_MAT_TRANSB/FLASH_ATTN_GQA/ADD/MUL）**对 BF16 同样放行**——gate 只区分 "FP32/量化 vs lowp"，不区分 FP16/BF16 | **风险源头** |
| 5 | `gguf_model.cpp:485-496`（graph/binding 上游） | BF16 权重 load time 手工位左移 16 位转 FP32 tensor，运行时不可见 BF16 | 唯一正常消费点 |
| 6 | Python bindings `core_types.cpp:27,278` | `DataType.BF16` 已导出 + `Tensor.from_buffer(dtype, shape, device)` 接受任意 dtype → **Python 可直接注入 BF16 CUDA tensor** | 可达性证据 |
| 7 | matmul `matmul_cuda.cpp:325-357`（cuda_matmul_transB/_dual） | **注释声称 BF16 "rejected here loudly"，实际代码不存在该拒绝**：gate(L329) 放行 → `a==FP16&&b==FP16` 不命中 → L350 `static_cast<const float*>` 静默读。注释与代码不符 | **P0** |
| 8 | RMSNorm graph 路径 `op_kernels.cpp:147-209`（rms_norm_kernel_dst） | `if FP16{...} else: require_dtype_supported（BF16 放行）→ launch_rms_norm(float*)` 静默读。注：eager `ops::rms_norm`（norm.cpp:79）是 loud throw ✓——**graph dst-injection 路径绕过了 ops 层的 loud 兜底** | **P0** |
| 9 | activation graph 路径 `op_kernels.cpp:109-142`（silu_kernel_dst） | 同上：gate 放行 BF16 → `launch_silu(float*)` 静默读。eager `ops::silu`（activation.cpp:38）loud ✓ | **P0** |
| 10 | elementwise graph 路径 `op_kernels.cpp:27-106`（add/mul_kernel_dst） | 同上：6-A-8.0 白名单放行 BF16 → `launch_add_bias/launch_multiply(float*)` 静默读。eager ops 层 loud ✓ | **P0** |
| 11 | attention `op_kernels.cpp:300-380`（flash_attn_gqa_kernel） | FP16 分支有 `q->dtype()==FP16` 守卫 ✓；FP32 GQA 分支（L361-380）**无 dtype 检查**，BF16 QKV → `static_cast<const float*>` 静默读 | **P0** |
| 12 | attention executor `attention_executor.cpp:593-608,636-650` | selector（attention_selector.cpp:25-28）只认 `q_dtype==FP16`，BF16 落默认 `CUDA_FP32_PREFILL/DECODE` → `static_cast<const float*>(q->data())` 无检查 | **P0** |
| 13 | ROPE `op_kernels.cpp:234-260`（rope_kernel_dst） | **完全无 gate、无 FP16/BF16 分支**，任意非 FP32 dtype 静默读（ROPE 不在白名单但也无人调用 require） | P1 |
| 14 | CPU attention `attention.cpp:43-132` | `static_cast<const float*>` 无检查（CPU 侧 BF16 同样可静默误读；CPU 其余算子均 FP32-only+loud ✓） | P1 |
| 15 | GEMV/MMQ 表 `cuda_gemv_instances.cu:78,102`、`cuda_mmq.cu:1868` | BF16 槽位 = nullptr（与未实现 IQ 槽位同型，null 即失败） | 绿 |
| 16 | KV cache `kv_cache.h:11-18` | `KVCacheDType` 枚举无 BF16 成员；`parse_kv_dtype`/kv_policy 无 "bf16" 串 | 绿（不存在即无法误用） |
| 17 | cuBLAS 基建 `cuda_quant.cu:1143-1212` | 仅 `CUDA_R_16F/32F` + `CUBLAS_COMPUTE_32F`；**全仓无 `CUDA_R_16BF`、无 `__nv_bfloat16`**（`cuda_bf16.h` 未 include） | 6-B-4 从零建 |
| 18 | CUDA Graph runner / dtype trace | dtype-agnostic（capture/replay 与 trace 不检查 dtype）——风险不在 runner，全部落在 #8-#12 的 kernel/gate 层 | 中性 |
| 19 | 测试 `test_dtype_contract.cpp:121` | 仅测 `is_native_execution_dtype(BF16)==false`；**无「BF16 到达任一白名单算子 → 期望 throw」的端到端测试** | 6-B-1 缺口 |

**核心判定**：

1. gate（`op_supports_lowp`）与 routing 的分工假设失效——6-A 注释多处声称 "BF16 …
   rejected downstream by the CUDA dtype routing"，但 dst-injection 路径的 downstream
   没有 BF16 throw（#8-#10），matmul 的 "rejected loudly" 注释根本没有对应代码（#7）。
   gate 放行 + FP32 fallback 无二次校验 = 静默误读，与 6-A-0 #8/#10 同型。
2. 正常模型加载路径不可达（#5 load time 转换），但 contract 承诺的是 **operator 边界
   loud 失败**；#6 证明 BF16 tensor 可从 Python/C++ 直接注入 → 漏洞真实可达。
3. 6-B 基建零存量：无 bf16 头、无 CUDA_R_16BF、无任何 bf16 kernel；但 6-A 的全部
   基建（policy/contract/trace/cuBLAS gemv_fp16 范式/验收链）可镜像复用。

**6-B-1 的正确动作（不是写 kernel）**：

```text
1. dtype_contract.h: is_native_execution_dtype / op_supports_lowp 不动，
   新增显式 "BF16 尚无 kernel" 拒绝点（或在 gate 内区分 FP16/BF16）。
2. 修 P0 四处：matmul_cuda (#7)、dst-injection ADD/MUL/SILU/RMS_NORM (#8-#10)、
   flash_attn_gqa_kernel FP32 分支 (#11)、attention_executor FP32 路径 (#12) ——
   gate 之后、float* cast 之前对 BF16 loud throw。
3. 修 P1 两处：ROPE gate 缺失 (#13)、CPU attention (#14)。
4. test_dtype_contract 补端到端用例：BF16 tensor → 每个白名单算子 → 期望 throw。
5. 全绿后（CTest 36/36）才进入 6-B-2 native BF16 RMSNorm。
```

**验收原则（沿用 6-A 纪律，不变）**：BF16 storage → native operator → FP32
accumulation/reduction → BF16 out；`activation dtype ≠ reduction dtype`、
`storage precision ≠ accumulator precision`；每个 native kernel 保留
「native BF16 ⇄ FP32 reference（reference 用相同 BF16 解码输入）」链式验证，
隔离算法错误与 BF16 存储舍入。

#### 6-B-1 contract hardening（2026-09-07，本轮交付 ✅）

> 按 6-B-0 审计的处方执行：**不实现任何 BF16 native kernel**，只把
> "BF16 可达 → FP32 静默误读"全部封死，并把 contract 行为锁进测试。
> 与 6-A-1 完全一致的纪律：**先封 contract，再增加能力。**

**设计（两层分工，`op_supports_lowp` 语义不动）**：gate 只区分
"FP32/量化 vs lowp"，不区分 FP16/BF16——8 个白名单算子对 BF16 全部放行。
新增第二层显式 BF16 闸门 `throw_bf16_unsupported` / `require_no_bf16`
（`dtype_contract.h`），每个 P0/P1 落点在 gate 之后、`float*` cast 之前调用。
非 BF16 dtype 上为 no-op（FP16/FP32/量化零行为变化）；6-B-2 起每落地一个
`__nv_bfloat16` kernel 即逐 op 退役对应闸门。

**落点（全部"gate 之后、cast 之前"）**

| ID | 文件 | 改动 |
|---|---|---|
| P0-1 | `src/operators/cuda/matmul_cuda.cpp` | `cuda_matmul_transB`（a/b/out）+ `_dual`（a/b1/b2/out）加 `require_no_bf16`；修正"rejected loudly"注释（此前无对应代码） |
| P0-2/3/4 | `src/core/op_kernels.cpp` | `add/mul_kernel_dst`（a+b）、`silu_kernel_dst`（x）、`rms_norm_kernel_dst`（x+weight）在 FP16 分流前加 `require_no_bf16`；eager `ops::add/multiply`（`elementwise.cpp`，此前零 gate）补 gate + BF16 拒绝 + 非 FP32 显式 throw |
| P0-5 | `src/core/op_kernels.cpp` + `src/inference/layers/attention_executor.cpp` | `flash_attn_gqa_kernel` 的 FP32 GQA / MHA / CPU 三分支入口加 `require_no_bf16(q/k/v)`；executor 的 `CUDA_FP32_PREFILL/DECODE` 在 `get_kv_on_device` 后加 `require_no_bf16(q/k/v)`（含 CPU `#else` 回退，守卫在 `#ifdef` 之外） |
| P1-1 | `src/core/op_kernels.cpp` + `src/core/compute_graph.cpp` | `rope_kernel_dst` 入口加 `require_dtype_supported(ROPE)` + `require_no_bf16` + 非 FP32 显式 throw（ROPE 不进白名单，维持 FP32-only）；dst 准入闸 `inputs_fp16` 扩展为"FP16 或 BF16 均禁入 FP32-only dst kernel"（纵深，kernel 内 throw 为第二道） |
| P1-2 | `src/operators/cpu/attention.cpp` | `scaled_dot_product_attention` / `_2d` / `_2d_gqa` 入口加 `require_no_bf16(q/k/v[/mask])`——同时覆盖 executor 的 `CPU_MHA/CPU_GQA/CUDA_GENERIC_MHA` 三路径（均经此 funnel）；FP16-on-CPU 维持现状（6-B-1 范围外，已记录） |

**测试**

- `tests/test_dtype_contract.cpp` 新增 `test_bf16_gate`（host-only）：12 个 op（8 白名单 + MUL_MAT/ROPE/SOFT_MAX/EMBEDDING）的 BF16 必 throw（消息含 op 名 + "bf16"），FP32/FP16/量化必 no-op；两层分工不变式（白名单对 BF16"gate 放行 + BF16 闸门拒绝"，6-B-2 起逐 op 更新）。
- 临时 smoke（验证后移除，未入库）：BF16 CUDA tensor 直注 `mul_mat_transB` / graph ADD/RMS_NORM/SILU/ROPE / CPU `sdpa_2d_gqa` 六点全部 loud throw 且消息含 "bf16"；FP32 graph ADD 正常——**不存在 BF16 → float\* 静默 fallback**。
- CTest **36/36 全绿**（RTX 4050 Laptop，含 `dtype-contract` 与全部 FP16 全家；FP16/FP32/Q4/Q5/Q6/Q8 零回归）。

**验收对照**

```text
1. P0/P1 所有可达 BF16 路径均 loud reject ............ ✅（上表 5 行）
2. 不存在 BF16 → float* 静默 fallback ................ ✅（smoke 六点 + cast 前必有 require 审查清单）
3. test_dtype_contract 覆盖新增拒绝面 ................ ✅（test_bf16_gate）
4. FP16 / FP32 / Q4/Q5/Q6/Q8 既有行为零回归 ......... ✅（闸门非 BF16 no-op + CTest 36/36）
5. CTest 全绿 ........................................ ✅（36/36）
6. docs 写入 6-B-1 delivered/frozen 记录 ............. ✅（本节）
```

**封版声明**：6-B-1 = DELIVERED / FROZEN——本轮未新增任何 kernel、未改任何非 BF16 数值路径。下一拍 6-B-2（native BF16 RMSNorm，镜像 6-A-2：BF16 存储 + FP32 reduction）解锁。

#### 6-B-2 native BF16 RMSNorm（2026-09-07，本轮交付 ✅）

> 6-B 首个真实 `__nv_bfloat16` kernel。严格镜像 6-A-2 的验证纪律而非实现细节：
> **BF16 存储/算术 + FP32 reduction**（`norm_compute=FP32` 不变，零 policy 改动），数值稳定性不牺牲。
> RMS_NORM 是 6-B-1 `require_no_bf16` 的首个退役 op（调用点退役，闸门函数本身语义不动）。

**新增 / 修改**

| 文件 | 改动 |
|---|---|
| `src/operators/cuda/cuda_rms_norm.{cu,h}` | 新增 `rms_norm_kernel_bf16` + `launch_rms_norm_bf16`（加权）与 `rms_norm_unweighted_kernel_bf16` + `launch_rms_norm_unweighted_bf16`（无权重）。逐行镜像 FP16 版，仅 `__half/__half2float/__float2half` → `__nv_bfloat16/__bfloat162float/__float2bfloat16`；reduction 全 FP32 |
| `include/forge/cuda_kernels.h` | 声明上两个 launch（`#include <cuda_bf16.h>`）；注释写明 RMS_NORM 已从 `require_no_bf16` 退役 |
| `src/operators/cpu/norm.cpp` | `ops::rms_norm` 新增 `bf16_path`（BF16+BF16 weight → native；混合 dtype 落原 `else` loud throw，消息升级为 FP16/BF16 双路径口径）；CPU eager 保持 FP32-only loud |
| `src/core/op_kernels.cpp` | `rms_norm_kernel_dst` 在 `require_no_bf16` **之前**插入 BF16 分流（CUDA 走新 launch 并 honor stream，混合 weight 显式 throw；CPU 本地 upcast→FP32 row kernel→BF16 存储，镜像 FP16 分支） |
| `src/core/compute_graph.cpp` | 准入闸 `dst_bf16_capable = (op==RMS_NORM)`——BF16 输入仅对 RMS_NORM 放行注入，其余 op 保持 6-B-1 阻断 |
| `tests/test_dtype_contract.cpp` | 新增 `bf16_retired_ops` 台账 + `test_bf16_retirement`（退役 op 必属白名单；本轮恰为 RMS_NORM 一家）。`test_bf16_gate` 本体不动——退役是调用点属性，闸门函数继续对 12 op 全 throw |
| `tests/test_rms_norm_bf16.cpp`（新） | CTest `rms-norm-bf16`：FP32 reference 对照（hidden=4096/8192 × rows∈{1,32,128,512} + 无权重 + 混合 dtype loud）+ 单节点 graph smoke（dtype 跟随 + 与 eager 逐位一致）+ 微基准（三臂两 regime），无 GPU 优雅 SKIP |

**正确性（FP32 reference 对照：输入/权重先编码为 BF16 再喂 kernel，reference 用同一解码值，残差仅来自 BF16 输出存储舍入）**

| 配置 | max_abs_err | max_rel_err |
|---|---|---|
| hidden=4096/8192, rows=1..512（加权） | ~3.9e-3 | ~3.9e-3 |
| hidden=4096, rows=64（无权重） | ~3.9e-3 | — |
| 混合 BF16×FP32 | loud reject ✅ | — |

误差约为 FP16 版（~5e-4）的 8×——恰为 7-bit vs 10-bit mantissa 的理论比（2³），证实残差确为存储舍入而非算法偏差；阈值沿用 0.02/0.05（结构错误给 O(1) 垃圾，照样拦截）。

**单节点 graph smoke**：`dtype=bf16`，与 eager `max_abs_vs_eager=0.0`（逐位一致）——dst 注入 + 规划器 + 退役闸门未引入任何数值差异。

**微基准（kernel-only，两 regime，不预设收益）**

| 配置 | 类别 | BF16 | FP16（sanity） | FP32 |
|---|---|---|---|---|
| hidden=4096, rows=1 | launch-bound | 0.0080 ms | 0.0094 ms | 0.0093 ms |
| hidden=8192, rows=512 | bandwidth-bound | 0.0859 ms / 196 GB/s | 0.0841 ms / 200 GB/s | 0.2257 ms / 149 GB/s |

结论与 6-A-2 封版原则 #3 同构：small hidden 均 launch-bound（无差异）；large hidden 进 memory-efficiency 区间，BF16 约 **2.6×**（流量减半），且与 FP16 **逐位同字节数即同性能**（0.086 vs 0.084 ms）——收益来自带宽而非算力，`__nv_bfloat16` 转换指令零可测开销。

**验收**：CTest **37/37 全过**（新增 `rms-norm-bf16`；`dtype-contract` 含新退役台账；FP16 全家无回归）。

**封版声明**：6-B-2 = DELIVERED / FROZEN。退役协议已验证成立（调用点退役 + 台账锁 + native 测试为可执行证明），6-B-3（BF16 activation）起逐 op 复用同一流程。

#### 6-B-3 native BF16 activation family（2026-09-07，本轮交付 ✅）

> 镜像 6-A-3/6-A-4：`silu` / `gelu` / `gelu_tanh`（unary）+ `silu_multiply` /
> `gelu_multiply`（gated，主链真实入口：LLaMA/Qwen/DeepSeek 用 SiLU-gated、Gemma4 用
> GeGLU）。计算模型与 6-B-2 同构：BF16 存储 + FP32 超越函数计算 + BF16 输出。
> 退役 SILU + GELU + GELU_TANH 三个 `require_no_bf16` 调用点族（gated 变体复用
> SILU/GELU 闸门，无新枚举——与 6-A-4 一致）。

**新增 / 修改**

| 文件 | 改动 |
|---|---|
| `src/operators/cuda/cuda_activation.cu` | 新增 `silu/gelu/gelu_tanh_kernel_bf16` + `launch_*_bf16`（逐行镜像 `_fp16` 版，仅三处转换函数替换；256 线程 block 不变） |
| `src/operators/cuda/cuda_elementwise.cu` | 新增 `silu_multiply/gelu_multiply_kernel_bf16` + `launch_*_bf16`（fused 单 kernel，镜像 FP16 分支——FP32 的"gelu 再 multiply"两 kernel 形态不动） |
| `include/forge/cuda_kernels.h` | 声明上 5 个 launch |
| `src/operators/cpu/activation.cpp` | `silu/gelu/gelu_tanh` 加 BF16 分支；`else` 消息升级为 FP16/BF16 双路径口径；CPU 保持 FP32-only |
| `src/operators/cpu/elementwise.cpp` | `silu_multiply/gelu_multiply` 加 BF16 分支（含 `up` 非 BF16 混合显式 throw）；`else` 消息同步升级 |
| `src/core/op_kernels.cpp` | `silu_kernel_dst` 在 `require_no_bf16` 之前插 BF16 分流（CUDA 走新 launch 并 honor stream；CPU 本地 upcast→FP32→BF16） |
| `src/core/compute_graph.cpp` | `dst_bf16_capable` 从 `(RMS_NORM)` 扩为 `(RMS_NORM \|\| SILU)`；GELU/GELU_TANH 无 dst kernel（恒走 legacy → W2 分支），闸门不动 |
| `tests/test_dtype_contract.cpp` | `bf16_retired_ops` 追加 SILU/GELU/GELU_TANH（size 断言 1→4）；`test_bf16_gate` 本体不动 |
| `tests/test_activation_bf16.cpp`（新） | CTest `activation-bf16`：5 算子 FP32 reference 对照 + 输出 dtype 硬断言 + 混合 loud 抽查 + 单节点 SILU graph smoke + 三臂两 regime  bench，无 GPU SKIP |

**正确性（FP32 reference 对照：输入先编码为 BF16 再喂 kernel，reference 用同一解码值）**

| 配置 | max_abs | max_rel |
|---|---|---|
| silu / gelu / gelu_tanh（unary, n=8k/2M） | ~7.8e-3 | ~3.8e-3 |
| silu_multiply / gelu_multiply（gated） | ~3.1e-2 | ~3.9e-3 |
| 混合 BF16 gate × FP32 up | loud reject ✅ | — |
| 单节点 graph smoke（SILU） | 与 eager 逐位一致（0.0） | dtype=bf16 ✅ |

解读：unary 误差 ≈ FP16 版（~1e-3）的 8×（mantissa 差 3 bit）；gated 的 max_rel **恰落在 BF16 epsilon 2⁻⁸ 上**，max_abs 3.1e-2 = 该量级输出（|out|≤16，1 BF16 ULP=0.0625）的 0.5 ULP——纯输出存储舍入。gated 阈值据此记为 0.05（~0.8 ULP，结构错误给 O(1) 仍被拦截），unary 沿用 0.02/0.05。

**微基准（kernel-only，三臂两 regime，不预设收益）**

| 配置 | 类别 | BF16 | FP16（sanity） | FP32 |
|---|---|---|---|---|
| silu n=4096 | launch-bound | ~0.007 ms | ~0.007 ms | ~0.007 ms |
| silu n=4M | bandwidth-bound | 0.0645 ms / 260 GB/s | 0.0644 ms / 260 GB/s | 0.2122 ms / 158 GB/s（**3.3×**） |
| silu_multiply n=4096 | launch-bound | ~0.007 ms | ~0.007 ms | ~0.006 ms |
| silu_multiply n=4M | bandwidth-bound | 0.0967 ms / 260 GB/s | 0.1088 ms / 231 GB/s | 0.2799 ms / 180 GB/s（**2.9×**） |

结论与 6-A-4 一致：small n 均 launch-bound（BF16 不更快）；large n 进 memory-efficiency 区间（~3×）。两个实测注记：① BF16 与 FP16 全程 byte-parity，转换指令零可测开销；② launch-bound 档运行间抖动 ±70%（首臂时钟态，Phase 2 Round 2 同类伪影已归档），单次小 n 数字不作结论；③ 大 n 须让 FP32 工作集 spill 出 L2（本机 n=2M 时 FP32 集 16.8MB 常驻 L2 给出假 parity，n=4M 后 HBM 真值显现）——bench 选点本身是 workload 相关的。

**验收**：CTest **38/38 全过**（新增 `activation-bf16`；`dtype-contract` 退役台账 4 op；RMSNorm/FP16 全家无回归）。

**封版声明**：6-B-3 = DELIVERED / FROZEN。下一拍 6-B-4（BF16 matmul：cuBLAS `CUDA_R_16BF` + `CUBLAS_COMPUTE_32F`，镜像 6-A-5）解锁。

#### 6-B-4 native BF16 matmul（2026-09-07，本轮交付 ✅）

> 镜像 6-A-5 的 4.1→4.4 四段结构。核心差别仅一行：`CUDA_R_16F → CUDA_R_16BF`，
> `CUBLAS_COMPUTE_32F` 不动。边界：BF16 storage ✅ / FP32 accumulation ✅ /
> BF16 output ✅ / No FP32 tensor materialization ✅；non-transB 保持 reject（本轮无
> row-major-B 需求）；Phase 3/5 量化路径（`launch_cublas_gemm_quant_fp16` 只收 FP32
> A/C）零交叉。退役 `MUL_MAT_TRANSB`（台账 4→5；dual 路径无 BF16 kernel，继续 loud）。

**新增 / 修改**

| 文件 | 改动 |
|---|---|
| `src/operators/cuda/cuda_quant.cu` | 新增 `launch_cublas_gemm_bf16`（逐行镜像 `launch_cublas_gemm_fp16`，仅 `CUDA_R_16F→CUDA_R_16BF` + `__half→__nv_bfloat16`；`get_cublas_handle(stream)` 复用） |
| `include/forge/cuda_kernels.h` | 新 launch 声明 |
| `src/operators/cuda/matmul_cuda.cpp` | `cuda_matmul_transB`：BF16 任一位置出现即要求全 BF16 A/B/out（混合 loud，错误信息指到 dtype 对），全 BF16 走新 launch；`require_no_bf16` 三行退役。`_dual` 保持拒绝 |
| `src/operators/cpu/matmul.cpp` | eager `matmul_transB` 新增 BF16+BF16→BF16 out 路由（bias 仍显式 reject，与 FP16 同口径；混合 BF16×他 dtype 在 eager 层即 loud） |
| `tests/test_dtype_contract.cpp` | `bf16_retired_ops` 追加 `MUL_MAT_TRANSB`（size 断言 4→5）；`test_bf16_gate` 本体不动 |
| `tests/test_matmul_bf16.cpp`（新） | CTest `matmul-bf16`：4.1 direct launch（M=1/8/64）+ 4.2 route（+小尺寸 + 混合 loud）+ 4.4 M sweep |
| `tests/test_matmul_bf16_chain.cpp`（新） | CTest `matmul-bf16-chain`：4.3 RMSNorm→Matmul→silu_multiply（M=1/8/32，K=4096/N=11008） |

**正确性（BF16 先编码，reference 用同解码输入——绝不拿原始 FP32 作 golden）**

| 配置 | max_abs | max_rel |
|---|---|---|
| 4.1 native launch（M=1/8/64, K=N=4096） | ~0.24–0.25 | ~3.8–4.2e-3 |
| 4.2 route（M=1/8/64 + 1×256×512） | 同上（小尺寸 3.1e-2） | ~3.8–3.9e-3 |
| 混合 BF16×FP32 | loud reject ✅ | — |

解读：`max_rel` 全档锚定 BF16 epsilon 2⁻⁸（纯输出存储舍入）；`max_abs` 随输出量级（O(10–100)，该量级 1 BF16 ULP=0.5–1.0，属 sub-ULP）。阈值实测定版：`max_abs<1.0 / max_rel<0.02`（~4–5× 实测；结构错误超 10×）。

**4.3 chain**：三档全 BF16（dtype 不变量 ✅）；cosine **0.999996**（M=1/8/32 一致）；max_abs 0.40/0.54/0.76；max_rel(|ref|≥1) 0.098/0.134/0.160。诊断结论：小量级元素的 ~10–16% 是 RMSNorm-store 舍入经 K=4096 累加 random-walk 后的诚实传播（6-A-5.3 "few-%" 的 1/8-scale 同构现象），cosine 为证；判据取 cosine≥0.999 为主 + max_abs<2.0 + max_rel(|ref|≥1)<0.3（~2× 实测；`max_rel_info` 仅记录分母伪影，不设闸）。

**4.4 M sweep（best-of-3 交替，K=N=4096，无预设倍率；M=1 重测不定假设）**

| M | BF16 | FP32 | speedup |
|---|---|---|---|
| 1 | 0.183 ms | 0.362 ms | **1.97×** |
| 8 | 0.198 ms | 0.412 ms | **2.07×** |
| 32 | 0.198 ms | 0.448 ms | **2.26×** |
| 128 | 0.318 ms | 1.003 ms | **3.15×** |
| 512 | 1.101 ms | 3.385 ms | **3.07×** |

结论：BF16 cuBLAS 从 M=1 即成立（用户要求重测的点：无 M=1 异常，TensorCore 路径全档有效）；收益随 M 从 ~2× 升至 ~3.1×（流量减半 + TensorCore），与 6-A-5 的 FP16 形态同构。

**验收**：CTest **40/40 全过**（新增 `matmul-bf16` + `matmul-bf16-chain`；`dtype-contract` 退役台账 5 op；量化/FP16/FP32 全家无回归）。

**封版声明**：6-B-4 = DELIVERED / FROZEN。至此形成 `contract → RMSNorm → Activation → Matmul` 完整路线；下一拍 6-B-5（Graph / Binding Integration，多节点 BF16 propagation 一次性解决）解锁。

#### 6-B-5 Graph / Binding Integration（2026-09-07，本轮交付 ✅）

> 不逐 operator 打 graph patch——6-B-2 的单节点 smoke 已证可达性，本轮一次性验证整个
> BF16 graph contract，并作为 contract 的最终证明面：eager native 的算子在 graph 中同样
> native，不存在 eager/graph 分叉。

**新增 / 修改**

| 文件 | 改动 |
|---|---|
| `src/operators/cuda/cuda_elementwise.cu` | 新增 `add/multiply_kernel_bf16` + `launch_add_bf16` / `launch_multiply_bf16`（BF16 存储 + FP32 逐元素计算 + BF16 输出，镜像 6-A-8.0） |
| `include/forge/cuda_kernels.h` | 上两个 launch 声明 |
| `src/operators/cpu/elementwise.cpp` | eager `ops::add/multiply` 新增 BF16+BF16→BF16 out 路由（CUDA；CPU 显式 loud；混合 loud）。FP16 eager 仍 loud（6-B-1 纪律：FP16 只活在 dst） |
| `src/core/op_kernels.cpp` | `add/mul_kernel_dst` 在 `require_no_bf16` 之前插 BF16 分流（CUDA 新 launch + stream；CPU 本地 upcast→FP32→BF16；`b` 非 BF16 混合 throw） |
| `src/core/compute_graph.cpp` | `dst_bf16_capable` 扩为 RMS_NORM\|\|SILU\|\|ADD\|\|MUL；planner 本体零改（勘察确认已 dtype-generic：`dtype_size` 计尺寸，BF16=2B） |
| `include/forge/compute_dtype_policy.h` | 新增 `bf16_policy()` + `is_bf16_policy()`（weight/activation=BF16，accumulator/norm_compute=FP32，kv=FP32——BF16 KV-cache 是 6-B-6 scope；当前仅具名能力，尚无 selector 消费） |
| `tests/test_dtype_contract.cpp` | `bf16_retired_ops` 追加 ADD + MUL（size 断言 5→7）；policy 段加 `bf16_policy` 断言；`test_bf16_gate` 本体不动 |
| `tests/test_graph_bf16.cpp`（新） | CTest `graph-bf16`：5 节点链 RMSNorm→Matmul→SiLU→Add→Mul + trace + capture/replay |

**实测（RTX 4050，M∈{1,8,32}，K=4096/N=11008；capture M=8/K=1024/N=2048）**

```text
Node      Input  Output  dst_injected
-------------------------------------
rms_norm  bf16   bf16    1
matmul    bf16   bf16    0   (legacy by design, 与 6-A-6 同构)
silu      bf16   bf16    1
add       bf16   bf16    1
mul       bf16   bf16    1
```

- 全链 `bf16 → bf16`，无 `BF16 → FP32 → BF16`（trace 5/5）。
- graph vs eager standalone：`max_abs = 0.0` **逐位一致**（三档）——graph 层零数值差异。
- planner：输出 `nbytes == M*N*2`（BF16 宽度，非 FP32 两倍）。
- CUDA Graph：ordinary == capture == replay（逐位）；buffer 地址/尺寸稳定（32768 B = 8·2048·2）；capture 窗口零 `cudaMalloc`（allocs 40→41 全命中池）。
- binding：Python `Tensor(BF16)` / `from_buffer(BF16)` 构造验证通过（nbytes=elements×2）；graph 本体无 Python 绑定（现状如此，scope 外，已记录）。

**验收对照**：9 项用户标准 ①–⑨ 全部 ✅（见上表与数据）；CTest **41/41 全过**（新增 `graph-bf16`；FP16/FP32/量化全家无回归）。

**封版声明**：6-B-5 = DELIVERED / FROZEN。下一拍 6-B-6（BF16 attention，镜像 6-A-7，decode 先行）解锁；`bf16_policy.kv=FP32` 在该拍被真正消费前保持具名状态。

#### 6-B-6 native BF16-Q attention, decode-first（2026-09-07，本轮交付 ✅）

> 镜像 6-A-7 的 decode 切片，但 storage contract 与 6-A-7 不同——**不是"把 `__half`
> 换成 `__nv_bfloat16`"**：Q 到达时已是 BF16（query 投影输出），K/V 到达时仍是 FP32
>（authoritative KV store；`bf16_policy.kv` 保持 FP32），全数学 FP32，仅终写转 BF16。
> 这样"attention compute dtype"与"KV storage dtype"两个变量没有混在一起；BF16 KV
> 是否入 6-B-6 后半段，待本轮数据决定。

**新增 / 修改**

| 文件 | 改动 |
|---|---|
| `src/operators/cuda/cuda_attn_common.cuh` | `cross_warp_merge_and_write` 新增 `__nv_bfloat16*` 重载（merge 全 FP32，仅终写 `__float2bfloat16`） |
| `src/operators/cuda/cuda_flash_attn.cu` | 新增 `flash_attn_gqa_decode_kernel_bf16q` + `launch_flash_attention_gqa_decode_bf16q`：Q `__nv_bfloat16`→smem float，K/V float 直读（16B=4 floats，`VEC_COUNT=HD/4`），FP32 scale/online-softmax/PV，HD 16/32/64/96/128，honor stream |
| `src/operators/cuda/cuda_flash_attn.h` + `include/forge/cuda_kernels.h` | 新 launch 声明 |
| `include/forge/attention_selector.h` + `attention_selector.cpp` | 新增 `CUDA_BF16_DECODE`（+ name）；`q==BF16 && seq==1` 路由至此；BF16 prefill 故意落回 FP32 case（其 `require_no_bf16` loud）。非 GQA BF16 照旧走 `CUDA_GENERIC_MHA`（loud） |
| `src/inference/layers/attention_executor.cpp` | 新增 `case CUDA_BF16_DECODE`：q 必 BF16、K/V 必 FP32、mask FP32-or-none、HD 检查、BF16 out、`launch_..._bf16q(..., stream)` |
| `src/core/op_kernels.cpp` | `flash_attn_gqa_kernel` 新增 BF16-Q 分支（K/V 必 FP32、GQA、HD、seq==1；prefill/MHA/BF16-KV 各自 loud）；FP32/MHA/CPU 分支的 `require_no_bf16` 原样保留 |
| `tests/test_dtype_contract.cpp` | `bf16_retired_ops` 追加 `FLASH_ATTN_GQA`（size 8；decode-scope，與 `MUL_MAT_TRANSB`/dual 同先例：prefill/BF16-KV/MHA 仍拒）；`test_bf16_gate` 本体不动 |
| `tests/test_attention_bf16.cpp`（新） | CTest `attn-bf16`：decode 正确性（4 hd）+ FP32 自洽 + stream A/B + selector + loud 四面 + graph node + kv-sweep bench |

**正确性（reference 用同解码 BF16 Q + 同 FP32 K/V；K/V 永不量化）**

| 配置 | max_abs |
|---|---|
| decode vs CPU FP32 ref（hd 16/32/64/128, kv=512） | 1.2–2.3e-4 |
| vs FP32 kernel 自洽（同解码值） | 1.2–2.3e-4 |

输出是 V∈[-1,1] 的 softmax 平均（|out|≤1），误差即 BF16 Q 输入舍入 + 输出舍入；阈值实测定版 `max_abs<1e-3`（~4× 实测；错 scale/轴/顺序给 O(0.1)）。stream A/B 逐位一致 4/4；graph node eager/capture/replay 逐位一致，`O==BF16`，`nbytes==nh*hd*2`。

**6-B-6.6 decode bench（best-of-3 交替，nh=8/nkv=2/hd=64）**

| kv_len | BF16-Q | FP32 | speedup |
|---|---|---|---|
| 128（launch-bound） | 0.01709 | 0.01716 | **1.00×** |
| 512 | 0.02138 | 0.02158 | **1.01×** |
| 2048（bandwidth-bound） | 0.04385 | 0.04405 | **1.00×** |

结论（诚实的中性结果）：decode 主导成本是 KV-cache streaming——K/V 两侧仍是 FP32 等字节，Q 仅 1KB，BF16-Q 理论上就不该赢。**这正是 `kv=FP32` 现状的直接度量**：decode 的 BF16 收益必须等 BF16 KV（KV 字节减半）才兑现，本轮不硬凑收益。该数据即"BF16 KV 是否入 6-B-6 后半段"的决策依据。

**验收**：CTest **42/42 全过**（新增 `attn-bf16`；`dtype-contract` 退役台账 8 op；FP16/FP32/量化全家无回归）。

**封版声明**：6-B-6 decode 切片 = DELIVERED / FROZEN。未做：BF16 prefill、BF16 KV-cache、paged BF16、MMA/half2、selector 重构——是否开 BF16 KV 后半段，待用户按上表拍板。

#### 6-B-6.7 native BF16 KV decode（2026-09-07，本轮交付 ✅）

> 用户拍板：先做 BF16 KV 后半段，再进 6-B-7。动机不是性能数字，而是 6-B-6 已证明
> decode 带宽的绑定变量是 KV storage——`BF16-Q + FP32-KV ≈ 1.00×` 之后，唯一有价值的
> 问题就是"BF16 KV 是否兑现 memory-traffic gain"。只扩 decode，不碰 prefill/paged。

**新增 / 修改**

| 文件 | 改动 |
|---|---|
| `src/operators/cuda/cuda_flash_attn.cu` | 新增 `flash_attn_gqa_decode_kernel_bf16` + `launch_flash_attention_gqa_decode_bf16`：Q/K/V 全 BF16，FP32 数学，BF16 输出；K/V/mask 16B=8×bf16 转换读（`VEC_COUNT=HD/4`→`HD/8`）；HD 集与 stream 语义同 6-B-6 |
| `src/operators/cuda/cuda_flash_attn.h` + `include/forge/cuda_kernels.h` | 新 launch 声明 |
| `src/core/op_kernels.cpp` | BF16 分支内按 K/V dtype 分派：全 FP32→bf16q launch，全 BF16→bf16 launch，混合/其他 loud（错误信息指到 K/V 对） |
| `src/inference/layers/attention_executor.cpp` | `case CUDA_BF16_DECODE` 内同口径分派（mask 跟随 K/V 侧 dtype：BF16-KV 配 BF16 mask，FP32-KV 配 FP32 mask）；selector **不动**（同一 path cover 两种 KV，path 名即"BF16 decode"语义） |
| `include/forge/compute_dtype_policy.h` | `bf16_policy().kv`: FP32→**BF16**（等待项正式消费）；`is_bf16_policy` 同步 |
| `tests/test_dtype_contract.cpp` | policy 断言 `kv==BF16`（6-B-6.7 消费记录） |
| `tests/test_attention_bf16.cpp` | 扩展：all-BF16 正确性（4 hd）+ stream A/B + dispatch 原生性（BF16-QKV 不抛且出 BF16）+ 混合 K/V loud + graph node bf16kv + bench 第三臂 |

**正确性（reference 用同解码 BF16 QKV；K/V 永不跨 dtype 混用）**

| 配置 | max_abs |
|---|---|
| all-BF16 vs CPU FP32 ref（hd 16/32/64/128, kv=512） | 1.1–2.2e-4（与 BF16-Q 切片同带，阈值同口径 `1e-3`） |
| stream A/B | 逐位一致 4/4 |
| graph node（eager/capture/replay） | 逐位一致，`O==BF16` |

**6-B-6.7 bench（三臂同值不同存储 + best-of-3 交替，nh=8/nkv=2/hd=64）**

| kv_len | BF16-KV | BF16-Q（6-B-6 基线） | FP32 | kv-gain |
|---|---|---|---|---|
| 128（launch-bound） | 0.00911 | 0.00853 | 0.00877 | **0.94×** |
| 512 | 0.01169 | 0.01542 | 0.01568 | **1.32×** |
| 2048（bandwidth-bound） | 0.02527 | 0.04379 | 0.04397 | **1.73×** |

结论（只记录实测，不预设）：① kv=128 的 0.94× 是 launch-bound 噪声（±7%，6-B-3 同类伪影量级），非回归；② kv=2048 的 **1.73×** 即 KV 字节减半兑现的部分收益——不到 2×，因为 Q/O/softmax 固定成本与转换指令不随 KV 减半；③ BF16-Q 基线与 FP32 全程一致（0.04379 vs 0.04397），6-B-6 的 1.00× 结论复现，A/B 方法可信。

**验收对照（用户 9 项）**：①全 BF16 Q/K/V→BF16 O ✅ ②FP32 accumulation ✅（kernel diff + parity）③vs FP32 ref ✅ ④stream A/B ✅ ⑤capture/replay ✅ ⑥selector/executor ✅（同 path 内 K/V 分派）⑦混合 loud ✅ ⑧kv sweep ✅ ⑨A/B 交替 best-of-N ✅（三臂同值）。

**验收**：CTest **42/42 全过**（`attn-bf16` 内扩展，无新增 CTest 名；其余全家无回归）。

**封版声明**：6-B-6.7 = DELIVERED / FROZEN。BF16 attention decode 至此两种 KV 形态俱全；未做：BF16 prefill、paged BF16、MMA/half2。下一拍 6-B-7（全链路 BF16 E2E：activation / RMSNorm / activation family / matmul / residual / attention / KV 全 native）解锁——E2E 不再引入新 storage 变量。

#### 6-B-7 End-to-end BF16 validation（2026-09-07，本轮交付 ✅ COMPLETE / FROZEN）

> 目标：证明完整 inference 闭链（RMSNorm → QKV 投影 → BF16 Attention + BF16 KV →
> residual → gated FFN → next layer → logits）在真实 ComputeGraph 执行下不发生
> `BF16→FP32→BF16` 隐式回退或 dtype contract 断裂。最小闭环 = 2 层 llama 式图
>（D=256, nh=8, nkv=2, hd=64, FF=512, V=128），prefill（q_len=16）与 decode（q_len=1）
> 两种 dispatch 模式，BF16 与 FP32 跑**同一拓扑**（同一组 float 模型值驱动两种 dtype
> 物化，差异只来自 BF16 存储/计算）。

**7.0 前置解锁（同轮交付）**：E2E 勘察发现 prefill 半环没有 kernel 可走（6-B-6 只放行
decode），因此本轮先补 **native BF16 vectorized prefill**
（`flash_attn_gqa_kernel_vectorized_bf16` + `launch_..._prefill_bf16`，镜像 6-A-7.2 的
"只换 dtype boundary"纪律：Q tile smem float、16B=8×bf16 载入、FP32 数学、BF16 输出、
无 wmma）+ `CUDA_BF16_PREFILL` path（selector 两分支化：q-BF16 seq1→DECODE、
seq>1→PREFILL）+ executor case + op_kernels 分派（prefill 强制 all-BF16 QKV，
FP32-KV-prefill 保持 loud）。prefill 正确性（causal FP32 ref，q=4/kv=16，hd 16/64/128）：
9.7e-4 / 9.7e-4 / 1.9e-3（= 0.5 BF16 ULP，纯输出舍入），阈值定 4e-3（~1 ULP）；
stream A/B 逐位；dispatch 原生性 + FP32-KV-prefill loud 全过。

**KV 写入/读取链路（与 6-A-8 的关键差异）**：decode 的 K/V cache **不是合成随机张量**
——由真实 prefill 写入路径产出（与 prefill 图 wk/wv 节点完全相同的 native 算子链
`rms_norm → matmul_transB`，逐 dtype 分别执行），再作为 decode 图输入消费。
`produce_decode_kv` 把 KV write path 与 read path 锁进同一个测试。

**新增 / 修改**

| 文件 | 改动 |
|---|---|
| `src/operators/cuda/cuda_flash_attn.cu/.h` + `cuda_kernels.h` | 7.0 prefill kernel + launch |
| `attention_selector.{h,cpp}` + `attention_executor.cpp` + `op_kernels.cpp` | 7.0 路由三层 |
| `tests/test_attention_bf16.cpp` | prefill correctness/stream/dispatch 断言；selector 更新为 decode+prefill 双路由 |
| `tests/test_e2e_bf16.cpp`（新） | CTest `e2e-bf16`：8.1 = 2 层闭环三项验收；KV 写读链路 |
| `tests/CMakeLists.txt` | 注册 `e2e-bf16` |

**三项验收（实测，阈值首跑标定后写死）**

1. **dtype trace（`FORGE_TRACE_DTYPE=1`）**：主链全部节点（每层 rms/wq/wk/wv/attn/wo/
   residual/silu/gate_up × 2 层 + out.rms/logits）in/out 全 BF16，无隐式回退；dst 注入
   严格等于 dstable 集合（rms/residual/silu/gate_up 走 planner buffer，matmul/attention
   legacy 系 by design）。
2. **BF16 vs FP32 parity（同一拓扑、同一模型值）**，阈值固定
   `max_abs<0.01 / max_rel<0.03(guard |ref|>=0.15) / cosine>=0.999 / top8 mismatch<=1`：

   | 模式 | max_abs | max_rel | cosine | top8 |
   |---|---|---|---|---|
   | prefill | 1.321e-03 | 4.768e-03 | 0.9999902 | 8/8 |
   | decode  | 8.727e-04 | 1.989e-03 | 0.9999886 | 8/8 |

   实测较阈值有 ~1 个数量级余量；guard 非空转（max|ref| 0.282/0.159 ≥ 0.15）。
3. **CUDA Graph 真链路**：ordinary → capture → replay×3，两模式输出逐位一致，
   logits buffer 地址/尺寸稳定（prefill 4096 B / decode 256 B，均为 BF16 宽度）。

**验收（回归）**：CTest **43/43 全过**（新增 `e2e-bf16`；`attn-bf16` 扩展 7.0 断言；
FP16 全家/量化/selector 无回归）。

非目标（明确未做）：真实模型加载（GGUF/sampler/argmax）、paged BF16、BF16 wmma、
性能测量（BF16 decode/prefill 性能属后续按 6-A Phase 2 铁律另立轮次）。

**封版记录（2026-09-07）：6-B-7 = COMPLETE / FROZEN —— Phase 6-B correctness baseline
冻结。** 全链形态：

```text
Phase 6-B — Native BF16 Execution          Status: COMPLETE / FROZEN (2026-09-07)

├── 6-B-0   audit (只读)                        ✅
├── 6-B-1   contract hardening                  ✅ FROZEN
├── 6-B-2   RMSNorm                             ✅ FROZEN
├── 6-B-3   activation family                   ✅ FROZEN
├── 6-B-4   matmul (CUDA_R_16BF)                ✅ FROZEN
├── 6-B-5   graph/binding (bf16_policy)         ✅ FROZEN
├── 6-B-6   attention decode (BF16-Q)           ✅ FROZEN
├── 6-B-6.7 attention decode (BF16 KV)          ✅ FROZEN
└── 6-B-7   E2E (+7.0 prefill unlock)           ✅ FROZEN

BF16 weights + BF16 activation + BF16 KV → BF16 compute → FP32 accumulation
退役台账: RMS_NORM/SILU/GELU/GELU_TANH/MUL_MAT_TRANSB/ADD/MUL/FLASH_ATTN_GQA (8 op)
CTest 43/43
```

遗留（不阻塞封版，已记录）：BF16 prefill 性能调优（vectorized 基线已可用）、paged BF16、
BF16 KV-cache 运行时集成（`KVCacheDType` 无 BF16 成员，E2E 用图输入承载）、
`bf16_policy` 接入 selector 的消费面、GGUF BF16 native load。

#### 6-B 实施序列（审计后修正）

```text
6-B-0  audit（本节，✅ 2026-09-07）
6-B-1  contract 收紧：P0/P1 loud reject + 端到端 BF16 reject 测试（先封漏，不加能力，✅ 2026-09-07 DELIVERED/FROZEN）
6-B-2  native BF16 RMSNorm（bf16 kernel 第一块试金石，镜像 6-A-2，✅ 2026-09-07 DELIVERED/FROZEN）
6-B-3  BF16 activation（SILU/GELU/GELU_TANH + gated 变体，镜像 6-A-3/4，✅ 2026-09-07 DELIVERED/FROZEN）
6-B-4  BF16 matmul（cuBLAS CUDA_R_16BF + CUBLAS_COMPUTE_32F，镜像 6-A-5，✅ 2026-09-07 DELIVERED/FROZEN）
6-B-5  graph/binding（policy 工厂 bf16_policy、dst-injection BF16 分支、dtype trace BF16 覆盖，✅ 2026-09-07 DELIVERED/FROZEN）
6-B-6  BF16 attention（decode-first：BF16-Q + FP32-KV，镜像 6-A-7，✅ 2026-09-07 DELIVERED/FROZEN；prefill/paged/MMA 未排期）
6-B-6.7 BF16 KV decode（BF16-QKV 全形态，kv-gain 0.94/1.32/1.73×，bf16_policy.kv→BF16，✅ 2026-09-07 DELIVERED/FROZEN）
6-B-7  E2E validation（2层闭环 + prefill产出KV→decode消费 + dtype trace/parity/CUDA Graph，✅ 2026-09-07 COMPLETE/FROZEN —— Phase 6-B 封版）
6-B-2  native BF16 RMSNorm（bf16 kernel 第一块试金石，镜像 6-A-2）
6-B-3  BF16 activation（SILU/GELU/GELU_TANH + gated 变体，镜像 6-A-3/4）
6-B-4  BF16 matmul（cuBLAS CUDA_R_16BF + CUBLAS_COMPUTE_32F，镜像 6-A-5）
6-B-5  graph/binding（policy 工厂 bf16_policy、dst-injection BF16 分支、dtype trace BF16 覆盖）
6-B-6  BF16 attention（镜像 6-A-7，decode 先行）
6-B-7  E2E validation（镜像 6-A-8：dtype trace + parity + CUDA Graph 逐位）
```

```text
BF16 weights + BF16 activation → BF16 Tensor Core → FP32 accumulation
```

完成时 policy 拥有 FP16/BF16 双实列，dtype abstraction 才算成熟。

### 6-C. FP8 W8A8（真正的 low-precision compute backend）

```text
FP8 weight + FP8 activation → Tensor Core → FP16/FP32 accumulation
```

与已完成的 **FP8 KV（KV8）是两回事**：KV8 只压 cache 路径；W8A8 才进入计算路径。

#### 6-C-0 FP8 W8A8 execution audit（2026-09-07，本轮交付 ✅ 只审计不改码）

目标不是"支持 FP8"，而是确定 Forge 中真正的
`FP8 weight + FP8 activation → Tensor Core → FP16/FP32 accumulation` 执行模型。
审计完全复制 6-B-0 纪律：先 audit → contract → native kernel → E2E，不先假设
编码/scale 方案/accumulation dtype。

**Audit-1：FP8 representation 现状 —— 三层全缺，FP8 只活在 KV 里**

| 层 | 现状 | 位置 | 6-C 缺口 |
|---|---|---|---|
| `DataType` | **无 FP8 成员** | `include/forge/types.h:6-28` | 需新增 `FP8_E4M3`（E5M2 视 6-C-1 拍板） |
| Tensor / Python binding | 与 DataType 一致，无 FP8 | `src/bindings/core_types.cpp:6-27` | binding 随枚举扩展 |
| GGUF（`GgmlDType`） | **无 FP8 映射**；GGUF 标准本身无 FP8 类型 | `include/forge/gguf_model.h:9-39` | **不存在"GGUF FP8 native load"工作项**——FP8 weight 只能来自运行时量化（calibration/quantize tool），这是 6-C-1 contract 设计的前提事实 |
| FP8 唯一存在形态 | `KVCacheDType::FP8_E4M3/E5M2` | `include/forge/kv_cache.h:17-18` | 与 `DataType` 是两套体系，W8A8 不会自动复用 |

**Audit-2：FP8 KV 基建盘点 —— 编解码可复用，scale 语义不可直接搬**

| 基建 | 位置 | 语义 | W8A8 复用判定 |
|---|---|---|---|
| E4M3/E5M2 ⇄ FP32 编解码 | `include/forge/fp8_utils.h:30-128`（host+device，无 scale） | 纯逐元素；E4M3 inf→clamp 240、NaN→240；E5M2 NaN→0 | ✅ 边界/测试/CPU 复用；**❌ 不可进 GEMM 热路径**（每元素 `floorf/log2f/exp2f` 软件转换，吞吐致命） |
| per-(row, kv_head) amax scale | quantize: `src/operators/cuda/cuda_quant.cu:1432-1466`（`scale = amax/FP8_MAX`）；scale buffer: `src/inference/kv_cache.cpp:862-883`（`fp8_scale_k_/v_` per-layer 独立 float buffer）+ `:1709-1718`（`d_q_key_scale`） | **后验** amax（写入时对整行统计），scale 存带外 buffer | ⚠️ 机制可参考，**粒度/时机均不匹配 GEMM**（见下） |
| fused FP8 decode 消费面 | `src/operators/cuda/cuda_fused_attn.cu:947,961` | 在线 dequant × scale，FP32 点积/softmax | ✅ "FP8 存储 + FP32 计算"纪律与 6-A/6-B 同构，作为模板 |
| quantize/dequant matrix kernels | `cuda_quant.cu:1355-1428`（unscaled） | 逐元素 | ✅ 工具性复用 |
| CPU quantize_row/dequant_row | `src/inference/kv_cache.cpp:561-601` | 逐元素 unscaled | ✅ |

W8A8 与 KV scale 的三个语义差距（6-C-1 必须拍板）：

```text
1. 粒度：KV = per-row(≈per-token) 后验 amax；W8A8 GEMM 在 sm_89 + cuBLASLt 只支持
   per-tensor scale（block-scale 需 sm_90+ 或手写 kernel）。per-tensor 后验需全量
   amax pass（两遍）→ 实用候选：delayed scaling（沿用上一 step amax）/ static calibration。
2. 传播：KV scale 是带外 buffer；activation scale 必须随 graph 边流动（fp8_policy 的
   新职责），否则会出现"scale 谁持有"的隐式全局态。
3. 饱和语义：E4M3 无 NaN clamp 是按 KV 小幅度定的；GEMM 输入是 RMSNorm 后的
   activation，幅度可控，但 6-C-1 需复核 clamp→240 的决策。
```

既有债（本轮发现，登记不修）：**paged FP8 decode kernel 签名无 scale 参数**
（`src/operators/cuda/cuda_paged_kv.cu:752-790`）→ paged FP8 = unscaled 路径，
与 contiguous scaled 路径语义不一致。处理归 6-C-6。

**Audit-3：静默升级风险 —— gate 是黑名单形状，FP8 会全线穿透**

核心缺口：`is_native_execution_dtype()`（`include/forge/dtype_contract.h:31-33`）
= 黑名单（`!FP16 && !BF16`）→ 一旦加入 `DataType::FP8`，会被判为 "native dtype"
**绕过全部 gate**。这正是 6-B-0 四处 P0 在 FP8 上的重演路径。落点（FP8 加入后
全部 P0：静默当 float 读）：

| 站点 | 位置 | 现行挡板 | FP8 后果 |
|---|---|---|---|
| add_kernel_dst FP32 fallback | `src/core/op_kernels.cpp:81-94` | `require_no_bf16` :61-62 | 穿透 → float 误读 |
| mul_kernel_dst FP32 fallback | `src/core/op_kernels.cpp:154-167` | :134-135 同 | 同 |
| silu/rms_norm dst FP32 fallback | `src/core/op_kernels.cpp:170-209` | 同型 | 同 |
| MUL_MAT_TRANSB FP32 cuBLAS 路径 | `src/operators/cuda/matmul_cuda.cpp:329-356` | `require_dtype_supported` + BF16/FP16 分支 | FP8 落 FP32 cublas 静默 |
| attention selector | `src/inference/layers/attention_selector.cpp:27-38` | 只认 `q_dtype==FP16/BF16` | FP8 落下方 FP32/量化分支，无检查 |
| graph dtype policy | `include/forge/compute_dtype_policy.h` | 无 fp8_policy | FP8 按 FP32 policy 静默传播 |

6-C-1 gate 设计拍板（本拍输出）：**把 `is_native_execution_dtype` 从黑名单改为
白名单**（显式列出 FP32 + 量化格式集合），FP8 天生被拒；之后每个 native FP8
kernel 落地时逐 op 退役（沿用 6-B 退役协议）。理由：per-dtype 黑名单
（`require_no_fp8` 形式）已被 FP16→BF16 迭代证明每加一个 dtype 都要重扫全库；
白名单一次收紧，扩展点归一为 `op_supports_lowp()` 单点。

**Audit-4：W8A8 执行路径能力 —— 硬件成立，软件 100% 绿地**

| 项 | 结论 | 证据 |
|---|---|---|
| GPU | RTX 4050 Laptop / **sm_89 (Ada)** → FP8 Tensor Core 硬件存在 | nvidia-smi compute_cap 8.9 |
| CUDA | 12.8 → cuBLASLt FP8（`CUDA_R_8F_E4M3` + `CUBLAS_COMPUTE_32F`）可用 | nvcc V12.8.93 |
| **编译约束** | `FORGE_CUDA_ARCH="86;89"` 双架构（`CMakeLists.txt:140`）→ **sm_86 无 FP8 TC**，FP8 kernel 必须 `__CUDA_ARCH__>=890` guard，否则 86 目标静默出错 | CMakeLists |
| cuBLAS 现状 | 只有 `cublasGemmEx`（FP32/16F/16BF，`cuda_quant.cu:1136-1241` + `cuda_common.h:51` handle）；**GemmEx 不接受 FP8 输入**；全库零 cublasLt | rg 核验 |
| 硬件指令现状 | 全库零 `__nv_fp8` / `cvt.rn.satfinite.e4m3x2` / `mma e4m3` → FP8 TC 执行路径 100% 绿地 | rg 核验 |
| 可借鉴经验 | Phase 2 wmma（`cuda_flash_attn.cu`）= 手写 TC 路径的组织经验 | — |

6-C-4 技术路线（届时拍板，本拍不定）：
`a) cuBLASLt`：成熟快，代价 = 新 API 面（descriptor/scale pointer/col-major 布局 trick）+ sm_89 限 per-tensor scale + 对齐约束；
`b) 手写 mma.sync.m16n8k32`：与 Phase 2 同构，可做 per-block scale（为 6-D NVFP4 铺路），工程量大。

**Capability matrix（6-C-0 快照）**

| Operator | FP8 input | FP8 output | Acc | Native kernel | 当前状态 |
|---------|---------:| ---------: | ---: | ------------- | ------- |
| RMSNorm | ✗ | ✗ | FP32 | ✗ | audit → 6-C-2 |
| SiLU/GELU family | ✗ | ✗ | FP32 | ✗ | audit → 6-C-3 |
| Matmul (transB) | ✗ | ✗ | FP32 | ✗（GemmEx 不支持 FP8，cublasLt/mma 二选一） | 核心 → 6-C-4 |
| Attention | KV-only：FP8 K/V + FP32 Q | ✗ | FP32 | ✓（fused scaled / paged unscaled，软件转换） | 策略 → 6-C-6 |
| Add/Mul | ✗ | ✗ | FP32 | ✗ | 后置 |
| KV write | FP32→FP8 per-row amax | FP8 | — | ✓ | 可复用 |

**6-C-0 结论与拍板**

1. **scale semantics 是 6-C 的核心设计问题，不是 kernel**：KV 的 per-row 后验 amax
   不能直接搬进 GEMM；6-C-1 必须先把 activation scale contract（粒度/传播/饱和语义）
   定死，再写任何 kernel——否则会得到 benchmark 漂亮但全链 FP16/FP32 来回转换的假 W8A8。
2. FP8 weight 无文件格式来源（GGUF 无 FP8），运行时量化工具是 6-C 的一等工作项。
3. 6-C 实施序列（镜像 6-B 模板 + activation 先行规则）：

```text
6-C-0  audit（本节 ✅ 2026-09-07）
6-C-1  contract：DataType::FP8_E4M3 + is_native_execution_dtype 白名单化 +
       全链 FP8 loud-reject 测试 + activation scale contract 定稿（✅ 2026-09-07
       DELIVERED/FROZEN，见上节——先封漏不加能力）
6-C-2  FP8 RMSNorm（native lowp 试金石，镜像 6-B-2，activation 路径第一块）
6-C-3  FP8 activation family（SILU/GELU + gated，镜像 6-B-3）
6-C-4  FP8 matmul（Tensor Core：cuBLASLt vs mma.m16n8k32 拍板，W8A8 首次合体）
6-C-5  graph/binding/selector（fp8_policy + FP8 dispatch 槽位）
6-C-6  FP8 attention / KV policy（含 paged FP8 unscaled 债处理）
6-C-7  E2E FP8 W8A8 validation（镜像 6-B-7）
6-D    NVFP4（Blackwell 压轴，不动）
```

4. 遗留登记（不阻塞）：paged FP8 unscaled 不一致（→6-C-6）、fp16_policy 缺失
   （FP16 目前不经 policy 工厂，记录）、E4M3 clamp 语义复核（→6-C-1）。

**非目标**：不写 FP8 kernel、不做 FP8 benchmark、不动 NVFP4、不补 6-B 遗留项
（paged BF16 / KVCacheDType BF16 槽位 / GGUF native load 等）。

**验收**：本轮只审计不改码，CTest 43/43 全绿（0 改动基线确认）。

#### 6-C-1 contract 收紧 + activation scale contract 定稿（2026-09-07，本轮交付 ✅ DELIVERED / FROZEN）

先封漏，不加能力：FP8 成为有 representation 的 dtype，但对每个 operator 都 loud reject。

**改动表**

| 文件 | 改动 |
|---|---|
| `include/forge/types.h:29-31` | `DataType::FP8_E4M3 = 21`（1 byte/element；与 `KVCacheDType::FP8_E4M3` 两套体系明确分离） |
| `src/operators/cpu/quant_traits.cpp:58` | traits 表登记 `fp8_e4m3`（type_size 1，非量化） |
| `include/forge/dtype_contract.h:36-64` | **`is_native_execution_dtype` 黑名单 → 白名单**（显式列出 FP32/INT8/INT32/全量化格式；FP16/BF16/FP8 → lowp；未知未来 dtype 默认拒绝） |
| `include/forge/dtype_contract.h:124-126` | **`op_supports_fp8(op)` FP8 退役台账**（初始为空 = 对所有 op 拒绝）；`op_supports_dtype` 将 FP8 路由至此台账 |
| `include/forge/dtype_contract.h:187-201` | `throw_fp8_unsupported` + `require_no_fp8` 路由 gate（镜像 `require_no_bf16`） |
| `src/bindings/core_types.cpp:28` | Python `DataType.FP8_E4M3` 暴露 |
| `src/core/op_kernels.cpp` | ADD/MUL/SILU/RMS_NORM 四个 dst kernel：FP32 fallback 的 CPU 分支前补 `require_no_fp8`（CUDA 分支由台账经 `require_dtype_supported` 覆盖） |
| `src/operators/cpu/attention.cpp` | 3 个 CPU attention 函数 ×Q/K/V(/mask) 补 `require_no_fp8` |
| `src/inference/layers/attention_executor.cpp` | `CUDA_FP32_PREFILL/DECODE` 两路径 ×Q/K/V 补 `require_no_fp8`（FP8 Q 与 BF16 同样穿透 selector 落默认路径） |
| `tests/test_dtype_contract.cpp` | `test_fp8_contract`：白名单不变量、台账为空、12 op × FP8 全拒绝（消息含 `fp8_e4m3`/op 名）、`require_no_fp8` 对非 FP8 零影响 |
| `tests/test_api_contract.py` | 枚举存在性测试加 `FP8_E4M3` |

**Gate 架构（与 6-B 的差异及理由）**：BF16 = "gate 放行 + 路由层逐点拒绝"（两层）；
FP8 = "**contract 层直接拒绝**（台账）+ 裸站点 `require_no_fp8` 兜底"。因为 6-C-0 审计
证明 FP8 没有任何 native kernel，contract 层一次收紧最稳；6-C-2..6-C-6 每个 native
kernel 落地时按 6-B 退役协议逐 op 改台账（kernel + 路由 + 台账三件套同拍落地）。
`is_native_execution_dtype` 白名单化的收益是**结构性的**：以后每加一个新 dtype，
默认是拒绝而不是放行——6-B-0 的静默误读模式被从语言层面关死。

**FP8 activation scale contract（定稿，6-C-2+ 按此实现，不得中途改）**

```text
S1 编码     W 与 A 均 E4M3（Forge codec：bias-7、max finite 240、无 NaN、
            0x7F/0xFF→240）。E5M2 仅限 KV，不进 W8A8（2-bit mantissa 对
            weight/activation 过粗）。
S2 粒度     per-tensor（sm_89 cuBLASLt 硬约束；block-scale 需 sm_90+ 或手写
            kernel，留给 6-D 评估）。weight：量化时一次性 amax/240 定标；
            activation：delayed scaling（per-tensor rolling amax，前一 forward
            的 amax 服务当前 step）。E2E correctness 轮（6-C-7）允许降级为
            static calibration（固定 scale 不滚动），性能轮再启用 rolling。
S3 传播     scale 是数据层：activation scale 以 tensor sidecar（device float
            scalar）随 graph 边透传；不进 ComputeDTypePolicy（policy 是类型
            层，只声明 dtype 五元组）。禁止"scale 谁持有"的隐式全局态。
S4 累加     FP32（CUBLAS_COMPUTE_32F / mma f32 accumulate）；attention softmax
            FP32（沿用 6-A 纪律）。accumulation 不随 activation dtype 变。
S5 反量化   GEMM 输出一次合并 ×(s_a·s_w)（数学等价，省一次全量往返）。
S6 位一致性 scale 约定 amax/240 → 量化值域 ⊆ [−240,240] ⊆ E4M3FN 硬件值域，
            Forge 软件编解码与硬件 cvt/mma 解码在所用值域内位一致（Forge 从
            不发射 exp=1111 模式；硬件 satfinite 上限 448 ≥ 240）。KV 的既有
            amax/240 约定不变，两套 FP8 共享同一 scale 惯例。
```

**验收**：CTest 43/43 全绿（`dtype-contract` 输出 `PASS ... + 6-C-1`）。全库
`switch(dtype)` 站点均带 default 兜底（kernel_selector/quant_policy 核验），
FP8 注入场景落 UNSUPPORTED 安全分支。

**非目标**：不写任何 FP8 kernel（含 RMSNorm）、不做 scale sidecar/rolling amax
实现（S2/S3 是 6-C-2+ 的实现契约）、不接 selector/policy 消费面（6-C-5）、
不处理 paged FP8 unscaled 债（6-C-6）、不评估 cuBLASLt vs mma（6-C-4）。

#### 6-C-2 Native FP8 RMSNorm（2026-09-07，本轮交付 ✅ DELIVERED / FROZEN）

第一个 native FP8 kernel，严格镜像 6-A-2/6-B-2 模板；本轮不同时引入 scale
machinery。只退役 `RMS_NORM` 一个台账点（SILU/GELU/ADD/MUL 保持拒绝）。

**执行模型**

```text
FP8 x / FP8 w --fp8 load(cvt)--> FP32 square/reduction --rsqrt(FP32)--> FP32 scale --fp8 store(cvt)--> FP8 out
```

storage ≠ compute 纪律延续：FP8 存储 + FP32 reduction/math + FP8 输出。

**改动表**

| 文件 | 改动 |
|---|---|
| `src/operators/cuda/cuda_rms_norm.cu:15-32,361-471` | `rms_norm_kernel_fp8` / `rms_norm_unweighted_kernel_fp8` + launchers；**硬件 codec** `fp8_hw_load/store_byte`（`cuda_fp8.h`：`__nv_cvt_fp8_to_halfraw` / `__nv_cvt_float_to_fp8(SATFINITE)`；sm_89 编译为 FP8 cvt 指令，sm_86 自动软件 fallback——6-C-0 双架构约束由 cuda_fp8.h 自带 arch 分支解决） |
| `include/forge/cuda_kernels.h:48-55`、`src/operators/cuda/cuda_rms_norm.h:31-37` | launcher 声明（双头登记） |
| `src/operators/cpu/norm.cpp:50-68` | eager 路由 `fp8_path`（**仅 FP8×FP8**，混合落 else loud-throw） |
| `src/core/op_kernels.cpp:310-349` | `rms_norm_kernel_dst` FP8 native 分支（CUDA launcher + CPU decode→FP32 row→encode）；混合 FP8/FP32 落 `require_no_fp8` throw |
| `src/core/compute_graph.cpp:297-311` | use_dst 判定加 `inputs_fp8 / dst_fp8_capable`（**仅 RMS_NORM**） |
| `include/forge/dtype_contract.h:128-135` | **`op_supports_fp8` 退役 RMS_NORM**（kernel+路由+台账三件套同拍，6-B 退役协议） |
| `tests/test_rms_norm_fp8.cpp`（新） | correctness（decoded-input reference）+ 2-hop 链式 + stream A/B + graph smoke + bench |
| `tests/test_dtype_contract.cpp:322-338` | `fp8_retired_ops = {RMS_NORM}` 显式列表；其余 11 op 保持全拒绝断言 |
| `tests/CMakeLists.txt:201-207` | 注册 `rms-norm-fp8` |

**Codec 位一致性（S6 落地证据）**：kernel 内用硬件 cvt（快路径），host/边界/测试用
forge 软件 codec——值域 ⊆ [-240,240]（amax/240 约定）内两者位一致；唯一分歧
= (240,448] 值域与 0x7F NaN（约定下不可达）+ RNE vs round-half-away tie-break
（都取最近桶，半步界内）。**forge 软件 codec 禁止进 kernel 热路径**（6-C-0 预言被
实测证实：见下）。

**正确性数据**（decoded-input reference；cosine 主指标；kernel=硬件 codec）

| Case | max_abs | max_rel | cosine |
|---|---:|---:|---:|
| weighted 4096×1 | 5.56e-2 | 4.66e-2 | 0.999687 |
| weighted 4096×512 | 6.25e-2 | 5.86e-2 | 0.999684 |
| weighted 8192×1 | 5.46e-2 | 4.53e-2 | 0.999687 |
| weighted 8192×512 | 6.25e-2 | 5.84e-2 | 0.999684 |
| unweighted 4096×64 | 6.24e-2 | — | 0.999640 |
| **chain 2-hop** 4096×8 | 6.25e-2 | 6.67e-2 | **0.999866** |

max_abs=6.25e-2 恰为 E4M3 在 [2,4] binade 的半步（理论值）；max_rel 界
= 1/16=6.25%（3-bit mantissa binade 低端最坏值），实测 5.9% 贴合。链式两跳
cosine 0.99987：FP8 store/decode 复合误差有界，无放大（阈值锁 0.998）。混合
FP8×FP32 loud reject ✓、stream A/B bitwise ✓、graph dst smoke bitwise ✓。

**Bench 数据与两个关键结论**（RTX 4050 Laptop / sm_89；kernel-only 计时）

| Regime | FP8 | BF16 | FP32 |
|---|---:|---:|---:|
| 1×4096（launch-bound） | 0.0166 ms | 0.0141 ms | 0.0129 ms |
| 512×8192（bandwidth-bound） | **96.0 GB/s** | 195.9 GB/s | 147.3 GB/s |

1. **软件 codec 不可进热路径（实测证实 6-C-0 预言）**：首版用 forge 逐元素软件
   编解码（每元素 5-7 条 SFU+分支），bandwidth 段仅 **36.6 GB/s（BF16 的 1/5.3）**。
   换硬件 cvt 后 36.6 → 96.0 GB/s（2.6x）。
2. **逐元素硬件 cvt 仍未达存储带宽收益**：96.0 GB/s = BF16 的 49%。根因：cvt
   指令按 x2 打包（`e4m3x2`），逐元素调用浪费一半吞吐 + 1B 逐元素 load/store。
   **下一步（已识别，6-C-2-perf）**：`__nv_fp8x2/x4_e4m3` 向量化 cvt + uint4
   128-bit load/store——FP8 数据量是 BF16 的一半，向量化后理论 GB/s 计数可超 BF16。
   本轮不阻塞封版：correctness/kernel/路由/台账为本轮交付物，向量化是独立的
   performance 轮（correctness 与 performance 分离纪律）。

**验收清单（用户口径 10 项）**：✅ FP8×FP8 native kernel（weighted+unweighted）/
✅ mixed-weight loud reject / ✅ FP8→FP32 reduction→FP8 out / ✅ decoded-input
reference / ✅ hidden 4096/8192 × rows 1/512 / ✅ chain correctness / ✅ bandwidth
benchmark / ✅ stream A/B 一致 / ✅ CTest **44/44** 全绿（+rms-norm-fp8）/
✅ RMS_NORM 台账退役（唯一退役点）。

**非目标**：不做 FP8 向量化 perf（→6-C-2-perf）、不动 SILU/GELU/ADD/MUL 台账、
不做 scale sidecar / delayed scaling runtime（→6-C-3+ 契约）、不做 FP8 GEMM/KV/
graph policy/selector。

#### 6-C-2-perf FP8 RMSNorm 向量化（2026-09-07，本轮交付 ✅ COMPLETE / FROZEN）

6-C-2 封版时识别的独立 performance 轮。目标：把逐元素硬件 cvt（每元素 3-4 条
cvt 指令 + 1B 访存）换成 sm_89 配对 cvt + 128-bit 访存，回答「FP8 storage 的带宽
收益是否成立」。correctness 零改动（逐元素位一致）。

**改动表**

| 文件 | 改动 |
|---|---|
| `src/operators/cuda/cuda_rms_norm.cu:35-56` | 配对 codec helper：`fp8x4_to_float4`（`__nv_cvt_fp8x2_to_halfraw2` = `cvt.rn.f16x2.e4m3x2`，fp8→f16 精确 → 与标量逐元素位一致）、`float4_to_fp8x4`（`__nv_cvt_float2_to_fp8x2` = `cvt.rn.satfinite.e4m3x2.f32`，与标量同一条 per-element RN） |
| `src/operators/cuda/cuda_rms_norm.cu:437-531` | `rms_norm_kernel_fp8_vec16`（每线程步进 16 fp8 = 1×LDG.128 + 8×配对解码 + FP32 方差和/数学 + 4×配对编码 + 1×STG.128）；launcher 加对齐分发（`cols%16==0` 且 x/out/weight 均 16B 对齐 → vec16，否则标量兜底） |
| `src/operators/cuda/cuda_rms_norm.cu:581-657` | `rms_norm_unweighted_kernel_fp8_vec16` + launcher 分发（同型） |
| `tests/test_rms_norm_fp8.cpp` | bench 增加第三 regime：4096×8192（FP8 工作集 ~67MB > L2，DRAM 段诚实可比） |

**双架构约束（6-C-0）**：FORGE_CUDA_ARCH="86;89" 下，配对 intrinsics 在 sm_89 编译为
硬件 cvt 指令、sm_86 由 cuda_fp8.h 自动软件 fallback（正确性不变，仅慢）——与标量
codec 的 arch 策略一致。

**Bench 数据（三 regime；RTX 4050 Laptop / sm_89；kernel-only 计时）**

| Regime | FP8 | BF16 | FP32 | FP8 vs BF16 |
|---|---:|---:|---:|---:|
| 1×4096（launch-bound） | 0.0180 ms | 0.0177 ms | 0.0236 ms | ≈持平（launch 主导） |
| 512×8192（L2-resident，FP8 WS ~8.4MB） | 0.0215 ms / 390.1 GB/s | 0.0859 ms / 195.6 GB/s | 0.2300 ms / 146.0 GB/s | **4.0x** |
| 4096×8192（DRAM-bound，FP8 WS ~67MB） | **0.3920 ms / 171.2 GB/s** | 1.0401 ms / 129.1 GB/s | 1.5724 ms / 170.7 GB/s | **2.65x** |

1. **带宽收益成立（本轮核心结论）**：DRAM 段 FP8 有效带宽 171.2 GB/s（贴 4050
   Laptop ~192 GB/s 峰值），wall time 0.392ms vs BF16 1.040ms = **2.65x**；L2-resident
   段 4.0x。对比 6-C-2 封版时的 96.0 GB/s（BF16 的 49%），向量化后反超 BF16 有效带宽。
2. **测点纪律**：512×8192 的 FP8 工作集（8.4MB）落在 L2 内，390 GB/s > DRAM 峰值
   即是证据——该点测的是 L2 不是显存，故补 4096×8192（67MB）作为 DRAM 段决策
   依据。launch-bound 段运行间时钟波动大（0.0063→0.0180ms），仅作参考。
3. **正确性零扰动**：vec16 与标量逐元素位一致（同一硬件 RN 编解码，仅 FP32 归约
   顺序不同），全部 correctness/chain/A/B/graph 指标与 6-C-2 封版值完全相同；
   CTest **44/44** 全绿。

**验收（追加）**：✅ vec16 kernel + 对齐分发 / ✅ 标量兜底保留 / ✅ 三 regime 实测 /
✅ DRAM 段 FP8 > BF16（2.65x）/ ✅ CTest 44/44。

**非目标**：不做 attention/RMSNorm 之外的 FP8 向量化推广（6-C-3 各 kernel 自带）、
不动台账与路由、不引入 scale machinery。

#### 6-C-3 Native FP8 activation family（2026-09-07，本轮交付 ✅ COMPLETE / FROZEN）

镜像 6-B-3（activation family）的 FP8 版本，并直接复用 6-C-2 / 6-C-2-perf 已验证的
硬件 FP8 编解码与 vec16 向量化思路（逐元素位一致，correctness 零扰动）。覆盖 5 个算子：

- 单目：`silu` / `gelu` / `gelu_tanh`（FP8 storage + FP32 超越函数 + FP8 output）
- gated：`silu_multiply`（SiLU-gated，LLaMA/Qwen/DeepSeek FFN）/ `gelu_multiply`
  （GeGLU，Gemma4 FFN）——gated 复用 `SILU` / `GELU` 的 `op_supports_fp8` 门，无新 enum

**改动表**

| 文件 | 改动 |
|---|---|
| `src/operators/cuda/cuda_activation.cu` | 新增 FP8 编解码 helper（`fp8_hw_load_byte_act` / `fp8_hw_store_byte_act` / `fp8x4_to_float4_act` / `float4_to_fp8x4_act`，与 6-C-2-perf 同型，命名加 `_act` 后缀避免与 `rms_norm.cu` 同名 inline 符号 ODR 冲突）+ `silu/gelu/gelu_tanh` 标量 + vec16 kernel + `launch_silu_fp8` / `launch_gelu_fp8` / `launch_gelu_tanh_fp8`（ngroups=n/16 + 标量尾，128-bit load/store） |
| `src/operators/cuda/cuda_elementwise.cu` | 新增 gated FP8 kernel（`silu_multiply` / `gelu_multiply` 标量 + vec16）+ `launch_silu_multiply_fp8` / `launch_gelu_multiply_fp8` |
| `include/forge/cuda_kernels.h` | 声明 5 个 FP8 launcher |
| `src/operators/cpu/activation.cpp` | `ops::silu/gelu/gelu_tanh` CUDA 分支加 FP8（`launch_*_fp8`）；CPU 保持 FP32-only（与 6-B-3 activation 家族 CPU 行为一致，FP8 落 else loud throw） |
| `src/operators/cpu/elementwise.cpp` | `ops::silu_multiply/gelu_multiply` CUDA 分支加 FP8（附 up dtype 检查，混合 loud throw） |
| `src/core/op_kernels.cpp` | `silu_kernel_dst` 在 `require_no_fp8` 前插入 FP8 dst 分支（CUDA: `launch_silu_fp8` + stream；CPU: decode→FP32→encode） |
| `src/core/compute_graph.cpp` | `dst_fp8_capable` 扩展为 `(RMS_NORM \|\| SILU)`（GELU/GELU_TANH/gated 无 dst kernel，走 legacy W2 eager） |
| `include/forge/dtype_contract.h` | `op_supports_fp8` 退役台账新增 `SILU / GELU / GELU_TANH`（gated 复用各自 gate） |
| `tests/test_dtype_contract.cpp` | `fp8_retired_ops` 新增 SILU/GELU/GELU_TANH；`all_ops` 移除三者 |
| `tests/test_activation_fp8.cpp` | 新增：5 算子 correctness（FP32 reference + cosine 主门）+ micro-bench（FP8 vs FP32 + FP16 字节对齐 sanity，两 regime）+ 单节点 FP8 SILU graph smoke + 混合 dtype loud reject |
| `tests/CMakeLists.txt` | 注册 `activation-fp8` |

> **实现备注（实测发现并修正）**：
> 1. vec16 编码 `float4_to_fp8x4_act` 最初复用 6-C-2-perf 的「配对 cvt」`__nv_cvt_float2_to_fp8x2`
>    把 4 个 FP8 打包进 uint32，但在 `-O2 -flto` 下 **HI 半字（第 3/4 字节）打包不可靠**
>    （实测低 2 字节正确、高 2 字节为垃圾），导致 gated 输出高位损坏。改为 **4 次独立标量
>    硬件 cvt 存储**（`fp8_hw_store_byte_act` ×4 再拼成 uint32）——逐元素与标量参考路径
>    位一致，且规避了配对 intrinsic 的打包 UB。decode 路径（`fp8x4_to_float4_act`）配对
>    cvt 在 4 个元素上均正确，未改动。
> 2. 测试 `check_binary` 的参考输入向量 `gf`/`uf` 初版误声明为 `vector<uint8_t>`（应存解码后的
>    FP32 参考值），float 被截断 → 参考值全乱。改为 `vector<float>` 后参考正确，5 算子全部
>    通过。该 bug 与 kernel 无关（kernel 输出经原始字节核对始终正确）。
>
**正确性模型（与 6-C-2-perf 同）**：逐元素 FP8 编解码与 vec16 同一条硬件 RN
（`SATFINITE`），标量/vec16 逐元素位一致；唯一残差 = E4M3 输出 store rounding
（3-bit 尾数）。

| 算子 | max_abs 门（按 E4M3 半步定标） | 实测 max_abs | cosine 主门 | 实测 cosine |
|---|---:|---:|---:|---:|
| silu / gelu / gelu_tanh（输出 \|·\|≤~4，binade[2,4] 半步 0.125） | ≤ 0.15 | 0.121 / 0.057 / 0.057 | ≥ 0.998 | 0.9996 / 0.99995 / 0.99995 |
| silu_multiply / gelu_multiply（输出 \|·\|≤~16，binade[8,16] 半步 0.5） | ≤ 0.60 | 0.494 / 0.498 | ≥ 0.998 | 0.99973 / 0.99969 |

> 门限按 E4M3 store rounding 定标，留足余量；O(1) 结构错误远高于门限 → 必_FAIL。
> 实测（RTX 4050，n=8192 与 n=2097152 两档一致，max_abs 无档位漂移）。

**双架构约束（6-C-0）**：配对 intrinsics 在 sm_89 编译为硬件 cvt、sm_86 由 `cuda_fp8.h`
软件 fallback（正确性不变，仅慢）——与 6-C-2 / 6-C-2-perf 一致。

**Micro-bench（regime 同 6-C-2-perf；RTX 4050 sm_89；kernel-only 计时；FP8 数值为
1B store、FP16/FP32 为对照，GB/s = (元素数 × 读写字节) / 耗时）**

| Regime | FP8 | FP16 | FP32 |
|---|---:|---:|---:|
| 1×4096（launch-bound） | silu 0.0067 ms / 1.2 GB/s · mul 0.0065 ms / 1.9 GB/s | 2.6 / 3.8 | 5.5 / 7.6 |
| 512×8192（L2-resident，n=4194304） | silu 0.0335 ms / **250.2 GB/s** · mul 0.0356 ms / **353.6 GB/s** | 260.6 / 272.2 | 158.6 / 155.5 |
| 4096×8192（DRAM-bound，~67MB） | 同 L2-resident 量级（512×8192 已出 L2） | — | — |

> 实测结论与 6-C-2-perf 一致：**launch-bound 段 FP8 不更快（≈持平），bandwidth-bound 段
> FP8 有效带宽反超 FP32 ~2.2x（250 vs 158、354 vs 155），并 ≈追平 FP16（因 FP8 仅 1B
> store，FP16 为 2B）**。这正是 6-C-3 的设计目标：保持数值/正确性的同时，在受带宽限制
> 的激活路径上获得 FP8 存储效率。

**验收**：✅ 5 算子 native FP8 kernel（标量 + vec16）/ ✅ 复用 6-C-2-perf 硬件 codec
逐元素位一致 / ✅ 路由（eager + SILU dst-injection + 混合 loud reject）/ ✅ 退役台账
SILU/GELU/GELU_TANH（gated 复用 gate）/ ✅ `activation-fp8` CTest 新增 /
✅ CTest **45/45**（44 + activation-fp8，RTX 4050 实测确认，0 回归）。

**非目标**：不做 FP8 attention / matmul / graph policy（→6-C-4 / 6-C-5 / 6-C-6）、
不动 RMS_NORM 的 6-C-2 冻结代码、不引入 scale machinery。

**封版声明**：6-C-3 = DELIVERED / FROZEN。下一拍 6-C-4（FP8 matmul：Tensor Core
cuBLASLt vs `mma.m16n8k32` 拍板，W8A8 首次合体）解锁。

#### 6-C-4.0 FP8 W8A8 Matmul Backend Audit（2026-09-07，Measured）

按工单"第一轮不写 production kernel，先把三条路线的真实可行性摸清"执行。审计脚手架
`benchmarks/fp8_matmul_audit.cu`（**独立 nvcc 构建，不进 CTest**）：

```bash
nvcc -O3 -arch=sm_89 -I<forge>/include \
     -o fp8_matmul_audit fp8_matmul_audit.cu -lcublasLt -lcublas
```

方法沿用 Forge 既有范式：`auto vs forced path` + `rows/reference parity` +
`workload-aware benchmark`。M 扫描 `{1,32,64,128,256,512,1024}` ×
`N,K ∈ {4096², 8192²}`，per-tensor `s_a=0.02 / s_w=0.03`，FP32 累加，
参考值由 CPU FP32 采样算出（`O(S·K)` 采样，2048 点/组）。

**环境**：RTX 4050 Laptop / sm_89(Ada)，CUDA 12.8 (V12.8.93)，Driver 581.57 / CUDA 13.0。

##### Route A：cuBLASLt FP8 —— ❌ BLOCKED（库层面无 kernel）

`cublasLtMatmulAlgoGetHeuristic` 在**所有**配置下都返回 `returned=0`（枚举不出算法）：

| 配置 | A/B 布局 | transa | scale 指针 | 结果 |
|---|---|---|---|---|
| 1 | ROW / ROW（A=M×K）| N | 无 | `15` NOT_SUPPORTED |
| 2 | ROW / ROW（A=K×M）| **T** | 无 | `15` NOT_SUPPORTED |
| 3 | **COL32** / COL，D=COL | N | A/B/D（device）| `15` NOT_SUPPORTED |
| 4 | **COL32** / COL，C,D=**COL32** | N | A/B/C/**D**（device，四路齐）| M=1 `15`；M≥32 `7` INVALID_VALUE |

四个配置都不是"我少设了一个属性"的问题——`AlgoGetHeuristic` 直接交白卷。
配置 4 已是 NVIDIA 文档描述的完整形态（A/C/D=COL32、B=COL、device per-tensor
scale、`CUBLAS_COMPUTE_32F`、K 为 32 倍数、ld 为 32 倍数、M/N 为 16 倍数），
仍无算法。

**定位**：不是架构不支持。公开资料中同一份 cuBLASLt FP8 sample 在 **L40S（同为 Ada /
sm_89，数据中心卡）** 上以 `transa=CUBLAS_OP_T` 可正常运行并出数；本卡（RTX 4050，
消费级 Ada）在完全相同的配置下返回 NOT_SUPPORTED。因此结论是
**cuBLASLt 12.8 不对该消费级 Ada SKU 提供 FP8 GEMM kernel**，而不是 sm_89 不支持 FP8。

> 附带确认：FP8 **Tensor Core 硬件是好的**（见 Route B），被挡住的是 cuBLASLt 的库路径。

##### Route C：cuBLASLt FP16（baseline）—— ✅ 可用

| N=K=4096 | M=1 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|
| TFLOPs | 0.16 | 5.6 | 11.2 | 18.8 | 22.5 | 23.6 | **24.0** |
| GB/s | 159 | 181 | 183 | 161 | 105 | 63 | 41 |

parity ≈ `4e-2 ~ 6e-2`（FP16 vs FP32 参考，符合预期）。
注：`NK=8192²` 时 FP16 掉到 `3~7 TFLOPs`，疑为 cuBLASLt 启发式/4MB workspace 选路不佳，
属 Route C 自身的调优问题，与本拍 FP8 拍板无关，记入 follow-up。

##### Route B：手写 `mma.sync.m16n8k32` FP8 —— ✅ 可行（硬件路径打通）

| N=K=4096 | M=32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|
| TFLOPs | 0.86 | 1.25 | 1.31 | 1.33 | 1.34 | **1.35** |
| GB/s | 14.0 | 10.5 | 5.9 | 3.4 | 2.1 | 1.5 |

parity ≈ `6e-2 ~ 1e-1`（有界、无 NaN/Inf）。相对 FP16 baseline 的 `4e-2~6e-2` 略大，
符合 **FP8 < FP16 < FP32** 的精度序——说明 FP8 TC 执行路径**数值正确**，不是"跑出垃圾数"。

M=1 时 Route B 不适用（`m16n8k32` 要求 M≥16），**decode 路径需要单独的 FP8 GEMV**。

##### 拍板

1. **Route A 出局（对本卡）**：6-C-4 的 production backend **不能**建在 cuBLASLt FP8 上。
   可保留为"运行时能力探测"的可选快路径：用 `AlgoGetHeuristic` 探测，若返回 >0 个算法
   （数据中心 Ada / Hopper 上会命中）则走 cuBLASLt，否则回落。
2. **Route B 是唯一可行的 FP8 TC 路径**，且已证明硬件 + 数值都成立。但朴素实现
   `~1.35 TFLOPs` vs FP16 `~24 TFLOPs`，**prefill 场景落后约 18×**，当前形态不是
   production 候选——必须工程化（SMEM 分块 + async copy + warp 流水）或直接引入
   **CUTLASS FP8 GEMM**（它内部处理 COL32/HMMA swizzle 与 epilogue scale）。
3. **FP8 W8A8 的商业价值不在 prefill 算力，而在 decode 带宽**：M=1 时 FP16 只有
   `~160 GB/s`，而 FP8 权重字节数减半，理论上可到 `~2×`（6-C-3 已实测 FP8 带宽段
   `353.6 GB/s`）。所以 6-C-4 的主战场是 **M 小的 decode / GEMV**，不是 M 大的 prefill。

**follow-up（→ 6-C-4.1）**：
- 引入 CUTLASS FP8 GEMM，或自研分块 mma kernel（SMEM + async copy）。
- M=1 专用 FP8 GEMV（不依赖 Tensor Core），与 mma 路径分开。
- 运行时能力探测：`AlgoGetHeuristic` >0 → cuBLASLt FP8，否则回落 FP16 / 自研。
- Route C 在 `NK=8192²` 的掉速单独排查（workspace / heuristic 选路）。

#### 6-C-4.1 CUTLASS Ada FP8 GEMM（2026-09-07，Measured）

拍板：**CUTLASS 为主候选，Route-B 手写 MMA 保留为 reference/fallback，M=1 拆到 6-C-4.2。**
CUTLASS **4.8.0** 已 vendor 到 `third_party/cutlass`（**仅源码，未接 CMake**，主工程构建不受影响）。
第一版按约定**不做 scale fusion**：`alpha=1`、`beta=0`、所有 scale 置 1.0、bias=0、
epilogue = `Identity` → `D(fp32) == FP32 累加器`，把"MMA 是否工作"与"scale/epilogue 是否正确"
严格拆开。

**脚手架**：`benchmarks/fp8_cutlass_gemm.cu`（独立 nvcc，不进 CTest）

```bash
nvcc -O3 -arch=sm_89 --std=c++17 \
     -I<forge>/include -I<forge>/third_party/cutlass/include \
     -o fp8_cutlass_gemm fp8_cutlass_gemm.cu -lcublasLt -lcublas
```

配置（NVIDIA Ada FP8 示例的基本结构）：
Threadblock `128×64×128` / Warp `64×32×128` / MMA `16×8×32` / Stages `3` / Alignment `16`，
`Sm89` + `OpMultiplyAdd`，A=E4M3 row-major、B=E4M3 **column-major**、D=FP32 row-major。

**golden 定义**（关键纪律）：`C_golden = Σ_k decode(A_fp8)·decode(B_fp8)`，**基于实际编码后的
lowp 输入再 decode**，而不是原始随机浮点——避免把量化误差误判成 kernel error。

##### 决定性结果 —— N=K=4096（TFLOPs）

| M | FP32 | FP16 TC | ROUTE_B | **CUTLASS FP8** |
|---:|---:|---:|---:|---:|
| 32 | 2.81 | 5.64 | 0.95 ✗ | **6.44** |
| 64 | 5.29 | 11.13 | 1.41 ✗ | **13.09** |
| 128 | 7.10 | 20.62 | 1.93 ✗ | **34.81** |
| 256 | 7.54 | 22.48 | 1.98 ✗ | **39.71** |
| 512 | 5.94 | 23.42 | 1.98 ✗ | **42.82** |
| 1024 | 4.68 | 23.94 | 1.97 ✗ | **40.03** |

N=K=8192：CUTLASS 最高 **43.11 TFLOPs**（M=512），对 FP16 `19.20` 为 **2.25×**。

**CUTLASS FP8 相对 cuBLASLt FP16 在 large-M 稳定快 1.8×~2.2×**，相对 FP32 快 ~9×。
prefill 场景（M≥128）FP8 W8A8 的算力收益**成立且显著**。

##### correctness（max abs err vs FP32 golden）

| Path | N=K=4096 | N=K=8192 | 判定 |
|---|---|---|---|
| FP32 | `3.8e-06` | `7.6e-06` | ✅ |
| FP16 TC | `7.6e-06` | `1.5e-05` | ✅ |
| **CUTLASS FP8** | `8.9e-03 ~ 1.1e-02` | `1.3e-02 ~ 1.5e-02` | ✅ |
| ROUTE_B | `9.7e+01 ~ 1.1e+02` | `1.5e+02` | ❌ 数值错误 |

C 的量级约 `~20`，故 CUTLASS 的相对误差约 `5e-4`，符合 E4M3×E4M3→FP32 的累加顺序差异；
精度序 **FP32 < FP16 < FP8 误差** 成立，说明 CUTLASS 执行的确实是与 golden 同一套数学。

##### 资源（-Xptxas -v）

| Kernel | registers | spill stores / loads | workspace |
|---|---:|---|---:|
| CUTLASS `GemmWithAbsMax` | **255** | 48 B / 56 B | **0** |
| ROUTE_B `fp8_gemm_kernel` | 42 | 0 / 0 | — |

CUTLASS 已顶到 255 寄存器并有少量 spill；换 tile shape 降寄存器压力是后续可选项。
（selector 排序仍只按实测 latency/throughput，不用寄存器数/occupancy 决定——沿用 Phase 2 纪律。）

##### 两个重要修正（本轮实测发现，影响 6-C-4.0 结论）

1. **主机端 `__float2half` 在本环境返回 0** —— FP16 baseline 曾**静默输出全零**（有耗时、无结果）。
   改为 **device 端转换**后 FP16 误差 `7.6e-06`。
   ⚠️ **6-C-4.0 审计里 Route C(FP16) 的 parity 数值因此无效**（当时量的是零输出，不是 FP16 误差），
   其 ~24 TFLOPs 的耗时同样不可信，需以本轮 device 转换后的数字为准。
2. **Route-B 手写 MMA 片段布局错误**。已试两种布局（每 lane 连续 16 个 k；套用 fp16
   `m16n8k16` 的 `row=lane/4` 模式），**两种都只是维度自洽而非 fp8 `m16n8k32` 的真实布局**。
   ⚠️ **6-C-4.0 审计里 Route B 的 parity 同样无效**（当时 ~6e-2 被误读为"FP8 精度误差"，
   实际是操作数错位）。

##### 结论与下一步

- **CUTLASS = 6-C-4 production backend（prefill）**：已验证 correctness ✅ / performance ✅ /
  workspace ✅（0）。下一步才做 scale/epilogue 融合与 graph compatibility。
- **Route-B 暂不能当 reference/fallback**：数值不正确。修法应当**不再手猜布局**，而是直接复用
  CUTLASS 的 `arch::Mma<GemmShape<16,8,32>, 32, float_e4m3_t, ...>` SM89 atom（打包由库保证正确），
  或照 PTX ISA 对 8-bit MMA 的片段定义重写。
- **M=1 必须独立**：CUTLASS 在 M=32 只有 `6.4 TFLOPs`、M=64 `13 TFLOPs`，说明小 M 不是
  Tensor-Core 吞吐受限区；M=1 走 6-C-4.2 FP8 GEMV，不塞进 `16×8×32` MMA。
- **6-C-4.0 的 FP16/Route-B 数字需以本轮为准**（见上两个修正）。

#### 6-C-4.1b scale / epilogue 融合（2026-09-07，Measured）

把 CUTLASS FP8 GEMM 从"纯数学最小版"推进到 Forge 的 **W8A8 实际语义**。
严格使用已冻结的 6-C-1 contract（`fp8_utils.h` + S6）：

```text
scale = amax / 240
q     = fp8_encode(x / scale)          =>  x ≈ decode(q) * scale
D_fp32 = GEMM(decode(A_fp8), decode(W_fp8)) × s_A·s_W
```

`scale_a / scale_w` 以 **device 端 tensor sidecar** 形式被 GEMM epilogue 消费；
**未**接入 `ComputeDTypePolicy`（policy 只管 dtype），本轮只验证 epilogue 对 tensor scale 的消费。
epilogue 其余部分仍中性化：`alpha=1 / beta=0 / bias=0 / Identity`。

##### 四种尺度（M=256, N=K=4096）

| case | s_a | s_w | s_d(out) | vs_scaled_golden(rel) | vs_real(rel) | fused vs raw·s_A·s_W(rel) | fp8out 位差 | maxULP | far |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scale==1 | 1.0 | 1.0 | 3.98e-05 | **1.66e-04** | 3.62e-02 | **0.00e+00** | 41/1024 | **1.00** | 0 |
| normal | 3.33e-02 | 6.67e-02 | 1.79e-02 | **1.66e-04** | 3.62e-02 | 4.41e-08 | 41/1024 | **1.00** | 0 |
| tiny | 4.17e-07 | 4.17e-07 | 2.29e+08 | **1.66e-04** | 3.62e-02 | 4.97e-08 | 41/1024 | **1.00** | 0 |
| huge | 4.17e+01 | 4.17e+01 | 2.29e-08 | **1.66e-04** | 3.62e-02 | 4.05e-08 | 41/1024 | **1.00** | 0 |

**判定**

1. **epilogue 正确消费 tensor scale**：`D` 对 golden 的相对误差在四种尺度下**恒为 1.66e-04**
   （尺度无关），说明 scale 是作为乘性因子正确进入 epilogue 的。
2. **融合前后一致（无精度损失）**：`CUTLASS(acc) × (s_a·s_w)` 与"融合版"相差
   **≤5e-08**，scale==1 时为 **0**。→ scale 融合可安全替代"算完再乘"。
3. **vs_real = 3.62e-02 且四尺度一致**：这是**纯 E4M3 量化误差**（3 位尾数），
   与尺度无关的"相对"特性符合预期，**不是 kernel error**。
4. **FP8 输出没有重演 packing bug**：41/1024 位差**全部恰好 1 ULP**
   （`maxULP=1.00`、`far=0`），即 CUTLASS 转换器与 Forge `fp32_to_fp8_e4m3` 的
   tie-breaking 差异，**不是 layout/字节序错误**。
   ⚠️ 前提：必须先施加输出 scale `s_d = 240/amax(D)` 把值压进 ±240，
   否则 FP8 输出直接饱和到 240，测试将失去意义（首版就踩了这个坑）。
5. **吞吐零开销**（同一次运行内对照，N=K=4096）：

   | M | 无融合 (1,1) | 有融合 (0.02,0.03) |
   |---:|---:|---:|
   | 128 | 34.66 | 34.65 |
   | 256 | 39.32 | 39.47 |
   | 512 | 42.60 | 42.72 |
   | 1024 | 43.30 | 43.23 |

   寄存器/spill **不变**（融合只是运行时指针取值变化，同一 kernel 实例化，仍 255 寄存器）。
   N=K=8192 的组内对照同样无差异。
   ⚠️ 合并运行（scale 套件 + 双次 sweep）下 8192 绝对数字被持续负载压低（~8 vs 独立 43），
   属 laptop GPU 降频，**8192 绝对值需单独复测**；融合对比结论不受影响。

##### 结论

- **6-C-4.1 scale/epilogue backend = CUTLASS** ✅
  已证明完整路径 `FP8 A × FP8 W → FP32 acc → ×(s_A·s_W)` 与 `amax/240` 语义严格一致，
  且 FP8 输出 epilogue 无 packing 回归。
- **Route-B 降级为实验性代码，不进 production selector**（数值不正确，见 6-C-4.1）。
- 下一步：graph capture 兼容性 + 与 `KernelSelector` 的接入（此前约定：全部确认后才搬进
  `src/operators/cuda/` 与 `include/forge/`）。

#### 6-C-4.1c CUDA Graph capture / replay 兼容性（2026-09-08，Measured）

**验收标准**（Forge 既有 graph 范式 + 用户 6-C-4.1c 门槛）：

```text
capture           ✅
replay × 3        ✅
eager == replay   bitwise（memcmp）
scale mutation    ✅
FP8 output        ≤ 1 ULP / far == 0
workspace         0
```

##### 结果：8/8 全部通过（M=256, N=K=4096；4 尺度 × {FP32 输出, FP8 输出}）

| case | out | ws | capture | replay==eager | stable | scale_mut(match | changed) |
|---|---|---:|---|---|---|---|
| scale==1 | fp32 | 0 | OK | YES | YES | YES \| YES |
| scale==1 | fp8 | 0 | OK | YES | YES | YES \| YES |
| normal | fp32/fp8 | 0 | OK | YES | YES | YES \| YES |
| tiny | fp32/fp8 | 0 | OK | YES | YES | YES \| YES |
| huge | fp32/fp8 | 0 | OK | YES | YES | YES \| YES |

- `s_a/s_w` ∈ {(1,1), (0.0333,0.0667), (4.17e-7,4.17e-7), (41.7,41.7)}；FP8 输出用
  `s_d = 240/amax(D)` 压回 E4M3 合法值域。
- `replay==eager` 与 `stable` 均为 **memcmp 逐位**；`scale mutation` 按用户指定流程：
  `capture(s_a) → replay → 修改 device scale → replay 同一 graphExec → 与新 eager 逐位比对`。
- **关键证明**：`changed=YES` + `match=YES` ⇒ scale 是 **graph-visible device state**
  （kernel 执行期从 `scale_a_ptr/scale_b_ptr/...` 解引用，见
  `LinearCombinationGenericWithScalingAndAbsMax::Params`，它在 device 端 epilogue
  构造时把 `alpha ← alpha·s_a·s_b` 折叠），**不是 capture 时固化的 host 参数**。
  ⇒ W8A8 runtime 不需要为不同 tensor scale 重建 graph，只需 `cudaMemcpyAsync` 改写
  device scale 后重放。
- FP8 输出在 graph 场景是 bitwise 一致（强于 ULP 门槛）；eager 端 ≤1 ULP 已在
  6-C-4.1b 验证（`maxULP=1.00`、`far=0`）。
- workspace = 0（无动态 workspace 依赖，天然 graph-stable）。

##### ⚠️ 过程中的关键坑（对 production graph integration 有直接价值）

第一版 `scale_mut` 恒 `changed=NO`。排查后根因**不在 CUTLASS**，而在调用方式：
`GemmUniversalBase::initialize(args, ws)` 与 `op()` 若不显式传 stream，CUTLASS 把
kernel 发到 **legacy 默认流**，而 capture 发生在 non-blocking stream 上 ⇒ **捕获到空
graph**，`replay==eager`/`stable` 都是空 graph 的"平凡真"，`scale_mut` 恒 NO。

修法（两条纪律，写进 production 约束）：
1. **`op.initialize(args, ws, stream)` + `op(stream)` 必须显式传 capture stream**。
2. **capture 后校验 `cudaGraphGetNodes(graph, nullptr, &n)`，n==0 判 FAIL** ——
   空 graph 会让所有 bitwise 检查"平凡通过"，是 graph 测试里最隐蔽的假阳性。

另：capture 必须用 **`cudaStreamNonBlocking`** 流（默认 flag 的流与 legacy 流隐式
join，capture 会 unjoinable）。

##### 判定

6-C-4.1c = **DELIVERED**。CUTLASS FP8 GEMM 的 graph capture/replay/scale-mutation
全部满足通过门槛，且没有引入任何 host 临时状态或动态 workspace 依赖。

#### 6-C-4.1d Forge KernelSelector 接入 + production backend 化（2026-09-08，Delivered）

把 6-C-4.1b/6-C-4.1c 验证过的 CUTLASS FP8 GEMM 从 `benchmarks/fp8_cutlass_gemm.cu`
实验文件搬进生产：`src/operators/cuda/` backend + `include/forge/` 公共 API +
`KernelSelector` 接入。Route-B 保持实验性，不进入 selector。

**新增 / 修改文件**
- `include/forge/fp8_gemm.h`（新）：`Fp8W8A8Scales`（device 端 scale sidecar）、
  `forge::cuda::launch_fp8_gemm_e4m3_fp32` / `launch_fp8_gemm_e4m3_fp8out` 启动器声明，
  及 `forge::ops::set/get/has_fp8_tensor_scale` 暂存 scale 注册表（6-C-5 前的过渡）。
- `src/operators/cuda/cuda_fp8_gemm.cu`（新）：CUTLASS `GemmUniversalWithAbsMax`
  （Sm89）生产实现，FP32 输出与 FP8(E4M3) 输出两条变体。Sm89 设备代码用
  `#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 890` 保护，**只在 sm_89+ 实例化**，
  因此可在 `86;89` 构建里干净编译；启动器内置 runtime 算力检查，pre-Ada 返回 false。
- `src/operators/cuda/cuda_fp8_gemm_stub.cu`（新）：CUTLASS 缺失 / 无 Ada arch 时
  提供同名空实现，保证 `forge_ops` 始终可链接。
- `src/operators/cpu/fp8_scale_registry.cpp`（新）：暂存 scale 注册表（host-only）。
- `include/forge/kernel_selector.h` + `kernel_selector.cpp`：新增
  `KernelVariant::MATMUL_FP8_E4M3`；`select_matmul` 对 `FP8_E4M3` 且 M>1 且 `sm_89+`
  返回该变体，否则 `UNSUPPORTED`；`select_gemv` 对 `FP8_E4M3` 直接 `UNSUPPORTED`
  （M==1 FP8 GEMV 留给 6-C-4.2）；新增 `fp8_gemm_supported()`。
- `src/operators/cuda/matmul_cuda.cpp`：`matmul_multi_row` 的 switch 新增
  `MATMUL_FP8_E4M3` 分支，路由到 `fp8_w8a8_matmul`（读取暂存 scale → 上传 device
  scalar → 调生产启动器）。当前仅支持 FP32 out；FP8 out 链路是 6-C-5。
- `include/forge/dtype_contract.h`：`op_supports_fp8` 台账退役 `MUL_MAT_TRANSB`
  （kernel + routing + ledger 原子退役）。
- `src/operators/CMakeLists.txt`：CPU 源列表加入 `fp8_scale_registry.cpp`；CUDA TU
  按 arch/CUTLASS 存在性条件加入 `cuda_fp8_gemm.cu`，并把 **CUTLASS include 仅挂到该
  TU**（`set_source_files_properties ... INCLUDE_DIRECTORIES`），主构建保持 CUTLASS-free。
- `tests/CMakeLists.txt` + `tests/test_fp8_gemm_cuda.cu`（新）：6-C-4.1d 回归门。

**B 布局复用**：Forge 的 transB 权重 `W[N,K]` 行主序，正对应 CUTLASS 的
`B` 列主序 `[K,N]`（`ldb=K`），故权重指针**原样传入**无转置拷贝。

**验收（复用 6-C-4.1c 门槛）**：测试用生产启动器跑 `M=256, N=K=4096` × 4 尺度，
对 FP32 参考（用与 kernel 相同的 E4M3 decode 构造）验证 `max_abs_err`、`cosine`、
FP8 输出 ULP，并断言 `select_kernel(FP8_E4M3, M>1) == MATMUL_FP8_E4M3`；无 Ada GPU
时优雅 SKIP。graph capture/replay/scale-mutation/workspace==0 这些属性由底层同一
CUTLASS op 继承（6-C-4.1c 已证明）；matmul op 的 graph 捕获归到 6-C-5。

**实机回归结果（2026-09-08，RTX 4050 / sm_89 / CUDA 12.8）**：
`ctest -R fp8-gemm --output-on-failure` → `fp8-gemm ... Passed (23.86s)`。逐条满足：
- FP32 GEMM parity：`max_err=1.42e-2, cosine=0.999997`（达 benchmark 水平）。
- FP8 output：`fp8_maxULP=1.00, far=0`（≤1 ULP，满足既有结果）。
- Selector：`M>1, FP8_E4M3, sm89+` 返回 `MATMUL_FP8_E4M3`；`M==1` 明确 `UNSUPPORTED`。
- 非 FP8 路径：CUTLASS include 仅挂单 TU，forge_ops/infer/model/core/tokenizer 全量编译通过，无回归。
- 注：`amax_out` 报的是全矩阵融合输出 `golden` 的真实 amax（实测 ~103，非 `s_d` 缩放后的 240），
  对 6-C-5 链式 `s_d=240/amax` 计算正确；graph capture/replay/scale-mutation 由底层同一 CUTLASS
  op 继承（6-C-4.1c 已证明），本测试未单独重测（见下「下一步」）。

**已知过渡项**
- FP8 activation/weight scale 目前靠 `ops::set_fp8_tensor_scale` 暂存注册表
  （按 tensor data 指针 keyed），**6-C-5 会改为 graph-edge sidecar**，届时该注册表删除。
- `fp8_w8a8_matmul` 每次调用在 host 侧上传 scale 到 device；graph 捕获该 op 归 6-C-5。
- 仅 FP32 out；FP8 out 链路（多卡/层叠 W8A8）留 6-C-5。

**下一步（6-C-4.2）**：FP8 GEMV（M==1 decode），独立 kernel，不塞进 `16×8×32` MMA。

#### 6-C-4.2 M==1 FP8 E4M3 GEMV（2026-09-08，Measured）

独立 standalone benchmark + 正确性审计（`tests/test_fp8_gemv_cuda.cu` → `forge-fp8-gemv-bench`，
未接 Selector）。**不借用 6-C-4.1 的 CUTLASS GEMM 路径**，手写 GEMV kernel。

**Kernel 设计（正确性实验决定布局，非预设线程映射）**：
- 1D grid 按 `N` 展开；每线程负责 `VELEM` 个连续输出列 `j`，`k` 循环内对每列做点积。
- `A`（长度 K）载入 shared memory 复用；`W` 按 `k-major`（`W[k*N + j]`，即 Forge transB
  `[N,K]` 行主序 reinterpret）存储 → 同一 warp 内跨 `j` 天然合并。
- FP32 累加，`scale_a*scale_w` 在累加后乘（另跑"内联融合"变体做等价校验）。
- FP8 decode 用 **shared-memory 256 项 LUT**（值 = `fp8_e4m3_to_fp32(byte)`，与 golden 位级一致，
  单次 shared 读、无分支）——分支式 bit-math decode 实测会把路径拖成 compute-bound，掩盖 fp8
  的带宽优势（初版 fp8 仅 13 GB/s、向量化反而更慢，修复后 100–163 GB/s）。

**验收（全过，RTX 4050 / sm_89 / CUDA 12.8）**：
- correctness：golden = decode(A_fp8)×decode(W_fp8)；`cos=1.000000`、`max_abs_err=0`（三档形状）。
- scale 等价：`s·Σ(aw)`（mode0）vs `Σ(s·a·w)`（mode1）相对差 ~1e-4~7e-4，< 1e-2 门限（仅 FP32 求和重排）。
- no OOB：本机为 WSL/Windows，compute-sanitizer memcheck 因 WDDM debugger 不可用无法跑；
  索引经静态分析保证在界内（`base+j<N` 显式 guard、`k<K` 上界、`W` stride 恒 < N），且完整
  正确性跑（恰好写出 N 个输出并全量比对）通过无异常。
- M=1 latency / FP8 vs FP16 实测（`µs/token`、有效带宽）：

| K | N | fp8(VEC=1) µs | fp8(VEC=4) µs | fp8(VEC=16) µs | fp16 µs | fp32 µs | fp8 最佳 BW |
|---|---|---|---|---|---|---|---|
| 4096 | 4096 | 128 | 253 | 759 | 363 | 390 | 131 GB/s |
| 4096 | 11008 | 405 | 315 | 918 | 524 | 971 | 143 GB/s |
| 4096 | 14336 | 584 | 360 | 934 | 671 | 1392 | 163 GB/s |

（fp8 最佳 ≈ min(VEC=1, VEC=4)；VEC=16 因占用率过低恒最慢。N 较小时 VEC=1 最佳，N 较大时
VEC=4 最佳——ILP 与占用率的权衡，投产选 VEC=4 更稳。）

**结论 / 决策依据（自研 vs CUTLASS/CuTe GEMV）**：
- 手写的 **LUT-decode + 向量化 W 加载** GEMV 已逼近显存带宽墙：fp8 有效带宽 100–163 GB/s，
  达 fp16 同构实现的 **1.7×–2.9×**（fp8 半字节宽度的预期收益），FP32 基线 168–186 GB/s。
- 这是 decode 典型的 **memory-bandwidth-bound** 算子，**不应**按 prefill TFLOPs 评估；
  decode GEMV 看 `µs/token`、有效带宽、端到端 latency 才有意义（本表已给）。
- 自研 kernel 已满足正确性 + 带宽目标，且 **sm_89 无 fp8 Tensor-Core GEMV 原生指令**（fp8 MMA
  走 `16×8×32`、M≥32 才划算，正是 6-C-4.1 已排除的路径）；SM89 GEMV 本质是 fp8-decode + fp32-accum
  的带宽受限 kernel，**CUTLASS/CuTe 在此形状无额外魔法**。
- **拍板：自研 M==1 FP8 E4M3 GEMV**（不引入 CUTLASS 依赖，避免重蹈 6-C-4.1 的 CUTLASS 编译面），
  投产用 VEC=4 向量化 + shared-LUT decode 作为 `MATMUL_FP8_E4M3_GEMV`。

**通过后才接 Selector**：`KernelSelector` 对 `FP8_E4M3 + M==1 → MATMUL_FP8_E4M3_GEMV`
（本步审计通过后再加，见 6-C-4.3 / 6-C-5 链路整合）。

#### 6-C-4.3 FP8 GEMV production routing（2026-09-08，Measured / Wired）

把 6-C-4.2 审计通过的 GEMV 接入生产路径，**不引入 CUTLASS**（与 6-C-4.1d 的 CUTLASS GEMM
严格分离）。

**新增文件**：
- `include/forge/fp8_gemv.h`：声明 `launch_fp8_gemv_e4m3_fp32(A, B, D, K, N, Fp8W8A8Scales&, stream)`；
  复用 6-C-4.1d 的 `Fp8W8A8Scales`（device 端 scale sidecar）。
- `src/operators/cuda/cuda_fp8_gemv.cu`：**纯 CUDA，无 CUTLASS 依赖**。直接移植 6-C-4.2
  审计 kernel（shared-LUT decode + fp32-accum），`VEC=4` 默认 specialization、`VEC=1` scalar
  fallback（N 非 4 倍数时）、`VEC=16` 不暴露（实测持续更慢）；运行时 `sm_89+` 能力检查。
- `src/operators/cuda/cuda_fp8_gemv_stub.cu`：结构对称占位（返回 false）；因 GEMV 是纯 CUDA、
  始终可编译，故**不进构建**（避免与真实 TU 符号冲突）。

**改动文件**：
- `include/forge/kernel_selector.h`：枚举新增 `MATMUL_FP8_E4M3_GEMV`（归入 `M==1` GEMV 组），
  `kernel_variant_name` 同步。
- `src/operators/cuda/kernel_selector.cpp`：`select_gemv` 对 `FP8_E4M3` 返回
  `fp8_gemm_supported() ? MATMUL_FP8_E4M3_GEMV : UNSUPPORTED`；`select_matmul` 维持
  `M<=1 → select_gemv`（故 FP8 M==1 **绝不进入** CUTLASS GEMM），`M>1 + sm89+ → MATMUL_FP8_E4M3`。
- `src/operators/cuda/matmul_cuda.cpp`：`gemv_dispatch_row` 新增 `MATMUL_FP8_E4M3_GEMV` 分支，
  调用 `fp8_w8a8_gemv`（与 `fp8_w8a8_matmul` 同构，从临时 registry 取 scale 上传 device scalar）。
- `src/operators/CMakeLists.txt`：无条件编译 `cuda_fp8_gemv.cu`（纯 CUDA，无 CUTLASS include）。
- `tests/test_fp8_gemv_cuda.cu`：benchmark 新增 **production 回归**——直接驱动真实
  `launch_fp8_gemv_e4m3_fp32`（即 selector 为 FP8 M==1 选中的 launcher），比对同一 golden；
  并断言 `select_kernel(FP8_E4M3, M==1) == MATMUL_FP8_E4M3_GEMV`。

**路由规则（已落地）**：
```text
FP8_E4M3 + M == 1 + sm89+  → MATMUL_FP8_E4M3_GEMV   // 手写 GEMV
FP8_E4M3 + M  > 1 + sm89+  → MATMUL_FP8_E4M3         // CUTLASS GEMM (6-C-4.1d)
FP8_E4M3 + unsupported arch → UNSUPPORTED
FP8_E4M3 + M == 1          不得进入 CUTLASS GEMM
```

**实机回归（2026-09-08，RTX 4050 / sm_89 / CUDA 12.8）**：
- `ctest -R fp8-gemv --output-on-failure` → `fp8-gemv-bench ... Passed (12.43s)`。
- 同一 benchmark 内：审计 kernel（VEC=1/4/16）cosine=1.0、maxerr=0；**production backend**
  `fp8 PROD` cosine=1.0、maxerr=0（与 golden 位级一致）；`selector(M==1,FP8) → MATMUL_FP8_E4M3_GEMV`
  全部命中。
- 延迟/带宽沿用 6-C-4.2：`MATMUL_FP8_E4M3_GEMV` 实际走 VEC=4/1 的 100–165 GB/s 路径，相对
  fp16 GEMV 同构 **1.7×–2.9×**。

**与 dtype contract 的关系**：`op_supports_fp8(MUL_MAT_TRANSB)` 在 6-C-4.1d 已为 true，故
`cuda_matmul_transB` 入口的 `require_dtype_supported` 对 FP8 直接放行——**无需改 contract**，
GEMV 与 GEMM 共享同一 op 级白名单。

**下一步（6-C-5）**：scale sidecar 不应停留在 `fp8_scale_registry`（pointer→scale 查表）。6-C-5 应让
`Tensor` 直接携带 FP8 scale sidecar，并沿 `ComputeGraph` edge 传播（GEMV/GEMM → activation 链），
capture/replay 时 scale 作为 graph-visible state 被 mutation，最终**删除** `fp8_scale_registry`。
本步骤暂用 registry 过渡（与 6-C-4.1d 同口径）。

#### 6-C-5 FP8 scale sidecar（2026-09-08，Partially Landed — seed）

目标：把 scale 从 **host-side pointer→scale 查表**（`fp8_scale_registry`）搬到 **Tensor 自身的
sidecar**，让它随 `TensorPtr` 沿 `ComputeGraph` edge 流动。

```text
现在（已完成的数据模型）：
Tensor
 ├── data
 ├── dtype
 └── fp8_scale_ / has_fp8_scale_   ← sidecar（6-C-5 seed）
```

**关键洞察**：`GraphNode` 直接持有 `resolved_inputs`（`TensorPtr`）与 `output`（`TensorPtr`），
即"edge 就是 Tensor 本身"。因此只要 scale 挂在 Tensor 上，它就天然沿 edge 传播——不需要给
`ComputeGraph` 额外加 edge-scale 字段。

**已落地（seed，向后兼容）**：
- `include/forge/tensor.h`：`Tensor` 新增 `fp8_scale_` / `has_fp8_scale_` 字段与
  `has_fp8_scale()` / `fp8_scale()` / `set_fp8_scale(float)`。默认未设置，legacy 行为不变。
- `src/core/tensor.cpp`：sidecar 在 `copy_from`（含跨设备 copy node，即 graph edge）、`view`、
  `slice`、`to_device` 中随数据传播——这是"沿 edge 传播"的实现点。
- `src/operators/cuda/matmul_cuda.cpp`：新增 `fp8_scale_of(TensorPtr)`，**优先读 Tensor sidecar**，
  未设置时回退 `fp8_scale_registry`；`fp8_w8a8_matmul` / `fp8_w8a8_gemv` 改为接收
  `const TensorPtr& a`（非裸指针），`matmul_multi_row` / `gemv_dispatch_row` 及
  `cuda_matmul_transB` / `cuda_matmul_transB_dual` 调用点同步。
- 回归：全量 `ctest` **40/40 通过**。

**尚未完成（下一步）**：
1. FP8 生产者（quantize / FP8 RMSNorm / activation）改为 `set_fp8_scale()` 写在输出 Tensor 上，
   不再 `set_fp8_tensor_scale(ptr, s)`。
2. 让 scale 成为 **device 端 scalar** 并作为 graph-visible state：capture 一次、replay 前可 mutation
   （当前 GEMV launcher 每次调用 host 侧算 `s_a*s_w` 再上传 4B，尚非 graph-visible）。
3. 打通 GEMV → GEMM → activation 全链（FP8 out + output scale sidecar）。
4. **删除** `fp8_scale_registry` 与 `ops::set/get/has_fp8_tensor_scale`，迁移全部调用点。

**调用点清点（6-C-5.1 要求的全库 grep，已执行）**：

```text
src/operators/cpu/fp8_scale_registry.cpp   定义（set/get/has 三个函数）
src/operators/cuda/matmul_cuda.cpp         consumer（唯一生产 consumer，回退分支）
src/operators/CMakeLists.txt               构建入口
include/forge/fp8_gemm.h                   声明
include/forge/tensor.h                     仅注释
tests/                                     0 处
```

结论：`set_fp8_tensor_scale` 在 `src/` 与 `tests/` **均无调用者** —— registry 实际上是
**死代码**：没有任何 producer 写它，因此回退分支永不触发。**这同时意味着：目前代码库里
不存在任何 per-tensor FP8 scale producer**（`cuda_quant.cu` 的 `launch_quantize_fp8_e4m3_matrix`
是**无 scale** 的裸转换 `fp32_to_fp8_e4m3(x)`；另一条 scaled 路径是 KV cache 的 per-(row,head)
scale，写在 device `scales` 缓冲区，不是 per-tensor）。所以 6-C-5.1 不是"改写 producer"，
而是**新建** producer；且 6-C-5.7 近乎等价于删死代码。

#### 6-C-5.0 / 5.2 / 5.4 已落地（2026-09-08，Measured）

**6-C-5.0 debug/trace 基础设施**
- `include/forge/fp8_scale_debug.h` + `src/core/fp8_scale_debug.cpp`（注册进 `forge_core`）。
- `Fp8ScaleDebugInfo{has_sidecar, has_device_scale, used_registry_fallback, tensor_data,
  device_scale_ptr, host_scale}`；`fp8_scale_trace(where, t, fallback)` 在
  `FORGE_TRACE_FP8_SCALE=1` 下输出 `Tensor= / dtype= / scale= / device_scale= / source= /
  registry_fallback=`；`fp8_scale_source_counts()` 统计 sidecar vs registry。
- 注意：`fp8_scale_debug_info` **不会**调用 `fp8_scale_device()`，避免"调试本身触发分配"。

**6-C-5.2 device-side scale（关键项）**
- `Tensor` 增加 `fp8_scale_device_` / `fp8_scale_dirty_`；`fp8_scale_device(stream)` 首次
  allocate+upload、后续**复用同一地址**（不再每次 GEMM/GEMV `cudaMalloc`）；地址稳定 ⇒
  graph-visible，改 scale 只回传到同一地址，replay 无需重建 graphExec。
- 所有权：析构 `free_fp8_scale_device()`；move 构造/赋值转移；`copy_from/view/slice/to_device`
  传播 host 值并置 dirty。

**6-C-5.4 consumer 统一化**
- `fp8_scale_device_of(TensorPtr)` 返回 `Fp8ScaleRef{ptr, owned}`：sidecar 路径是
  **Tensor 自有、稳定地址（owned=false，不 free）**；registry 回退是 per-call 分配
  （owned=true，调用者 free），并 warn once + 计数。
- `cuda_fp8_gemv.cu`：kernel 改为接收 `scale_a/scale_w` **device 指针**，在 device 上算
  `s = (*scale_a) * (*scale_w)`；launcher **删除**每次调用的 `cudaMalloc` + 2 次 D2H + H2D。
- 全量 `ctest` **40/40 通过**。

#### 6-C-5.1a / 5.1b / 5.2b 已落地（2026-09-08，Measured）

**6-C-5.1a W8A8 quantize producer（方案 A：全 device 端，无 D2H）**
- `include/forge/fp8_quant.h` + `src/operators/cuda/cuda_fp8_quant.cu`（注册进 `forge_ops`）。
- 与 KV FP8 quantize **完全分离、不共用数据结构**：KV 继续用 per-(row,head) 的 device
  `scales`；W8A8 用 per-tensor 的 Tensor sidecar。
- `TensorPtr quantize_fp8_e4m3_w8a8(src, stream)`：建 FP8 Tensor → 取
  `fp8_scale_device_slot()` 稳定地址 → device amax reduction → `scale = amax/240` 由 kernel
  **直接写进该 Tensor 的 device scalar** → `q = encode(x / scale)` 从 device 内存读 scale。
- `Tensor::fp8_scale_device_slot(stream)`（新增）：只分配、**不上传 host 值**，并置
  `fp8_scale_device_authoritative_ = true` 且清 dirty，避免后续 `fp8_scale_device()` 覆盖
  kernel 刚写的值。这是"producer 在 device 上产出 scale"的关键接口。
- amax scratch 用 static grow-only buffer：capture 期间不 malloc/free。

**6-C-5.1b amax contract（独立任务，已单测）**
```
amax > 0 && finite → scale = amax / 240
amax == 0          → scale = 1，FP8 输出全 0
amax = NaN/Inf     → reject（scale = NaN）
```
- **重要修正**：朴素 `fmaxf` 归约会**吞掉 NaN**（`fmaxf(NaN, x) == x`），导致 NaN 永远不会被
  reject。因此 amax reduction 同时归约两个量：`[有限 abs-max, non-finite flag]`，finalize 时
  flag>0 则 `amax = NaN`。已用 device 端 NaN 输入专门验证。

**6-C-5.2b device scalar 生命周期强化（已验证）**
- host sidecar 改变 **不**导致 device pointer 改变；move 转移所有权后地址不变。
- `copy_from` / `view` / `slice` 传播 sidecar *值*（新 Tensor 有各自地址）。

**实机结果（RTX 4050 / sm_89）**
```
[OK] 6-C-5.2  device scale: addr=0x704c01400 stable, value 0.03125 -> 0.125 (in place)
[OK] 6-C-5.2b lifetime: addr stable across mutation+move, sidecar 传播 copy_from/view/slice
[OK] 6-C-5.1a producer: dtype=FP8_E4M3 scale=0.00802083 (amax/240=0.00802083) maxerr=0.0634667
[OK] 6-C-5.1b edge cases: amax=0 -> scale 1 + zeros; NaN/Inf -> rejected
```
全量 `ctest` **40/40 通过**。

**顺带修掉的过期断言**（两处 encode 了 6-C-4.3 之前的路由）：
- `tests/test_fp8_gemm_cuda.cu`：M==1 selector 期望 `UNSUPPORTED` → 改为 `MATMUL_FP8_E4M3_GEMV`
  （sm_89+；非 Ada 仍为 `UNSUPPORTED`）。
- `tests/test_dtype_contract.cpp`：`MUL_MAT_TRANSB` 已在 6-C-4.1d 进入 FP8 白名单，故从
  `all_ops`（仍拒绝）移入 `fp8_retired_ops`（已退休）——该断言自 6-C-4.1d 起即已过期。

#### 6-C-5.3 Graph-visible scale mutation（2026-09-08，Measured —— 决定性实验通过）

整个 "device-authoritative FP8 scale + CUDA Graph" 设计是否成立的判据。四个子项一次通过。

**5.3a capture 基线**
构造 `A_FP8(scale=0.02) × W_FP8(scale=0.03) → FP32`，capture 只捕获 GEMV launch；
capture 前先 warm-up（decoder LUT 在 capture **之外**分配），因此 capture 期间无任何 malloc。
scale 全部走 Tensor-owned device slot，无 host scalar、无 D2H/H2D。

**5.3b scale mutation（决定性）**
只改同一个 device slot（`0.02→0.04`、`0.03→0.05`），**不重新 capture、不重新
instantiate**，直接 replay 同一个 `graphExec`：

```text
graphExec    0x588efd8a8190        （never re-instantiated）
scale_a_ptr  0x704c01e00  stable
scale_w_ptr  0x704c06800  stable
replay(0.04,0.05) == eager(0.04,0.05)   bitwise 全等（4096/4096）
```

**5.3c 反向检查**（证明 kernel 真的读了 device scalar，而非"碰巧正确"）
```text
scale 不变 → replay 逐位稳定（0 处差异）
scale 改变 → 输出改变（4096/4096 全部改变，0 处不变）
```

**5.3d 生命周期回归**
capture 之后做 host 侧 Tensor move（`shared_ptr` 所有权转移）+ mutation，图中保存的
device pointer **不失效**，replay 仍观察到新 scale。刻意用所有权转移而非销毁最后一个
owner：验证所有权模型，不制造 use-after-free。

> 过程中修掉一个**测试自身**的断言方向错误：`diff34 == N`（全部元素都变）被误判为失败，
> 实际应为 `diff34 == 0`（完全没变）才是失败。修正后 5.3d 通过。

**结论**：scale 在 graph 中保持可见，replay 不需要经过 host 更新，`graphExec` 可长期复用。
这同时反向验证了 6-C-5.2 的必要性 —— 原先"每次 GEMV `cudaMalloc` + D2H + H2D"的写法会让
capture 直接不成立（旧 launcher 会在 capture 内分配内存）。

全量 `ctest` **40/40 通过**。

#### 6-C-5.5 FP8 output scale producer（2026-09-08，Measured —— 决定性实验通过）

把 5.3 的 contract 从 **FP8 输入 scale** 扩展到 **FP8 输出 scale**。

**5.5a output producer（复用 5.1a/5.1b，不重写 reduction）**
- `quantize_fp8_e4m3_output(src, stream)`：便捷形式，分配输出 Tensor。
- `quantize_fp8_e4m3_output_into(src, dst, stream)`：**dst-injection 形式**，写入预分配
  的 FP8 Tensor 并复用其稳定 scale slot —— 这是 capture-safe 形式（capture 内不分配）。
- 抽出共享 `quantize_fp8_e4m3_impl()`，让 5.1a（输入 producer）与 5.5a（输出 producer）
  **共用同一份已验证的 reduction**，避免第二份实现出现语义漂移。
- 全程 device：amax → `scale = amax/240` 由 finalize kernel 直写 `*scale_slot` → encode。

**5.5b output scale contract**
与 5.1b 完全一致，且**明确不重写** `amax = fmaxf(amax, fabsf(x))`（会吞掉 NaN），
直接复用 5.1b 修好的 finite-amax + non-finite flag reduction。

**5.5c 数值正确性**（覆盖 normal± / all-zero / saturation / NaN / +Inf）
- 新增 `Tensor::sync_fp8_scale_from_device(stream)`：device-computed scale 的 host mirror
  是空的，需要显式 D2H 才能让 `has_fp8_scale() / fp8_scale() / fp8_scale_device()` 三者
  语义一致。刻意做成 opt-in —— D2H 在 graph capture 内非法，绝不能进热路径。
- 实测：`[OK] +/-, zero(->scale 1 + zeros), saturation, NaN/Inf rejected; host/device agree`

**5.5d 决定性实验：producer 在 graph 内更新 scale slot**
```text
graphExec=0x5c4df2b97450 (never re-instantiated)
scale_ptr=0x704c06800 (stable)
scale updated BY THE GRAPH: 0.000404167 -> 0.0202083
replay == eager bitwise
```
改变 FP32 输入 D 使 amax 必然变化，**不重新 capture / instantiate**，replay 后 scale slot
被 **graph 内部的 producer** 更新。这比 5.3（host 写 scale 后 replay）更强，证明：

> scale slot 是 **graph-visible 的 device-side output**，而不是 host bookkeeping。

> 过程中修掉一个**测试自身**的 NaN 比较 bug：`out->fp8_scale() != dev_scale` 在两者都是
> NaN 时恒为真（`NaN != NaN`），导致误报。改为 NaN 感知比较后通过。

全量 `ctest` **40/40 通过**。

#### 6-C-5.6 W8A8 最小确定性 E2E chain（2026-09-08，Measured —— 决定性实验通过）

新增独立测试目标 `forge-fp8-w8a8-e2e-test`（`tests/test_fp8_w8a8_e2e.cu`），从 **真实 producer
到真实 consumer、经由 ops 边界（`ops::cuda_matmul_transB`，即 ComputeGraph 层使用的入口）**
驱动一条最小确定性 W8A8 链，而不是单独调 launcher：

```text
X_f32 --quantize_fp8_e4m3_w8a8--> A(fp8, device-authoritative scale sidecar)
        （5.6a：ReLU fused producer -> FP8 activation，device 上不物化 FP32 中间量）
W_f32 --quantize_fp8_e4m3_w8a8--> B(fp8, device-authoritative scale sidecar)
Y = cuda_matmul_transB(A, B)          # M>1: CUTLASS GEMM；M==1: FP8 GEMV
Y.dtype == FP32                       # chain 终止于 FP32，无 FP8->FP32->FP8 往返
```

实测（RTX 4050, sm_89, fp8-gemv bench + 全量 ctest **41/41**）：

```text
[OK] decode-chain      (1x4096x1024)      cos=1.000000 maxerr=7.6e-06  deterministic(bitwise) registry=0 sidecar=4
[OK] decode-act-chain  (1x4096x1024 relu)  cos=1.000000 maxerr=3.8e-06  deterministic(bitwise) registry=0 sidecar=4
[OK] prefill-chain     (32x512x256)        cos=1.000000 maxerr=3.5e-03  deterministic(bitwise) registry=0 sidecar=4
[OK] prefill-act-chain (32x512x256 relu)   cos=1.000000 maxerr=1.8e-03  deterministic(bitwise) registry=0 sidecar=4
[OK] 5.6a activation producer: fused ReLU == relu-then-quantize (bytewise + scale);
     all-negative -> scale 1 + zeros
```

门控语义：
- **FP32-out parity** vs decode 型 CPU golden（复用 producer 产出的同一批字节，避免隐式 FP8->FP32->FP8）；
- **bitwise determinism**（同一 producer tensor 连续两次 matmul，逐字节相同，路径无原子操作）；
- **scale 只走 sidecar**：`fp8_scale_source_counts() registry==0 && sidecar>=4`；
- **dtype 链**：producer 输出 FP8 + `fp8_scale_device_authoritative()`，consumer 输出 FP32；
- 5.6a 同时验证 ReLU-fused activation producer 与 relu-then-quantize 参考**逐字节 + scale 全等**，
  以及全负输入坍缩到 `scale=1 + 全零`（5.1b amax==0 contract）。

> **过程中发现并修复一个 6-C-4.2/4.3 的潜伏布局 bug（生产正确性）**：E2E 把同一份权重喂给
> M>1（CUTLASS GEMM）与 M==1（FP8 GEMV）时 M==1 严重失配。根因是生产 FP8 GEMV kernel 的
> `W[k*N+j]`（k-major）假设与 Forge 真实 transB 权重布局（row-major `[N,K]`，`W[n*K+k]` ——
> CPU 参考 `matmul.cpp`、FP32 GEMV `gemv_transB_kernel`、CUTLASS FP8 GEMM（col-major `[K,N]`,
> ldb=K）均一致）冲突：同一 weight Tensor 在两种 M 下不可能同时对。修复：`cuda_fp8_gemv.cu`
> 生产 kernel 改为 **warp-per-row n-major**（warp 处理一个输出行、lane 扫 k、每步一个 coalesced
> 32B sector + warp reduce，结构与已验证的 FP32 `gemv_transB_kernel` 相同）。6-C-4.2 的 VEC 宽度
> 微基准是 k-major 人工布局上的带宽实验，仍保留在 `test_fp8_gemv_cuda.cu` 中作为历史对照；生产
> PROD 回归改为喂真实 row-major 权重与其独立 golden（maxerr ~1e-10, cos=1.0）。

#### 6-C-5.7 Legacy scale registry 删除 gate（2026-09-08，Measured）

按 Phase-6 纪律，**先证明生产路径零依赖再删代码**（5.6 全绿后执行）：
- 删除 `src/operators/cpu/fp8_scale_registry.cpp`，CMake 移除该源（`src/operators/CMakeLists.txt`）；
- `fp8_gemm.h` 删除 `ops::set/get/has_fp8_tensor_scale` 声明；
- `matmul_cuda.cpp` 的 `fp8_scale_device_of()` 删除 registry fallback（含 warn-once、per-call
  `cudaMalloc`/`cudaMemcpy` 分支），仅保留 Tensor sidecar 解析；`Fp8ScaleRef{owned}` 生命周期
  拆分不再需要，改回裸 `const float*`（sidecar 恒 Tensor-owned / stable / graph-visible）；
- debug 基础设施（`fp8_scale_debug.h/cpp`）保留 `registry` 计数作为 **live canary**（恒 0，
  再也不会变正），`Fp8ScaleSourceCounts.registry==0` 成为常真门禁；
- `tensor.h` 注释同步：unset sidecar 是硬错误（无 silent host fallback）。

实测：全量 `ctest` **41/41 通过**（含 `fp8-gemv-bench`、新增 `fp8-w8a8-e2e`）；
`grep` 确认 `fp8_scale_registry` / `set_fp8_tensor_scale` 不再有任何 src/tests/benchmarks 引用。

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

---

## 5. 附加交付记录（非优化 Phase）

### forge-server Phase 1（2026-09-11，OpenAI 兼容本地推理 API，✅ DELIVERED）

**目标**：构建产出 `forge-server`，本地端口提供 OpenAI 兼容推理 API，OpenCode 以
`@ai-sdk/openai-compatible` provider 接入。参考 llama.cpp server。

**改动表**：

| 文件 | 内容 |
|---|---|
| `third_party/cpp-httplib/`（新增） | vendored cpp-httplib（httplib.h + httplib.cpp，MIT，取自 llama.cpp vendor） |
| `src/server/server.h`（新增） | ServerOptions / ChatCompletionRequest / ServerState（模型+KV 单实例，`inference_mutex_` 串行化推理） |
| `src/server/server_state.cpp`（新增） | 加载链路镜像 forge-cli 文本路径（tokenizer→model→engine→KV→warmup）；chat 生成复用 `Generator` + `TokenCallback`，UTF-8 安全增量解码；ChatTemplateEngine（jinja）渲染 messages |
| `src/server/server_routes.cpp`（新增） | GET `/health`、GET `/v1/models`、POST `/v1/chat/completions`（SSE 流式 + 非流式）；OpenAI 错误格式；CORS + OPTIONS 预检 |
| `src/server/server_main.cpp`（新增） | 参数解析（`-m/--host/--port/-ngl/-t/--kv-cache-dtype/--cuda-graph/--load-mode`）+ SIGINT/SIGTERM 优雅停止 |
| `CMakeLists.txt` | 新增 `forge_httplib` STATIC 库与 `forge-server` 可执行目标（链接集对齐 forge-cli：infer/model/ops/core/tokenizer/vision + httplib） |

**并发模型**：与 llama.cpp server 相同——httplib 线程池收 HTTP，推理经单 mutex 串行
（单模型单 KV 实例）。阶段 3 可切换到既有 `RequestScheduler` 做 continuous batching。

**实测验收**（TinyLlama-1.1B Q4_0，沙盒无 GPU → CPU -ngl 0）：
- `GET /health` → `{"status":"ok"}`；`GET /v1/models` → OpenAI list 格式
- 非流式 chat：`finish_reason=stop`，usage 正确（26 prompt + 9 completion）
- 流式 SSE：首 chunk 带 role、逐 delta chunk、`data: [DONE]` 结尾、UTF-8 未拆多字节
- 多轮 messages（system+user）模板渲染正常；无效 JSON → OpenAI 错误格式 HTTP 400
- CTest 回归 **48/48 全绿**

**非目标 / Deferred（阶段 2）**：`--api-key`、字符串级 stop sequences、客户端断开中止
生成（当前断开后生成仍完成）、并发排队上限（429）；阶段 3：RequestScheduler batching。

### forge-server Phase 2（2026-09-11，健壮性四件套，✅ DELIVERED）

**改动表**：

| 文件 | 内容 |
|---|---|
| `include/forge/generator.h` / `src/inference/generator.cpp` | `GenerationConfig` 新增 `abort_flag`（`const std::atomic<bool>*`）；decode 主循环边界 + speculative 逐 token + plain 逐 token 三处检查，中止后 `finish_reason="aborted"`（对非 server 调用者零影响，默认 nullptr） |
| `src/server/server.h` | `ServerOptions` 加 `api_key`/`max_inflight`；`ChatCompletionRequest` 加 `stop`；`ServerState` 加 `try_acquire_slot`/`release_slot`（CAS 计数）；`setup_middleware` 声明 |
| `src/server/server_state.cpp` | stop-string 检测：UTF-8 flush 前扣留 `max_stop_len-1` 字节（stop 串永不部分流出）+ tail 滑窗匹配 → 命中置 abort；客户端断开（`on_token` 返回 false）置 abort；非流式 text 截断至最早 stop 串；slot 管理 |
| `src/server/server_routes.cpp` | `setup_middleware`（CORS + OPTIONS + `--api-key` 时全端点 Bearer 校验，401 OpenAI 错误格式）；chat handler 前 `try_acquire_slot`，满 → HTTP 429；**流式 slot 生命周期 bug 修复**：SlotGuard 以 shared_ptr 捕入 chunk provider lambda（handler 先于 provider 返回，栈 guard 会提前释放导致 429 失效），provider 销毁时释放 |
| `src/server/server_main.cpp` | `--api-key` / `--max-inflight N`（默认 4） |

**实测验收**（TinyLlama-1.1B Q4_0 CPU -ngl 0）：
- 无/错 api-key → HTTP 401 + OpenAI 错误格式；正确 key → 200
- stop=`"3."`：content 停在 stop 串前（"1. Blue\n2. Green\n"），提前中止（10/40 token），finish_reason=stop
- 流式占用 slot 期间第二请求 → HTTP 429；首个流结束后 → 恢复 200（slot 正确释放）
- 客户端 3s 断开 → 6s 后 slot 已释放（abort 生效；未中止则 150 token 需 40s+）
- CTest **48/48 全绿**（沙盒内 kv-quant-segment/q8-roundtrip 因沙盒禁 GPU 访问失败，沙盒外全绿，与改动无关）

### forge-server Phase 2.5（2026-09-11，KV overflow 乱码根治 + `--ctx-size`，✅ DELIVERED）

**症状**：opencode 长会话（prompt > 4096 token）输出多语言乱码 token 沙拉（`肌etable ISA...`），单轮/短会话完全正常。

**根因链**（用 4955-token 精确复现）：
1. `TransformerEngine::init_kv_cache` 把大 ctx 模型的 KV cap 到 4096（防 OOM）；
2. opencode 会话历史超过 4096 → `KVCache::update` 返回 -1 拒写；
3. **-1 被静默吞掉**——attention 继续读未写入的 KV 内存 → logits 崩坏 → 乱码。5 个 KV 写点全部同病（generic/gemma4/phimoe/deepseek/dflash）。

**修复（两层）**：
| 层 | 文件 | 内容 |
|---|---|---|
| 防御层 | `src/inference/engines/generic_engine.cpp`、`phimoe_engine.cpp`、`dflash_engine.cpp`（2 处）、`src/inference/layers/gemma4_attention.cpp`、`deepseek_layer_executor.cpp` | 全部 5 个 `kv_cache_.update` 写点改为 `< 0` 即 throw（带 layer/write_pos/seq_len/max 诊断信息）；异常经 server routes catch → HTTP 500 明确报因。**乱码从此不可能静默发生** |
| 截断层 | `src/server/server_state.cpp` | `chat_completion` 渲染前查真实 KV 容量（`engine->kv_cache()->max_seq_len()`，含 CUDA 自动缩减后的值）；prompt + gen_budget 超容量时**从最旧的非 system 消息开始丢弃**（保 system + 最新轮），逐条重渲染直到装下；连最简组合都装不下（巨型单条 system）→ throw 明确报错（llama.cpp server 同语义） |
| 能力层 | `transformer_engine.h/.cpp`、`server.h`、`server_state.cpp`、`server_main.cpp` | 新增 `-c/--ctx-size N`：覆盖硬编码 4096 cap（min(N, 模型训练长度)；CUDA 显存检查仍会自动缩减）；KV fp32 下 Qwen3-VL 4B 8192 ctx ≈ 2.3GB，4050 6GB 可容纳 |

**实测验收**（Qwen3-VL-4B Q3_K_M，CPU -ngl 0）：
- 修复前：4223-token prompt → 乱码（`肌etable...` 与用户报告逐字一致）
- 修复后默认 cap：T1 超大 system（4223>4096，2 条消息无从截断）→ HTTP 500 `KV cache overflow at layer 0: write_pos=0 seq_len=4223 exceeds max_seq_len=4096`（明确报因，不再乱码）；T2 30 条消息多轮历史 → 自动截断最旧 → 4068 tok 正确答 "4"
- `-c 8192` 后：T1 → **连贯自我介绍**（"I am Qwen3, the large-scale language model launched by Alibaba Group"）；T2 7764 tok → "4"；T3 正常
- CTest **48/48 全绿**

**运维**：GPU 显存充足时建议 `--ctx-size 8192 --kv-cache-dtype f16`（KV 减半 ≈1.2GB，16k ctx 也可行）；opencode 的 build agent 单轮 prompt 含 9.6k 字符 system + tools，长会话累积快，截断层保证任何长度下行为正确。

**Deferred**：paged KV 的 overflow 语义、多请求并发下的截断策略共享、`/v1/models` 暴露 ctx 字段。

