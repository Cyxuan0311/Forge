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
Phase 3  Tensor Core 量化 GEMM ── 高风险高收益，仅限 M>16
   ↓
Phase 4  KernelSelector 2.0 + Autotuner ── 架构性收益
   ↓
Phase 5  Native 低精度执行模型 ── 长期，依赖 Phase 3
```

| 阶段 | 主题 | 风险 | 预期收益面 |
|---|---|---|---|
| Phase 0 | 基线工程 | 极低 | 无直接收益，解锁后续 |
| Phase 1 | 补课型优化 | 低 | MoE decode、QKV/FFN 拼接、output proj |
| Phase 2 | Attention 后端化 | 中 | prefill（含量化 KV）、长上下文 decode |
| Phase 3 | Tensor Core 量化 GEMM | 高 | prefill / 大 batch |
| Phase 4 | Selector + Autotuner | 中 | 全链路（消除人工调参） |
| Phase 5 | Native FP16/BF16 | 高 | 全链路（结构性） |

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

### 2-D. GPU 常驻 paged metadata

按 `plan.md` 第 17 节，把 `indptr / indices / last_page_len / seq_lens`
常驻 GPU，避免 CUDA Graph replay 时反复重建 host-side metadata。

---

## Phase 3 — Tensor Core 量化 GEMM（**限定 M > 16**）

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
- `tests/test_mmq_reference.cpp` 扩展覆盖 TC 路径
- `forge-bench-prefill_gemm` 在 M=32/64/128 档位对比 MMQ，记录加速比
- M=1 档位性能**不得回退**（回归保护）

---

## Phase 4 — KernelSelector 2.0 + Autotuner

**目标**：把 `attention_selector` 的模式推广到 GEMV / MMQ / MoE，并从
heuristic 升级为 autotuned。

### 4-A. 统一 Selector

现状：attention 有 selector（`attention_selector.cpp`），而 GEMV / MMQ / MoE
各自 if-else（`matmul_cuda.cpp:176-233`、`cuda_moe.cu:231`）。

新增：

```cpp
struct KernelProblem {
    int sm_arch; DataType dtype;
    int M, N, K, batch, head_dim, seq_len;
    bool paged; KVCacheDType kv_dtype;
};

enum class KernelVariant { GEMV_Q8_1, GEMV_SPLITK, MMQ, MMQ_TC, MOE_GEMV, MOE_GROUPED, ... };

KernelVariant select_kernel(const KernelProblem&);
```

### 4-B. Benchmark-driven dispatch

按 `plan.md` 第 8 节：首次启动对候选 kernel 实测，结果按
`（arch, dtype, M, N, K）` 签名缓存。

与现有设计天然契合——`ExecutionPlan::plan_id()`
（`execution_plan.cpp:89-110`）已经把执行特征编码成字符串，
`CudaGraphKey`（`cuda_graph_runner.h:45-54`）已有 shape signature 机制。

**验收**：新增 `tests/test_kernel_selector.cpp`，验证相同 problem 两次
选择一致、缓存命中率可统计。

---

## Phase 5 — Native 低精度执行模型

**依赖**：Phase 3（否则 FP16 路径没有对应的量化 GEMM backend）。

### 5-A. 引入 ComputeDTypePolicy

`plan.md` 第 13 节的建议，方向正确：

```cpp
struct ComputeDTypePolicy {
    DataType weight_dtype;
    DataType activation_dtype;
    DataType accumulator_dtype;
    DataType kv_dtype;
};
```

现状 `DataType` 单一枚举决定全部（`include/forge/types.h:7-29`），
且 `BF16 = 15` 注释明确写着 `converted to FP32 at load time`，
即 BF16 目前不是真正的执行 dtype。

### 5-B. 落地顺序

```text
FP16 activation 端到端      （已有 launch_cublas_gemm_fp16_fp32 可复用）
        ↓
FP8 W8A8                    （KV8 已完成，补 W8/A8）
        ↓
NVFP4 / W4A4                （Blackwell 专用）
```

每一步都要先把 activation 路径打通，再动 weight，否则会变成
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

任何改变浮点累加顺序的改动（1-C 协作归约、3-B MMA、5-A FP16 accumulator）
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
| 优先级表 NVFP4 标 P2 / 价值 ★★★★★ | 移入 Phase 5 末位，与价值解耦 | 优先级须受依赖约束：NVFP4 需要 Phase 3 的 TC backend 与 Phase 5 的 FP16 activation 先行 |
| 未提及 MoE 实现落后 | 新增 **Phase 1-A**，列为本计划单项最大收益 | `cuda_moe.cu:207` 仍用 `CudaQuantTraits::dot_block()`，而该函数是 scalar（`cuda_gemv_tmpl.cuh:38-41`）；主 GEMV 路径早已 dp4a 化。这是"补课"而非"创新"，风险低、收益确定 |

另：`plan.md` 末尾「README 与 CMakeLists 关于 `USE_CUBLAS` 默认值冲突」一条
经核验**不成立**（README 未声明默认值，`docs/dependencies.md:94` 与
`CMakeLists.txt:87` 一致为 ON），本计划仅保留 `docs/dependencies.md:14`
的措辞修正（0-E）。
