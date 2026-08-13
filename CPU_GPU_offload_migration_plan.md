# CPU/GPU 内存卸载渐进式迁移方案

> 目标:消除 Forge 与 llama.cpp 在 CPU/GPU 内存卸载上的主要差距。
> 策略:按风险从低到高分 5 个阶段推进,每阶段独立可交付、可回归、可回滚。

---

## 0. 现状盘点(基线)

| 维度 | Forge 现状 | llama.cpp 参照 |
|------|-----------|---------------|
| 按层卸载 | `set_gpu_layers` (`src/inference/engines/transformer_engine.cpp:53`) | `load_tensors` 逐层选 buft |
| 加载策略 | 整模型先载入单 device,再逐层搬移 (`src/cli/forge_cli.cpp:435`) | 逐层直接分配目标后端,无中间副本 |
| GPU 权重 | 全量 `cudaMemcpy` (`src/model/gguf_model.cpp:508`) | mmap + `buffer_from_host_ptr` + `mlock` |
| KV 卸载 | 跟随层设备,无独立开关 (`transformer_engine.cpp:394`) | `offload_kqv` 独立于 `-ngl` |
| KV 量化 | fp32/f16/q8_0/q4_0/q4_k (paged 仅 fp32) | `-ctk/-ctv` + Hadamard 旋转 |
| 计算图缓冲 | 单后端整块 `GraphBuffer` (`compute_graph.cpp:82`) | gallocr 按 split 后端独立分配 |
| 跨设备调度 | 层边界手动 `transfer_hidden` / `ensure_device` | `ggml_backend_sched` 五遍分配 + 自动 copy |
| 多 GPU | 仅 CPU/CUDA 二值 | tensor_split / main_gpu / meta device |
| 输入/输出层 | 跟随首/末层设备 | 输入层固定 CPU,输出层单独计层 |
| 重算回退 | 无(跨设备直接失败 `op_dispatch.cpp:147`) | op_offload 优先级提升 |

---

## 阶段 1:加载路径改造(风险:中,收益:高,无 API 变更)

**问题**:`-ngl N` 时整模型先载入 CUDA 再逐层搬回 CPU,模型超过显存即失败,且浪费一次 H2D+D2H 拷贝。

### 1.1 拆分层级设备决策

`src/model/model.cpp:load_from_loader` 目前接收单一 `DeviceType`。改造为接收 `const std::vector<DeviceType>& layer_devices`:

- `load_tensors` 遍历每个 tensor name,从 name 推导归属层(`"blk.%d."` 前缀),查 `layer_devices[layer]`。
- 推理层表(`model_weights.h` 中的 `layers[i]`)已有分层结构,直接按层目标 device 调 `loader.get_tensor(name, dev)`。
- 保留 `device` 单参数重载作为向后兼容(等价于 `layer_devices` 全同值)。

### 1.2 消除 `set_gpu_layers` 的二次搬移

`set_gpu_layers` (`transformer_engine.cpp:53-90`) 改为"仅断言"模式:

- 加载时已按 `layer_devices` 放置完毕,`set_gpu_layers` 只校验权重实际设备与目标一致,不再调用 `move_layer_weights`。
- 保留 `move_layer_weights` 用于"运行时热改 offload 层数"(可选),但默认路径不再触发。

### 1.3 CLI 对齐

`src/cli/cli_args.cpp:115` 的 `-ngl` 已存在;补充:
- `-ngl` 支持 `"auto"`/`"all"`(llama.cpp 语义:`-1`=auto、`-2`=all、`>=0`=指定层数)。当前 Forge 默认 `-1` 即"全部",与 llama.cpp 默认 auto 不同,文档需注明。
- 卸载层数超过层数时取 `min` 并 warn。

### 验收
- `-ngl 16` 加载 32 层模型,显存占用约等于 16 层权重 + KV,而非整模型。
- `verify_inference.sh` 全绿(CPU / CUDA / 部分卸载各跑一遍)。

---

## 阶段 2:独立 KV 卸载开关(风险:低,收益:中)

**问题**:KV 设备跟随层设备,无法"权重上 GPU、KV 留 CPU"(典型场景:长上下文 + 小显存)。

### 2.1 新增 `offload_kqv`

- `include/forge/context.h` 的 `ContextParams` 增加 `bool offload_kqv = true`。
- CLI 增加 `-kvo/--kv-offload` 与 `-nkvo/--no-kv-offload`(`src/cli/cli_args.cpp`)。
- Python 绑定 `src/bindings/common.h` 透传。

### 2.2 决策逻辑改造

`transformer_engine.cpp:394`:

```cpp
DeviceType kv_dev = (gpu_layers_ >= cfg.num_layers)
                        ? DeviceType::CUDA
                        : DeviceType::CPU;
```

改为:

```cpp
// offload_kqv == true 时跟随层设备(现状);
// offload_kqv == false 时强制 CPU(即使层在 GPU)。
DeviceType kv_dev = ctx_.params().offload_kqv
                        ? (gpu_layers_ >= cfg.num_layers ? CUDA : CPU)
                        : DeviceType::CPU;
```

同时 `set_layer_devices` 在 `offload_kqv == false` 时被跳过,KV 全部留在 CPU。

### 2.3 注意力路径适配

当某层权重在 CUDA 而 KV 在 CPU 时:
- decode 路径(M=1)GQA kernel 需要读 KV。当前实现 `llama_graph_builder.cpp:109` 已有 `k_sliced` 跨设备时手动拷到 CUDA 的先例,沿用该模式:CPU KV → CUDA 暂存 → kernel → 回写。
- 需引入 `kv_cache.h` 的 per-layer "remote KV" 标记,避免每步全量拷贝,只拷当前层需要的 `seq_len` 切片。
- flash-attn kernel 在 CPU-KV 分支退化为逐个 slice 拷贝(可先只支持 fp32 KV 的远程访问)。

### 验收
- `-ngl 32 -nkvo` 在 8G 显存上跑 7B 模型 + 长上下文,权重全 GPU、KV 全 CPU,不 OOM。
- `-ngl 32 -kvo`(默认)行为与现状完全一致(回归)。

---

## 阶段 3:计算图跨后端调度(风险:高,收益:高)

**问题**:`ComputeGraph::allocate_graph` 单后端整块缓冲;跨设备仅靠 `ensure_device` 同步拷贝,无自动 copy 节点、无异步、无事件。

### 3.1 引入 per-backend 缓冲(对应 gallocr)

- 新增 `GraphAllocator`(`src/core/`),替代 `MemoryPlanner` 的单块规划:
  - 按节点 `device` 分组,为**每个后端**独立分配 `GraphBuffer`。
  - 复用现有 `MemoryPool` 作为后端缓冲池,`device_` 按后端区分。
  - 保留 `MemoryPlanner` 的 lifetime 复用逻辑,作用域缩小到单后端内。

### 3.2 自动跨设备 copy 节点(对应 backend_sched split)

`ComputeGraph` 增加 pass(仿照 `ggml_backend_sched` 思路,但按 Forge 现有 DAG 结构实现):

1. **Pass A(设备传播)**:节点设备从输入权重设备推导;无权重节点跟随生产节点设备。
2. **Pass B(插入 copy)**:对每条跨设备边插入显式 `COPY` 节点(`OpType::COPY`),并注册到 `OpDispatch`。
3. **Pass C(异步化,可选)**:CUDA 侧用 `cudaMemcpyAsync` + `cudaStream_t`,`BackendScheduler` 记录依赖事件。

`GraphRuntime::build` (`src/inference/graph/graph_runtime.cpp:18`) 在图构建后调用上述 pass,`BackendScheduler::schedule` 的贪心分配作为 Pass A 的输入之一保留。

### 3.3 图运行时兼容

- `graph_cuda_runner.cpp:122` 的 CUDA Graph 兼容性检查:含 CPU 节点或 COPY 节点时 `is_compatible` 返回 false,自动回落普通执行(现状已有该机制)。
- 保证所有引擎(transformer / deepseek / qwen35 / gemma4)的图构建器不感知 copy 节点存在(仅由图层插入)。

### 验收
- 部分卸载下 prompt 处理(prefill)吞吐不低于现状,decode 无回归。
- 日志/`MemoryCounters` 能按设备统计 H2D/D2H 拷贝字节数。
- CUDA Graph 路径不受影响(未全量卸载时自动禁用)。

---

## 阶段 4:权重加载内存优化(风险:中,收益:中)

**问题**:CUDA 权重全量 `cudaMemcpy`,内存占用 = 模型全量常驻显存;无 mlock/load-mode 语义。

### 4.1 支持 GPU 侧 host 缓冲(对应 `buffer_from_host_ptr`)

- `CudaBackend` 增加 `allocate_host_registered` / `from_host_ptr`:
  - 加载 CUDA 权重时,将 mmap 区域经 `cudaHostRegister`(或 `cuMemHostRegister`)注册为 pinned host 内存,张量 `data()` 直接指向 mmap,算子按需零拷贝读取。
  - 启动参数:GPU 权重使用 `buffer_from_host_ptr`(llama.cpp 默认路径)或全量 `cudaMemcpy`(现状,便于 kernel 直接访问)。
- `GgufModel::get_tensor` (`gguf_model.cpp:458`) 的 CUDA 分支增加此选项。

### 4.2 mlock 语义

- CLI `--load-mode` 的三态(视平台可用性):
  - `mmap`(默认,现状)
  - `mlock`(将 CPU 权重 mmap 页 `mlock` 驻留,`--load-mode mlock`)
  - `mmap_mlock`(llama.cpp 默认)
- 仅影响 CPU/主机侧权重,不改变推理逻辑。

### 4.3 输入层留 CPU(对齐 llama.cpp)

- llama.cpp 显式将 embedding 输入层留在 CPU("offloading 无收益")。Forge 当前 `transformer_engine.cpp:68` 把 token_embedding 搬到第一层设备。
- 评估:当 `-ngl` 部分卸载时,将 token_embedding + 首个 RMSNorm 固定 CPU,仅在 `-ngl` 全部卸载时上 GPU。需 benchmark 验证收益后再默认开启,先做成开关 `--offload-embedding`(默认 off)。

### 验收
- 使用 host-registered 后,`-ngl 32` 7B 模型在显存紧张时加载成功,推理数值与全拷贝一致(对比 logits)。
- `free`/`nvidia-smi` 观测:mlock 后 CPU 侧不换页。

---

## 阶段 5:多 GPU 与张量切分(风险:高,收益:按需)

**问题**:`set_gpu_layers` 只有 CPU/CUDA 二值;`BackendScheduler` 虽能枚举多卡,但推理引擎未使用多卡。

### 5.1 层切分(layer split,低配版)

- `ContextParams` 增加 `int main_gpu = 0` 与 `std::vector<int> gpu_layers_per_dev`(或复用 `tensor_split` 比例)。
- `set_gpu_layers` 的 `layer_devices_` 从 `DeviceType` 改为 `struct { DeviceType type; int device_id; }`,贯穿 `LayerExecutionContext` 与 KV 放置。
- 各层算子在执行前用 `BackendManager::get_cuda_backend(device_id)` 切换到对应设备。

### 5.2 张量切分(tensor split,高阶,可选)

- 对齐 llama.cpp `SPLIT_MODE_TENSOR`:对 `MUL_MAT_TRANSB` 权重按行切分到多 GPU,输出部分和 + 归约。
- 需要跨卡通信。llama.cpp 用 meta device 抽象;Forge 建议先做"半切 + 主机归约"(免 NCCL 依赖),验证收益后再考虑 `n_copies` 流水线并行。

### 验收
- `-ngl 16:16`(两卡各 16 层)推理结果与单卡一致。
- 提供 `-sm layer|tensor` 与 `-ts` 解析,文档说明限制。

---

## 通用工程要求(贯穿所有阶段)

1. **回归基线**:每个阶段合入前跑 `python -m report.runner` 与 `examples/verify_inference.sh`,CPU / CUDA / 部分卸载三态全绿。
2. **内存记账**:扩展 `src/core/memory_counters.h`,输出 JSON 格式的 `{device, type, bytes, op, layer}` 明细,供报告模块渲染。
3. **开关默认保守**:新行为默认关闭或与现状等价,`-ngl 0` / `-ngl -1` / 全量卸载路径零改动。
4. **性能回归门限**:decode tok/s 与 prefill t/s 不得低于基线 5%(除非文档注明 trade-off)。
5. **文档同步**:每阶段更新 `docs/architecture_zh.md`(核心层/计算图章节)与 `docs/cli_zh.md`(新增参数)。

## 阶段依赖图

```
阶段1(加载路径)  ──►  阶段2(offload_kqv)  ──►  阶段3(图调度)
                                     └────────►  阶段4(权重内存)
阶段1/3 完成后可并行开工阶段5(多GPU, 依赖 1 的 layer_devices 重构)
```

## 建议执行顺序

| 顺序 | 阶段 | 工作量 | 风险 | 理由 |
|------|------|--------|------|------|
| 1 | 阶段 1 | 中 | 中 | 解除"模型必须小于显存"的硬限制,收益最大 |
| 2 | 阶段 2 | 低 | 低 | 独立开关,小步验证,不碰计算路径 |
| 3 | 阶段 4 | 中 | 中 | 内存占用优化,独立于调度改造 |
| 4 | 阶段 3 | 大 | 高 | 图调度重构,建议放在前三项稳定后 |
| 5 | 阶段 5 | 大 | 高 | 依赖阶段 1 的 `layer_devices` 重构,最后做 |
