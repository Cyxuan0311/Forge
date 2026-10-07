# CLI 使用

[English](cli.md)

## 基本用法

```bash
./forge-cli -m <模型路径> [选项]
```

## 选项

### 模型与输入

| 选项 | 默认值 | 说明 |
|------|-------|------|
| `-m, --model PATH` | （必需） | GGUF / NINF 模型文件路径 |
| `-p, --prompt TEXT` | — | 输入提示（非交互模式） |
| `-n, --n-predict N` | `256` | 最大生成 token 数 |
| `--mmproj PATH` | — | 多模态视觉编码器路径（MiniCPM-V） |
| `--image PATH` | — | 多模态模型的输入图片 |

### 性能

| 选项 | 默认值 | 说明 |
|------|-------|------|
| `-ngl, --n-gpu-layers N` | `-1` | 卸载到 GPU 的层数（`-1` = 全部，`0` = 仅 CPU） |
| `-t, --threads N` | auto | CPU 线程数 |
| `-b, --batch-size N` | `512` | Prompt 处理时的批大小 |
| `--kv-cache-dtype TYPE` | `f16` | K 与 V **同时**使用的 KV 缓存数据类型：`fp32`, `f16`, `q8_0`, `q4_0`, `q4_k`, `fp8_e4m3`, `fp8_e5m2` |
| `-ctk, --cache-type-k TYPE` | | **仅 K** 使用的 KV 缓存数据类型（覆盖 `--kv-cache-dtype`） |
| `-ctv, --cache-type-v TYPE` | | **仅 V** 使用的 KV 缓存数据类型（覆盖 `--kv-cache-dtype`） |

### 采样

| 选项 | 默认值 | 说明 |
|------|-------|------|
| `--temp FLOAT` | `0.7` | 采样温度（`0` = 贪婪采样） |
| `--top-k N` | `40` | Top-K 采样（`0` = 禁用） |
| `--top-p FLOAT` | `0.9` | Top-P（核）采样 |
| `--repeat-penalty FLOAT` | `1.1` | 重复惩罚 |
| `--seed N` | 随机 | 随机种子（可重现） |

### 模式

| 选项 | 默认 | 说明 |
|------|------|------|
| `-i, --interactive` | 关闭 | 交互式对话模式 |
| `--no-stream` | 关闭 | 禁用流式输出 |
| `--system-prompt TEXT` | — | 系统提示词 |

### 信息与调试

| 选项 | 说明 |
|------|------|
| `--info` | 显示模型信息并退出 |
| `--bench` | 运行性能基准测试 |
| `-v, --verbose` | 启用详细日志 |
| `-h, --help` | 显示帮助信息 |

### Web 仪表盘

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dashboard` | 关闭 | 在独立的本地端口上提供实时性能仪表盘 |
| `--dashboard-port N` | `9090` | 期望端口；被占用时自动使用下一个空闲端口。`0` = 由系统分配 |
| `--dashboard-host H` | `127.0.0.1` | 仪表盘绑定地址 |
| `--dashboard-ops` | 关闭 | 额外记录逐算子耗时拆解面板 |
| `--no-open` | — | 不自动拉起浏览器（stdout 非 TTY 或设置了 `FORGE_NO_OPEN=1` 时同样不会打开） |

`forge-server` 也支持同样的参数。仪表盘使用**独立**端口而不是 API 端口，
因此 `--api-key` 仍然保护着每一个 API 请求，OpenAI 接口行为完全不变。

面板内容：实时与全程解码速率、TTFT p50/p90/p99、token 间延迟 p50/p90/p99
及其直方图、prefill 与 decode 耗时、累计 token 数、请求数、排队等待、
前缀缓存命中率、显存与 KV 缓存仪表，以及逐请求明细表。
`--dashboard-ops` 会额外加上 `PerfProfiler` 的逐算子耗时拆解。

> 算子面板需要 `-DFORGE_PROFILING=ON` 构建（见 [build_zh.md](build_zh.md)）。
> 它是显式开启的，因为启用 Profiler 会拖慢 decode。

表头处、uptime 读数旁边有两个控件：

**主题。** 单个按钮按 浅色 → 深色 → 跟随系统 循环。`跟随系统` 会读取操作系统的
设置，并在页面打开期间持续跟随；选择保存在 `localStorage` 中，刷新后仍然有效。
注意该存储按「源 + 端口」区分，因此当 9090 被占用、引擎退到其他端口时，主题会
回到 `跟随系统`。

**导出。** 序列化图表当前绘制的同一份 `/api/metrics` 快照，因此保存下来的文件
与屏幕上看到的内容一致。共四种格式：

| 格式 | 内容 |
| --- | --- |
| JSON | 完整快照，无损，并附带导出时间与生成方 |
| CSV | 时间序列、请求与算子三个数据块，放在同一个文件里 |
| Markdown | 摘要报告，复制到剪贴板，便于粘贴进 issue |
| Prometheus | 文本曝光格式的 gauge，以文件形式给出 |

四种格式都做了正确的转义（CSV 遵循 RFC 4180，Prometheus 标签转义 `\`、`"`
与换行），所以包含引号的模型路径不会破坏输出。

> CSV 是**多数据块**文件，并非单一矩形表格——严格的 CSV 无法容纳嵌套数据。
> Excel 与 LibreOffice 会把各数据块显示为独立区域；`pandas.read_csv` 无法解析，
> 需要程序化读取时请改用 JSON 导出。
>
> Prometheus 导出是一个**文件**，不是抓取端点。它适用于 node_exporter 的
> textfile collector 或手动导入；没有任何程序会去抓取它。若要实时抓取，需要服务端
> 路由（`/api/export`），而本仪表盘刻意没有提供——导出发生在你的浏览器里，
> 因此没有可供 `curl` 调用的接口。

在 `-p` / `--prompt` 模式下，`--dashboard` 会让进程在生成完成后继续驻留，
以便仪表盘保持可访问；按 `Ctrl-C` 退出。交互模式与 `--bench` 则照常常驻。

编译进二进制需要 Node.js 20.19+（Vite 7）：

```bash
cmake -B build -DFORGE_BUILD_DASHBOARD=ON
cmake --build build -j
```

不开启时仪表盘仍会提供 `/api/metrics`，而 `/dashboard/` 会说明如何重新构建。
前端源码位于 [`web/`](../web)（Vue 3 + TypeScript + Vite + ECharts）；
`npm run dev` 可对着正在运行的引擎做热重载。

### `forge-inspect`

独立的 GGUF 分析工具。读取模型文件，输出元数据、tensor 布局、dtype 直方图、
算子流水线，并可选给出 KV/显存估算。

```bash
# 人类可读摘要
forge-inspect model.gguf

# 能不能装下？按 4K 上下文和 6 GiB 显卡计算
forge-inspect -M --ctx 4096 --vram 6144 model.gguf

# q8_0 KV 相比 f16 可省一半缓存
forge-inspect -M --ctx 4096 --kv-dtype q8_0 model.gguf

# 机器可读，便于脚本处理
forge-inspect -M -f json model.gguf | jq .memory
```

| 选项 | 说明 |
|------|------|
| `-M, --memory` | KV 缓存 + 显存估算 |
| `--ctx N` | 假定的上下文长度（默认用模型自身的训练长度） |
| `--kv-dtype D` | `f16`、`fp32`、`q8_0`、`q4_0`、`q4_k`、`fp8_e4m3`、`fp8_e5m2` |
| `--vram MB` | 用于比对的预算；输出超出/剩余量以及需卸载的层数 |
| `-f, --format` | `text`、`table`、`markdown`、`json`、`yaml`、`csv`、`ini` |
| `-d, --tensor-diff` | 与第二个模型逐 tensor 对比（需要两个文件路径） |
| `-V, --validate` | 用 tensor 实际内容交叉校验元数据；发现错误时退出码为 2 |
| `--filter SUBSTR` | 只显示名字包含 `SUBSTR` 的 tensor（不区分大小写） |
| `--tensor NAME` | 按精确名字显示单个 tensor 的完整信息 |
| `--metadata` | 打印全部元数据键值 |
| `-t, --tree [D]` | 深度为 `D` 的 tensor 树 |

估算口径与 Forge 引擎自身一致，因此数值与引擎输出的 `KV Cache` 相同。
量化 KV 的大小包含每块的 scale，且每行向上取整到整块；可选 dtype 列表与引擎
实际可存储的类型完全一致（没有 BF16，也没有 K-quant 的 KV cache）。
混合架构（LFM2、Qwen3.5）还会标注真正含注意力的层数——它通常远小于总层数。

`--tensor-diff` 对比两个模型的 tensor 索引：哪些 tensor 只存在于其中一个、
共有的 tensor 中哪些在 dtype、shape 或大小上不同。它只读 GGUF 索引、不读权重
数据，因此索引相同但数值不同的模型会被判为相同。`--top N` 可限制列表长度
（统计数字不受影响）。

`--validate` 会用 tensor 实际内容交叉校验文件头：层数与实际存在的层是否一致、
head 几何关系、tensor 重名、词表与 embedding 矩阵是否匹配、tensor 范围是否超出
文件长度。若存在错误则退出码为 2，可用于 CI 或下载后自动校验。

## 示例

### 交互式对话

```bash
./forge-cli -m model.gguf

# 直接输入消息。命令：/quit、/clear、/system、/help
```

### 单次提示

```bash
./forge-cli -m model.gguf -p "什么是机器学习？" --stream
```

### 仅 CPU 推理

```bash
./forge-cli -m model.gguf -ngl 0
```

### 自定义采样参数

```bash
./forge-cli -m model.gguf -p "写一首关于 AI 的诗" \
  --temp 0.9 --top-p 0.95 --repeat-penalty 1.2 -n 512
```

### 多模态（视觉语言）

```bash
./forge-cli -m minicpmv.gguf --mmproj mmproj.gguf \
  --image photo.jpg -p "描述这张图片"
```

### Benchmark 模式

```bash
./forge-cli -m model.gguf --bench
```

### 查看模型信息

```bash
./forge-cli -m model.gguf --info
```

## 交互命令

| 命令 | 说明 |
|------|------|
| `/quit` 或 `/exit` | 退出对话 |
| `/clear` | 清除对话历史 |
| `/system TEXT` | 设置系统提示词 |
| `/image PATH` | 加载图片（多模态模型） |
| `/save PATH` | 保存对话到文件 |
| `/help` | 显示帮助 |

## 性能调优

- **GPU**：使用 `--n-gpu-layers -1` 将所有层卸载到 GPU。设置 `-ctk q8_0 -ctv q4_0` 可在质量损失远小于直接量化 K 的前提下把 KV 显存降到 f16 的约 1/2.5（见 `kv-quant.md`）。
- **CPU**：增加 `--threads` 匹配 CPU 核心数。使用 `--batch-size 1024` 加快 prompt 处理速度。
- **采样**：降低 `--temp` 获得更确定的输出；提高则增加创造性。
