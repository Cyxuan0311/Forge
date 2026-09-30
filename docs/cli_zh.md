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
| `-V, --validate` | 用 tensor 实际内容交叉校验元数据；发现错误时退出码为 2 |
| `--metadata` | 打印全部元数据键值 |
| `-t, --tree [D]` | 深度为 `D` 的 tensor 树 |

估算口径与 Forge 引擎自身一致，因此数值与引擎输出的 `KV Cache` 相同。
量化 KV 的大小包含每块的 scale，且每行向上取整到整块；可选 dtype 列表与引擎
实际可存储的类型完全一致（没有 BF16，也没有 K-quant 的 KV cache）。
混合架构（LFM2、Qwen3.5）还会标注真正含注意力的层数——它通常远小于总层数。

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
