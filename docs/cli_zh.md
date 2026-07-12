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
| `--kv-cache-dtype TYPE` | `fp32` | KV 缓存数据类型（`fp32` 或 `q4_0`） |

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
| `--stream` | 开启 | 逐 token 流式输出 |
| `--no-stream` | 关闭 | 禁用流式输出 |
| `--system-prompt TEXT` | — | 系统提示词 |

### 信息与调试

| 选项 | 说明 |
|------|------|
| `--info` | 显示模型信息并退出 |
| `--bench` | 运行性能基准测试 |
| `-v, --verbose` | 启用详细日志 |
| `-h, --help` | 显示帮助信息 |

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
./forge-cli -m model.gguf --device cpu --n-gpu-layers 0
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

- **GPU**：使用 `--n-gpu-layers -1` 将所有层卸载到 GPU。设置 `--kv-cache-dtype q4_0` 可减少显存占用。
- **CPU**：增加 `--threads` 匹配 CPU 核心数。使用 `--batch-size 1024` 加快 prompt 处理速度。
- **采样**：降低 `--temp` 获得更确定的输出；提高则增加创造性。
