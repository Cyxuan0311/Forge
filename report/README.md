# Forge 性能报告模块

## 目录结构

```
report/
├── base.py          # BenchmarkBase 抽象基类（模板）
├── tinyllama.py     # TinyLlama-1.1B  benchmark
├── qwen.py          # Qwen2.5-7B-Instruct benchmark
├── runner.py        # 统一运行器（自动发现所有模型）
├── chart.py         # 图表生成模块
└── README.md        # 本文档

resource/reports/    # 输出目录（结果 JSON + 图表 PNG）
├── tinyllama/
│   ├── results.json
│   ├── performance_report.png
│   └── prefill_*.png
└── qwen/
    ├── results.json
    ├── performance_report.png
    └── prefill_*.png
```

## 快速运行

```bash
# 运行所有模型 benchmark
python -m report.runner

# 只跑 TinyLlama
python -m report.runner --models tinyllama

# 只跑 Qwen
python -m report.runner --models qwen

# 跳过 llama.cpp 对比（只需 Forge 数据）
python -m report.runner --skip-llama-cpp

# 跳过 CPU benchmark（省时间）
python -m report.runner --skip-cpu
```

## 单模型运行

```bash
python -m report.tinyllama --skip-llama-cpp
python -m report.qwen --skip-cpu --num-runs 5
```

## 输出

所有结果和图表保存在 `resource/reports/<model_name>/` 下：

- `results.json` — benchmark 原始数据
- `performance_report.png` — 多面板性能报告图
- `prefill_*.png` — 各模型 prefill 速度图

## 添加新模型

1. 在 `report/` 下创建新文件，例如 `deepseek.py`：

```python
from base import BenchmarkBase

class DeepSeekBenchmark(BenchmarkBase):
    MODEL_NAME = "DeepSeek-R1-Distill-Qwen-7B Q4_K_M"
    MODEL_PATH = "/path/to/model.gguf"
    BENCH_TOKENS = 100
```

2. `python -m report.runner` 会自动发现并运行。

## 依赖

- `forge`（本地 build 目录）
- `numpy`
- `matplotlib`
- `llama-cpp-python`（仅对比 llama.cpp 时需要）
