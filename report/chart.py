"""
图表生成模块：从 JSON 结果生成性能对比图表。

用法：
    from report.chart import generate_report
    generate_report(results_list, output_dir)
"""

import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec

C_NI_GPU = "#2196F3"
C_NI_CPU = "#90CAF9"
C_LC_GPU = "#FF9800"
C_LC_CPU = "#FFCC80"
C_BG = "#FAFAFA"
C_GRID = "#E0E0E0"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.facecolor": C_BG,
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.color": C_GRID,
    "grid.alpha": 0.5,
})


def safe_get(d, key, default=0):
    if d is None:
        return default
    val = d.get(key, default)
    return val if val is not None else default


def _plot_decode_bar(results, ax):
    models = []
    ni_gpu, lc_gpu, ni_cpu, lc_cpu = [], [], [], []

    for r in results:
        models.append(r["model_name"].replace(" Q4_0", ""))
        ni_gpu.append(safe_get(r.get("forge_gpu_decode"), "median"))
        lc_gpu.append(safe_get(r.get("llama_cpp_gpu_decode"), "median"))
        ni_cpu.append(safe_get(r.get("forge_cpu_decode"), "median"))
        lc_cpu.append(safe_get(r.get("llama_cpp_cpu_decode"), "median"))

    x = np.arange(len(models))
    n = 0
    bars_data = []
    if any(v > 0 for v in ni_gpu):
        bars_data.append(("Forge GPU", ni_gpu, C_NI_GPU)); n += 1
    if any(v > 0 for v in lc_gpu):
        bars_data.append(("llama.cpp GPU", lc_gpu, C_LC_GPU)); n += 1
    if any(v > 0 for v in ni_cpu):
        bars_data.append(("Forge CPU", ni_cpu, C_NI_CPU)); n += 1
    if any(v > 0 for v in lc_cpu):
        bars_data.append(("llama.cpp CPU", lc_cpu, C_LC_CPU)); n += 1

    width = 0.7 / max(n, 1)
    for i, (label, values, color) in enumerate(bars_data):
        offset = i * width - (n - 1) * width / 2
        bars = ax.bar(x + offset, values, width, label=label, color=color,
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_title("Decode Speed (tok/s)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("tok/s")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(bottom=0)


def _plot_prefill_lines(results, ax):
    for r in results:
        name = r["model_name"].replace(" Q4_0", "")
        ni_pf = r.get("forge_gpu_prefill")
        lc_pf = r.get("llama_cpp_gpu_prefill")

        if ni_pf:
            lengths = sorted([int(k) for k in ni_pf.keys()])
            speeds = [ni_pf[str(l)]["mean"] if str(l) in ni_pf
                      else ni_pf.get(l, {}).get("mean", 0) for l in lengths]
            ax.plot(lengths, speeds, "o-", color=C_NI_GPU, label=f"{name} (Forge)",
                    linewidth=2, markersize=5)

        if lc_pf:
            common_lengths = sorted([int(k) for k in lc_pf.keys()])
            speeds = [lc_pf[str(l)]["mean"] if str(l) in lc_pf
                      else lc_pf.get(l, {}).get("mean", 0) for l in common_lengths]
            ax.plot(common_lengths, speeds, "s--", color=C_LC_GPU, label=f"{name} (llama.cpp)",
                    linewidth=2, markersize=5, alpha=0.8)

    ax.set_title("Prefill Speed vs Prompt Length", fontsize=13, fontweight="bold")
    ax.set_xlabel("Prompt Length (tokens)")
    ax.set_ylabel("tok/s")
    ax.set_xscale("log", base=2)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.legend(fontsize=8, loc="best")
    ax.set_ylim(bottom=0)


def _plot_gpu_speedup(results, ax):
    models = []
    ni_speedups, lc_speedups = [], []

    for r in results:
        ni_gpu = safe_get(r.get("forge_gpu_decode"), "median")
        ni_cpu = safe_get(r.get("forge_cpu_decode"), "median")
        lc_gpu = safe_get(r.get("llama_cpp_gpu_decode"), "median")
        lc_cpu = safe_get(r.get("llama_cpp_cpu_decode"), "median")

        if ni_cpu > 0:
            models.append(r["model_name"].replace(" Q4_0", ""))
            ni_speedups.append(ni_gpu / ni_cpu)
            lc_speedups.append(lc_gpu / lc_cpu if lc_cpu > 0 else 0)

    if not models:
        ax.text(0.5, 0.5, "No CPU data available", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        ax.set_title("GPU Acceleration Factor", fontsize=13, fontweight="bold")
        return

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width / 2, ni_speedups, width, label="Forge",
                   color=C_NI_GPU, edgecolor="white")
    for bar, val in zip(bars1, ni_speedups):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.1f}x", ha="center", va="bottom", fontsize=10, fontweight="bold")

    if any(v > 0 for v in lc_speedups):
        bars2 = ax.bar(x + width / 2, lc_speedups, width, label="llama.cpp",
                       color=C_LC_GPU, edgecolor="white")
        for bar, val in zip(bars2, lc_speedups):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                        f"{val:.1f}x", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_title("GPU Acceleration Factor (vs CPU)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("Speedup")
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)


def _plot_ratio_bar(results, ax):
    models = []
    ratios = []

    for r in results:
        ni_gpu = safe_get(r.get("forge_gpu_decode"), "median")
        lc_gpu = safe_get(r.get("llama_cpp_gpu_decode"), "median")
        if lc_gpu > 0:
            models.append(r["model_name"].replace(" Q4_0", ""))
            ratios.append(ni_gpu / lc_gpu * 100)

    if not models:
        ax.text(0.5, 0.5, "No llama.cpp data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        ax.set_title("Forge / llama.cpp Ratio", fontsize=13, fontweight="bold")
        return

    colors = ["#4CAF50" if r >= 80 else "#FFC107" if r >= 50 else "#F44336" for r in ratios]
    bars = ax.barh(models, ratios, color=colors, edgecolor="white", height=0.5)
    ax.axvline(x=100, color="#333", linestyle="--", linewidth=1, alpha=0.7)

    for bar, val in zip(bars, ratios):
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}%", ha="left", va="center", fontsize=11, fontweight="bold")

    ax.set_title("Forge / llama.cpp GPU Decode Ratio", fontsize=13, fontweight="bold")
    ax.set_xlabel("% of llama.cpp speed")
    ax.set_xlim(0, max(ratios) * 1.2 + 10)
    ax.invert_yaxis()


def _plot_summary_table(results, ax):
    ax.axis("off")

    headers = ["Model", "Forge\nGPU", "Forge\nCPU", "llama.cpp\nGPU", "llama.cpp\nCPU",
               "NI/LC\nGPU Ratio", "GPU\nSpeedup"]

    rows = []
    for r in results:
        ni_gpu = safe_get(r.get("forge_gpu_decode"), "median")
        ni_cpu = safe_get(r.get("forge_cpu_decode"), "median")
        lc_gpu = safe_get(r.get("llama_cpp_gpu_decode"), "median")
        lc_cpu = safe_get(r.get("llama_cpp_cpu_decode"), "median")
        ratio = ni_gpu / lc_gpu * 100 if lc_gpu > 0 else 0
        speedup = ni_gpu / ni_cpu if ni_cpu > 0 else 0

        rows.append([
            r["model_name"].replace(" Q4_0", ""),
            f"{ni_gpu:.1f}" if ni_gpu > 0 else "-",
            f"{ni_cpu:.1f}" if ni_cpu > 0 else "-",
            f"{lc_gpu:.1f}" if lc_gpu > 0 else "-",
            f"{lc_cpu:.1f}" if lc_cpu > 0 else "-",
            f"{ratio:.0f}%" if ratio > 0 else "-",
            f"{speedup:.1f}x" if speedup > 0 else "-",
        ])

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2196F3")
            cell.set_text_props(color="white", fontweight="bold")
        elif col == 5 and row > 0:
            val_str = rows[row - 1][5]
            val = float(val_str.replace("%", "")) if val_str != "-" else 0
            if val >= 80:
                cell.set_facecolor("#C8E6C9")
            elif val >= 50:
                cell.set_facecolor("#FFF9C4")
            else:
                cell.set_facecolor("#FFCDD2")
        else:
            cell.set_facecolor("#f5f5f5" if row % 2 == 0 else "white")
        cell.set_edgecolor("#BDBDBD")

    ax.set_title("Performance Summary Table", fontsize=13, fontweight="bold", pad=20)


def _plot_model_breakdown(results, ax):
    categories = ["Embedding", "Attention\n(QKV+Attn+O)", "FFN\n(Gate+Up+Down)", "Output\nLogits"]
    ni_pct = [5, 26, 59, 10]

    x = np.arange(len(categories))
    bars = ax.bar(x, ni_pct, color=[C_NI_GPU, "#42A5F5", "#1565C0", "#0D47A1"],
                  edgecolor="white", width=0.6)

    for bar, val in zip(bars, ni_pct):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_title("Forge Layer Time Breakdown (TinyLlama Decode)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylabel("% of Layer Time")
    ax.set_ylim(0, 70)
    ax.grid(axis="x", visible=False)


def _generate_individual_charts(results, output_dir):
    for r in results:
        name = r["model_name"]
        safe_name = name.replace(" ", "_").replace("/", "_").replace("Q4_0", "").strip("_")

        ni_pf = r.get("forge_gpu_prefill")
        lc_pf = r.get("llama_cpp_gpu_prefill")

        if ni_pf:
            fig, ax = plt.subplots(figsize=(8, 5))
            lengths = sorted([int(k) for k in ni_pf.keys()])
            ni_speeds = [ni_pf[str(l)]["mean"] if str(l) in ni_pf
                         else ni_pf.get(l, {}).get("mean", 0) for l in lengths]

            ax.bar(range(len(lengths)), ni_speeds, color=C_NI_GPU, edgecolor="white", width=0.6)
            ax.set_xticks(range(len(lengths)))
            ax.set_xticklabels([str(l) for l in lengths])
            ax.set_xlabel("Prompt Length (tokens)")
            ax.set_ylabel("Prefill Speed (tok/s)")
            ax.set_title(f"Forge Prefill: {name}", fontweight="bold")

            for i, (l, s) in enumerate(zip(lengths, ni_speeds)):
                ax.text(i, s + 0.5, f"{s:.1f}", ha="center", fontsize=9, fontweight="bold")

            if lc_pf:
                ax2 = ax.twinx()
                lc_lengths = sorted([int(k) for k in lc_pf.keys()])
                lc_speeds = [lc_pf[str(l)]["mean"] if str(l) in lc_pf
                             else lc_pf.get(l, {}).get("mean", 0) for l in lc_lengths]
                ax2.plot(range(len(lc_lengths)), lc_speeds, "s-", color=C_LC_GPU,
                         linewidth=2, markersize=8, label="llama.cpp")
                ax2.set_ylabel("llama.cpp Prefill (tok/s)", color=C_LC_GPU)
                ax2.tick_params(axis="y", labelcolor=C_LC_GPU)
                ax2.legend(loc="upper left")

            plt.tight_layout()
            path = os.path.join(output_dir, f"prefill_{safe_name}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {path}")


def generate_full_report(results, output_dir):
    """生成 4×2 多面板性能报告图。"""
    os.makedirs(output_dir, exist_ok=True)

    gpu_name = results[0].get("gpu_name", "Unknown GPU") if results else "Unknown GPU"

    fig = plt.figure(figsize=(20, 24))
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.3)

    fig.suptitle(
        f"Forge Performance Report\n{gpu_name}  |  Q4_0 Quantization",
        fontsize=18, fontweight="bold", y=0.98
    )

    _plot_decode_bar(results, fig.add_subplot(gs[0, 0]))
    _plot_prefill_lines(results, fig.add_subplot(gs[0, 1]))
    _plot_gpu_speedup(results, fig.add_subplot(gs[1, 0]))
    _plot_ratio_bar(results, fig.add_subplot(gs[1, 1]))
    _plot_summary_table(results, fig.add_subplot(gs[2, :]))
    _plot_model_breakdown(results, fig.add_subplot(gs[3, :]))

    path = os.path.join(output_dir, "performance_report.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    _generate_individual_charts(results, output_dir)


def generate_report(results, output_dir):
    """生成报告并保存到 output_dir。"""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nGenerating report charts in {output_dir}/...")
    generate_full_report(results, output_dir)
    print("Report generation complete!")
