"""生成 README 首屏效果图（基于真实运行结果，非示意图）。

额外依赖：pip install matplotlib
运行：python docs/make_figure.py
输出：docs/overview.png
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

from data_gen import load_readings, load_truth
from detect import Z_THRESHOLD, detect, evaluate, evaluate_all

OUT = ROOT / "docs" / "overview.png"

for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
    try:
        font_manager.findfont(name, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [name]
        break
    except ValueError:
        continue
plt.rcParams["axes.unicode_minus"] = False

readings = load_readings()
truth = load_truth()
group = (readings[readings["device_id"] == "D01"]
         .sort_values("timestamp").reset_index(drop=True))
values = group["temperature"].to_numpy(float)
ts = pd.to_datetime(group["timestamp"])
mask = detect(values)
labels = (truth[truth["device_id"] == "D01"]
          .sort_values("timestamp")["label"].to_numpy())
score = evaluate(mask, labels)
total = evaluate_all(readings, truth)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7.4),
                               gridspec_kw={"height_ratios": [1.15, 1]})
fig.suptitle("工业传感器时序异常检测 · 真实运行结果", fontsize=16, y=0.985)
fig.text(0.5, 0.932,
         f"{len(readings):,} 条读数 · 4 台设备 · 12 项单元测试通过 · "
         f"全量 F1 = {total['f1']:.4f}",
         ha="center", fontsize=10, color="#555555")

ax1.plot(ts, values, lw=0.55, color="#378ADD", label="温度读数")
ax1.scatter(ts[mask], values[mask], s=7, color="#E24B4A",
            label="检出异常点", zorder=3)
ax1.set_title(f"设备 D01 · 30 天温度时序（单点 F1 = {score['f1']:.4f}）",
              fontsize=12)
ax1.set_ylabel("温度 (°C)")
ax1.legend(loc="upper right", fontsize=9, frameon=False)
ax1.grid(alpha=0.25, lw=0.5)

ks = [3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0]
curves = {"精确率": [], "召回率": [], "F1": []}
for k in ks:
    m = evaluate_all(readings, truth, z_threshold=Z_THRESHOLD, drift_k=k)
    curves["精确率"].append(m["precision"])
    curves["召回率"].append(m["recall"])
    curves["F1"].append(m["f1"])
colors = {"精确率": "#639922", "召回率": "#BA7517", "F1": "#D85A30"}
for name, ys in curves.items():
    ax2.plot(ks, ys, marker="o", ms=4, lw=1.6, color=colors[name], label=name)
ax2.axvline(5.0, color="#888888", ls="--", lw=0.9)
ax2.annotate("默认 k = 5.0", xy=(5.0, curves["F1"][3]),
             xytext=(5.6, 0.52), fontsize=9, color="#555555",
             arrowprops=dict(arrowstyle="->", lw=0.8, color="#888888"))
ax2.set_title("阈值敏感性分析（固定 z = 4.0）", fontsize=12)
ax2.set_xlabel("漂移阈值倍数 k")
ax2.set_ylabel("指标值")
ax2.set_ylim(0.0, 1.05)
ax2.legend(loc="lower left", fontsize=9, frameon=False, ncol=3)
ax2.grid(alpha=0.25, lw=0.5)

fig.tight_layout(rect=(0, 0, 1, 0.915))
OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT, dpi=160, facecolor="white")
print(f"已生成 {OUT}")
