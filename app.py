"""Streamlit 可视化看板：时序曲线、异常点标注、告警列表与指标卡。

启动：streamlit run app.py
浏览器打开 http://localhost:8501
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from data_gen import DEVICES, generate
from detect import DRIFT_K, Z_THRESHOLD, detect, evaluate, merge_segments, run_pipeline

st.set_page_config(page_title="工业传感器异常检测看板", layout="wide")
alt.data_transformers.disable_max_rows()


@st.cache_data(show_spinner="正在生成模拟传感器数据…")
def load_all() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = generate()
    readings = df.drop(columns=["label"])
    truth = df[["timestamp", "device_id", "label"]]
    return readings, truth


readings, truth = load_all()

st.title("工业传感器时序异常检测看板")
st.caption("模拟数据 · 两级检测（局部突变 + 缓慢漂移） · 阈值可在左侧实时调整")

with st.sidebar:
    st.header("参数")
    device = st.selectbox("设备", DEVICES)
    z_threshold = st.slider("突变阈值 z", 2.0, 8.0, Z_THRESHOLD, 0.5,
                            help="局部 robust z-score 超过该值判为突变")
    drift_k = st.slider("漂移倍数 k", 3.0, 12.0, DRIFT_K, 0.5,
                        help="窗口中位数偏离全局基线超过 k 倍噪声尺度判为漂移")
    st.divider()
    st.caption("数据由 data_gen.py 生成，替换同名 CSV 即可接入真实传感器读数。")

group = (readings[readings["device_id"] == device]
         .sort_values("timestamp").reset_index(drop=True))
values = group["temperature"].to_numpy(dtype=float)
mask = detect(values, z_threshold=z_threshold, drift_k=drift_k)
labels = (truth[truth["device_id"] == device]
          .sort_values("timestamp")["label"].to_numpy())
score = evaluate(mask, labels)

c1, c2, c3, c4 = st.columns(4)
c1.metric("数据点数", f"{len(group):,}")
c2.metric("告警区间", f"{len(merge_segments(mask))}")
c3.metric("召回率", f"{score['recall']:.1%}")
c4.metric("F1", f"{score['f1']:.3f}")

plot = group[["timestamp", "temperature"]].copy()
plot["anomaly"] = mask
line = (alt.Chart(plot.iloc[::3])
        .mark_line(color="#378ADD", strokeWidth=1.2)
        .encode(x=alt.X("timestamp:T", title="时间"),
                y=alt.Y("temperature:Q", scale=alt.Scale(zero=False),
                        title="温度 (°C)")))
dots = (alt.Chart(plot[plot["anomaly"]])
        .mark_circle(color="#E24B4A", size=20, opacity=0.85)
        .encode(x="timestamp:T", y="temperature:Q",
                tooltip=["timestamp:T", "temperature:Q"]))
st.altair_chart((line + dots).properties(height=340), width="stretch")

st.subheader("告警明细")
segments = merge_segments(mask, group["timestamp"].to_numpy())
seg_df = pd.DataFrame([
    {"开始时间": s["start_time"][:19], "结束时间": s["end_time"][:19],
     "持续点数": s["length"],
     "最大偏差 (°C)": round(float(np.max(np.abs(
         values[s["start_idx"]:s["end_idx"] + 1] - np.median(values)))), 2)}
    for s in segments
])
if seg_df.empty:
    st.info("当前阈值下未检出告警，可尝试降低左侧阈值。")
else:
    st.dataframe(seg_df, width="stretch", hide_index=True)
    st.download_button(
        "下载告警明细 CSV",
        seg_df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"anomalies_{device}.csv", mime="text/csv")

with st.expander("全设备告警概览"):
    overview = run_pipeline(readings, z_threshold=z_threshold, drift_k=drift_k)
    st.dataframe(
        overview.groupby(["device_id", "kind"]).size().unstack(fill_value=0),
        width="stretch")

with st.expander("指标口径说明"):
    st.markdown(
        "- 标签仅用于**评估**，检测过程不读取标签列。\n"
        "- 召回率 = 检出的异常点 / 全部真实异常点；F1 为精确率与召回率的调和平均。\n"
        "- 精确率、误报率可在 `python detect.py` 的终端输出中查看完整数值。")
