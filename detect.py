"""两级异常检测：局部突变 + 缓慢漂移，并输出可解释的告警区间。

运行：python detect.py
输出：data/anomalies.csv，并在终端打印精确率 / 召回率 / F1。

只依赖 numpy + pandas，可单独运行、单独测试。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

WINDOW = 288          # 局部窗口 = 1 天（5 分钟 × 288）
DRIFT_WINDOW = 96     # 漂移窗口 = 8 小时
Z_THRESHOLD = 4.0     # 局部 robust z-score 阈值
DRIFT_K = 5.0         # 漂移阈值 = k × 噪声尺度
MIN_SEGMENT = 2       # 少于 2 点的孤立点视为噪声
SEGMENT_COLUMNS = ["device_id", "kind", "start_time", "end_time",
                   "length", "max_deviation", "start_idx", "end_idx"]


def robust_z_score(values: np.ndarray, window: int = WINDOW) -> np.ndarray:
    """滑动窗口中位数 + MAD 的稳健 z 分数，对离群点不敏感。"""
    s = pd.Series(np.asarray(values, dtype=float))
    med = s.rolling(window, center=True, min_periods=8).median()
    mad = (s - med).abs().rolling(window, center=True, min_periods=8).median()
    scale = np.maximum(mad * 1.4826, 0.5)  # MAD 转近似标准差，下限防除零
    return ((s - med).abs() / scale).fillna(0.0).to_numpy()


def detect(values: np.ndarray, window: int = WINDOW,
           z_threshold: float = Z_THRESHOLD,
           drift_k: float = DRIFT_K) -> np.ndarray:
    """返回与输入等长的布尔数组，True 表示异常点。

    第一级：局部 robust z-score 抓尖峰、断连等突变。
    第二级：窗口中位数相对全局基线的偏移抓缓慢漂移；
            阈值用一阶差分的 MAD 估计噪声尺度，避免写死绝对温度。
    """
    v = np.asarray(values, dtype=float)
    spike = robust_z_score(v, window) > z_threshold

    diff = np.diff(v)
    sigma = max(1.4826 * float(np.median(np.abs(diff - np.median(diff)))), 0.5)
    ref = float(np.median(v))
    roll_med = (pd.Series(v)
                .rolling(DRIFT_WINDOW, center=True, min_periods=8)
                .median().to_numpy())
    drift = np.abs(roll_med - ref) > drift_k * sigma

    return np.asarray(spike | drift, dtype=bool)


def merge_segments(mask: np.ndarray, timestamps: np.ndarray | None = None,
                   min_length: int = MIN_SEGMENT) -> list[dict]:
    """把连续的异常点合并成告警区间，过滤孤立噪声点。"""
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    if idx.size == 0:
        return []
    groups = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    segments = []
    for g in groups:
        if len(g) < min_length:
            continue
        seg = {"start_idx": int(g[0]), "end_idx": int(g[-1]),
               "length": int(len(g))}
        if timestamps is not None:
            seg["start_time"] = pd.Timestamp(timestamps[g[0]]).strftime("%Y-%m-%d %H:%M:%S")
            seg["end_time"] = pd.Timestamp(timestamps[g[-1]]).strftime("%Y-%m-%d %H:%M:%S")
        segments.append(seg)
    return segments


def _classify(values: np.ndarray, seg: dict, ref: float) -> str:
    part = values[seg["start_idx"]:seg["end_idx"] + 1]
    seg["max_deviation"] = round(float(np.max(np.abs(part - ref))), 2)
    if np.min(part) < 5.0:
        return "断连"
    if seg["length"] <= 10:
        return "尖峰"
    return "漂移"


def run_pipeline(readings: pd.DataFrame, window: int = WINDOW,
                 z_threshold: float = Z_THRESHOLD,
                 drift_k: float = DRIFT_K) -> pd.DataFrame:
    """对全部设备逐台检测，返回告警区间表。"""
    rows = []
    for dev, group in readings.groupby("device_id", sort=True):
        group = group.sort_values("timestamp").reset_index(drop=True)
        v = group["temperature"].to_numpy(dtype=float)
        ts = group["timestamp"].to_numpy()
        mask = detect(v, window=window, z_threshold=z_threshold, drift_k=drift_k)
        ref = float(np.median(v))
        for seg in merge_segments(mask, ts):
            seg["device_id"] = dev
            seg["kind"] = _classify(v, seg, ref)
            rows.append(seg)
    return pd.DataFrame(rows, columns=SEGMENT_COLUMNS)


def evaluate(mask: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """点级别评估：精确率 / 召回率 / F1 / 误报率。"""
    pred = np.asarray(mask, dtype=bool)
    truth = np.asarray(labels) > 0
    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn,
            "false_positive_rate": round(fp / max(len(pred), 1), 4)}


def evaluate_all(readings: pd.DataFrame, truth: pd.DataFrame,
                 window: int = WINDOW, z_threshold: float = Z_THRESHOLD,
                 drift_k: float = DRIFT_K) -> dict[str, float]:
    merged = readings.merge(truth, on=["timestamp", "device_id"], how="inner")
    masks, labels = [], []
    for _, group in merged.groupby("device_id", sort=True):
        group = group.sort_values("timestamp")
        masks.append(detect(group["temperature"].to_numpy(float),
                            window=window, z_threshold=z_threshold,
                            drift_k=drift_k))
        labels.append(group["label"].to_numpy())
    return evaluate(np.concatenate(masks), np.concatenate(labels))


def main() -> None:
    from data_gen import load_readings, load_truth

    DATA_DIR.mkdir(exist_ok=True)
    readings = load_readings()
    anomalies = run_pipeline(readings)
    anomalies.to_csv(DATA_DIR / "anomalies.csv", index=False, encoding="utf-8")

    print(f"读数 {len(readings)} 条，检出告警区间 {len(anomalies)} 段")
    print(anomalies.groupby(["device_id", "kind"]).size().to_string())
    truth = load_truth()
    if truth is None:
        print("未找到 ground_truth.csv，先运行 python data_gen.py 再评估")
    else:
        print("评估指标：", evaluate_all(readings, truth))
    print(f"告警明细已写入：{DATA_DIR / 'anomalies.csv'}")


if __name__ == "__main__":
    main()
