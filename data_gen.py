"""生成模拟工业设备传感器时序数据与故障标签。

运行：python data_gen.py
输出：data/sensor_readings.csv（读数，不含标签）
      data/ground_truth.csv（标签，仅用于评估，检测程序不会读取）

数据为程序生成的模拟数据，不含任何真实企业信息，可安全公开。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

DAYS = 30
FREQ_MIN = 5
DEVICES = ("D01", "D02", "D03", "D04")
SEED = 20261114
START = "2026-10-15 00:00"

# 标签：0 正常，1 尖峰，2 漂移，3 断连
LABEL_NAME = {0: "正常", 1: "尖峰", 2: "漂移", 3: "断连"}


def _inject_faults(rng: np.random.Generator, n: int, temp: np.ndarray,
                   label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """在固定且互不重叠的时间槽注入三类故障，保证标签与信号一一对应。"""
    spike_len = min(8, max(3, n // 1000))
    drift_len = min(180, max(50, n // 40))
    drop_len = min(60, max(20, n // 100))

    def put(frac: float, length: int, kind: int, mutate) -> None:
        s = int(n * frac)
        length = min(length, n - s)
        if length < 2:
            return
        temp[s:s + length] = mutate(temp[s:s + length], length)
        label[s:s + length] = kind

    # 尖峰：3 段，读数瞬时抬升 18~28 度
    for frac in (0.10, 0.38, 0.66):
        put(frac, spike_len, 1,
            lambda x, L: x + rng.uniform(18, 28))

    # 漂移：2 段，读数缓慢抬升 14~18 度
    for frac in (0.20, 0.48):
        put(frac, drift_len, 2,
            lambda x, L: x + np.linspace(0, rng.uniform(14, 18), L))

    # 断连：2 段，读数掉到 0
    for frac in (0.80, 0.90):
        put(frac, drop_len, 3,
            lambda x, L: np.zeros(L))

    return temp, label


def generate(days: int = DAYS, freq_min: int = FREQ_MIN,
             devices: tuple[str, ...] = DEVICES,
             seed: int = SEED) -> pd.DataFrame:
    """生成带标签的完整数据集（含 label 列）。"""
    rng = np.random.default_rng(seed)
    n = days * 24 * 60 // freq_min
    timestamps = pd.date_range(START, periods=n, freq=f"{freq_min}min")
    period = 24 * 60 / freq_min
    t = np.arange(n)

    frames = []
    for i, dev in enumerate(devices):
        base = 65.0 + i * 1.5
        daily = 4.0 * np.sin(2 * np.pi * t / period + i * 0.7)
        temp = base + daily + rng.normal(0, 0.6, n)
        vib = (2.0 + 0.25 * np.sin(2 * np.pi * t / (period / 4))
               + rng.normal(0, 0.08, n))
        label = np.zeros(n, dtype=np.int8)
        temp, label = _inject_faults(rng, n, temp, label)

        frames.append(pd.DataFrame({
            "timestamp": timestamps,
            "device_id": dev,
            "temperature": np.round(temp, 2),
            "vibration": np.round(vib, 3),
            "label": label,
        }))

    return pd.concat(frames, ignore_index=True)


def load_readings() -> pd.DataFrame:
    """优先读取 CSV；不存在时直接在内存生成，保证任何入口都能一键运行。"""
    path = DATA_DIR / "sensor_readings.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=["timestamp"])
    else:
        df = generate().drop(columns=["label"])
    return df


def load_truth() -> pd.DataFrame | None:
    path = DATA_DIR / "ground_truth.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["timestamp"])


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    df = generate()
    df.drop(columns=["label"]).to_csv(
        DATA_DIR / "sensor_readings.csv", index=False, encoding="utf-8")
    df[["timestamp", "device_id", "label"]].to_csv(
        DATA_DIR / "ground_truth.csv", index=False, encoding="utf-8")
    print(f"生成 {len(df)} 条读数，{df['device_id'].nunique()} 台设备，"
          f"时间跨度 {df['timestamp'].min()} ~ {df['timestamp'].max()}")
    print(f"异常点占比 {(df['label'] > 0).mean():.2%}")
    print(f"输出目录：{DATA_DIR}")


if __name__ == "__main__":
    main()
