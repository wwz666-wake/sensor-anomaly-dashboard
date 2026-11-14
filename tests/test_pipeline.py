"""单元测试：用固定信号验证三类故障都能被检出，且正常信号误报可控。

运行：python -m unittest discover -s tests -v
或：  pytest tests -q
"""

from __future__ import annotations

import unittest

import numpy as np

from data_gen import generate
from detect import detect, evaluate, merge_segments, robust_z_score, run_pipeline


def _clean_signal(n: int = 600, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 65.0 + 2.0 * np.sin(np.arange(n) / 40) + rng.normal(0, 0.5, n)


class TestDataGeneration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = generate(days=3, devices=("D01",), seed=7)

    def test_01_schema_and_size(self):
        self.assertEqual(
            set(["timestamp", "device_id", "temperature", "vibration", "label"]),
            set(self.df.columns))
        self.assertEqual(len(self.df), 3 * 24 * 12)
        self.assertFalse(self.df.isna().any().any())

    def test_02_three_fault_types_present(self):
        self.assertEqual(set(self.df["label"].unique()), {0, 1, 2, 3})

    def test_03_labels_are_minority(self):
        ratio = (self.df["label"] > 0).mean()
        self.assertGreater(ratio, 0.01)
        self.assertLess(ratio, 0.30)


class TestDetection(unittest.TestCase):

    def test_04_spike_detected(self):
        v = _clean_signal()
        v[300:305] += 25
        mask = detect(v)
        self.assertTrue(mask[300:305].all())

    def test_05_dropout_detected(self):
        v = _clean_signal()
        v[300:340] = 0.0
        mask = detect(v)
        self.assertGreater(mask[300:340].mean(), 0.8)

    def test_06_drift_detected(self):
        v = np.full(600, 65.0) + np.random.default_rng(1).normal(0, 0.3, 600)
        v[400:] += np.linspace(0, 10, 200)
        mask = detect(v)
        self.assertGreater(mask[480:].mean(), 0.5)

    def test_07_normal_signal_low_false_positive(self):
        v = _clean_signal(1200, seed=9)
        self.assertLess(detect(v).mean(), 0.05)

    def test_08_robust_z_score_is_finite(self):
        z = robust_z_score(_clean_signal())
        self.assertTrue(np.isfinite(z).all())
        self.assertGreater(z.max(), 0)


class TestSegmentsAndMetrics(unittest.TestCase):

    def test_09_merge_consecutive_points(self):
        mask = np.array([0, 1, 1, 0, 1, 1, 1, 0, 1], dtype=bool)
        segs = merge_segments(mask)
        self.assertEqual([(s["start_idx"], s["end_idx"]) for s in segs],
                         [(1, 2), (4, 6)])

    def test_10_perfect_prediction_scores_one(self):
        truth = np.array([0, 1, 1, 0])
        m = evaluate(truth.astype(bool), truth)
        self.assertEqual(m["precision"], 1.0)
        self.assertEqual(m["recall"], 1.0)
        self.assertEqual(m["f1"], 1.0)

    def test_11_metrics_stay_in_range(self):
        rng = np.random.default_rng(0)
        m = evaluate(rng.random(100) > 0.7, rng.random(100) > 0.8)
        for key in ("precision", "recall", "f1", "false_positive_rate"):
            self.assertGreaterEqual(m[key], 0.0)
            self.assertLessEqual(m[key], 1.0)

    def test_12_pipeline_output_columns(self):
        df = generate(days=2, devices=("D01", "D02"), seed=11)
        out = run_pipeline(df.drop(columns=["label"]))
        self.assertEqual(
            list(out.columns),
            ["device_id", "kind", "start_time", "end_time", "length",
             "max_deviation", "start_idx", "end_idx"])
        self.assertGreater(len(out), 0)


if __name__ == "__main__":
    unittest.main()
