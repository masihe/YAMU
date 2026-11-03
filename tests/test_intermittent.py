"""TSB를 최근 window일만으로 계산하면 그 구간만 잘라 처음부터 돌린 TSB와 같은지."""

import unittest

import numpy as np

from forecast_ml import intermittent


class TsbWindow(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(1)
        self.spend = ((rng.random((3, 200)) < 0.3) * rng.integers(1_000, 50_000, (3, 200))).astype(np.int64)

    def test_window_covering_all_history_equals_full_tsb(self):
        card = np.array([0, 1, 2])
        t = np.array([89, 89, 89])

        windowed = intermittent.tsb_window(self.spend, card, t, alpha=0.2, beta=0.1, window=90)
        full = intermittent.tsb_daily(self.spend, alpha=0.2, beta=0.1)[card, t]

        np.testing.assert_allclose(windowed, full)

    def test_short_window_equals_tsb_started_at_window_start(self):
        card = np.array([0, 2, 1])
        t = np.array([150, 199, 120])

        windowed = intermittent.tsb_window(self.spend, card, t, alpha=0.1, beta=0.05, window=90)
        expected = [intermittent.tsb_daily(self.spend[c:c + 1, d - 89:d + 1], alpha=0.1, beta=0.05)[0, -1]
                    for c, d in zip(card, t)]

        np.testing.assert_allclose(windowed, expected)
        # 전체 기록으로 계산한 값과는 다르다(창 밖의 소비를 보지 않음)
        full = intermittent.tsb_daily(self.spend, alpha=0.1, beta=0.05)[card, t]
        self.assertFalse(np.allclose(windowed, full))

    def test_window_before_first_day_counts_as_zero_spend(self):
        windowed = intermittent.tsb_window(self.spend, np.array([0]), np.array([10]), alpha=0.2, beta=0.1, window=90)
        full = intermittent.tsb_daily(self.spend, alpha=0.2, beta=0.1)[0, 10]

        self.assertAlmostEqual(float(windowed[0]), float(full))


if __name__ == "__main__":
    unittest.main()
