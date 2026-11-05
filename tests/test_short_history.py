"""짧은 기록 평가: 앞쪽을 0원으로 채운 입력이 맞는지, TSB가 앞쪽 0원과 상관없이 '최근 L일만 본 TSB'와 같은지."""

import unittest

import numpy as np

from forecast_ml import intermittent, short_history
from forecast_ml.serving import panel_from_history


class ShortHistory(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(3)
        daily = ((rng.random((2, 90)) < 0.4) * rng.integers(1_000, 30_000, (2, 90))).astype(np.int64)
        self.panel = panel_from_history(daily, np.datetime64("2024-03-31").astype(object))
        self.card = np.array([0, 1])
        self.t = np.array([89, 89])

    def test_days_before_history_start_are_zero_and_recent_days_kept(self):
        daily = short_history.truncated_history(self.panel, self.card, self.t, 30)

        self.assertEqual(daily.shape, (2, 90))
        self.assertTrue((daily[:, :60] == 0).all())
        np.testing.assert_array_equal(daily[:, 60:], self.panel.spend[:, 60:])

    def test_full_length_keeps_everything(self):
        daily = short_history.truncated_history(self.panel, self.card, self.t, 90)

        np.testing.assert_array_equal(daily, self.panel.spend)

    def test_tsb_on_zero_padded_history_equals_tsb_window_of_length(self):
        for length in (14, 30, 60):
            padded = short_history.truncated_history(self.panel, self.card, self.t, length)
            on_padded = intermittent.tsb_daily(padded, alpha=0.2, beta=0.1)[:, -1]
            windowed = intermittent.tsb_window(self.panel.spend, self.card, self.t, alpha=0.2, beta=0.1, window=length)
            np.testing.assert_allclose(on_padded, windowed)

    def test_activity_bucket_boundaries(self):
        buckets = short_history.activity_bucket(np.array([0, 1, 3, 4, 10, 11, 20, 21, 90]))

        np.testing.assert_array_equal(buckets, [-1, 0, 0, 1, 1, 2, 2, 3, 3])

    def test_rejects_length_outside_history(self):
        with self.assertRaises(ValueError):
            short_history.truncated_history(self.panel, self.card, self.t, 0)
        with self.assertRaises(ValueError):
            short_history.truncated_history(self.panel, self.card, self.t, 91)


if __name__ == "__main__":
    unittest.main()
