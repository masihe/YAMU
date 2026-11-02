"""특징과 간헐적 수요 방식을 손으로 계산한 값과 비교한다."""

import unittest

import numpy as np

from forecast_ml import intermittent
from forecast_ml.features import make_features
from tests.test_baselines import panel_from


class Features(unittest.TestCase):

    def setUp(self):
        # 100일. 84일 전부터 7일마다 1,000원(쓴 날 12번), 기준일 이틀 전 5,000원
        spend = [0] * 100
        for d in range(99 - 83, 100, 7):
            spend[d] = 1_000
        spend[97] = 5_000
        self.panel = panel_from({"A": spend}, "2023-01-01")
        self.f = make_features(self.panel, np.array([0]), np.array([99]), 7).iloc[0]

    def test_windows(self):
        self.assertEqual(self.f["sum_7d"], 1_000 + 5_000)
        self.assertEqual(self.f["days_7d"], 2)
        self.assertEqual(self.f["sum_84d"], 12 * 1_000 + 5_000)
        self.assertEqual(self.f["days_84d"], 13)

    def test_recency_and_size(self):
        self.assertEqual(self.f["days_since_last"], 2)  # 7일 간격 소비는 93일째가 마지막, 97일째 5,000원
        self.assertAlmostEqual(self.f["mean_active_84d"], 17_000 / 13)
        self.assertAlmostEqual(self.f["adi_84d"], 84 / 13)

    def test_no_future_days(self):
        panel = panel_from({"A": self.panel.spend[0].tolist()[:100] + [999_999] * 10}, "2023-01-01")
        f = make_features(panel, np.array([0]), np.array([99]), 7).iloc[0]
        self.assertTrue((f == self.f).all())


class Intermittent(unittest.TestCase):

    def test_tsb_drops_when_no_spend(self):
        spend = np.array([[0, 10, 0, 10, 0, 0, 0, 0]])
        daily = intermittent.tsb_daily(spend, alpha=0.5, beta=0.5)
        self.assertEqual(daily[0, 0], 0)
        # 첫 소비일: z=10, d=0.5 → 5
        self.assertAlmostEqual(daily[0, 1], 5.0)
        # 안 쓴 날이 이어지면 계속 낮아진다
        self.assertTrue(daily[0, 7] < daily[0, 4] < daily[0, 3])

    def test_sba_steady_series(self):
        # 이틀마다 10원 → z=10, p=2 근처, 하루 예측 ≈ (1 − α/2)·5
        spend = np.array([[10, 0] * 50])
        daily = intermittent.sba_daily(spend, alpha=0.2)
        self.assertAlmostEqual(daily[0, -1], 0.9 * 10 / 2, places=1)


if __name__ == "__main__":
    unittest.main()
