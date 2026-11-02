"""Baseline 계산이 Java LivingCostEstimator와 같고, 작은 가짜 데이터에서 손으로 계산한 값과 같은지 본다."""

import unittest

import numpy as np
import pandas as pd

from forecast_ml import baselines
from forecast_ml.backtest import origins
from forecast_ml.data import Panel


def panel_from(daily: dict[str, list[int]], start: str, observed_months: dict[str, list[bool]] | None = None) -> Panel:
    """카드별 일별 소비 목록으로 Panel을 만든다. 확인된 달은 기본으로 모두 True."""
    cards = np.array(sorted(daily))
    length = len(next(iter(daily.values())))
    days = pd.date_range(start, periods=length, freq="D")
    months = pd.period_range(days[0], days[-1], freq="M")
    spend = np.array([daily[c] for c in cards], dtype=np.int64)
    if observed_months is None:
        observed = np.ones((len(cards), len(months)), dtype=bool)
    else:
        observed = np.array([observed_months[c] for c in cards], dtype=bool)
    return Panel(cards, days, months, spend, observed, 0.0)


class Avg3mMatchesJava(unittest.TestCase):
    """be/src/test/.../LivingCostEstimatorTest.java의 기대값."""

    def test_three_months_divided_by_real_days(self):
        # 7·8·9월 합 630,000 ÷ 92일 = 6,847.8 → 6,848
        self.assertEqual(baselines.avg_3m_daily([210_000, 210_000, 210_000], [31, 31, 30]), 6_848)

    def test_zero_month_counts_as_zero(self):
        # 300,000 + 300,000 + 0 = 600,000 ÷ 92 = 6,521.7 → 6,522
        self.assertEqual(baselines.avg_3m_daily([300_000, 300_000, 0], [31, 31, 30]), 6_522)

    def test_leap_february_is_29_days(self):
        self.assertEqual(baselines.avg_3m_daily([87_000, 93_000, 90_000], [29, 31, 30]), 3_000)

    def test_rounds_half_up_to_won(self):
        self.assertEqual(baselines.avg_3m_daily([45], [30]), 2)
        self.assertEqual(baselines.avg_3m_daily([44], [30]), 1)

    def test_no_month_is_none(self):
        self.assertIsNone(baselines.avg_3m_daily([], []))

    def test_spec_example_projected_amount(self):
        # 명세 「생활비 예측 상세」: 일평균 7,000원 × 10일 = 70,000원
        self.assertEqual(baselines.avg_3m_daily([217_000, 217_000, 210_000], [31, 31, 30]) * 10, 70_000)


class PanelBaselines(unittest.TestCase):

    def setUp(self):
        # 2023-01-01부터 120일. 1월 31,000(매일 1,000), 2월 0, 3월 매주 일요일 7,000, 4월은 기준일 이후 정답용
        days = pd.date_range("2023-01-01", periods=120, freq="D")
        spend = []
        for d in days:
            if d.month == 1:
                spend.append(1_000)
            elif d.month == 3 and d.day_of_week == 6:
                spend.append(7_000)
            elif d.month == 4:
                spend.append(500)
            else:
                spend.append(0)
        self.panel = panel_from({"A": spend}, "2023-01-01")
        self.card = np.array([0])
        self.t = np.array([int(np.where(days == pd.Timestamp("2023-04-03"))[0][0])])  # 4월 3일(월)

    def test_avg_3m_uses_previous_three_months_only(self):
        # 1~3월: 31,000 + 0 + 7,000×4(3/5·12·19·26) = 59,000 ÷ 90일 = 655.6 → 656, × 7
        self.assertEqual(int(baselines.avg_3m(self.panel, self.card, self.t, 7)[0]), 656 * 7)

    def test_avg_3m_skips_unobserved_month(self):
        panel = panel_from({"A": self.panel.spend[0].tolist()}, "2023-01-01",
                           {"A": [False, True, True, True]})
        # 2·3월만: 28,000 ÷ 59일 = 474.6 → 475
        self.assertEqual(int(baselines.avg_3m(panel, self.card, self.t, 7)[0]), 475 * 7)

    def test_recent_28_days_includes_today(self):
        # 3/7~4/3: 3/12·19·26 일요일 21,000 + 4/1~3 1,500 = 22,500 ÷ 28 = 803.6 → 804
        self.assertEqual(int(baselines.avg_recent(self.panel, self.card, self.t, 7)[0]), 804 * 7)

    def test_last_week_is_last_seven_days_sum(self):
        # 3/28~4/3: 4/1·2·3 = 1,500 (3/26 일요일은 범위 밖)
        self.assertEqual(int(baselines.last_week(self.panel, self.card, self.t, 7)[0]), 1_500)
        self.assertEqual(int(baselines.last_week(self.panel, self.card, self.t, 14)[0]), 3_000)

    def test_weekly_median_of_four_weeks(self):
        # 주별(7일씩 거꾸로): 1,500 / 7,000 / 7,000 / 7,000 → 중앙값 7,000
        self.assertEqual(int(baselines.weekly_median(self.panel, self.card, self.t, 7)[0]), 7_000)

    def test_weekday_average_puts_sunday_spend_on_sunday(self):
        # 다음 7일(4/4 화 ~ 4/10 월). 일요일 4/9는 지난 8번의 일요일(4/2 500, 3/26·19·12·5 7,000씩, 2월 0) = 28,500,
        # 토요일 4/8은 4/1 500, 월요일 4/10은 4/3 500, 나머지 요일은 0. 합 29,500 ÷ 8 = 3,687.5 → 3,688
        self.assertEqual(int(baselines.weekday_average(self.panel, self.card, self.t, 7)[0]), 3_688)

    def test_actual_is_next_days_sum(self):
        self.assertEqual(int(baselines.actual(self.panel, self.card, self.t, 7)[0]), 3_500)


class Origins(unittest.TestCase):

    def test_drops_origins_with_unobserved_months(self):
        spend = [1_000] * 200
        all_months = panel_from({"A": spend}, "2023-01-01")
        missing_feb = panel_from({"A": spend}, "2023-01-01", {"A": [True, False, True, True, True, True, True]})
        _, t_all = origins(all_months, 7)
        _, t_missing = origins(missing_feb, 7)
        self.assertGreater(len(t_all), len(t_missing))
        months = missing_feb.month_of_day()
        for t in t_missing:
            self.assertFalse(1 in {months[t] - b for b in range(4)})

    def test_checks_every_month_the_horizon_spans(self):
        # 2023-01-01부터 220일(1~8월). 2월만 미확인이면, 1/31 기준 30일 예측(2/1~3/2)은 빼야 한다
        spend = [1_000] * 220
        panel = panel_from({"A": spend}, "2023-01-01", {"A": [True, False, True, True, True, True, True, True]})
        _, t = origins(panel, 30, step=1)
        months = panel.month_of_day()
        for ti in t:
            spanned = set(months[ti + 1: ti + 31]) | {months[ti] - b for b in range(4)}
            self.assertNotIn(1, spanned)

    def test_drops_origins_without_recent_check_spend(self):
        # 처음 100일만 쓰고 그 뒤로 안 씀 → 기준일 전 90일에 소비가 없으면 평가하지 않는다
        spend = [1_000] * 100 + [0] * 150
        panel = panel_from({"A": spend}, "2023-01-01")
        _, t = origins(panel, 7, active_days=90)
        self.assertTrue(len(t) > 0)
        self.assertTrue(all(ti - 89 <= 99 for ti in t))


if __name__ == "__main__":
    unittest.main()
