import unittest

import pandas as pd

from forecast_ml import recurring


def rows(*items: tuple[str, str, str, int]) -> pd.DataFrame:
    """(날짜, 카드, 가맹점, 금액) 목록으로 거래 표를 만든다."""
    return pd.DataFrame({
        "date": pd.to_datetime([d for d, _, _, _ in items]),
        "card": [c for _, c, _, _ in items],
        "store": [s for _, _, s, _ in items],
        "amount": [a for _, _, _, a in items],
    })


def coffee_three_times_a_week(months: int) -> list[tuple[str, str, str, int]]:
    """월·수·금 4,500원 커피."""
    days = pd.date_range("2024-01-01", periods=months * 30, freq="D")
    return [(str(d.date()), "c1", "cafe", 4500) for d in days if d.day_of_week in (0, 2, 4)]


def netflix(months: int, day: int = 15) -> list[tuple[str, str, str, int]]:
    return [(f"2024-{m:02d}-{day:02d}", "c1", "netflix", 13500) for m in range(1, months + 1)]


class FlagTest(unittest.TestCase):
    def test_current_rule_removes_frequent_same_price_coffee(self):
        df = rows(*coffee_three_times_a_week(4))
        flagged = recurring.flag(df, "CURRENT")
        march = df["date"].dt.month >= 3
        self.assertTrue(flagged[march].all())
        self.assertFalse(flagged[~march].any())

    def test_once_a_month_and_interval_keep_coffee(self):
        df = rows(*coffee_three_times_a_week(4))
        self.assertFalse(recurring.flag(df, "ONCE_A_MONTH").any())
        self.assertFalse(recurring.flag(df, "INTERVAL").any())

    def test_every_rule_removes_monthly_subscription_from_third_month(self):
        df = rows(*netflix(5))
        for rule in recurring.RULES:
            with self.subTest(rule=rule):
                self.assertEqual(recurring.flag(df, rule).tolist(), [False, False, True, True, True])

    def test_interval_accepts_short_february_and_month_end(self):
        df = rows(("2024-01-31", "c1", "tel", 55000), ("2024-02-29", "c1", "tel", 55000),
                  ("2024-03-31", "c1", "tel", 55000))
        self.assertEqual(recurring.flag(df, "INTERVAL").tolist(), [False, False, True])

    def test_interval_rejects_gap_over_35_days(self):
        df = rows(("2024-01-01", "c1", "gym", 50000), ("2024-02-05", "c1", "gym", 50000),
                  ("2024-03-12", "c1", "gym", 50000))
        self.assertEqual(recurring.flag(df, "INTERVAL").tolist(), [False, False, False])

    def test_current_counts_only_past_six_months(self):
        # 1·2월과 8월: 8월에서 보면 앞 5개 달(3~7월)에 없으므로 3개 달이 안 된다
        df = rows(("2024-01-10", "c1", "s", 1000), ("2024-02-10", "c1", "s", 1000), ("2024-08-10", "c1", "s", 1000))
        self.assertEqual(recurring.flag(df, "CURRENT").tolist(), [False, False, False])
        df = rows(("2024-03-10", "c1", "s", 1000), ("2024-05-10", "c1", "s", 1000), ("2024-08-10", "c1", "s", 1000))
        self.assertEqual(recurring.flag(df, "CURRENT").tolist(), [False, False, True])

    def test_keys_do_not_mix(self):
        df = rows(("2024-01-10", "c1", "s", 1000), ("2024-02-10", "c2", "s", 1000), ("2024-03-10", "c1", "s", 1000),
                  ("2024-03-10", "c1", "s", 1001))
        for rule in recurring.RULES:
            with self.subTest(rule=rule):
                self.assertFalse(recurring.flag(df, rule).any())

    def test_once_a_month_drops_key_with_any_double_month_in_window(self):
        df = rows(*netflix(3), ("2024-02-20", "c1", "netflix", 13500))
        self.assertEqual(recurring.flag(df, "ONCE_A_MONTH").tolist(), [False, False, False, False])
        self.assertEqual(recurring.flag(df, "CURRENT").tolist(), [False, False, True, False])

    def test_result_follows_input_index(self):
        df = rows(*netflix(3)).iloc[::-1]
        df.index = [7, 8, 9]
        for rule in recurring.RULES:
            with self.subTest(rule=rule):
                result = recurring.flag(df, rule)
                self.assertEqual(list(result.index), [7, 8, 9])
                self.assertEqual(result.tolist(), [True, False, False])

    def test_unknown_rule(self):
        with self.assertRaises(ValueError):
            recurring.flag(rows(*netflix(1)), "NOPE")


if __name__ == "__main__":
    unittest.main()
