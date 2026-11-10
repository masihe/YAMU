"""제외 규칙: 큰 지출을 분위수 대신 한 건 금액으로 뺄 수 있는지."""

import unittest

import pandas as pd

from forecast_ml.data import CREDIT_CODES, check_rows


def rows(amounts: list[int]) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.to_datetime(["2024-01-01"] * len(amounts)), "card": "C1",
                         "amount": amounts, "fraud": "0", "code": "10",
                         "store": [f"S{i}" for i in range(len(amounts))]})


class LargeAmount(unittest.TestCase):

    def test_amount_rule_removes_only_rows_above_the_amount(self):
        df = rows([10_000, 1_000_000, 1_000_001, 3_000_000])

        kept, removed_share = check_rows(df, frozenset({"large"}), large_amount=1_000_000)

        # 100만 원 "초과"만 뺀다. 정확히 100만 원은 남는다
        self.assertEqual(kept["amount"].tolist(), [10_000, 1_000_000])
        self.assertAlmostEqual(removed_share, 4_000_001 / 5_010_001)

    def test_without_amount_rule_uses_quantile(self):
        df = rows(list(range(1, 101)))

        kept, _ = check_rows(df, frozenset({"large"}), large_quantile=0.9)

        self.assertEqual(int(kept["amount"].max()), int(df["amount"].quantile(0.9)))


def dated(items: list[tuple[str, str, int]]) -> pd.DataFrame:
    """(날짜, 가맹점, 금액) 목록. 카드 하나, 이상거래 아님, 체크 승인."""
    return pd.DataFrame({"date": pd.to_datetime([d for d, _, _ in items]), "card": "C1",
                         "store": [s for _, s, _ in items], "amount": [a for _, _, a in items],
                         "fraud": "0", "code": "10"})


def coffee_and_netflix() -> pd.DataFrame:
    """1~4월 매달 15일 넷플릭스 13,500원, 매주 월요일 커피 4,500원."""
    netflix = [(f"2024-{m:02d}-15", "netflix", 13_500) for m in range(1, 5)]
    coffee = [(str(d.date()), "cafe", 4_500) for d in pd.date_range("2024-01-01", "2024-04-30", freq="W-MON")]
    return dated(netflix + coffee)


class RecurringOnce(unittest.TestCase):
    """recurring_once는 서비스 규칙(과거 6개월, 매달 한 번)으로 뺀다(S15P31E103-304)."""

    def test_removes_monthly_subscription_from_third_month_and_keeps_frequent_coffee(self):
        kept, _ = check_rows(coffee_and_netflix(), frozenset({"recurring_once"}))

        netflix = kept[kept["store"] == "netflix"]
        self.assertEqual(netflix["date"].dt.month.tolist(), [1, 2])
        self.assertEqual(int((kept["store"] == "cafe").sum()), 18)

    def test_old_rule_still_removes_every_month_of_both(self):
        kept, _ = check_rows(coffee_and_netflix(), frozenset({"recurring"}))

        self.assertTrue(kept.empty)

    def test_counts_rows_that_other_rules_remove(self):
        # 2월 구독이 이상거래로 빠져도 반복 판정에서는 센다(BE는 분류와 상관없이 셈). 세지 않으면 3월이 남는다
        df = dated([(f"2024-{m:02d}-15", "netflix", 13_500) for m in range(1, 4)])
        df.loc[1, "fraud"] = "1"

        kept, _ = check_rows(df, frozenset({"fraud", "recurring_once"}))

        self.assertEqual(kept["date"].dt.month.tolist(), [1])


if __name__ == "__main__":
    unittest.main()


class Codes(unittest.TestCase):
    """codes로 신용(00) 거래만 골라 같은 제외 규칙을 적용한다(S15P31E103-404)."""

    def test_credit_codes_keep_only_lump_sum_credit(self):
        df = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"] * 3), "card": "C1", "amount": [1_000, 2_000, 3_000],
                           "fraud": "0", "code": ["00", "01", "10"], "store": ["S1", "S2", "S3"]})

        kept, _ = check_rows(df, frozenset(), codes=CREDIT_CODES)

        self.assertEqual(kept["amount"].tolist(), [1_000])

    def test_recurring_is_counted_within_the_chosen_codes_only(self):
        # 같은 가맹점·금액을 1·2월은 체크, 3월은 신용으로 냄. 신용만 보면 한 달뿐이라 반복이 아니다
        df = dated([("2024-01-15", "netflix", 13_500), ("2024-02-15", "netflix", 13_500),
                    ("2024-03-15", "netflix", 13_500)])
        df.loc[2, "code"] = "00"

        kept, _ = check_rows(df, frozenset({"recurring_once"}), codes=CREDIT_CODES)

        self.assertEqual(len(kept), 1)
