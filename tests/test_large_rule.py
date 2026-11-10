"""BE 큰 결제 중앙값 규칙의 AI-Hub 근사(S15P31E103-404)."""

import unittest

import pandas as pd

from forecast_ml.large_rule import median_large, with_median_large_excluded


def rows(items: list[tuple[str, int, str]]) -> pd.DataFrame:
    """(날짜, 금액, 승인거래코드) 목록. 카드 하나, 같은 시간대, 순번은 목록 순서."""
    return pd.DataFrame({"date": pd.to_datetime([d for d, _, _ in items]), "seq": range(len(items)), "hour": 10,
                         "card": "C1", "amount": [a for _, a, _ in items], "code": [c for _, _, c in items]})


class MedianLarge(unittest.TestCase):

    def test_ten_times_median_of_last_three_months(self):
        coffee = [(f"2024-01-0{i}", 4_000, "10") for i in range(1, 6)]
        flags = median_large(rows(coffee + [("2024-01-10", 40_000, "00"), ("2024-01-11", 39_999, "00")])).tolist()

        # 직전 5건 중앙값 4,000원의 10배(40,000원)부터 LARGE. 39,999원은 LIVING
        self.assertEqual(flags, [False] * 5 + [True, False])

    def test_large_and_installment_do_not_join_the_sample(self):
        coffee = [(f"2024-01-0{i}", 4_000, "10") for i in range(1, 5)]
        flags = median_large(rows(coffee + [("2024-01-06", 500_000, "01"), ("2024-01-07", 2_000_000, "00"),
                                            ("2024-01-08", 40_000, "00")])).tolist()

        # 할부·100만 원 초과는 표본이 아니라 직전 LIVING이 4건뿐 → 5건 미만이라 이 규칙을 쓰지 않는다
        self.assertEqual(flags, [False] * 7)

    def test_sample_older_than_three_months_drops_out(self):
        coffee = [(f"2024-01-0{i}", 4_000, "10") for i in range(1, 6)]
        flags = median_large(rows(coffee + [("2024-04-06", 40_000, "00")])).tolist()

        self.assertEqual(flags, [False] * 6)


if __name__ == "__main__":
    unittest.main()


class Excluded(unittest.TestCase):

    def test_flagged_rows_become_fraud_and_others_stay(self):
        coffee = [(f"2024-01-0{i}", 4_000, "10") for i in range(1, 7)]
        df = rows(coffee + [("2024-01-10", 40_000, "00")]).assign(fraud="0")
        df.loc[0, "fraud"] = "1"  # 이미 이상거래인 행은 판정 표본에서 빠지고 그대로 남는다

        out = with_median_large_excluded(df, frozenset({"00", "10"}))

        # 이상거래를 뺀 직전 표본 5건(중앙값 4,000원)의 10배라 40,000원이 이상거래 표시로 바뀐다
        self.assertEqual(out["fraud"].tolist(), ["1"] + ["0"] * 5 + ["1"])
        self.assertEqual(df["fraud"].tolist(), ["1"] + ["0"] * 6)  # 원본은 그대로
