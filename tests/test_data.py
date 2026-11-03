"""제외 규칙: 큰 지출을 분위수 대신 한 건 금액으로 뺄 수 있는지."""

import unittest

import pandas as pd

from forecast_ml.data import check_rows


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


if __name__ == "__main__":
    unittest.main()
