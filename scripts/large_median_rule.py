"""BE SpendingClassifier의 "직전 3개월 LIVING 중앙값 10배 이상 → LARGE" 규칙을 AI-Hub에 적용하면 학습 데이터에서 얼마나 더 빠지나
(S15P31E103-404 Codex 검토, 2026-11-10). 측정만 하고 학습 데이터·모델은 바꾸지 않는다.

BE 규칙(be/.../SpendingClassifier.java): 할부(2개월 이상)이거나 한 건 100만 원 초과면 LARGE. 아니면 사용자의 모든 카드에서
그 결제 직전 3개월(승인 시각 기준)의 LIVING 결제가 5건 이상이고, 그 건당 중앙값(짝수면 가운데 둘 평균)의 10배 이상이면 LARGE.
LARGE로 정한 결제는 표본에 넣지 않는다.

AI-Hub 근사: 사람 = 카드KEY(체크·신용 결제 모두 표본). 승인 시각이 없어 (날짜, 승인시간대, 승인SEQ) 순서로 "직전"을 정하고,
3개월 창은 날짜로 자른다(그 결제 날짜 − 3개월 이후 날짜). 이상거래는 없는 거래로 본다(서비스에 없는 개념). 국내 일시불·체크(00·10·11)와
할부(01)만 본다.

사용: .venv\\Scripts\\python.exe -I scripts\\large_median_rule.py
"""

import argparse
from pathlib import Path

import numpy as np

from forecast_ml import recurring
from forecast_ml.data import CHECK_CODES, CREDIT_CODES
from forecast_ml.large_rule import INSTALLMENT, LARGE_AMOUNT, MIN_SAMPLES, MULTIPLE, load_ordered as load, median_large
from forecast_ml.runs import new_run, write_json

REPO = Path(__file__).resolve().parents[1]
CARD_ZIPS = ("TL_카드거래.zip", "VL_카드거래.zip")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO / "data" / "raw" / "aihub-71925")
    parser.add_argument("--runs-dir", type=Path, default=REPO / "runs")
    args = parser.parse_args()
    zips = [args.data_dir / name for name in CARD_ZIPS]
    run_dir = new_run(args.runs_dir, "large-median-rule", {
        "script": "scripts/large_median_rule.py", "inputs": zips, "min_samples": MIN_SAMPLES, "multiple": MULTIPLE,
        "large_amount": LARGE_AMOUNT, "approx": "사람=카드KEY, 직전=(날짜,승인시간대,승인SEQ) 순, 3개월 창은 날짜"}, REPO)
    df = load(zips)
    df = df[(df["fraud"] != "1") & df["code"].isin(CHECK_CODES | CREDIT_CODES | {INSTALLMENT})].copy()
    flag = median_large(df)
    results = {}
    for name, codes in (("credit", CREDIT_CODES), ("check", CHECK_CODES)):
        part = df[df["code"].isin(codes)]
        rec = recurring.flag(part[["date", "card", "store", "amount"]], "ONCE_A_MONTH")
        # 지금 학습 데이터: 이상거래(이미 뺌)·100만 원 초과·반복 지출 제외 뒤
        living = part[(part["amount"] <= LARGE_AMOUNT) & ~rec]
        hit = flag.loc[living.index]
        amount = living["amount"]
        per_card = living.assign(hit=hit).groupby("card")["hit"].any()
        results[name] = {
            "living_rows": int(len(living)), "living_amount": int(amount.sum()),
            "median_large_rows": int(hit.sum()), "median_large_amount": int(amount[hit].sum()),
            "row_share": round(float(hit.mean()), 4), "amount_share": round(float(amount[hit].sum() / amount.sum()), 4),
            "cards_with_hit_share": round(float(per_card.mean()), 4),
            "hit_amount_quantiles": [int(v) for v in np.percentile(amount[hit], [10, 50, 90])] if hit.any() else [],
        }
        print(name, results[name], flush=True)
    write_json(run_dir / "metrics.json", results)
    print(run_dir)


if __name__ == "__main__":
    main()
