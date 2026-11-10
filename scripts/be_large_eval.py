"""서비스 조건(BE 큰 결제 중앙값 규칙까지 뺀 입력·정답) 재평가(S15P31E103-404 Codex 검토, 2026-11-11).
사전 등록은 reports/7.3.10-credit-estimate.md 「서비스 조건 재평가 사전 등록」.

신용·체크 × 사람 5묶음 × 2023·2024 × H. 묶음마다
- AI_NOW: 지금 학습 데이터(중앙값 규칙 없음)로 학습한 TWO_STAGE_TREE
- AI_BE: 중앙값 규칙에 걸린 결제를 뺀 데이터로 학습한 TWO_STAGE_TREE(참고, 서비스용 모델 아님)
- TSB_FIXED: α 0.3·β 0.2, 최근 90일
평가 행·입력·정답은 모두 중앙값 규칙을 뺀 데이터 기준이다.

사용: .venv\\Scripts\\python.exe -I scripts\\be_large_eval.py [--kinds credit check] [--horizons 7 14 30]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_ml import baselines, crossval, intermittent, models
from forecast_ml.backtest import VARIANTS, metrics
from forecast_ml.data import CHECK_CODES, CREDIT_CODES, build_panel
from forecast_ml.experiment import BIAS_LIMIT, TSB_WINDOW, split
from forecast_ml.features import make_features
from forecast_ml.large_rule import INSTALLMENT, load_ordered, with_median_large_excluded
from forecast_ml.runs import new_run, write_json

REPO = Path(__file__).resolve().parents[1]
CARD_ZIPS = ("TL_카드거래.zip", "VL_카드거래.zip")
YEARS = {"2023": ("2022-07-01", "2023-01-01", "2024-01-01"), "2024": ("2023-07-01", "2024-01-01", None)}
VARIANT = "fraud+large+recurring_once"
LARGE_AMOUNT = 1_000_000
BE_PARAMS = {"alpha": 0.3, "beta": 0.2}
KINDS = {"credit": CREDIT_CODES, "check": CHECK_CODES}
NAMES = ("AI_NOW", "AI_BE", "TSB_FIXED")


def fit(panel, horizon: int, val_from, test_from, test_to, bucket: int):
    """panel의 학습 행으로 TWO_STAGE_TREE를 학습한다."""
    s = split(panel, horizon, val_from, test_from, test_to, bucket)
    X = make_features(panel, s.card, s.t, horizon, "core")
    return models.build("TWO_STAGE_TREE", {}).fit(X[s.train], s.y[s.train].astype(np.float64))


def fold(now, be, horizon: int, val_from, test_from, test_to, bucket: int) -> dict:
    """한 묶음: 서비스 조건 평가 행의 세 예측과, 같은 행의 지금 데이터 기준 정답(정답이 얼마나 줄었는지 보려고)."""
    s = split(be, horizon, val_from, test_from, test_to, bucket)
    card, t, y = s.card[s.test], s.t[s.test], s.y[s.test]
    X = make_features(be, card, t, horizon, "core")
    tsb = np.floor(intermittent.tsb_window(be.spend, card, t, window=TSB_WINDOW, **BE_PARAMS) * horizon + 0.5)
    return {"card": card, "y": y, "y_now": baselines.actual(now, card, t, horizon),
            "pred": {"AI_NOW": fit(now, horizon, val_from, test_from, test_to, bucket).predict(X),
                     "AI_BE": fit(be, horizon, val_from, test_from, test_to, bucket).predict(X),
                     "TSB_FIXED": tsb}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO / "data" / "raw" / "aihub-71925")
    parser.add_argument("--runs-dir", type=Path, default=REPO / "runs")
    parser.add_argument("--kinds", nargs="+", default=list(KINDS), choices=list(KINDS))
    parser.add_argument("--horizons", type=int, nargs="+", default=[7, 14, 30])
    args = parser.parse_args()
    zips = [args.data_dir / name for name in CARD_ZIPS]
    run_dir = new_run(args.runs_dir, "be-large-eval", {
        "script": "scripts/be_large_eval.py", "inputs": zips, "variant": VARIANT, "large_amount": LARGE_AMOUNT,
        "be_params": BE_PARAMS, "years": YEARS, "folds": crossval.FOLDS, "kinds": args.kinds, "horizons": args.horizons,
        "judge": "AI_NOW 두 해·세 기간 |편향| <= 10% 그리고 TSB_FIXED 대비 95% 상한 < 0이면 재학습 불필요",
        "preregistration": "reports/7.3.10-credit-estimate.md"}, REPO)
    df = load_ordered(zips)
    # 걸린 결제는 이상거래처럼 빼되 행은 남겨 확인된 달·반복 판정은 그대로 둔다(BE도 반복은 분류와 상관없이 셈)
    df_be = with_median_large_excluded(df, CHECK_CODES | CREDIT_CODES | {INSTALLMENT})
    exclude = VARIANTS[VARIANT]
    results = {}
    for kind in args.kinds:
        now = build_panel(df, exclude, large_amount=LARGE_AMOUNT, codes=KINDS[kind])
        be = build_panel(df_be, exclude, large_amount=LARGE_AMOUNT, codes=KINDS[kind])
        if not (np.array_equal(now.cards, be.cards) and now.days.equals(be.days)
                and np.array_equal(now.observed, be.observed)):
            raise ValueError("두 패널의 카드·날짜·확인된 달이 다릅니다")
        print(kind, "removed now", round(now.removed_share, 4), "be", round(be.removed_share, 4), flush=True)
        for horizon in args.horizons:
            for year, (val_from, test_from, test_to) in YEARS.items():
                blocks = [fold(now, be, horizon, pd.Timestamp(val_from), pd.Timestamp(test_from),
                               pd.Timestamp(test_to) if test_to else None, b) for b in range(crossval.FOLDS)]
                pooled = {"card": np.concatenate([b["card"] for b in blocks]),
                          "y": np.concatenate([b["y"] for b in blocks]),
                          "pred": {n: np.concatenate([b["pred"][n] for b in blocks]).astype(np.float64) for n in NAMES}}
                y_now = np.concatenate([b["y_now"] for b in blocks])
                boot = crossval.bootstrap(pooled, list(NAMES), references=("TSB_FIXED",))
                out = {"rows": int(len(pooled["y"])), "people": boot["people"],
                       "target_ratio_vs_now": round(float(pooled["y"].mean() / y_now.mean()), 4), "methods": {}}
                for n in NAMES:
                    m = metrics(pooled["pred"][n], pooled["y"])
                    out["methods"][n] = {"metrics": m, "bias_ci": boot["bias"][n],
                                         "vs_tsb_fixed": boot["mae_vs"].get(f"{n}|TSB_FIXED"),
                                         "bias_ok": bool(abs(m["rel_bias"]) <= BIAS_LIMIT)}
                key = f"{kind} · H{horizon} · {year}"
                results[key] = out
                print(key, "rows", out["rows"], "target", out["target_ratio_vs_now"],
                      {n: (round(out["methods"][n]["metrics"]["MAE"]), out["methods"][n]["metrics"]["rel_bias"],
                           out["methods"][n]["vs_tsb_fixed"]) for n in NAMES}, flush=True)
    write_json(run_dir / "metrics.json", results)
    print(run_dir)


if __name__ == "__main__":
    main()
