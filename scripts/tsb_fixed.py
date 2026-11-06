"""BE TsbEstimator 고정값(α 0.3·β 0.2) TSB와 검증 구간에서 고른 TSB_90D, TWO_STAGE_TREE 비교(S15P31E103-304, 2026-11-06).

교차검증(scripts/crossval.py)과 같은 사람 5묶음 × 평가 연도 2023·2024 × H, 같은 평가 행이다.
- TSB_FIXED: 최근 90일만 보는 TSB(tsb_window), α 0.3·β 0.2 고정(BE와 같은 값)
- TSB_90D: 같은 계산, α·β는 묶음마다 검증 구간에서 고름(교차검증 판정 기준선)
- TWO_STAGE_TREE: 교차검증과 같은 학습(설정값 없음)
판정은 사전 등록과 같다: 두 해 모두 편향 ±10% 안이고 기준 대비 MAE 비율의 사람 부트스트랩 95% 상한 < 0.

사용: .venv\\Scripts\\python.exe -I scripts\\tsb_fixed.py [--variant fraud+large+recurring_once] [--horizons 7 14 30]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_ml import crossval, intermittent, models
from forecast_ml.backtest import VARIANTS, metrics
from forecast_ml.data import build_panel, load_transactions
from forecast_ml.experiment import BIAS_LIMIT, TSB_GRID, TSB_WINDOW, choose, split
from forecast_ml.features import make_features
from forecast_ml.runs import new_run, write_json

REPO = Path(__file__).resolve().parents[1]
CARD_ZIPS = ("TL_카드거래.zip", "VL_카드거래.zip")
YEARS = {"2023": ("2022-07-01", "2023-01-01", "2024-01-01"), "2024": ("2023-07-01", "2024-01-01", None)}
LARGE_AMOUNT = 1_000_000
BE_PARAMS = {"alpha": 0.3, "beta": 0.2}
NAMES = ("TSB_FIXED", "TSB_90D", "TWO_STAGE_TREE")


def tsb_total(panel, s, horizon: int, params: dict) -> np.ndarray:
    """최근 90일 TSB의 H일 합계(원 단위 반올림)."""
    daily = intermittent.tsb_window(panel.spend, s.card, s.t, window=TSB_WINDOW, **params)
    return np.floor(daily * horizon + 0.5)


def fold(panel, horizon: int, val_from, test_from, test_to, bucket: int) -> dict:
    """한 묶음의 평가 행 예측 세 가지와 고른 TSB 설정값."""
    s = split(panel, horizon, val_from, test_from, test_to, bucket)
    scores = [(p, metrics(tsb_total(panel, s, horizon, p)[s.val], s.y[s.val])) for p in TSB_GRID]
    chosen, _ = choose(scores)
    X = make_features(panel, s.card, s.t, horizon, "core")
    model = models.build("TWO_STAGE_TREE", {}).fit(X[s.train], s.y[s.train].astype(np.float64))
    return {"card": s.card[s.test], "y": s.y[s.test], "params": chosen,
            "pred": {"TSB_FIXED": tsb_total(panel, s, horizon, BE_PARAMS)[s.test],
                     "TSB_90D": tsb_total(panel, s, horizon, chosen)[s.test],
                     "TWO_STAGE_TREE": model.predict(X[s.test])}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO / "data" / "raw" / "aihub-71925")
    parser.add_argument("--runs-dir", type=Path, default=REPO / "runs")
    parser.add_argument("--horizons", type=int, nargs="+", default=[7, 14, 30])
    parser.add_argument("--variant", default="fraud+large+recurring_once", choices=[k for k in VARIANTS if "large" in k])
    args = parser.parse_args()
    zips = [args.data_dir / name for name in CARD_ZIPS]
    config = {"script": "scripts/tsb_fixed.py", "inputs": zips, "variant": args.variant, "large_amount": LARGE_AMOUNT,
              "be_params": BE_PARAMS, "tsb_grid": TSB_GRID, "years": YEARS, "folds": crossval.FOLDS,
              "horizons": args.horizons,
              "judge": "두 해 모두 |편향| <= 10% 그리고 기준 대비 MAE 비율 95% 상한 < 0"}
    run_dir = new_run(args.runs_dir, "tsb-fixed", config, REPO)
    panel = build_panel(load_transactions(zips), VARIANTS[args.variant], large_amount=LARGE_AMOUNT)
    results = {}
    for horizon in args.horizons:
        for year, (val_from, test_from, test_to) in YEARS.items():
            blocks = [fold(panel, horizon, pd.Timestamp(val_from), pd.Timestamp(test_from),
                           pd.Timestamp(test_to) if test_to else None, b) for b in range(crossval.FOLDS)]
            pooled = {"card": np.concatenate([b["card"] for b in blocks]),
                      "y": np.concatenate([b["y"] for b in blocks]),
                      "pred": {n: np.concatenate([b["pred"][n] for b in blocks]).astype(np.float64) for n in NAMES}}
            boot = crossval.bootstrap(pooled, list(NAMES), references=("TSB_FIXED", "TSB_90D"))
            out = {"rows": int(len(pooled["y"])), "people": boot["people"],
                   "tsb_params_by_fold": [b["params"] for b in blocks], "methods": {}}
            for n in NAMES:
                m = metrics(pooled["pred"][n], pooled["y"])
                out["methods"][n] = {"metrics": m, "bias_ci": boot["bias"][n],
                                     "vs": {r: boot["mae_vs"][f"{n}|{r}"] for r in ("TSB_FIXED", "TSB_90D") if r != n},
                                     "bias_ok": bool(abs(m["rel_bias"]) <= BIAS_LIMIT)}
            results[f"H{horizon} · {year}"] = out
            tree = out["methods"]["TWO_STAGE_TREE"]
            print(f"H{horizon} {year}", "fixed MAE", round(out["methods"]["TSB_FIXED"]["metrics"]["MAE"]),
                  "tsb90 MAE", round(out["methods"]["TSB_90D"]["metrics"]["MAE"]),
                  "tree vs fixed", tree["vs"]["TSB_FIXED"], flush=True)
    write_json(run_dir / "metrics.json", results)
    print(run_dir)


if __name__ == "__main__":
    main()
