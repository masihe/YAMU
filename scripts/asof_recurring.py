"""평가 입력을 서비스 시점 반복 판정으로 고쳐 다시 판정(Codex 검토 반영, S15P31E103-304, 2026-11-06).

학습 전처리(recurring_once)는 달 전체 건수로 반복을 정해, 기준일 뒤 같은 달 결제가 있으면 그 달 첫 결제를 생활비에 남긴다.
서비스는 기준일까지만 세므로 그 결제를 뺀다(recurring.provisional_once). 학습은 그대로 두고, 평가 행의 90일 입력에서만
그 결제를 빼서 AI(TWO_STAGE_TREE)·고정값 TSB(α 0.3·β 0.2)·검증으로 고른 TSB_90D를 다시 계산한다. 정답은 그대로다.
판정은 사전 등록과 같다: 두 해 모두 |AI 편향| <= 10% 그리고 기준 대비 MAE 비율 95% 상한 < 0.

사용: .venv\\Scripts\\python.exe -I scripts\\asof_recurring.py [--horizons 7 14 30]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_ml import crossval, intermittent, models, recurring, short_history
from forecast_ml.backtest import VARIANTS, metrics
from forecast_ml.data import CHECK_CODES, build_panel, load_transactions
from forecast_ml.experiment import TSB_GRID, TSB_WINDOW, choose, split
from forecast_ml.features import make_features
from forecast_ml.runs import new_run, write_json

REPO = Path(__file__).resolve().parents[1]
CARD_ZIPS = ("TL_카드거래.zip", "VL_카드거래.zip")
YEARS = {"2023": ("2022-07-01", "2023-01-01", "2024-01-01"), "2024": ("2023-07-01", "2024-01-01", None)}
VARIANT = "fraud+large+recurring_once"
LARGE_AMOUNT = 1_000_000
BE_PARAMS = {"alpha": 0.3, "beta": 0.2}
NAMES = ("AI", "TSB_FIXED", "TSB_90D")


def provisional_rows(df: pd.DataFrame) -> pd.DataFrame:
    """학습 데이터에 남아 있지만 서비스는 [date, until) 기준일에 빼는 결제. 반복 판정은 모든 체크 거래로 센다(data.check_rows와 같음)."""
    check = df[df["code"].isin(CHECK_CODES)]
    got = recurring.provisional_once(check[["date", "card", "store", "amount"]])
    kept = check.loc[got.index]
    return got[(kept["fraud"] != "1") & (kept["amount"] <= LARGE_AMOUNT)]


def adjust(daily: np.ndarray, card_id: np.ndarray, day: pd.DatetimeIndex, provisional: pd.DataFrame,
           cards: pd.Index) -> tuple[np.ndarray, np.ndarray]:
    """평가 행의 90일 입력(마지막 칸 = 기준일)에서 서비스라면 뺐을 결제를 뺀다. 고친 입력과 고친 행 표시."""
    out = daily.copy()
    touched = np.zeros(len(out), dtype=bool)
    for card, date, until, amount in zip(cards.get_indexer(provisional["card"]), provisional["date"],
                                         provisional["until"], provisional["amount"]):
        rows = np.where((card_id == card) & (day >= date) & (day < until))[0]
        if len(rows) == 0:
            continue
        back = (day[rows] - date).days.to_numpy()
        out[rows, short_history.HISTORY_DAYS - 1 - back] -= amount
        touched[rows] = True
    if (out < 0).any():
        raise ValueError("빼고 나서 음수가 된 날이 있습니다")
    return out, touched


def tsb_from_daily(daily: np.ndarray, horizon: int, params: dict) -> np.ndarray:
    """90일 입력만으로 계산한 TSB H일 합계(원 단위 반올림)."""
    n = len(daily)
    d = intermittent.tsb_window(daily, np.arange(n), np.full(n, daily.shape[1] - 1), window=TSB_WINDOW, **params)
    return np.floor(d * horizon + 0.5)


def fold(panel, provisional, horizon, val_from, test_from, test_to, bucket) -> dict:
    s = split(panel, horizon, val_from, test_from, test_to, bucket)
    scores = []
    for params in TSB_GRID:
        d = intermittent.tsb_window(panel.spend, s.card, s.t, window=TSB_WINDOW, **params)
        scores.append((params, metrics(np.floor(d * horizon + 0.5)[s.val], s.y[s.val])))
    chosen, _ = choose(scores)
    X = make_features(panel, s.card, s.t, horizon, "core")
    model = models.build("TWO_STAGE_TREE", {}).fit(X[s.train], s.y[s.train].astype(np.float64))
    card, t, y = s.card[s.test], s.t[s.test], s.y[s.test]
    daily = short_history.truncated_history(panel, card, t, short_history.HISTORY_DAYS)
    fixed, touched = adjust(daily, card, panel.days[t], provisional, pd.Index(panel.cards))
    out = {"card": card, "y": y, "touched": touched}
    for label, inp in (("before", daily), ("after", fixed)):
        out[label] = {"AI": model.predict(short_history.serving_features(inp, horizon)),
                      "TSB_FIXED": tsb_from_daily(inp, horizon, BE_PARAMS),
                      "TSB_90D": tsb_from_daily(inp, horizon, chosen)}
    return out


def judge(card: np.ndarray, y: np.ndarray, pred: dict) -> dict:
    pooled = {"card": card, "y": y, "pred": {n: np.asarray(pred[n], dtype=np.float64) for n in NAMES}}
    boot = crossval.bootstrap(pooled, list(NAMES), references=("TSB_FIXED", "TSB_90D"))
    ai = metrics(pooled["pred"]["AI"], y)
    out = {"MAE": {n: metrics(pooled["pred"][n], y)["MAE"] for n in NAMES}, "AI_bias": ai["rel_bias"],
           "AI_bias_ci": boot["bias"]["AI"]}
    for ref in ("TSB_FIXED", "TSB_90D"):
        ci = boot["mae_vs"][f"AI|{ref}"]
        out[f"vs_{ref}"] = ci
        out[f"pass_{ref}"] = bool(abs(ai["rel_bias"]) <= 0.10 and ci["high"] < 0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO / "data" / "raw" / "aihub-71925")
    parser.add_argument("--runs-dir", type=Path, default=REPO / "runs")
    parser.add_argument("--horizons", type=int, nargs="+", default=[7, 14, 30])
    args = parser.parse_args()
    zips = [args.data_dir / name for name in CARD_ZIPS]
    config = {"script": "scripts/asof_recurring.py", "inputs": zips, "variant": VARIANT, "large_amount": LARGE_AMOUNT,
              "be_params": BE_PARAMS, "years": YEARS, "folds": crossval.FOLDS, "horizons": args.horizons}
    run_dir = new_run(args.runs_dir, "asof-recurring", config, REPO)
    df = load_transactions(zips)
    provisional = provisional_rows(df)
    panel = build_panel(df, VARIANTS[VARIANT], large_amount=LARGE_AMOUNT)
    results = {"provisional": {"rows": int(len(provisional)), "amount": int(provisional["amount"].sum())}}
    print("provisional", results["provisional"], flush=True)
    for horizon in args.horizons:
        for year, (val_from, test_from, test_to) in YEARS.items():
            blocks = [fold(panel, provisional, horizon, pd.Timestamp(val_from), pd.Timestamp(test_from),
                           pd.Timestamp(test_to) if test_to else None, b) for b in range(crossval.FOLDS)]
            card = np.concatenate([b["card"] for b in blocks])
            y = np.concatenate([b["y"] for b in blocks])
            touched = np.concatenate([b["touched"] for b in blocks])
            entry = {"rows": int(len(y)), "touched_rows": int(touched.sum())}
            for label in ("before", "after"):
                pred = {n: np.concatenate([b[label][n] for b in blocks]) for n in NAMES}
                entry[label] = judge(card, y, pred)
            key = f"H{horizon} · {year}"
            results[key] = entry
            print(key, entry["touched_rows"], {lab: (round(entry[lab]["vs_TSB_FIXED"]["high"], 4),
                                                     round(entry[lab]["vs_TSB_90D"]["high"], 4),
                                                     round(entry[lab]["AI_bias"], 4)) for lab in ("before", "after")},
                  flush=True)
    write_json(run_dir / "metrics.json", results)
    print(run_dir)


if __name__ == "__main__":
    main()
