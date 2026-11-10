"""신용카드 마감 전 사용 추정 비교(S15P31E103-404, 2026-11-10). 사전 등록은 reports/7.3.10-credit-estimate.md.

맞힐 값: 기준일 다음 H일의 신용(00) 생활 소비 합계. 평가 행은 신용 패널의 split(...).test(사람 5묶음 × 2023·2024).
- AVG_3M: 지난 완료 3개월 신용 일평균 × H(지금 BE 방식)
- TSB_FIXED: 최근 90일 신용 TSB, α 0.3·β 0.2(BE TsbEstimator)
- CHECK_MODEL: 체크 생활비로 학습한 TWO_STAGE_TREE를 신용 특징에 그대로 적용
- CREDIT_AI: 신용 생활 소비로 학습한 TWO_STAGE_TREE
- TOTAL_AI: 체크+신용 합으로 학습한 TWO_STAGE_TREE × 최근 84일 신용 비율(둘 다 0원이면 1)
판정: 두 해 모두 |편향| <= 10% 그리고 AVG_3M·TSB_FIXED 대비 MAE 비율 95% 상한이 둘 다 < 0.

사용: .venv\\Scripts\\python.exe -I scripts\\credit_estimate.py [--horizons 7 14 30]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_ml import baselines, crossval, intermittent, models
from forecast_ml.backtest import VARIANTS, metrics
from forecast_ml.data import CREDIT_CODES, Panel, build_panel, load_transactions
from forecast_ml.experiment import BIAS_LIMIT, TSB_WINDOW, split
from forecast_ml.features import make_features
from forecast_ml.runs import new_run, write_json

REPO = Path(__file__).resolve().parents[1]
CARD_ZIPS = ("TL_카드거래.zip", "VL_카드거래.zip")
YEARS = {"2023": ("2022-07-01", "2023-01-01", "2024-01-01"), "2024": ("2023-07-01", "2024-01-01", None)}
VARIANT = "fraud+large+recurring_once"
LARGE_AMOUNT = 1_000_000
BE_PARAMS = {"alpha": 0.3, "beta": 0.2}
SHARE_WINDOW = 84
REFERENCES = ("AVG_3M", "TSB_FIXED")
CANDIDATES = ("CHECK_MODEL", "CREDIT_AI", "TOTAL_AI")
NAMES = REFERENCES + CANDIDATES


def combine(check: Panel, credit: Panel) -> Panel:
    """두 패널을 카드KEY 합집합으로 맞춰 일별 소비를 더한다. 날짜·달은 같은 원자료에서 만들어 같아야 한다."""
    if not (check.days.equals(credit.days) and check.months.equals(credit.months)):
        raise ValueError("두 패널의 날짜가 다릅니다")
    cards = np.union1d(check.cards, credit.cards)
    spend = np.zeros((len(cards), len(check.days)), dtype=np.int64)
    observed = np.zeros((len(cards), len(check.months)), dtype=bool)
    for p in (check, credit):
        rows = np.searchsorted(cards, p.cards)
        spend[rows] += p.spend
        observed[rows] |= p.observed
    return Panel(cards, check.days, check.months, spend, observed, removed_share=float("nan"))


def rows_in(panel: Panel, cards: np.ndarray) -> np.ndarray:
    """카드KEY 문자열 → panel 행 번호. 없으면 -1."""
    idx = np.searchsorted(panel.cards, cards)
    idx = np.clip(idx, 0, len(panel.cards) - 1)
    return np.where(panel.cards[idx] == cards, idx, -1)


def window_sum(panel: Panel, card: np.ndarray, t: np.ndarray, window: int) -> np.ndarray:
    """기준일 t 포함 최근 window일 합계. card가 -1이면 0."""
    cum = baselines.cumulative(panel)
    safe = np.maximum(card, 0)
    total = cum[safe, t + 1] - cum[safe, t + 1 - window]
    return np.where(card >= 0, total, 0)


def fit_tree(panel: Panel, horizon: int, val_from, test_from, test_to, bucket: int):
    """panel의 학습 행(평가 묶음이 아닌 사람 × 평가 시작 − H 이전)으로 TWO_STAGE_TREE를 학습한다."""
    s = split(panel, horizon, val_from, test_from, test_to, bucket)
    X = make_features(panel, s.card, s.t, horizon, "core")
    model = models.build("TWO_STAGE_TREE", {}).fit(X[s.train], s.y[s.train].astype(np.float64))
    return model, int(s.train.sum())


def fold(panels: dict, horizon: int, val_from, test_from, test_to, bucket: int) -> dict:
    """한 묶음의 평가 행과 다섯 방식 예측."""
    credit, check, total = panels["credit"], panels["check"], panels["total"]
    s = split(credit, horizon, val_from, test_from, test_to, bucket)
    card, t, y = s.card[s.test], s.t[s.test], s.y[s.test]
    keys = credit.cards[card]

    avg = baselines.avg_3m(credit, card, t, horizon)
    if (avg < 0).any():
        raise ValueError("확인된 달이 없는 평가 행이 있습니다")
    tsb = np.floor(intermittent.tsb_window(credit.spend, card, t, window=TSB_WINDOW, **BE_PARAMS) * horizon + 0.5)

    X_credit = make_features(credit, card, t, horizon, "core")
    check_model, n_check = fit_tree(check, horizon, val_from, test_from, test_to, bucket)
    credit_model, n_credit = fit_tree(credit, horizon, val_from, test_from, test_to, bucket)
    total_model, n_total = fit_tree(total, horizon, val_from, test_from, test_to, bucket)

    total_card = rows_in(total, keys)
    if (total_card < 0).any():
        raise ValueError("합친 패널에 없는 평가 카드가 있습니다")
    X_total = make_features(total, total_card, t, horizon, "core")
    credit_84 = window_sum(credit, card, t, SHARE_WINDOW)
    check_84 = window_sum(check, rows_in(check, keys), t, SHARE_WINDOW)
    both = credit_84 + check_84
    share = np.divide(credit_84, both, out=np.ones(len(t)), where=both > 0)

    return {"card": card, "y": y, "share": share,
            "train_rows": {"CHECK_MODEL": n_check, "CREDIT_AI": n_credit, "TOTAL_AI": n_total},
            "pred": {"AVG_3M": avg.astype(np.float64), "TSB_FIXED": tsb,
                     "CHECK_MODEL": check_model.predict(X_credit),
                     "CREDIT_AI": credit_model.predict(X_credit),
                     "TOTAL_AI": total_model.predict(X_total) * share}}


def passes(out: dict, name: str) -> bool:
    """한 해 판정: |편향| <= 10% 그리고 두 기준 대비 MAE 비율 95% 상한 < 0."""
    m = out["methods"][name]
    return m["bias_ok"] and all(m["vs"][r]["high"] < 0 for r in REFERENCES)


def to_markdown(results: dict) -> str:
    """조건마다 방식별 MAE·편향·두 기준 대비 비율 표."""
    lines = []
    for key, out in results.items():
        lines.append(f"\n### {key} (평가 {out['rows']:,}건, {out['people']}명, 실제 평균 "
                     f"{out['methods']['AVG_3M']['metrics']['mean_actual']:,.0f}원, 신용 비율 중앙값 {out['share_median']:.2f})\n")
        lines.append("| 방식 | MAE | 편향(%) | vs AVG_3M(%) [95%] | vs TSB_FIXED(%) [95%] | 한 해 통과 |")
        lines.append("|---|---|---|---|---|---|")
        for name in NAMES:
            m = out["methods"][name]

            def ci(ref):
                c = m["vs"].get(ref)
                return "-" if c is None else f"{c['mean'] * 100:+.1f} [{c['low'] * 100:+.1f}, {c['high'] * 100:+.1f}]"

            mark = "-" if name in REFERENCES else ("O" if passes(out, name) else "x")
            lines.append(f"| {name} | {m['metrics']['MAE']:,.0f} | {m['metrics']['rel_bias'] * 100:+.1f} | "
                         f"{ci('AVG_3M')} | {ci('TSB_FIXED')} | {mark} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO / "data" / "raw" / "aihub-71925")
    parser.add_argument("--runs-dir", type=Path, default=REPO / "runs")
    parser.add_argument("--horizons", type=int, nargs="+", default=[7, 14, 30])
    args = parser.parse_args()
    zips = [args.data_dir / name for name in CARD_ZIPS]
    config = {"script": "scripts/credit_estimate.py", "inputs": zips, "variant": VARIANT, "large_amount": LARGE_AMOUNT,
              "be_params": BE_PARAMS, "share_window": SHARE_WINDOW, "years": YEARS, "folds": crossval.FOLDS,
              "horizons": args.horizons, "references": REFERENCES, "candidates": CANDIDATES,
              "judge": "두 해 모두 |편향| <= 10% 그리고 AVG_3M·TSB_FIXED 대비 MAE 비율 95% 상한 둘 다 < 0",
              "preregistration": "reports/7.3.10-credit-estimate.md"}
    run_dir = new_run(args.runs_dir, "credit-estimate", config, REPO)
    df = load_transactions(zips)
    exclude = VARIANTS[VARIANT]
    check = build_panel(df, exclude, large_amount=LARGE_AMOUNT)
    credit = build_panel(df, exclude, large_amount=LARGE_AMOUNT, codes=CREDIT_CODES)
    panels = {"check": check, "credit": credit, "total": combine(check, credit)}
    print("cards check", len(check.cards), "credit", len(credit.cards), "total", len(panels["total"].cards),
          "removed check", round(check.removed_share, 4), "credit", round(credit.removed_share, 4), flush=True)
    results = {}
    for horizon in args.horizons:
        for year, (val_from, test_from, test_to) in YEARS.items():
            blocks = [fold(panels, horizon, pd.Timestamp(val_from), pd.Timestamp(test_from),
                           pd.Timestamp(test_to) if test_to else None, b) for b in range(crossval.FOLDS)]
            pooled = {"card": np.concatenate([b["card"] for b in blocks]),
                      "y": np.concatenate([b["y"] for b in blocks]),
                      "pred": {n: np.concatenate([b["pred"][n] for b in blocks]) for n in NAMES}}
            share = np.concatenate([b["share"] for b in blocks])
            boot = crossval.bootstrap(pooled, list(NAMES), references=REFERENCES)
            out = {"rows": int(len(pooled["y"])), "people": boot["people"], "share_median": float(np.median(share)),
                   "train_rows_by_fold": [b["train_rows"] for b in blocks], "methods": {}}
            for n in NAMES:
                m = metrics(pooled["pred"][n], pooled["y"])
                out["methods"][n] = {"metrics": m, "bias_ci": boot["bias"][n],
                                     "vs": {r: boot["mae_vs"][f"{n}|{r}"] for r in REFERENCES if r != n},
                                     "bias_ok": bool(abs(m["rel_bias"]) <= BIAS_LIMIT)}
            key = f"H{horizon} · {year}"
            results[key] = out
            print(key, "rows", out["rows"], "people", out["people"],
                  {n: round(out["methods"][n]["metrics"]["MAE"]) for n in NAMES}, flush=True)
    write_json(run_dir / "metrics.json", results)
    (run_dir / "metrics.md").write_text(to_markdown(results), encoding="utf-8")
    print(run_dir)


if __name__ == "__main__":
    main()
