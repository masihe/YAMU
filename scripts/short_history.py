"""짧은 기록 평가 실행(냉시작, 2026-11-05): 14·30일 × 평가 연도 2023·2024 × 사람 5묶음 × 기록 길이 14·30·60·90일.

판정(실행 전에 정함): 같은 길이로 자른 TSB보다 AI MAE가 확실히 낮고(사람 부트스트랩 95% 상한 < 0) 편향이 ±10% 안이면 통과.
두 해 모두 통과한 가장 짧은 길이부터 AI를 쓴다. 판정은 기록 안에 체크 사용이 있는 행만(BE가 AI를 부르는 경우) 본다.
90일은 기존 교차검증(runs/20261103-110529-crossval)의 TWO_STAGE_TREE와 같은 값이 나와야 한다.

사용: .venv\\Scripts\\python.exe -I scripts\\short_history.py [--horizons 14 30] [--lengths 14 30 60 90]
      [--variant fraud+large+recurring_once]  # 반복 지출 제외 규칙(기본은 11/5 실행과 같은 전체 기간 규칙)
      [--tsb-fixed 0.3 0.2]  # BE 고정값 TSB와의 비교·판정도 냄(2026-11-06, 7일)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_ml import crossval, short_history
from forecast_ml.backtest import VARIANTS, metrics
from forecast_ml.data import build_panel, load_transactions
from forecast_ml.experiment import BIAS_LIMIT
from forecast_ml.runs import new_run, write_json

REPO = Path(__file__).resolve().parents[1]
CARD_ZIPS = ("TL_카드거래.zip", "VL_카드거래.zip")
YEARS = {"2023": ("2022-07-01", "2023-01-01", "2024-01-01"), "2024": ("2023-07-01", "2024-01-01", None)}
VARIANT = "fraud+large+recurring"
LARGE_AMOUNT = 1_000_000


def summarize(blocks: list[dict], length: int) -> dict:
    """묶음을 합쳐 한 길이의 지표·판정을 만든다."""
    card = np.concatenate([b["card"] for b in blocks])
    y = np.concatenate([b["y"] for b in blocks])
    part = [b["by_length"][length] for b in blocks]
    ai, tsb, avg = (np.concatenate([p[k] for p in part]) for k in ("AI", "TSB", "AVG_3M"))
    active = np.concatenate([p["active"] for p in part])
    c, yy = card[active], y[active]
    boot = short_history.bootstrap(c, yy, ai[active], tsb[active].astype(np.float64))
    ai_m, tsb_m = metrics(ai[active], yy), metrics(tsb[active].astype(np.float64), yy)
    has_avg = active & (avg >= 0)
    result = {
        "rows": int(len(y)), "active_rows": int(active.sum()), "active_share": round(float(active.mean()), 4),
        "people": boot["people"], "AI": ai_m, "TSB": tsb_m, "bootstrap": boot,
        "pass": bool(abs(ai_m["rel_bias"]) <= BIAS_LIMIT and boot["mae_vs_tsb"]["high"] < 0),
        "AVG_3M_available_share": round(float(has_avg.sum() / max(active.sum(), 1)), 4),
        "tsb_params_by_fold": [p["tsb_params"] for p in part],
    }
    if has_avg.any():
        result["AVG_3M_where_available"] = metrics(avg[has_avg].astype(np.float64), y[has_avg])
        result["TSB_where_AVG_3M_available"] = metrics(tsb[has_avg].astype(np.float64), y[has_avg])
    if "TSB_FIXED" in part[0]:
        fixed = np.concatenate([p["TSB_FIXED"] for p in part])[active].astype(np.float64)
        fixed_boot = short_history.bootstrap(c, yy, ai[active], fixed)
        result["TSB_FIXED"] = metrics(fixed, yy)
        result["bootstrap_vs_fixed"] = fixed_boot
        result["pass_vs_fixed"] = bool(abs(ai_m["rel_bias"]) <= BIAS_LIMIT and fixed_boot["mae_vs_tsb"]["high"] < 0)
    bucket =short_history.activity_bucket(np.concatenate([p["spend_days"] for p in part]))
    result["by_activity"] = {}
    for i, (low, high) in enumerate(short_history.ACTIVITY_BUCKETS):
        rows = bucket == i
        if rows.sum() < 200 or y[rows].sum() == 0:
            continue
        b_boot = short_history.bootstrap(card[rows], y[rows], ai[rows], tsb[rows].astype(np.float64))
        b_ai = metrics(ai[rows], y[rows])
        result["by_activity"][f"{low}-{high}"] = {
            "rows": int(rows.sum()), "people": b_boot["people"], "AI": b_ai,
            "TSB": metrics(tsb[rows].astype(np.float64), y[rows]), "bootstrap": b_boot,
            "pass": bool(abs(b_ai["rel_bias"]) <= BIAS_LIMIT and b_boot["mae_vs_tsb"]["high"] < 0)}
    return result


def activity_markdown(results: dict) -> str:
    """결제한 날 수 구간별 AI vs TSB."""
    lines = ["| 조건 | 기록 | 결제한 날 | 행 | 실제 평균 | AI MAE | TSB MAE | AI vs TSB MAE(%) [95%] | AI 편향(%) | TSB 편향(%) | 통과 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, by_len in results.items():
        for length, r in by_len.items():
            for name, b in r["by_activity"].items():
                ci = b["bootstrap"]["mae_vs_tsb"]
                lines.append(f"| {key} | {length}일 | {name}일 | {b['rows']:,} | {b['AI']['mean_actual']:,.0f} | "
                             f"{b['AI']['MAE']:,.0f} | {b['TSB']['MAE']:,.0f} | {ci['mean'] * 100:+.1f} "
                             f"[{ci['low'] * 100:+.1f}, {ci['high'] * 100:+.1f}] | {b['AI']['rel_bias'] * 100:+.1f} | "
                             f"{b['TSB']['rel_bias'] * 100:+.1f} | {'O' if b['pass'] else 'x'} |")
    return "\n".join(lines) + "\n"


def to_markdown(results: dict) -> str:
    lines = ["| 조건 | 기록 | 판정 행(사용 있음) | AI MAE | TSB MAE | AI vs TSB MAE(%) [95%] | AI 편향(%) | 통과 | "
             "AVG_3M 계산 가능 | AVG_3M MAE / 같은 행 TSB MAE |", "|---|---|---|---|---|---|---|---|---|---|"]
    for key, by_len in results.items():
        for length, r in by_len.items():
            ci = r["bootstrap"]["mae_vs_tsb"]
            avg = r.get("AVG_3M_where_available")
            avg_text = "-" if avg is None else f"{avg['MAE']:,.0f} / {r['TSB_where_AVG_3M_available']['MAE']:,.0f}"
            lines.append(f"| {key} | {length}일 | {r['active_rows']:,} ({r['active_share']:.0%}) | {r['AI']['MAE']:,.0f} | "
                         f"{r['TSB']['MAE']:,.0f} | {ci['mean'] * 100:+.1f} [{ci['low'] * 100:+.1f}, {ci['high'] * 100:+.1f}] | "
                         f"{r['AI']['rel_bias'] * 100:+.1f} | {'O' if r['pass'] else 'x'} | "
                         f"{r['AVG_3M_available_share']:.0%} | {avg_text} |")
    fixed_rows = [(key, length, r) for key, by_len in results.items() for length, r in by_len.items() if "TSB_FIXED" in r]
    if fixed_rows:
        lines += ["", "### 고정값 TSB 기준", "",
                  "| 조건 | 기록 | 고정값 TSB MAE | 고정값 TSB 편향(%) | AI vs 고정값 TSB MAE(%) [95%] | 통과 |", "|---|---|---|---|---|---|"]
        for key, length, r in fixed_rows:
            ci = r["bootstrap_vs_fixed"]["mae_vs_tsb"]
            lines.append(f"| {key} | {length}일 | {r['TSB_FIXED']['MAE']:,.0f} | {r['TSB_FIXED']['rel_bias'] * 100:+.1f} | "
                         f"{ci['mean'] * 100:+.1f} [{ci['low'] * 100:+.1f}, {ci['high'] * 100:+.1f}] | "
                         f"{'O' if r['pass_vs_fixed'] else 'x'} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO / "data" / "raw" / "aihub-71925")
    parser.add_argument("--runs-dir", type=Path, default=REPO / "runs")
    parser.add_argument("--horizons", type=int, nargs="+", default=[14, 30])
    parser.add_argument("--lengths", type=int, nargs="+", default=list(short_history.LENGTHS))
    parser.add_argument("--variant", default=VARIANT, choices=[k for k in VARIANTS if "large" in k])
    parser.add_argument("--tsb-fixed", type=float, nargs=2, metavar=("ALPHA", "BETA"),
                        help="이 고정값 TSB(BE TsbEstimator와 같은 계산)와의 비교·판정도 낸다")
    args = parser.parse_args()
    fixed_tsb = None if args.tsb_fixed is None else {"alpha": args.tsb_fixed[0], "beta": args.tsb_fixed[1]}
    zips = [args.data_dir / name for name in CARD_ZIPS]
    config = {"script": "scripts/short_history.py", "inputs": zips, "variant": args.variant, "large_amount": LARGE_AMOUNT,
              "model": short_history.MODEL, "years": YEARS, "folds": crossval.FOLDS, "horizons": args.horizons,
              "lengths": args.lengths, "tsb_fixed": fixed_tsb,
              "judge": "기록에 사용 있는 행만. |AI 편향| <= 10% 그리고 AI÷TSB MAE − 1의 95% 상한 < 0. 두 해 모두 통과한 가장 짧은 길이"}
    run_dir = new_run(args.runs_dir, "short-history", config, REPO)
    panel = build_panel(load_transactions(zips), VARIANTS[args.variant], large_amount=LARGE_AMOUNT)
    results = {}
    saved = {}
    for horizon in args.horizons:
        for year, (val_from, test_from, test_to) in YEARS.items():
            blocks = [short_history.run(panel, horizon, pd.Timestamp(val_from), pd.Timestamp(test_from),
                                        pd.Timestamp(test_to) if test_to else None, bucket, tuple(args.lengths), fixed_tsb)
                      for bucket in range(crossval.FOLDS)]
            key = f"H{horizon} · {year}"
            results[key] = {length: summarize(blocks, length) for length in args.lengths}
            # 행별 예측(scripts/activity_rule.py가 읽음). runs/는 git에 올리지 않는다
            for length in args.lengths:
                prefix = f"H{horizon}_{year}_L{length}_"
                saved[prefix + "card"] = np.concatenate([b["card"] for b in blocks])
                saved[prefix + "y"] = np.concatenate([b["y"] for b in blocks])
                for field in ("AI", "TSB", "spend_days"):
                    saved[prefix + field] = np.concatenate([b["by_length"][length][field] for b in blocks])
            print(key, {length: (r["AI"]["MAE"], r["TSB"]["MAE"], r["pass"]) for length, r in results[key].items()},
                  flush=True)
    np.savez_compressed(run_dir / "predictions.npz", **saved)
    write_json(run_dir / "metrics.json", {k: {str(l): v for l, v in by.items()} for k, by in results.items()})
    (run_dir / "metrics.md").write_text(to_markdown(results) + "\n## 결제한 날 수별\n\n" + activity_markdown(results),
                                        encoding="utf-8")
    print(run_dir)


if __name__ == "__main__":
    main()
