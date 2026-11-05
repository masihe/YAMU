"""결제 비율로 AI/TSB를 고르는 기준(냉시작, 2026-11-05). scripts/short_history.py가 남긴 predictions.npz를 읽는다.

사전 등록(2026-11-05, 결과를 보기 전에 사용자와 정함):
- 기준: 기록 중 결제한 날의 비율 r = 결제한 날 ÷ 기록 길이가 X 이상이면 AI, 아니면 TSB. 결제가 0일인 행은 AI를 부르지 않으므로 뺀다.
- X 후보: 0(항상 AI), 0.1, 0.2, 0.25, 0.33, 0.4, 0.5.
- 고르기(2023년만): 14·30일 예측 × 기록 14·30·60·90일 8조건에서 "기준 예측 MAE ÷ 항상 TSB MAE"의 평균이 가장 작은 X.
  단 8조건 모두 기준 예측의 편향이 ±10% 안이어야 한다. 그런 X가 없으면 8조건 중 가장 나쁜 |편향|이 가장 작은 X.
- 확인(2024년, 한 번): 고른 X를 그대로 적용해 8조건마다 "항상 TSB"보다 확실히 낫고(사람 부트스트랩 95% 상한 < 0)
  편향이 ±10% 안인지 본다. "항상 AI"(X = 0)와도 비교한다.

사용: .venv\\Scripts\\python.exe scripts\\activity_rule.py runs\\<날짜시각>-short-history
"""

import argparse
import json
from pathlib import Path

import numpy as np

from forecast_ml import short_history
from forecast_ml.backtest import metrics
from forecast_ml.experiment import BIAS_LIMIT

CANDIDATES = (0.0, 0.1, 0.2, 0.25, 0.33, 0.4, 0.5)
HORIZONS = (14, 30)
LENGTHS = (14, 30, 60, 90)


def condition(data, horizon: int, year: str, length: int) -> dict[str, np.ndarray]:
    """결제가 1일 이상인 행만."""
    prefix = f"H{horizon}_{year}_L{length}_"
    rows = {k: data[prefix + k] for k in ("card", "y", "AI", "TSB", "spend_days")}
    keep = rows["spend_days"] > 0
    return {k: v[keep] for k, v in rows.items()}


def policy(rows: dict[str, np.ndarray], length: int, x: float) -> np.ndarray:
    """r >= x이면 AI, 아니면 TSB."""
    use_ai = rows["spend_days"] / length >= x
    return np.where(use_ai, rows["AI"], rows["TSB"].astype(np.float64))


def choose(data) -> tuple[float, dict]:
    """2023년 8조건으로 X를 고른다(사전 등록 규칙)."""
    table = {}
    for x in CANDIDATES:
        ratios, biases = [], []
        for h in HORIZONS:
            for length in LENGTHS:
                rows = condition(data, h, "2023", length)
                m = metrics(policy(rows, length, x), rows["y"])
                tsb = metrics(rows["TSB"].astype(np.float64), rows["y"])
                ratios.append(m["MAE"] / tsb["MAE"])
                biases.append(m["rel_bias"])
        table[x] = {"mean_mae_ratio": round(float(np.mean(ratios)), 4),
                    "worst_abs_bias": round(float(np.max(np.abs(biases))), 4),
                    "all_bias_ok": bool(all(abs(b) <= BIAS_LIMIT for b in biases))}
    ok = [x for x in CANDIDATES if table[x]["all_bias_ok"]]
    chosen = (min(ok, key=lambda x: table[x]["mean_mae_ratio"]) if ok
              else min(CANDIDATES, key=lambda x: table[x]["worst_abs_bias"]))
    return chosen, table


def check(data, x: float) -> dict:
    """2024년 8조건에서 고른 X를 항상 TSB·항상 AI와 비교한다."""
    out = {}
    for h in HORIZONS:
        for length in LENGTHS:
            rows = condition(data, h, "2024", length)
            pred = policy(rows, length, x)
            m = metrics(pred, rows["y"])
            boot = short_history.bootstrap(rows["card"], rows["y"], pred, rows["TSB"].astype(np.float64))
            share_ai = float((rows["spend_days"] / length >= x).mean())
            out[f"H{h} · L{length}"] = {
                "rows": int(len(rows["y"])), "ai_share": round(share_ai, 4), "rule": m,
                "always_TSB": metrics(rows["TSB"].astype(np.float64), rows["y"]),
                "always_AI": metrics(rows["AI"], rows["y"]), "vs_TSB": boot["mae_vs_tsb"],
                "pass": bool(abs(m["rel_bias"]) <= BIAS_LIMIT and boot["mae_vs_tsb"]["high"] < 0)}
    return out


def to_markdown(x: float, table: dict, result: dict) -> str:
    lines = ["## 2023년으로 고르기\n", "| X | 8조건 평균 MAE ÷ TSB | 가장 나쁜 |편향| | 모두 ±10% 안 |", "|---|---|---|---|"]
    for c, t in table.items():
        mark = " ← 선택" if c == x else ""
        lines.append(f"| {c} | {t['mean_mae_ratio']:.3f} | {t['worst_abs_bias'] * 100:.1f}% | "
                     f"{'O' if t['all_bias_ok'] else 'x'} |{mark}")
    lines += [f"\n## 2024년 확인 (X = {x})\n",
              "| 조건 | 행 | AI로 간 비율 | 기준 MAE | 항상 TSB MAE | 항상 AI MAE | 기준 vs TSB(%) [95%] | 기준 편향(%) | 통과 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for key, r in result.items():
        ci = r["vs_TSB"]
        lines.append(f"| {key} | {r['rows']:,} | {r['ai_share']:.0%} | {r['rule']['MAE']:,.0f} | "
                     f"{r['always_TSB']['MAE']:,.0f} | {r['always_AI']['MAE']:,.0f} | {ci['mean'] * 100:+.1f} "
                     f"[{ci['low'] * 100:+.1f}, {ci['high'] * 100:+.1f}] | {r['rule']['rel_bias'] * 100:+.1f} | "
                     f"{'O' if r['pass'] else 'x'} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    data = np.load(args.run_dir / "predictions.npz")
    x, table = choose(data)
    result = check(data, x)
    (args.run_dir / "activity_rule.json").write_text(
        json.dumps({"chosen_x": x, "selection_2023": {str(k): v for k, v in table.items()}, "check_2024": result},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (args.run_dir / "activity_rule.md").write_text(to_markdown(x, table, result), encoding="utf-8")
    print(to_markdown(x, table, result))


if __name__ == "__main__":
    main()
