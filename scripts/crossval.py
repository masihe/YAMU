"""평가 보강 실행: 제외 규칙 × H × 평가 연도 2023·2024 × 사람 5묶음. runs/<날짜시각>-crossval/에 남긴다.

사용: .venv\\Scripts\\python.exe -I scripts\\crossval.py [--feature-set core] [--horizons 7 14 30] [--variants fraud+large+recurring]
"""

import argparse
from pathlib import Path

import pandas as pd

from forecast_ml import crossval, experiment, models
from forecast_ml.backtest import HORIZONS, VARIANTS
from forecast_ml.data import build_panel, load_transactions
from forecast_ml.runs import new_run, write_json

REPO = Path(__file__).resolve().parents[1]
CARD_ZIPS = ("TL_카드거래.zip", "VL_카드거래.zip")
# 평가 연도 → (검증 시작, 평가 시작, 평가 끝). 검증은 평가 전년 하반기
YEARS = {"2023": ("2022-07-01", "2023-01-01", "2024-01-01"), "2024": ("2023-07-01", "2024-01-01", None)}


def to_markdown(results: dict) -> str:
    """조건마다 합친 지표와 판정 표."""
    out = []
    for key, block in results.items():
        boot = block["bootstrap"]
        out.append(f"\n### {key} (평가 {block['rows']}건, {boot['people']}명)\n")
        out.append("| 방식 | MAE | 편향(%) [95%] | vs LAST_7D MAE(%) [95%] | vs TSB MAE(%) [95%] | 통과 LAST_7D 점/구간 | 통과 TSB 점/구간 |")
        out.append("|---|---|---|---|---|---|---|")
        for name, v in sorted(block["verdict"].items(), key=lambda kv: kv[1]["metrics"]["MAE"]):
            m = v["metrics"]
            b = boot["bias"][name]

            def ci(ref):
                c = boot["mae_vs"].get(f"{name}|{ref}")
                return "-" if c is None else f"{c['mean'] * 100:+.1f} [{c['low'] * 100:+.1f}, {c['high'] * 100:+.1f}]"

            def mark(ref):
                j = v["judge"].get(ref)
                return "-" if j is None else f"{'O' if j['point'] else 'x'}/{'O' if j['interval'] else 'x'}"

            out.append(f"| {name} | {m['MAE']:,.0f} | {m['rel_bias'] * 100:+.1f} [{b['low'] * 100:+.1f}, {b['high'] * 100:+.1f}] | "
                       f"{ci('LAST_7D')} | {ci('TSB')} | {mark('LAST_7D')} | {mark('TSB')} |")
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO / "data" / "raw" / "aihub-71925")
    parser.add_argument("--runs-dir", type=Path, default=REPO / "runs")
    parser.add_argument("--feature-set", default="core", choices=["core", "core+calendar"])
    parser.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    args = parser.parse_args()
    zips = [args.data_dir / name for name in CARD_ZIPS]
    config = {"script": "scripts/crossval.py", "inputs": zips, "feature_set": args.feature_set, "years": YEARS,
              "folds": crossval.FOLDS, "references": crossval.REFERENCES, "horizons": args.horizons,
              "variants": {k: sorted(VARIANTS[k]) for k in args.variants}, "model_grids": models.GRIDS,
              "judge": "점 = 합친 지표로 |편향| <= 10% 그리고 MAE < 기준, 구간 = 점 편향 조건 + MAE 비율 95% 상한 < 0"}
    run_dir = new_run(args.runs_dir, "crossval", config, REPO)
    df = load_transactions(zips)
    results = {}
    for variant in args.variants:
        panel = build_panel(df, VARIANTS[variant])
        for horizon in args.horizons:
            for year, (val_from, test_from, test_to) in YEARS.items():
                blocks = [experiment.run(panel, horizon, pd.Timestamp(val_from), pd.Timestamp(test_from), args.feature_set,
                                         test_to=pd.Timestamp(test_to) if test_to else None, bucket=b,
                                         keep_predictions=True) for b in range(crossval.FOLDS)]
                pooled = crossval.pool(blocks)
                names = list(pooled["pred"])
                boot = crossval.bootstrap(pooled, names)
                key = f"{variant} · H{horizon} · {year}"
                results[key] = {"rows": int(len(pooled["y"])), "bootstrap": boot,
                                "verdict": crossval.verdict(pooled, boot)}
                print(key, "rows", len(pooled["y"]), "people", boot["people"], flush=True)
    write_json(run_dir / "metrics.json", results)
    (run_dir / "metrics.md").write_text(to_markdown(results), encoding="utf-8")
    print(run_dir)


if __name__ == "__main__":
    main()
