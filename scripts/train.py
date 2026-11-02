"""7.3.3 모델 실험을 실행해 runs/<날짜시각>-models-<특징 묶음>/에 설정·지표·표를 남긴다.

사용: .venv\\Scripts\\python.exe -I scripts\\train.py [--feature-set core|core+calendar]
"""

import argparse
from pathlib import Path

import pandas as pd

from forecast_ml import experiment, models
from forecast_ml.backtest import HORIZONS, VARIANTS
from forecast_ml.data import build_panel, load_transactions
from forecast_ml.runs import new_run, write_json

REPO = Path(__file__).resolve().parents[1]
CARD_ZIPS = ("TL_카드거래.zip", "VL_카드거래.zip")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO / "data" / "raw" / "aihub-71925")
    parser.add_argument("--runs-dir", type=Path, default=REPO / "runs")
    parser.add_argument("--feature-set", default="core", choices=["core", "core+calendar"])
    parser.add_argument("--val-from", default="2023-07-01")
    parser.add_argument("--test-from", default="2024-01-01")
    args = parser.parse_args()
    zips = [args.data_dir / name for name in CARD_ZIPS]
    config = {"script": "scripts/train.py", "inputs": zips, "feature_set": args.feature_set,
              "val_from": args.val_from, "test_from": args.test_from, "horizons": HORIZONS,
              "variants": {k: sorted(v) for k, v in VARIANTS.items()}, "model_grids": models.GRIDS,
              "sba_grid": experiment.SBA_GRID, "tsb_grid": experiment.TSB_GRID,
              "gate": f"|편향| <= {experiment.BIAS_LIMIT:.0%} 그리고 MAE < {experiment.GATE_BASELINE}",
              "split": "fit=학습 사람 & 기준일 < val_from-H, val=학습 사람 & val_from <= 기준일 < test_from-H, "
                       "train=fit+val 구간 전체, test=평가 사람(crc32 % 5 == 0) & 기준일 >= test_from"}
    name = "models-" + args.feature_set.replace("+", "-")
    run_dir = new_run(args.runs_dir, name, config, REPO)
    df = load_transactions(zips)
    results = {}
    for variant, exclude in VARIANTS.items():
        panel = build_panel(df, exclude)
        results[variant] = {}
        for horizon in HORIZONS:
            block = experiment.run(panel, horizon, pd.Timestamp(args.val_from), pd.Timestamp(args.test_from),
                                   args.feature_set)
            results[variant][f"H{horizon}"] = block
            passed = [n for n, r in block["results"].items() if r["passes_gate"]]
            print(variant, horizon, "rows", block["rows"], "passed", passed)
    write_json(run_dir / "metrics.json", results)
    (run_dir / "metrics.md").write_text(experiment.to_markdown(results), encoding="utf-8")
    print(run_dir)


if __name__ == "__main__":
    main()
