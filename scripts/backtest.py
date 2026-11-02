"""7.3.2 Baseline 백테스트를 실행해 runs/<날짜시각>-baselines/에 설정·지표·표를 남긴다.

사용: .venv\\Scripts\\python.exe -I scripts\\backtest.py [--data-dir data\\raw\\aihub-71925] [--test-from 2024-01-01]
"""

import argparse
from pathlib import Path

import pandas as pd

from forecast_ml.backtest import BASELINES, HORIZONS, VARIANTS, evaluate, to_markdown
from forecast_ml.data import build_panel, load_transactions
from forecast_ml.runs import new_run, write_json

REPO = Path(__file__).resolve().parents[1]
CARD_ZIPS = ("TL_카드거래.zip", "VL_카드거래.zip")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO / "data" / "raw" / "aihub-71925")
    parser.add_argument("--runs-dir", type=Path, default=REPO / "runs")
    parser.add_argument("--test-from", default="2024-01-01")
    args = parser.parse_args()
    zips = [args.data_dir / name for name in CARD_ZIPS]
    config = {"script": "scripts/backtest.py", "inputs": zips, "test_from": args.test_from,
              "horizons": HORIZONS, "baselines": list(BASELINES), "variants": {k: sorted(v) for k, v in VARIANTS.items()},
              "origin_step_days": 7, "active_days": 90, "test_cards": "crc32(card) % 5 == 0",
              "observed_month": "그달 그 카드KEY의 거래(신용·체크)가 하나라도 있음",
              "large": "체크 거래 금액 99% 분위수 초과", "recurring": "같은 카드·가맹점·금액이 3개 이상의 달에 나타남"}
    run_dir = new_run(args.runs_dir, "baselines", config, REPO)
    df = load_transactions(zips)
    results = {}
    for variant, exclude in VARIANTS.items():
        panel = build_panel(df, exclude)
        results[variant] = {"removed_check_amount_share": round(panel.removed_share, 4),
                            "horizons": evaluate(panel, pd.Timestamp(args.test_from))}
        print(variant, "removed", results[variant]["removed_check_amount_share"])
    write_json(run_dir / "metrics.json", results)
    (run_dir / "metrics.md").write_text(to_markdown(results), encoding="utf-8")
    print(run_dir)


if __name__ == "__main__":
    main()
