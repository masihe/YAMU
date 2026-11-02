"""7.3.1 데이터 확인 집계를 다시 실행해 runs/<날짜시각>-eda/에 남긴다.

사용: .venv\\Scripts\\python.exe -I scripts\\eda.py [--data-dir data\\raw\\aihub-71925]
"""

import argparse
from pathlib import Path

from forecast_ml import eda_codes, eda_continuity
from forecast_ml.runs import new_run, write_json

REPO = Path(__file__).resolve().parents[1]
CARD_ZIPS = ("TL_카드거래.zip", "VL_카드거래.zip")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO / "data" / "raw" / "aihub-71925")
    parser.add_argument("--runs-dir", type=Path, default=REPO / "runs")
    args = parser.parse_args()
    zips = [args.data_dir / name for name in CARD_ZIPS]
    run_dir = new_run(args.runs_dir, "eda", {"script": "scripts/eda.py", "inputs": zips}, REPO)
    write_json(run_dir / "card_continuity.json", eda_continuity.scan(zips))
    df = eda_codes.load(zips)
    write_json(run_dir / "approval_code_check.json",
               {"rows": int(len(df)), "pairing_with_00": eda_codes.pairing(df), "density_00": eda_codes.density(df)})
    print(run_dir)


if __name__ == "__main__":
    main()
