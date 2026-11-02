"""팀 레포 ai/의 생활비 예측 서빙이 이 저장소와 같은 값을 내는지 확인할 기대값(golden)을 만든다.

입력 몇 가지(빈 기록, 창 경계의 소비, 무작위, 아주 큰 금액)마다 serving과 같은 경로
(panel_from_history → make_features "core")로 특징을 계산하고, 최종 모델로 7·14·30일 예측을 낸다.

사용: .venv\\Scripts\\python.exe -I scripts\\make_golden.py --model-dir models\\<모델 버전> --out <팀 레포>\\ai\\tests\\golden_living_cost.json
"""

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np

from forecast_ml import artifacts, serving
from forecast_ml.features import make_features

AS_OF = date(2026, 11, 2)
HISTORY = artifacts.HISTORY_DAYS


def cases() -> dict[str, list[int]]:
    """이름 → 90일 일별 소비. 마지막 값이 AS_OF 당일이다."""
    rng = np.random.default_rng(20261102)
    zero = [0] * HISTORY

    def only(day: int, amount: int) -> list[int]:
        row = zero.copy()
        row[day] = amount
        return row

    sparse = ((rng.random(HISTORY) < 0.25) * rng.integers(1_000, 80_000, HISTORY)).tolist()
    dense = rng.integers(0, 50_000, HISTORY).tolist()
    weekly = [15_000 if d % 7 == 0 else 0 for d in range(HISTORY)]
    return {
        "all_zero": zero,
        # 84일 창 밖(0~5번째 날)에만 소비: 합계·쓴 날은 0이고 경과일은 89
        "only_before_84d_window": only(0, 30_000),
        "first_day_of_84d_window": only(HISTORY - 84, 30_000),
        "only_today": only(HISTORY - 1, 12_345),
        "week_boundaries": [7_000 if d in (HISTORY - 7, HISTORY - 8, HISTORY - 14, HISTORY - 15, HISTORY - 28)
                            else 0 for d in range(HISTORY)],
        "single_active_day_in_84d": only(HISTORY - 40, 99_999),
        "weekly": weekly,
        "sparse_random": sparse,
        "dense_random": dense,
        "max_amount_every_day": [1_000_000_000_000] * HISTORY,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    loaded = artifacts.load_all(args.model_dir)
    named = cases()
    daily = np.array(list(named.values()), dtype=np.int64)
    n = daily.shape[0]
    panel = serving.panel_from_history(daily, AS_OF)
    X = make_features(panel, np.arange(n), np.full(n, HISTORY - 1), 7, "core")
    predictions = {str(h): serving.predict(m, daily, AS_OF).tolist() for h, m in sorted(loaded.items())}
    versions = {m.meta.get("model_version") for m in loaded.values()}
    if len(versions) != 1:
        raise ValueError("horizon마다 모델 버전이 다릅니다")

    head = {"asOf": AS_OF.isoformat(), "modelVersion": versions.pop(), "features": list(X.columns)}
    rows = [
        {"name": name, "dailySpending": named[name],
         # 정수 특징은 int, 실수 특징은 float로 남긴다. json의 float는 repr이라 값이 그대로 돌아온다
         "features": {col: X[col].iloc[i].item() for col in X.columns},
         "predictions": {h: values[i] for h, values in predictions.items()}}
        for i, name in enumerate(named)
    ]
    # 입력 하나를 한 줄로 써서 diff를 읽기 쉽게 한다
    text = json.dumps(head, ensure_ascii=False)[:-1] + ', "cases": [\n'
    text += ",\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n]}\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"{len(named)}개 입력 → {args.out}")


if __name__ == "__main__":
    main()
