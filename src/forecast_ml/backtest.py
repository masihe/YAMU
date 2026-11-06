"""시간순 백테스트: 기준일을 7일씩 옮기며 Baseline의 7·14일 체크 소비 예측을 실제와 비교한다(Task 7.3.2). 실행은 scripts/backtest.py."""

import zlib

import numpy as np
import pandas as pd

from forecast_ml import baselines
from forecast_ml.data import Panel

HORIZONS = (7, 14)
# 2026-11-02 첫 실행(5종 비교) 뒤 두 가지만 남김: 이상거래만 / 이상거래+큰 지출+반복 지출
VARIANTS = {
    "fraud": frozenset({"fraud"}),
    "fraud+large+recurring": frozenset({"fraud", "large", "recurring"}),
    "fraud+large+recurring_once": frozenset({"fraud", "large", "recurring_once"}),
}
ACTIVE_DAYS = 90
BASELINES = {
    "AVG_3M": baselines.avg_3m,
    "AVG_28D": baselines.avg_recent,
    "WEEKDAY_8W": baselines.weekday_average,
    "LAST_7D": baselines.last_week,
    "MEDIAN_4W": baselines.weekly_median,
}


def is_test_card(card: str, test_share: int = 5, bucket: int = 0) -> bool:
    """카드KEY 해시로 사람을 test_share묶음으로 나누고, bucket번 묶음이면 평가용이다."""
    return zlib.crc32(card.encode()) % test_share == bucket


def origins(panel: Panel, horizon: int, step: int = 7, active_days: int = ACTIVE_DAYS) -> tuple[np.ndarray, np.ndarray]:
    """평가할 (카드, 기준일). 기준일 달과 앞 3개월, 예측 기간이 걸친 달이 모두 확인되고,
    기준일까지 active_days일 안에 체크 소비(제외 규칙 적용 뒤)가 있는 경우만 쓴다."""
    month_of_day = panel.month_of_day()
    start = max(active_days, int(np.searchsorted(month_of_day, 3)))
    t = np.arange(start, len(panel.days) - horizon - 1, step)
    cards, times = np.meshgrid(np.arange(len(panel.cards)), t, indexing="ij")
    cards, times = cards.ravel(), times.ravel()
    m = month_of_day[times]
    ok = np.ones(len(times), dtype=bool)
    for back in (0, 1, 2, 3):
        ok &= panel.observed[cards, m - back]
    # 예측 기간(t+1 ~ t+horizon)이 걸친 달을 모두 본다. 30일은 가운데 달을 건너뛸 수 있다(1/31 + 30일 = 3/2)
    last = month_of_day[times + horizon]
    for ahead in range(1, int((last - m).max(initial=0)) + 1):
        month = np.minimum(m + ahead, last)
        ok &= panel.observed[cards, month]
    cum = np.concatenate([np.zeros((panel.spend.shape[0], 1), dtype=np.int64), np.cumsum(panel.spend, axis=1)], axis=1)
    ok &= (cum[cards, times + 1] - cum[cards, times + 1 - active_days]) > 0
    return cards[ok], times[ok]


def metrics(pred: np.ndarray, real: np.ndarray) -> dict:
    """MAE, 편향(예측-실제 평균), 과소예측 비율, WAPE(오차 합 ÷ 실제 합)."""
    error = pred - real
    return {
        "n": int(len(real)),
        "mean_actual": round(float(real.mean()), 1),
        "MAE": round(float(np.abs(error).mean()), 1),
        "bias": round(float(error.mean()), 1),
        "under_rate": round(float((pred < real).mean()), 4),
        "WAPE": round(float(np.abs(error).sum() / real.sum()), 4) if real.sum() else None,
        "rel_bias": round(float(error.mean() / real.mean()), 4) if real.mean() else None,
        "zero_share": round(float((real == 0).mean()), 4),
        "MAE_zero": round(float(np.abs(error[real == 0]).mean()), 1) if (real == 0).any() else None,
        "MAE_pos": round(float(np.abs(error[real > 0]).mean()), 1) if (real > 0).any() else None,
    }


def evaluate(panel: Panel, test_from: pd.Timestamp) -> dict:
    """horizon·Baseline·평가 구간(전체, 평가용 사람×2024년)별 지표."""
    test_cards = np.array([is_test_card(c) for c in panel.cards])
    result = {}
    for horizon in HORIZONS:
        card, t = origins(panel, horizon)
        real = baselines.actual(panel, card, t, horizon)
        slices = {"all": np.ones(len(t), dtype=bool),
                  "test_cards_2024": test_cards[card] & (panel.days[t] >= test_from)}
        by_baseline = {}
        for name, predict in BASELINES.items():
            pred = predict(panel, card, t, horizon)
            if (pred < 0).any():
                raise ValueError(f"{name}: 확인된 달이 없는 기준일이 평가 집합에 들어 있습니다")
            by_baseline[name] = {s: metrics(pred[mask], real[mask]) for s, mask in slices.items()}
        result[f"H{horizon}"] = {"cards": int(len(np.unique(card))), "origins": int(len(t)), "baselines": by_baseline}
    return result


def to_markdown(results: dict) -> str:
    """변형·horizon·Baseline별 평가 구간 지표 표."""
    lines = ["| 제외 | H | Baseline | 구간 | n | 실제 평균 | MAE | 편향 | 과소 비율 | WAPE |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for variant, res in results.items():
        for h, block in res["horizons"].items():
            for name, slices in block["baselines"].items():
                for s, m in slices.items():
                    wape = "-" if m["WAPE"] is None else f"{m['WAPE']:.3f}"
                    lines.append(f"| {variant} | {h} | {name} | {s} | {m['n']} | {m['mean_actual']:,.0f} | {m['MAE']:,.0f} | "
                                 f"{m['bias']:,.0f} | {m['under_rate']:.1%} | {wape} |")
    return "\n".join(lines) + "\n"
