"""짧은 기록 평가(2026-11-05, 냉시작): 기록이 L일뿐인 사용자에게 지금 모델(TWO_STAGE_TREE)이 TSB보다 나은지 잰다.

교차검증과 같은 사람·기준일·학습을 쓰고, 평가할 때만 기준일까지 90일 중 앞쪽 (90 − L)일을 0원으로 채운다.
BE가 기록이 L일인 계좌를 AI 서버에 보낼 때와 같은 입력이다(serving.predict와 같은 특징 계산).
TSB는 첫 결제부터 계산을 시작하므로 앞쪽 0원은 영향이 없고 "최근 L일만 본 TSB"와 같다. 설정값은 같은 L로 자른 검증 구간에서 고른다.
한계: 기록이 긴 사람의 앞부분을 가린 것이라, 가입 초기 소비가 평소와 다르면 그 차이는 잡지 못한다.
"""

import numpy as np
import pandas as pd

from forecast_ml import intermittent, models
from forecast_ml.artifacts import HISTORY_DAYS
from forecast_ml.backtest import metrics
from forecast_ml.data import Panel
from forecast_ml.experiment import TSB_GRID, choose, split
from forecast_ml.features import make_features
from forecast_ml.serving import panel_from_history

LENGTHS = (14, 30, 60, 90)
MODEL = "TWO_STAGE_TREE"


def truncated_history(panel: Panel, card: np.ndarray, t: np.ndarray, length: int) -> np.ndarray:
    """(행 수, 90) 기준일까지 90일. 앞쪽 (90 − length)일은 0원(기록 시작 전)."""
    if not 1 <= length <= HISTORY_DAYS:
        raise ValueError(f"기록 길이는 1~{HISTORY_DAYS}일이어야 합니다")
    offsets = np.arange(-HISTORY_DAYS + 1, 1)
    daily = panel.spend[card[:, None], t[:, None] + offsets[None, :]].astype(np.int64)
    daily[:, : HISTORY_DAYS - length] = 0
    return daily


def serving_features(daily: np.ndarray, horizon: int) -> pd.DataFrame:
    """서비스와 같은 경로(90일 Panel의 마지막 날 기준)로 특징을 만든다. 날짜는 core 특징에 쓰이지 않는다."""
    panel = panel_from_history(daily, pd.Timestamp("2024-03-31").date())
    n = daily.shape[0]
    return make_features(panel, np.arange(n), np.full(n, HISTORY_DAYS - 1), horizon, "core")


def avg_3m_truncated(panel: Panel, card: np.ndarray, t: np.ndarray, horizon: int, length: int) -> np.ndarray:
    """지금 서비스의 AVG_3M: 기준일 달을 뺀 지난 3개월 중, 기록 시작(t − length + 1) 뒤에 1일부터 다 들어온 달만 쓴다.
    쓸 달이 없으면 -1(서비스는 생활비 '자료 없음')."""
    month_of_day = panel.month_of_day()
    starts = np.searchsorted(month_of_day, np.arange(len(panel.months)))
    monthly = np.add.reduceat(panel.spend, starts, axis=1)
    month_len = panel.months.days_in_month.to_numpy()
    m = month_of_day[t]
    first_day = t - length + 1
    total = np.zeros(len(t), dtype=np.int64)
    days = np.zeros(len(t), dtype=np.int64)
    for back in (1, 2, 3):
        mm = np.clip(m - back, 0, None)
        use = (m - back >= 0) & panel.observed[card, mm] & (starts[mm] >= first_day)
        total += np.where(use, monthly[card, mm], 0)
        days += np.where(use, month_len[mm], 0)
    daily = np.where(days > 0, (total * 2 + np.maximum(days, 1)) // (np.maximum(days, 1) * 2), -1)
    return np.where(daily >= 0, daily * horizon, -1)


def run(panel: Panel, horizon: int, val_from: pd.Timestamp, test_from: pd.Timestamp,
        test_to: pd.Timestamp | None, bucket: int, lengths: tuple[int, ...] = LENGTHS) -> dict:
    """한 묶음: 학습(교차검증과 같음) 한 번, 길이마다 평가 구간 예측. JSON에 쓰지 않는 numpy 값을 돌려준다."""
    s = split(panel, horizon, val_from, test_from, test_to, bucket)
    X = make_features(panel, s.card, s.t, horizon, "core")
    model = models.build(MODEL, {}).fit(X[s.train], s.y[s.train].astype(np.float64))
    card, t, y = s.card[s.test], s.t[s.test], s.y[s.test]
    out = {"card": card, "y": y, "by_length": {}}
    for length in lengths:
        scores, cache = [], {}
        for params in TSB_GRID:
            cache[tuple(params.items())] = d = intermittent.tsb_window(panel.spend, s.card, s.t, window=length, **params)
            scores.append((params, metrics(np.floor(d * horizon + 0.5)[s.val], s.y[s.val])))
        params, _ = choose(scores)
        tsb = np.floor(cache[tuple(params.items())][s.test] * horizon + 0.5).astype(np.int64)
        daily = truncated_history(panel, card, t, length)
        ai = np.asarray(model.predict(serving_features(daily, horizon)), dtype=np.float64)
        out["by_length"][length] = {"AI": ai, "TSB": tsb, "AVG_3M": avg_3m_truncated(panel, card, t, horizon, length),
                                    "active": daily.sum(axis=1) > 0, "spend_days": (daily > 0).sum(axis=1),
                                    "tsb_params": params}
    return out


# 기록 안에서 결제한 날 수 구간(2026-11-05, 사용자 가설 "활동이 많으면 AI가 낫다" 확인용). 0일은 AI를 부르지 않아 뺀다.
ACTIVITY_BUCKETS = ((1, 3), (4, 10), (11, 20), (21, HISTORY_DAYS))


def activity_bucket(spend_days: np.ndarray) -> np.ndarray:
    """결제한 날 수 → ACTIVITY_BUCKETS 순번. 0일이면 -1."""
    out = np.full(len(spend_days), -1)
    for i, (low, high) in enumerate(ACTIVITY_BUCKETS):
        out[(spend_days >= low) & (spend_days <= high)] = i
    return out


def bootstrap(card: np.ndarray, y: np.ndarray, ai: np.ndarray, ref: np.ndarray, rounds: int = 1000,
              seed: int = 0) -> dict:
    """사람을 복원 추출해 (AI MAE ÷ 기준 MAE − 1)과 AI 편향 비율의 평균·95% 구간."""
    rng = np.random.default_rng(seed)
    people = np.unique(card)
    rows = {c: np.where(card == c)[0] for c in people}
    ratios, biases = [], []
    for _ in range(rounds):
        pick = np.concatenate([rows[c] for c in rng.choice(people, len(people))])
        yy = y[pick]
        ratios.append(np.abs(ai[pick] - yy).mean() / np.abs(ref[pick] - yy).mean() - 1)
        biases.append((ai[pick] - yy).mean() / yy.mean())

    def summary(values: list[float]) -> dict:
        return {"mean": round(float(np.mean(values)), 4), "low": round(float(np.percentile(values, 2.5)), 4),
                "high": round(float(np.percentile(values, 97.5)), 4)}

    return {"people": int(len(people)), "mae_vs_tsb": summary(ratios), "bias": summary(biases)}
