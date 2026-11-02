"""서비스 입력(계좌별 최근 90일 일별 생활 소비)을 학습 때와 같은 특징으로 바꿔 예측한다."""

from datetime import date

import numpy as np
import pandas as pd

from forecast_ml.artifacts import HISTORY_DAYS, LoadedModel
from forecast_ml.data import Panel
from forecast_ml.features import make_features


def panel_from_history(daily: np.ndarray, as_of: date) -> Panel:
    """(계좌 수, 90) 일별 소비를 as_of가 마지막 날인 Panel로 만든다. 서비스 입력은 모든 날이 확인된 날로 본다."""
    if daily.ndim != 2 or daily.shape[1] != HISTORY_DAYS:
        raise ValueError(f"일별 기록은 계좌마다 {HISTORY_DAYS}일이어야 합니다")
    if (daily < 0).any():
        raise ValueError("일별 소비는 0 이상이어야 합니다")
    days = pd.date_range(end=pd.Timestamp(as_of), periods=HISTORY_DAYS, freq="D")
    months = pd.period_range(days[0], days[-1], freq="M")
    cards = np.array([str(i) for i in range(daily.shape[0])])
    observed = np.ones((daily.shape[0], len(months)), dtype=bool)
    return Panel(cards, days, months, daily.astype(np.int64), observed, 0.0)


def predict(loaded: LoadedModel, daily: np.ndarray, as_of: date) -> np.ndarray:
    """계좌마다 다음 horizon일 체크 생활 소비 합계(원, 0 이상 정수)."""
    panel = panel_from_history(daily, as_of)
    n = daily.shape[0]
    X = make_features(panel, np.arange(n), np.full(n, HISTORY_DAYS - 1), loaded.horizon, "core")
    if list(X.columns) != loaded.features:
        raise ValueError("모델이 학습한 특징과 지금 계산한 특징이 다릅니다")
    pred = np.asarray(loaded.model.predict(X), dtype=np.float64)
    return np.maximum(np.floor(pred + 0.5), 0).astype(np.int64)
