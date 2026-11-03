"""간헐적 수요용 단순 방식. 날마다 상태를 갱신해 하루 예측을 내고 horizon을 곱한다.

- SBA(Syntetos·Boylan 2001): Croston처럼 쓴 날에만 금액 z와 사용 간격 p를 지수평활하고, 하루 예측 z/p에 (1 − α/2)를 곱해 편향을 보정한다.
- TSB(Teunter·Syntetos·Babai 2011): 날마다 "쓸 확률" d를 갱신하고(안 쓴 날도 낮춤), 쓴 날에만 금액 z를 갱신한다. 하루 예측 d·z.
처음 쓴 날 전에는 0으로 예측한다.
"""

import numpy as np

from forecast_ml.data import Panel


def sba_daily(spend: np.ndarray, alpha: float) -> np.ndarray:
    """카드 × 날짜 하루 예측(그날까지 본 뒤의 다음 날 예측)."""
    if not 0 < alpha <= 1:
        raise ValueError("alpha는 0 초과 1 이하")
    n_cards, n_days = spend.shape
    z = np.zeros(n_cards)
    p = np.ones(n_cards)
    last = np.full(n_cards, -1)
    out = np.zeros((n_cards, n_days))
    for day in range(n_days):
        x = spend[:, day].astype(np.float64)
        pos = x > 0
        first = pos & (last < 0)
        later = pos & (last >= 0)
        z = np.where(first, x, np.where(later, z + alpha * (x - z), z))
        interval = day - last
        p = np.where(first, day + 1, np.where(later, p + alpha * (interval - p), p))
        last = np.where(pos, day, last)
        out[:, day] = np.where(last >= 0, (1 - alpha / 2) * z / p, 0.0)
    return out


def tsb_daily(spend: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """카드 × 날짜 하루 예측. alpha는 금액, beta는 쓸 확률의 평활 계수."""
    if not (0 < alpha <= 1 and 0 < beta <= 1):
        raise ValueError("alpha·beta는 0 초과 1 이하")
    n_cards, n_days = spend.shape
    z = np.zeros(n_cards)
    d = np.zeros(n_cards)
    started = np.zeros(n_cards, dtype=bool)
    out = np.zeros((n_cards, n_days))
    for day in range(n_days):
        x = spend[:, day].astype(np.float64)
        pos = x > 0
        first = pos & ~started
        later = pos & started
        z = np.where(first, x, np.where(later, z + alpha * (x - z), z))
        started |= pos
        d = np.where(started, d + beta * (pos.astype(np.float64) - d), 0.0)
        out[:, day] = d * z
    return out


def tsb_window(spend: np.ndarray, card: np.ndarray, t: np.ndarray, alpha: float, beta: float,
               window: int = 90) -> np.ndarray:
    """(카드, 기준일)마다 기준일까지 최근 window일(기준일 포함)만 보고 계산한 TSB 하루 예측.

    서비스는 계좌마다 최근 90일 남짓만 받으므로, 전체 기록을 훑는 tsb_daily 대신 이것이 실제 대체 계산과 같다.
    상태는 창의 첫날 0에서 시작한다. 창이 기록 앞으로 넘어가는 날은 0원으로 본다.
    """
    if not (0 < alpha <= 1 and 0 < beta <= 1):
        raise ValueError("alpha·beta는 0 초과 1 이하")
    n = len(card)
    z = np.zeros(n)
    d = np.zeros(n)
    started = np.zeros(n, dtype=bool)
    start = t - window + 1
    for k in range(window):
        day = start + k
        x = np.where(day >= 0, spend[card, np.maximum(day, 0)], 0).astype(np.float64)
        pos = x > 0
        first = pos & ~started
        later = pos & started
        z = np.where(first, x, np.where(later, z + alpha * (x - z), z))
        started |= pos
        d = np.where(started, d + beta * (pos.astype(np.float64) - d), 0.0)
    return d * z


def forecast(daily: np.ndarray, card: np.ndarray, t: np.ndarray, horizon: int) -> np.ndarray:
    """기준일 t의 하루 예측 × horizon, 원 단위 반올림."""
    return np.floor(daily[card, t] * horizon + 0.5).astype(np.int64)


def sba(panel: Panel, card: np.ndarray, t: np.ndarray, horizon: int, alpha: float = 0.1) -> np.ndarray:
    """SBA로 t+1 ~ t+horizon 합계 예측."""
    return forecast(sba_daily(panel.spend, alpha), card, t, horizon)


def tsb(panel: Panel, card: np.ndarray, t: np.ndarray, horizon: int, alpha: float = 0.1, beta: float = 0.1) -> np.ndarray:
    """TSB로 t+1 ~ t+horizon 합계 예측."""
    return forecast(tsb_daily(panel.spend, alpha, beta), card, t, horizon)
