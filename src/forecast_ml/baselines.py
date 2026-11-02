"""통계 Baseline. 모두 기준일 t까지의 소비로 t+1 ~ t+H의 체크 소비 합계를 예측한다(원 단위)."""

import numpy as np

from forecast_ml.data import Panel


def rounded_divide(dividend: np.ndarray | int, divisor: np.ndarray | int) -> np.ndarray | int:
    """0 이상 정수 나눗셈을 원 단위로 반올림한다. Java LivingCostEstimator.roundedDivide와 같은 식."""
    return (dividend * 2 + divisor) // (divisor * 2)


def avg_3m_daily(month_amounts: list[int], month_days: list[int]) -> int | None:
    """확인된 완료 달들의 합 ÷ 그 달들의 실제 일수(명세 AVG_3M 일평균). 달이 없으면 None."""
    if len(month_amounts) != len(month_days):
        raise ValueError("달 금액과 일수의 개수가 다릅니다")
    if not month_amounts:
        return None
    if any(a < 0 for a in month_amounts) or any(d <= 0 for d in month_days):
        raise ValueError("금액은 0 이상, 일수는 1 이상이어야 합니다")
    return int(rounded_divide(sum(month_amounts), sum(month_days)))


def avg_3m(panel: Panel, card: np.ndarray, t: np.ndarray, horizon: int) -> np.ndarray:
    """AVG_3M: 기준일이 속한 달을 빼고 지난 3개월 중 확인된 달의 일평균 × horizon. 확인된 달이 없으면 -1."""
    month_of_day = panel.month_of_day()
    monthly = np.add.reduceat(panel.spend, np.searchsorted(month_of_day, np.arange(len(panel.months))), axis=1)
    month_len = panel.months.days_in_month.to_numpy()
    m = month_of_day[t]
    total = np.zeros(len(t), dtype=np.int64)
    days = np.zeros(len(t), dtype=np.int64)
    for back in (1, 2, 3):
        mm = m - back
        valid = mm >= 0
        use = valid & panel.observed[card, np.clip(mm, 0, None)]
        total += np.where(use, monthly[card, np.clip(mm, 0, None)], 0)
        days += np.where(use, month_len[np.clip(mm, 0, None)], 0)
    daily = np.where(days > 0, rounded_divide(total, np.maximum(days, 1)), -1)
    return np.where(daily >= 0, daily * horizon, -1)


def cumulative(panel: Panel) -> np.ndarray:
    """앞에 0을 붙인 날짜 누적합. cum[:, d+1] - cum[:, a] = a..d 합계."""
    return np.concatenate([np.zeros((panel.spend.shape[0], 1), dtype=np.int64), np.cumsum(panel.spend, axis=1)], axis=1)


def avg_recent(panel: Panel, card: np.ndarray, t: np.ndarray, horizon: int, window: int = 28) -> np.ndarray:
    """최근 window일(t 포함) 일평균(원 단위 반올림) × horizon."""
    cum = cumulative(panel)
    total = cum[card, t + 1] - cum[card, t + 1 - window]
    return rounded_divide(total, window) * horizon


def weekday_average(panel: Panel, card: np.ndarray, t: np.ndarray, horizon: int, weeks: int = 8) -> np.ndarray:
    """예측할 날마다 같은 요일의 최근 weeks번(t 이전) 평균을 더한다. 마지막에 원 단위 반올림."""
    total = np.zeros(len(t), dtype=np.int64)
    for k in range(1, horizon + 1):
        first = -(-k // 7)  # t 이하가 되는 가장 가까운 같은 요일: t+k-7*first
        for j in range(first, first + weeks):
            total += panel.spend[card, t + k - 7 * j]
    return rounded_divide(total, weeks)


def last_week(panel: Panel, card: np.ndarray, t: np.ndarray, horizon: int) -> np.ndarray:
    """지난 7일(t 포함) 합계를 7일당 소비로 보고 horizon에 맞춘다. 7일이면 지난 7일 합계 그대로."""
    cum = cumulative(panel)
    total = cum[card, t + 1] - cum[card, t - 6]
    return rounded_divide(total * horizon, 7)


def weekly_median(panel: Panel, card: np.ndarray, t: np.ndarray, horizon: int, weeks: int = 4) -> np.ndarray:
    """최근 weeks주(t에서 7일씩 끊은 주) 합계의 중앙값을 7일당 소비로 보고 horizon에 맞춘다."""
    cum = cumulative(panel)
    sums = np.stack([cum[card, t + 1 - 7 * w] - cum[card, t + 1 - 7 * (w + 1)] for w in range(weeks)], axis=1)
    median = np.median(sums, axis=1)
    return np.floor(median * horizon / 7 + 0.5).astype(np.int64)


def actual(panel: Panel, card: np.ndarray, t: np.ndarray, horizon: int) -> np.ndarray:
    """t+1 ~ t+horizon의 실제 체크 소비 합계."""
    cum = cumulative(panel)
    return cum[card, t + 1 + horizon] - cum[card, t + 1]
