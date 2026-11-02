"""기준일 t까지의 일별 체크 소비로 모델 입력 특징을 만든다. YAMU가 계좌별 일별 생활 소비로 만들 수 있는 값만 쓴다."""

import numpy as np
import pandas as pd

from forecast_ml.baselines import cumulative
from forecast_ml.data import Panel

WINDOWS = (7, 14, 28, 84)
FEATURE_SETS = ("core", "core+calendar")


def _last_spend_day(panel: Panel) -> np.ndarray:
    """날짜 d까지 마지막으로 소비한 날의 인덱스. 아직 없으면 -1."""
    idx = np.where(panel.spend > 0, np.arange(panel.spend.shape[1]), -1)
    return np.maximum.accumulate(idx, axis=1)


def make_features(panel: Panel, card: np.ndarray, t: np.ndarray, horizon: int, feature_set: str = "core") -> pd.DataFrame:
    """(카드, 기준일)마다 특징 한 행. 기준일 당일까지 쓰고 그 뒤 날짜는 보지 않는다.

    core: 최근 7·14·28·84일 합계와 쓴 날 수, 2~4주 전 주별 합계, 마지막 소비 후 경과일,
          84일 동안 쓴 날의 평균 금액, 평균 사용 간격(ADI), 쓴 날 금액의 변동성(CV²)
    core+calendar: core + 예측 기간의 주말 일수, 예측 첫날의 일자, 기준일에서 월말까지 남은 일수
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"모르는 특징 묶음: {feature_set}")
    cum = cumulative(panel)
    active = (panel.spend > 0).astype(np.int64)
    cum_active = np.concatenate([np.zeros((active.shape[0], 1), dtype=np.int64), np.cumsum(active, axis=1)], axis=1)
    square = panel.spend.astype(np.float64) ** 2
    cum_square = np.concatenate([np.zeros((square.shape[0], 1)), np.cumsum(square, axis=1)], axis=1)

    feats: dict[str, np.ndarray] = {}
    for w in WINDOWS:
        feats[f"sum_{w}d"] = cum[card, t + 1] - cum[card, t + 1 - w]
        feats[f"days_{w}d"] = cum_active[card, t + 1] - cum_active[card, t + 1 - w]
    for k in (2, 3, 4):
        feats[f"week_minus{k}"] = cum[card, t + 1 - 7 * (k - 1)] - cum[card, t + 1 - 7 * k]

    last = _last_spend_day(panel)[card, t]
    feats["days_since_last"] = np.where(last >= 0, t - last, 999)

    n = feats["days_84d"]
    total = feats["sum_84d"].astype(np.float64)
    mean = np.divide(total, n, out=np.zeros_like(total), where=n > 0)
    sq = cum_square[card, t + 1] - cum_square[card, t + 1 - 84]
    var = np.divide(sq, n, out=np.zeros_like(sq), where=n > 0) - mean ** 2
    feats["mean_active_84d"] = mean
    feats["adi_84d"] = np.divide(84.0, n, out=np.full(len(n), 84.0), where=n > 0)
    feats["cv2_84d"] = np.divide(np.maximum(var, 0), mean ** 2, out=np.zeros_like(mean), where=(n > 1) & (mean > 0))

    if feature_set == "core+calendar":
        days = panel.days
        first = days[t + 1]
        weekend = np.zeros(len(t), dtype=np.int64)
        for k in range(1, horizon + 1):
            weekend += (days[t + k].day_of_week >= 5).astype(np.int64)
        feats["weekend_days_in_horizon"] = weekend
        feats["first_day_of_month"] = first.day.to_numpy()
        feats["days_to_month_end"] = (days[t].days_in_month - days[t].day).to_numpy()
    return pd.DataFrame(feats)
