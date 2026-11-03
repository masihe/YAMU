"""AI-Hub 카드거래 zip을 읽어 카드KEY × 날짜 체크 소비 행렬을 만든다."""

import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# 활용 가이드라인 31쪽: 10 체크계좌승인, 11 체크신용승인
CHECK_CODES = frozenset({"10", "11"})
EXCLUSIONS = frozenset({"fraud", "large", "recurring"})
_COLUMNS = {1: "date", 4: "card", 5: "amount", 10: "fraud", 15: "code", 30: "store"}


def load_transactions(zip_paths: list[Path]) -> pd.DataFrame:
    """zip 안 분기별 CSV를 모두 읽어 date·card·amount·fraud·code·store 컬럼으로 합친다. 원본은 바꾸지 않는다."""
    frames = []
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                with archive.open(info) as raw:
                    frame = pd.read_csv(raw, encoding="utf-8-sig", usecols=list(_COLUMNS), dtype=str)
                frame.columns = [_COLUMNS[i] for i in sorted(_COLUMNS)]
                frames.append(frame)
    if not frames:
        raise ValueError("읽은 CSV가 없습니다")
    df = pd.concat(frames, ignore_index=True)
    if df["amount"].isna().any() or not df["amount"].str.fullmatch(r"\d+").all():
        raise ValueError("통합승인금액에 0 이상 정수가 아닌 값이 있습니다")
    df["amount"] = df["amount"].astype("int64")
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return df


@dataclass(frozen=True)
class Panel:
    """카드KEY별 일별 체크 소비와 확인된 달.

    spend[c, d]: 카드 c의 날짜 d 체크 소비 합계(원, 제외 규칙 적용 뒤)
    observed[c, m]: 카드 c가 달 m에 거래(신용·체크 모두)가 하나라도 있으면 True. 체크 0원도 0원으로 확인된 달로 본다
    """

    cards: np.ndarray
    days: pd.DatetimeIndex
    months: pd.PeriodIndex
    spend: np.ndarray
    observed: np.ndarray
    removed_share: float

    def month_of_day(self) -> np.ndarray:
        """날짜 인덱스 → 달 인덱스."""
        return np.searchsorted(self.months.start_time, self.days, side="right") - 1


def check_rows(df: pd.DataFrame, exclude: frozenset[str], large_quantile: float = 0.99,
               recurring_months: int = 3, large_amount: int | None = None) -> tuple[pd.DataFrame, float]:
    """체크 거래만 남기고 제외 규칙을 적용한다. 돌려주는 비율은 제외로 빠진 체크 금액의 비중이다.

    - fraud: 이상거래여부 1
    - large: 체크 거래 금액 분위수(large_quantile) 초과. large_amount(원)를 주면 분위수 대신 그 금액 초과
    - recurring: 같은 카드·가맹점·금액이 서로 다른 recurring_months개 이상의 달에 나타남
    """
    unknown = set(exclude) - EXCLUSIONS
    if unknown:
        raise ValueError(f"모르는 제외 규칙: {sorted(unknown)}")
    check = df[df["code"].isin(CHECK_CODES)].copy()
    keep = pd.Series(True, index=check.index)
    if "fraud" in exclude:
        keep &= check["fraud"] != "1"
    if "large" in exclude:
        limit = check["amount"].quantile(large_quantile) if large_amount is None else large_amount
        keep &= check["amount"] <= limit
    if "recurring" in exclude:
        month = check["date"].dt.to_period("M")
        months = check.assign(month=month).groupby(["card", "store", "amount"])["month"].transform("nunique")
        keep &= months < recurring_months
    total = check["amount"].sum()
    removed = check.loc[~keep, "amount"].sum()
    return check[keep], float(removed / total) if total else 0.0


def build_panel(df: pd.DataFrame, exclude: frozenset[str] = frozenset(), large_quantile: float = 0.99,
                recurring_months: int = 3, large_amount: int | None = None) -> Panel:
    """체크를 한 번이라도 쓴 카드KEY만 모아 일별 체크 소비 행렬과 확인된 달을 만든다."""
    check, removed_share = check_rows(df, exclude, large_quantile, recurring_months, large_amount)
    cards = np.array(sorted(df.loc[df["code"].isin(CHECK_CODES), "card"].unique()))
    days = pd.date_range(df["date"].min().to_period("M").start_time, df["date"].max().to_period("M").end_time.normalize(),
                         freq="D")
    months = pd.period_range(days[0], days[-1], freq="M")
    card_index = pd.Index(cards)

    spend = np.zeros((len(cards), len(days)), dtype=np.int64)
    daily = check.groupby(["card", "date"])["amount"].sum()
    rows = card_index.get_indexer(daily.index.get_level_values(0))
    cols = days.get_indexer(daily.index.get_level_values(1))
    spend[rows, cols] = daily.to_numpy()

    observed = np.zeros((len(cards), len(months)), dtype=bool)
    active = df[df["card"].isin(card_index)]
    pairs = active.groupby(["card", active["date"].dt.to_period("M")]).size().index
    observed[card_index.get_indexer(pairs.get_level_values(0)), months.get_indexer(pairs.get_level_values(1))] = True
    return Panel(cards, days, months, spend, observed, removed_share)
