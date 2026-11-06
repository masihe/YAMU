"""반복 지출 판정 규칙 비교(Task 7.3.7, S15P31E103-295).

모든 규칙은 BE PaymentLivingCostReader처럼 과거만 본다. 키는 같은 카드·가맹점·금액이다.

- CURRENT: 그 결제의 달과 앞 5개 달(6개월) 중 같은 키가 나온 서로 다른 달이 3개 이상이면 반복(BE 지금 규칙)
- ONCE_A_MONTH: CURRENT이면서, 그 6개월 안에 같은 키가 두 번 이상 나온 달이 하나도 없을 때만 반복
- INTERVAL: 바로 앞 같은 키 결제와의 간격, 그 앞 간격이 모두 25~35일이면 반복(매달 한 번씩 세 번째부터)
"""

import numpy as np
import pandas as pd

WINDOW_MONTHS = 6
MIN_MONTHS = 3
INTERVAL_DAYS = (25, 35)
RULES = ("CURRENT", "ONCE_A_MONTH", "INTERVAL")


def flag(check: pd.DataFrame, rule: str) -> pd.Series:
    """check(date·card·store·amount 컬럼)의 행마다 반복 지출로 빼면 True. 인덱스는 check와 같다."""
    if rule not in RULES:
        raise ValueError(f"모르는 규칙: {rule}")
    if rule == "INTERVAL":
        return _interval(check)
    counts = month_counts(check)
    months, multi = _window(counts)
    recurring = months >= MIN_MONTHS
    if rule == "ONCE_A_MONTH":
        recurring &= multi == 0
    per_key_month = pd.Series(recurring, index=counts.index)
    return _rows(check, per_key_month)


def month_counts(check: pd.DataFrame) -> pd.Series:
    """(card, store, amount, month 번호)별 결제 수. 키·달 순으로 정렬돼 있다."""
    month = check["date"].dt.year * 12 + check["date"].dt.month
    return check.assign(month=month).groupby(["card", "store", "amount", "month"]).size().sort_index()


def provisional_once(check: pd.DataFrame) -> pd.DataFrame:
    """ONCE_A_MONTH에서 기준일에 따라 판정이 갈리는 결제와 그 기간.

    그 달 같은 키의 첫 결제이고 다른 조건(6개월 중 3개 달, 다른 달에 두 번 나온 적 없음)은 맞지만 같은 달에 같은 키 결제가
    또 있는 경우다. 서비스(기준일까지만 셈)는 기준일이 [date, until) 안이면 반복으로 빼고, 달 전체로 세는 flag()는 남긴다.
    돌려주는 표: check의 해당 행(date·card·store·amount)과 until(같은 달 두 번째 같은 키 결제일). 같은 날이면 기간이 비어 있다.
    """
    counts = month_counts(check)
    months, multi = _window(counts)
    own_multi = counts.to_numpy() >= 2
    per_key_month = pd.Series((months >= MIN_MONTHS) & own_multi & (multi - own_multi == 0), index=counts.index)
    candidate = _rows(check, per_key_month).astype(bool)
    month = check["date"].dt.year * 12 + check["date"].dt.month
    ordered = check.assign(month=month).sort_values(["card", "store", "amount", "month", "date"], kind="stable")
    rank = ordered.groupby(["card", "store", "amount", "month"]).cumcount()
    second = ordered[rank == 1].set_index(["card", "store", "amount", "month"])["date"]
    first = ordered[(rank == 0) & candidate.loc[ordered.index]]
    until = second.reindex(pd.MultiIndex.from_frame(first[["card", "store", "amount", "month"]])).to_numpy()
    return first[["date", "card", "store", "amount"]].assign(until=until)


def own_month_count(check: pd.DataFrame) -> pd.Series:
    """행마다 같은 키가 그 결제의 달에 나온 수."""
    return _rows(check, month_counts(check))


def _window(counts: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """(키, 달)마다 그 달과 앞 5개 달에 같은 키가 나온 달 수와, 그중 두 번 이상 나온 달 수."""
    keys = counts.index.droplevel("month")
    key_code = pd.factorize(keys)[0].astype(np.int64)
    month = counts.index.get_level_values("month").to_numpy(dtype=np.int64)
    # 키 번호가 같으면 달 순서, 다르면 키 순서로 커지는 값. 정렬돼 있어 searchsorted로 창의 첫 위치를 찾는다
    composite = key_code * 1_000_000 + month
    if np.any(np.diff(composite) <= 0):
        raise ValueError("키·달 순서가 정렬돼 있지 않습니다")
    first = np.searchsorted(composite, composite - (WINDOW_MONTHS - 1), side="left")
    position = np.arange(len(composite))
    months = position - first + 1
    multi_cum = np.concatenate([[0], np.cumsum(counts.to_numpy() >= 2)])
    multi = multi_cum[position + 1] - multi_cum[first]
    return months, multi


def _rows(check: pd.DataFrame, per_key_month: pd.Series) -> pd.Series:
    """(키, 달) 값을 check 행으로 펼친다."""
    month = check["date"].dt.year * 12 + check["date"].dt.month
    index = pd.MultiIndex.from_arrays([check["card"], check["store"], check["amount"], month],
                                      names=["card", "store", "amount", "month"])
    return pd.Series(per_key_month.reindex(index).to_numpy(), index=check.index)


def _interval(check: pd.DataFrame) -> pd.Series:
    """바로 앞 두 간격이 모두 INTERVAL_DAYS 안이면 True. 같은 날 같은 키가 여럿이면 간격 0으로 본다."""
    low, high = INTERVAL_DAYS
    ordered = check.sort_values(["card", "store", "amount", "date"], kind="stable")
    gap = ordered.groupby(["card", "store", "amount"])["date"].diff().dt.days
    previous = gap.groupby([ordered["card"], ordered["store"], ordered["amount"]]).shift(1)
    recurring = gap.between(low, high) & previous.between(low, high)
    return recurring.reindex(check.index)
