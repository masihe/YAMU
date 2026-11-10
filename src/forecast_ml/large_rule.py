"""BE SpendingClassifier의 큰 결제(LARGE) 규칙을 AI-Hub 거래에 적용한다(S15P31E103-404 Codex 검토, 2026-11-10).

BE 규칙(be/.../SpendingClassifier.java): 할부(2개월 이상)이거나 한 건 100만 원 초과면 LARGE. 아니면 사용자의 모든 카드에서
그 결제 직전 3개월(승인 시각 기준)의 LIVING 결제가 5건 이상이고, 그 건당 중앙값(짝수면 가운데 둘 평균)의 10배 이상이면 LARGE.
LARGE로 정한 결제는 표본에 넣지 않는다.

AI-Hub 근사: 사람 = 카드KEY(체크·신용 결제 모두 표본). 승인 시각이 없어 (날짜, 승인시간대, 승인SEQ) 순서로 "직전"을 정하고,
3개월 창은 날짜로 자른다. 이상거래는 부르는 쪽이 먼저 뺀다(서비스에 없는 개념).
"""

import bisect
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = {1: "date", 2: "seq", 3: "hour", 4: "card", 5: "amount", 10: "fraud", 15: "code", 30: "store"}
LARGE_AMOUNT = 1_000_000
MIN_SAMPLES = 5
MULTIPLE = 10
INSTALLMENT = "01"


def load_ordered(zips: list[Path]) -> pd.DataFrame:
    """date·seq·hour·card·amount·fraud·code·store 컬럼으로 읽는다. 원본은 바꾸지 않는다."""
    frames = []
    for path in zips:
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                with archive.open(info) as raw:
                    frame = pd.read_csv(raw, encoding="utf-8-sig", usecols=list(COLUMNS), dtype=str)
                frame.columns = [COLUMNS[i] for i in sorted(COLUMNS)]
                frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    df["amount"] = df["amount"].astype("int64")
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df["hour"] = pd.to_numeric(df["hour"], errors="raise").astype("int64")
    df["seq"] = pd.to_numeric(df["seq"], errors="raise").astype("int64")
    return df


def median_large(df: pd.DataFrame) -> pd.Series:
    """행마다 BE 중앙값 규칙으로 LARGE가 되는지(True). 할부·100만 원 초과는 규칙 1로 LARGE라 여기서는 False로 두고 표본에서 뺀다."""
    out = pd.Series(False, index=df.index)
    ordered = df.sort_values(["card", "date", "hour", "seq"], kind="stable")
    rule1 = (ordered["code"] == INSTALLMENT) | (ordered["amount"] > LARGE_AMOUNT)
    for _, part in ordered.groupby("card", sort=False):
        dates = part["date"].to_numpy()
        amounts = part["amount"].to_numpy()
        first_rule = rule1.loc[part.index].to_numpy()
        window_dates: list = []   # 표본(LIVING)의 날짜, 시간순
        window_amounts: list = []  # 같은 순서의 금액
        sorted_amounts: list = []  # 창 안 금액 정렬본
        start = 0
        flags = np.zeros(len(part), dtype=bool)
        for i in range(len(part)):
            if first_rule[i]:
                continue
            limit = (pd.Timestamp(dates[i]) - pd.DateOffset(months=3)).to_datetime64()
            while start < len(window_dates) and window_dates[start] < limit:
                del sorted_amounts[bisect.bisect_left(sorted_amounts, window_amounts[start])]
                start += 1
            n = len(sorted_amounts)
            large = False
            if n >= MIN_SAMPLES:
                mid = n // 2
                median = sorted_amounts[mid] if n % 2 else (sorted_amounts[mid - 1] + sorted_amounts[mid]) / 2
                large = median > 0 and amounts[i] >= MULTIPLE * median
            flags[i] = large
            if not large:
                window_dates.append(dates[i])
                window_amounts.append(amounts[i])
                bisect.insort(sorted_amounts, amounts[i])
        out.loc[part.index] = flags
    return out


def with_median_large_excluded(df: pd.DataFrame, codes: frozenset[str]) -> pd.DataFrame:
    """median_large에 걸린 결제를 이상거래(fraud "1")로 바꾼 복사본. codes(체크·신용·할부)와 이상거래가 아닌 결제만 판정한다.

    행을 지우지 않아 확인된 달(build_panel observed)과 반복 지출 판정(모든 결제로 셈)은 그대로다. 제외 규칙에 fraud가 있어야 빠진다.
    """
    base = df[(df["fraud"] != "1") & df["code"].isin(codes)]
    flagged = median_large(base)
    out = df.copy()
    out.loc[flagged[flagged].index, "fraud"] = "1"
    return out
