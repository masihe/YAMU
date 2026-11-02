"""승인거래코드별로 같은 카드·가맹점·금액의 정상(00) 거래와 짝이 맞는지, 카드별 월 거래 밀도를 본다(Task 7.3.1).

원본 zip은 수정하지 않는다. 실행은 scripts/eda.py.
"""

import zipfile
from pathlib import Path

import pandas as pd

COLUMNS = {0: "base_month", 1: "date", 4: "card", 5: "amount", 10: "fraud", 15: "approval_code",
           17: "category", 26: "plan", 30: "store"}


def load(zip_paths: list[Path]) -> pd.DataFrame:
    """zip 안 CSV에서 필요한 컬럼만 문자열로 읽어 합친다."""
    frames = []
    for zp in zip_paths:
        with zipfile.ZipFile(zp) as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                with z.open(info) as raw:
                    df = pd.read_csv(raw, encoding="utf-8-sig", usecols=list(COLUMNS), header=0, dtype=str)
                df.columns = [COLUMNS[i] for i in sorted(COLUMNS)]
                frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["amount"] = df["amount"].astype("int64")
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return df


def pairing(df: pd.DataFrame) -> dict:
    """코드별로 같은 카드·가맹점·금액의 00 거래가 있는 비율과, 있으면 며칠 차이인지."""
    normal = df[df["approval_code"] == "00"][["card", "store", "amount", "date"]]
    result = {}
    for code, part in df[df["approval_code"] != "00"].groupby("approval_code"):
        merged = part[["card", "store", "amount", "date"]].reset_index().merge(
            normal, on=["card", "store", "amount"], how="left", suffixes=("", "_normal"))
        merged["gap"] = (merged["date"] - merged["date_normal"]).dt.days
        # 짝이 여러 개면 날짜가 가장 가까운 것 하나만
        merged["abs_gap"] = merged["gap"].abs()
        best = merged.sort_values("abs_gap").drop_duplicates("index")
        matched = best["gap"].notna()
        result[code] = {
            "rows": int(len(part)),
            "matched_ratio": round(float(matched.mean()), 4),
            "gap_days_quantiles": {str(q): float(best.loc[matched, "gap"].quantile(q)) for q in (0.1, 0.5, 0.9)}
            if matched.any() else None,
            "amount_median": int(part["amount"].median()),
            "plan_values": part["plan"].value_counts().head(5).to_dict(),
            "base_month_equals_date_month": round(float((part["base_month"] == part["date"].dt.strftime("%Y%m")).mean()), 4),
        }
    return result


def density(df: pd.DataFrame) -> dict:
    """카드별 활동한 달의 거래 건수·금액 분포(정상 00만)."""
    normal = df[df["approval_code"] == "00"]
    monthly = normal.groupby(["card", normal["date"].dt.to_period("M")]).agg(n=("amount", "size"), total=("amount", "sum"))
    return {
        "card_months": int(len(monthly)),
        "tx_per_active_month_quantiles": {str(q): float(monthly["n"].quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
        "amount_per_active_month_quantiles": {str(q): float(monthly["total"].quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
    }
