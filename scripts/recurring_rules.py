"""반복 지출 판정 규칙 측정(Task 7.3.7, S15P31E103-295, 2026-11-05).

AI-Hub 체크 거래(이상거래·한 건 100만 원 초과 제외 = 서비스 생활비 입력과 같은 조건)에서 규칙마다 빠지는 금액·건수를 잰다.
합성 데이터라 무엇이 진짜 구독인지 정답이 없다. 규칙이 얼마나 빼는지와, 지금 규칙이 빼는 것 중 "한 달에 여러 번 나오는
같은 가격 소비"가 얼마인지를 본다. 6개월 창이 다 차도록 평가는 2021-06-01 이후 거래만 한다(앞 달은 창 계산에만 씀).

사용: .venv\\Scripts\\python.exe -I scripts\\recurring_rules.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_ml import recurring
from forecast_ml.data import CHECK_CODES, load_transactions
from forecast_ml.runs import new_run, write_json

REPO = Path(__file__).resolve().parents[1]
CARD_ZIPS = ("TL_카드거래.zip", "VL_카드거래.zip")
LARGE_AMOUNT = 1_000_000
EVAL_FROM = "2021-06-01"


def share(part: float, whole: float) -> float:
    return round(float(part / whole), 4) if whole else 0.0


def amount_quantiles(amounts: pd.Series) -> dict:
    if amounts.empty:
        return {}
    return {f"p{int(q * 100)}": int(amounts.quantile(q)) for q in (0.1, 0.5, 0.9)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=REPO / "data" / "raw" / "aihub-71925")
    parser.add_argument("--runs-dir", type=Path, default=REPO / "runs")
    args = parser.parse_args()
    zips = [args.data_dir / name for name in CARD_ZIPS]
    config = {"inputs": [str(z) for z in zips], "large_amount": LARGE_AMOUNT, "eval_from": EVAL_FROM,
              "window_months": recurring.WINDOW_MONTHS, "min_months": recurring.MIN_MONTHS,
              "interval_days": list(recurring.INTERVAL_DAYS), "rules": list(recurring.RULES)}
    run_dir = new_run(args.runs_dir, "recurring", config, REPO)

    df = load_transactions(zips)
    check = df[df["code"].isin(CHECK_CODES) & (df["fraud"] != "1") & (df["amount"] <= LARGE_AMOUNT)]
    check = check[["date", "card", "store", "amount"]].copy()
    if check["store"].isna().any():
        raise ValueError("가맹점이 빈 체크 거래가 있습니다")

    flags = {rule: recurring.flag(check, rule) for rule in recurring.RULES}
    own_month = recurring.own_month_count(check)
    in_eval = check["date"] >= EVAL_FROM
    ev = check[in_eval]
    total, rows = int(ev["amount"].sum()), int(len(ev))
    frequent = own_month[in_eval] >= 2

    result = {"rows": rows, "amount": total, "cards": int(ev["card"].nunique()), "rules": {}}
    for rule, flagged in flags.items():
        f = flagged[in_eval]
        removed = ev[f]
        per_card = (removed.groupby("card")["amount"].sum() / ev.groupby("card")["amount"].sum()).fillna(0)
        per_card = per_card.reindex(ev["card"].unique(), fill_value=0)
        result["rules"][rule] = {
            "removed_rows": int(f.sum()), "removed_rows_share": share(f.sum(), rows),
            "removed_amount_share": share(removed["amount"].sum(), total),
            "frequent_share_of_removed_amount": share(removed.loc[frequent[f], "amount"].sum(),
                                                       removed["amount"].sum()),
            "removed_amount_quantiles": amount_quantiles(removed["amount"]),
            "per_card_removed_share": {"p50": round(float(per_card.quantile(0.5)), 4),
                                       "p90": round(float(per_card.quantile(0.9)), 4),
                                       "max": round(float(per_card.max()), 4),
                                       "cards_with_any": int((per_card > 0).sum())},
        }

    current = flags["CURRENT"][in_eval]
    overlap = {}
    for rule in ("ONCE_A_MONTH", "INTERVAL"):
        other = flags[rule][in_eval]
        overlap[rule] = {
            "current_removed_kept_amount_share": share(ev.loc[current & ~other, "amount"].sum(),
                                                       ev.loc[current, "amount"].sum()),
            "removed_but_not_by_current_amount": int(ev.loc[other & ~current, "amount"].sum()),
        }
    once, interval = flags["ONCE_A_MONTH"][in_eval], flags["INTERVAL"][in_eval]
    overlap["ONCE_A_MONTH_vs_INTERVAL"] = {
        "both_amount_share": share(ev.loc[once & interval, "amount"].sum(), total),
        "only_once_amount_share": share(ev.loc[once & ~interval, "amount"].sum(), total),
        "only_interval_amount_share": share(ev.loc[interval & ~once, "amount"].sum(), total),
    }
    result["overlap"] = overlap
    # 지금 규칙이 빼는 것 중 자주(그 달 2번 이상) 나온 같은 가격 소비의 금액대
    result["current_frequent_removed_quantiles"] = amount_quantiles(ev.loc[current & frequent, "amount"])
    result["current_once_removed_quantiles"] = amount_quantiles(ev.loc[current & ~frequent, "amount"])
    # 학습 때 규칙(전체 기간에서 서로 다른 3개 달)과 비교용
    months_full = check.assign(m=check["date"].dt.to_period("M")).groupby(["card", "store", "amount"])["m"] \
        .transform("nunique")
    result["training_rule_removed_amount_share"] = share(ev.loc[months_full[in_eval] >= 3, "amount"].sum(), total)

    write_json(run_dir / "metrics.json", result)
    print(run_dir)
    print(pd.DataFrame({r: {k: v for k, v in m.items() if not isinstance(v, dict)}
                        for r, m in result["rules"].items()}).T.to_string())
    print({k: result[k] for k in ("rows", "amount", "cards", "training_rule_removed_amount_share")})
    print(overlap)


if __name__ == "__main__":
    main()
