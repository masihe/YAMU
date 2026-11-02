"""카드거래 합성데이터에서 카드KEY별 이력이 여러 달 이어지는지 집계한다(Task 7.3.1).

zip을 풀지 않고 표준 라이브러리로 스트리밍해서 읽는다. 원본은 수정하지 않는다. 실행은 scripts/eda.py.
"""

import csv
import io
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

EXPECTED_COLUMNS = 32
COL_MONTH, COL_DATE, COL_CARD, COL_AMOUNT = 0, 1, 4, 5
COL_FRAUD, COL_APPROVAL_CODE, COL_CATEGORY = 10, 15, 17


def month_index(yyyymm: str) -> int:
    """'202104' → 연·월을 이어 붙인 월 번호."""
    return int(yyyymm[:4]) * 12 + int(yyyymm[4:6]) - 1


def longest_run(months: set[int]) -> int:
    """연속된 월 번호의 가장 긴 길이."""
    best = run = 0
    prev = None
    for m in sorted(months):
        run = run + 1 if prev is not None and m == prev + 1 else 1
        best = max(best, run)
        prev = m
    return best


def scan(zip_paths: list[Path]) -> dict:
    """zip 안의 모든 CSV를 읽어 카드별 월·건수·금액과 전체 분포를 모은다."""
    months_by_card: dict[str, set[int]] = defaultdict(set)
    rows_by_card: Counter = Counter()
    split_by_card: dict[str, set[str]] = defaultdict(set)
    rows = bad_rows = fraud_rows = 0
    approval_codes: Counter = Counter()
    categories: Counter = Counter()
    month_rows: Counter = Counter()
    amounts: list[int] = []

    for zp in zip_paths:
        split = zp.stem  # TL_카드거래 / VL_카드거래
        with zipfile.ZipFile(zp) as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                with z.open(info) as raw:
                    reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
                    header = next(reader)
                    if len(header) != EXPECTED_COLUMNS:
                        raise ValueError(f"{info.filename}: 컬럼 {len(header)}개, {EXPECTED_COLUMNS}개 기대")
                    for row in reader:
                        if len(row) != EXPECTED_COLUMNS or not row[COL_AMOUNT].lstrip("-").isdigit():
                            bad_rows += 1
                            continue
                        rows += 1
                        card = row[COL_CARD]
                        date = row[COL_DATE]
                        months_by_card[card].add(month_index(date[:6]))
                        rows_by_card[card] += 1
                        split_by_card[card].add(split)
                        month_rows[date[:6]] += 1
                        approval_codes[row[COL_APPROVAL_CODE]] += 1
                        categories[row[COL_CATEGORY]] += 1
                        if row[COL_FRAUD] == "1":
                            fraud_rows += 1
                        if rows % 20 == 0:  # 분위수용 표본(5%)
                            amounts.append(int(row[COL_AMOUNT]))

    runs = Counter(longest_run(m) for m in months_by_card.values())
    active = Counter(len(m) for m in months_by_card.values())
    per_card_rows = sorted(rows_by_card.values())
    amounts.sort()

    def q(values: list[int], p: float) -> int:
        return values[min(len(values) - 1, int(p * len(values)))] if values else 0

    cards = len(months_by_card)
    return {
        "rows": rows,
        "bad_rows": bad_rows,
        "fraud_rows": fraud_rows,
        "cards": cards,
        "cards_in_both_splits": sum(1 for s in split_by_card.values() if len(s) > 1),
        "rows_per_card_quantiles": {p: q(per_card_rows, p) for p in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)},
        "active_months_hist": dict(sorted(active.items())),
        "longest_consecutive_months_hist": dict(sorted(runs.items())),
        "cards_with_run_ge_4": sum(n for r, n in runs.items() if r >= 4),
        "cards_with_run_ge_6": sum(n for r, n in runs.items() if r >= 6),
        "rows_by_month": dict(sorted(month_rows.items())),
        "approval_codes": dict(approval_codes.most_common()),
        "top_categories": dict(categories.most_common(30)),
        "category_count": len(categories),
        "amount_quantiles_sampled": {p: q(amounts, p) for p in (0.01, 0.1, 0.5, 0.9, 0.99)},
    }
