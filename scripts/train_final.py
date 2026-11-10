"""최종 모델 학습·저장: HGB_POISSON 또는 TWO_STAGE_TREE(--model), 세 규칙(이상거래·큰 지출·반복 지출) 제외 데이터, 모든 사람·가능한 전 기간.
반복 지출 규칙은 --variant로 고른다: fraud+large+recurring(전체 기간 3개 달, 11/6 이전) 또는
fraud+large+recurring_once(서비스 규칙, S15P31E103-304).

사용: .venv\\Scripts\\python.exe -I scripts\\train_final.py [--horizons 7 14 30] [--eval-runs <runs 폴더 이름>...]
      [--model TWO_STAGE_TREE] [--large-amount 1000000] [--variant fraud+large+recurring_once]
      [--codes credit]  # 신용(00) 생활 소비 모델(S15P31E103-404, 2026-11-10). 기본은 체크
      [--be-large]  # BE 큰 결제 중앙값 10배 규칙에 걸린 결제도 뺌(서비스 조건, 2026-11-11, large_rule)
결과: models/<모델 버전>/h{H}.joblib, h{H}.meta.json (커밋하지 않음)
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from forecast_ml import artifacts, baselines, models
from forecast_ml.backtest import VARIANTS, origins
from forecast_ml.data import CHECK_CODES, CREDIT_CODES, build_panel, load_transactions
from forecast_ml.large_rule import INSTALLMENT, load_ordered, with_median_large_excluded
from forecast_ml.features import make_features

REPO = Path(__file__).resolve().parents[1]
CARD_ZIPS = ("TL_카드거래.zip", "VL_카드거래.zip")
VARIANT = "fraud+large+recurring"
MODELS = ("HGB_POISSON", "TWO_STAGE_TREE")
RECURRING_RULES = {"recurring": "같은 카드·가맹점·금액이 3개 이상의 달",
                   "recurring_once": "그 결제의 달과 앞 5개 달 중 같은 카드·가맹점·금액이 3개 이상의 달에 나오고 "
                                     "그중 두 번 이상 나온 달이 없음(그 결제의 달은 달 전체 건수로 셈)"}
# 고를 수 있는 변형 → 그 변형의 반복 지출 제외 이름
RECURRING_VARIANTS = {"fraud+large+recurring": "recurring", "fraud+large+recurring_once": "recurring_once"}
# --codes → (승인거래코드, 설명 이름, 모델 버전에 붙는 말)
CODES = {"check": (CHECK_CODES, "체크(승인거래코드 10·11)", ""),
         "credit": (CREDIT_CODES, "신용(승인거래코드 00 국내일시불)", "credit-")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO / "data" / "raw" / "aihub-71925")
    parser.add_argument("--models-dir", type=Path, default=REPO / "models")
    parser.add_argument("--horizons", type=int, nargs="+", default=[7, 14, 30])
    parser.add_argument("--eval-runs", nargs="*", default=[], help="이 모델을 평가한 runs 폴더 이름(설명에 남김)")
    parser.add_argument("--large-amount", type=int, default=None, help="큰 지출 기준(원). 없으면 체크 거래 상위 1%%")
    parser.add_argument("--model", default="HGB_POISSON", choices=MODELS)
    parser.add_argument("--variant", default=VARIANT, choices=list(RECURRING_VARIANTS))
    parser.add_argument("--codes", default="check", choices=list(CODES))
    parser.add_argument("--be-large", action="store_true", help="BE 큰 결제 중앙값 10배 규칙에 걸린 결제도 뺀다")
    args = parser.parse_args()
    codes, kind, prefix = CODES[args.codes]
    zips = [args.data_dir / name for name in CARD_ZIPS]
    variant = args.variant if args.large_amount is None else args.variant.replace("large", f"large{args.large_amount}")
    if args.be_large:
        variant += "+bemedian"
    version = f"{args.model.lower().replace('_', '-')}-{prefix}{datetime.now():%Y%m%d}-{variant.replace('+', '-')}"
    out_dir = args.models_dir / version
    if out_dir.exists():
        raise SystemExit(f"이미 있습니다: {out_dir}")
    df = (with_median_large_excluded(load_ordered(zips), CHECK_CODES | CREDIT_CODES | {INSTALLMENT}) if args.be_large
          else load_transactions(zips))
    panel = build_panel(df, VARIANTS[args.variant], large_amount=args.large_amount, codes=codes)
    kind_short = kind.split("(")[0]
    large_rule = (f"{kind_short} 거래 금액 99% 분위수 초과" if args.large_amount is None
                  else f"{kind_short} 거래 한 건 {args.large_amount:,}원 초과")
    for horizon in args.horizons:
        card, t = origins(panel, horizon)
        X = make_features(panel, card, t, horizon, "core")
        y = baselines.actual(panel, card, t, horizon).astype(np.float64)
        model = models.build(args.model, {}).fit(X, y)
        trees = ({"classifier": int(model.classifier.n_iter_), "regressor": int(model.regressor.n_iter_)}
                 if isinstance(model, models.TwoStage) else int(model.n_iter_))
        meta = {"model_version": version, "model": args.model, "variant": variant,
                "exclusions": {"fraud": "이상거래여부 1", "large": large_rule,
                               "recurring": RECURRING_RULES[RECURRING_VARIANTS[args.variant]],
                               **({"be_median_large": "사용자(카드KEY)의 모든 체크·신용 결제에서 직전 3개월 LIVING 5건 이상이고 그 중앙값의 10배 이상(BE SpendingClassifier, 승인 시각 대신 날짜·시간대·순번 순)"} if args.be_large else {})},
                "target": f"다음 {horizon}일 {kind} 소비 합계(원)",
                "training": {"rows": int(len(y)), "people": int(len(np.unique(card))),
                             "from": str(panel.days[t.min()].date()), "to": str(panel.days[t.max()].date()),
                             "trees": trees},
                "evaluated_in": args.eval_runs, "trained_at": datetime.now().isoformat(timespec="seconds")}
        artifacts.save(out_dir, horizon, model, list(X.columns), meta)
        print(horizon, "rows", len(y), "trees", trees)
    print(out_dir)


if __name__ == "__main__":
    main()
