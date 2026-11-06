"""최종 모델 학습·저장: HGB_POISSON 또는 TWO_STAGE_TREE(--model), 세 규칙(이상거래·큰 지출·반복 지출) 제외 데이터, 모든 사람·가능한 전 기간.
반복 지출 규칙은 --variant로 고른다: fraud+large+recurring(전체 기간 3개 달, 11/6 이전) 또는
fraud+large+recurring_once(서비스 규칙, S15P31E103-304).

사용: .venv\\Scripts\\python.exe -I scripts\\train_final.py [--horizons 7 14 30] [--eval-runs <runs 폴더 이름>...]
      [--model TWO_STAGE_TREE] [--large-amount 1000000] [--variant fraud+large+recurring_once]
결과: models/<모델 버전>/h{H}.joblib, h{H}.meta.json (커밋하지 않음)
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from forecast_ml import artifacts, baselines, models
from forecast_ml.backtest import VARIANTS, origins
from forecast_ml.data import build_panel, load_transactions
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO / "data" / "raw" / "aihub-71925")
    parser.add_argument("--models-dir", type=Path, default=REPO / "models")
    parser.add_argument("--horizons", type=int, nargs="+", default=[7, 14, 30])
    parser.add_argument("--eval-runs", nargs="*", default=[], help="이 모델을 평가한 runs 폴더 이름(설명에 남김)")
    parser.add_argument("--large-amount", type=int, default=None, help="큰 지출 기준(원). 없으면 체크 거래 상위 1%%")
    parser.add_argument("--model", default="HGB_POISSON", choices=MODELS)
    parser.add_argument("--variant", default=VARIANT, choices=list(RECURRING_VARIANTS))
    args = parser.parse_args()
    zips = [args.data_dir / name for name in CARD_ZIPS]
    variant = args.variant if args.large_amount is None else args.variant.replace("large", f"large{args.large_amount}")
    version = f"{args.model.lower().replace('_', '-')}-{datetime.now():%Y%m%d}-{variant.replace('+', '-')}"
    out_dir = args.models_dir / version
    if out_dir.exists():
        raise SystemExit(f"이미 있습니다: {out_dir}")
    panel = build_panel(load_transactions(zips), VARIANTS[args.variant], large_amount=args.large_amount)
    large_rule = ("체크 거래 금액 99% 분위수 초과" if args.large_amount is None
                  else f"체크 거래 한 건 {args.large_amount:,}원 초과")
    for horizon in args.horizons:
        card, t = origins(panel, horizon)
        X = make_features(panel, card, t, horizon, "core")
        y = baselines.actual(panel, card, t, horizon).astype(np.float64)
        model = models.build(args.model, {}).fit(X, y)
        trees = ({"classifier": int(model.classifier.n_iter_), "regressor": int(model.regressor.n_iter_)}
                 if isinstance(model, models.TwoStage) else int(model.n_iter_))
        meta = {"model_version": version, "model": args.model, "variant": variant,
                "exclusions": {"fraud": "이상거래여부 1", "large": large_rule,
                               "recurring": RECURRING_RULES[RECURRING_VARIANTS[args.variant]]},
                "target": f"다음 {horizon}일 체크(승인거래코드 10·11) 소비 합계(원)",
                "training": {"rows": int(len(y)), "people": int(len(np.unique(card))),
                             "from": str(panel.days[t.min()].date()), "to": str(panel.days[t.max()].date()),
                             "trees": trees},
                "evaluated_in": args.eval_runs, "trained_at": datetime.now().isoformat(timespec="seconds")}
        artifacts.save(out_dir, horizon, model, list(X.columns), meta)
        print(horizon, "rows", len(y), "trees", trees)
    print(out_dir)


if __name__ == "__main__":
    main()
