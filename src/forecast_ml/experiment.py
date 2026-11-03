"""7.3.3 실험: 설정값 고르기(학습 사람 × 검증 기간) → 다시 학습(학습 사람 × 2023년까지) → 평가(평가 사람 × 2024년) 한 번.

통과 기준(2026-11-02 결정): 편향이 실제 평균의 ±10% 안이고, MAE가 LAST_7D보다 낮다.
설정값도 같은 순서(편향 ±10% 통과 → MAE 최소, 통과가 없으면 편향 절댓값 최소)로 고른다.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecast_ml import baselines, intermittent, models
from forecast_ml.backtest import BASELINES, is_test_card, metrics, origins
from forecast_ml.data import Panel
from forecast_ml.features import make_features

BIAS_LIMIT = 0.10
GATE_BASELINE = "LAST_7D"
SBA_GRID = [{"alpha": a} for a in (0.05, 0.1, 0.2, 0.3)]
TSB_WINDOW = 90  # 서비스가 받는 기록 길이
BLEND_GRID = [{"w": round(w / 10, 1)} for w in range(11)]
TSB_GRID = [{"alpha": a, "beta": b} for a in (0.05, 0.1, 0.2, 0.3) for b in (0.02, 0.05, 0.1, 0.2)]


@dataclass(frozen=True)
class Split:
    """같은 (카드, 기준일) 목록 위의 구간 표시."""

    card: np.ndarray
    t: np.ndarray
    y: np.ndarray
    fit: np.ndarray
    val: np.ndarray
    train: np.ndarray
    test: np.ndarray


def split(panel: Panel, horizon: int, val_from: pd.Timestamp, test_from: pd.Timestamp,
          test_to: pd.Timestamp | None = None, bucket: int = 0) -> Split:
    """정답 기간이 다음 구간과 겹치는 기준일(경계 앞 horizon일)은 앞 구간에서 뺀다.

    평가: bucket번 묶음 사람 × test_from <= 기준일 < test_to(없으면 끝까지). 학습·검증: 나머지 사람 × test_from 이전.
    """
    card, t = origins(panel, horizon)
    y = baselines.actual(panel, card, t, horizon)
    date = panel.days[t]
    gap = pd.Timedelta(days=horizon)
    test_card = np.array([is_test_card(c, bucket=bucket) for c in panel.cards])[card]
    train = ~test_card & (date < test_from - gap)
    in_test_period = (date >= test_from) & (date < test_to if test_to is not None else True)
    return Split(card, t, y,
                 fit=train & (date < val_from - gap),
                 val=~test_card & (date >= val_from) & (date < test_from - gap),
                 train=train,
                 test=test_card & in_test_period)


def choose(scores: list[tuple[dict, dict]]) -> tuple[dict, dict]:
    """(설정값, 검증 지표) 중 편향 ±10% 안에서 MAE 최소. 없으면 편향 절댓값 최소."""
    ok = [s for s in scores if s[1]["rel_bias"] is not None and abs(s[1]["rel_bias"]) <= BIAS_LIMIT]
    if ok:
        return min(ok, key=lambda s: s[1]["MAE"])
    return min(scores, key=lambda s: abs(s[1]["rel_bias"] or 0))


def passes(test_metrics: dict, gate_mae: float) -> bool:
    """통과 기준: 편향 ±10% 안이고 LAST_7D보다 MAE가 낮다."""
    return abs(test_metrics["rel_bias"]) <= BIAS_LIMIT and test_metrics["MAE"] < gate_mae


def run(panel: Panel, horizon: int, val_from: pd.Timestamp, test_from: pd.Timestamp, feature_set: str,
        test_to: pd.Timestamp | None = None, bucket: int = 0, keep_predictions: bool = False) -> dict:
    """Baseline 7종(+ TSB_90D)과 모델 6종(+ STACK_HGB·BLEND_HGB_TSB·H30_SCALED)의 검증 선택 결과와 평가 지표.

    keep_predictions면 결과마다 "pred"(평가 구간 예측)와 블록에 "test_card"·"test_y"를 더한다(JSON에 쓰지 않는 numpy 값).
    """
    s = split(panel, horizon, val_from, test_from, test_to, bucket)
    results: dict[str, dict] = {}

    for name, predict in BASELINES.items():
        pred = predict(panel, s.card, s.t, horizon)
        results[name] = {"kind": "baseline", "params": {}, "test": metrics(pred[s.test], s.y[s.test]),
                         "pred": pred[s.test]}

    for name, grid, daily in (("SBA", SBA_GRID, lambda p: intermittent.sba_daily(panel.spend, **p)),
                              ("TSB", TSB_GRID, lambda p: intermittent.tsb_daily(panel.spend, **p))):
        scores = []
        cache = {}
        for params in grid:
            cache[tuple(params.items())] = d = daily(params)
            pred = intermittent.forecast(d, s.card, s.t, horizon)
            scores.append((params, metrics(pred[s.val], s.y[s.val])))
        params, val = choose(scores)
        pred = intermittent.forecast(cache[tuple(params.items())], s.card, s.t, horizon)
        results[name] = {"kind": "baseline", "params": params, "val": val, "test": metrics(pred[s.test], s.y[s.test]),
                         "pred": pred[s.test]}

    # 서비스와 같은 조건의 TSB(최근 90일만). 2026-11-03 사전 등록의 판정 기준선
    scores = []
    cache = {}
    for params in TSB_GRID:
        cache[tuple(params.items())] = d = intermittent.tsb_window(panel.spend, s.card, s.t, window=TSB_WINDOW, **params)
        scores.append((params, metrics(np.floor(d * horizon + 0.5)[s.val], s.y[s.val])))
    params, val = choose(scores)
    tsb90 = np.floor(cache[tuple(params.items())] * horizon + 0.5).astype(np.int64)
    results["TSB_90D"] = {"kind": "baseline", "params": params, "val": val, "test": metrics(tsb90[s.test], s.y[s.test]),
                          "pred": tsb90[s.test]}

    X = make_features(panel, s.card, s.t, horizon, feature_set)
    y = s.y.astype(np.float64)
    hgb_val = hgb_test = None
    for name, grid in models.GRIDS.items():
        scores = []
        for params in grid:
            model = models.build(name, params).fit(X[s.fit], y[s.fit])
            val_pred = model.predict(X[s.val])
            scores.append((params, metrics(val_pred, s.y[s.val])))
        params, val = choose(scores)
        model = models.build(name, params).fit(X[s.train], y[s.train])
        pred = model.predict(X[s.test])
        results[name] = {"kind": "model", "params": params, "val": val, "test": metrics(pred, s.y[s.test]), "pred": pred}
        if name == "HGB_POISSON":
            hgb_val, hgb_test = val_pred, pred

    # STACK_HGB: 단순 방식 예측 3개를 특징으로 더한 HGB_POISSON
    stacked = X.assign(tsb90_pred=tsb90, last7_pred=baselines.last_week(panel, s.card, s.t, horizon),
                       avg28_pred=baselines.avg_recent(panel, s.card, s.t, horizon))
    model = models.build("HGB_POISSON", {}).fit(stacked[s.fit], y[s.fit])
    val = metrics(model.predict(stacked[s.val]), s.y[s.val])
    model = models.build("HGB_POISSON", {}).fit(stacked[s.train], y[s.train])
    pred = model.predict(stacked[s.test])
    results["STACK_HGB"] = {"kind": "model", "params": {}, "val": val, "test": metrics(pred, s.y[s.test]), "pred": pred}

    # BLEND_HGB_TSB: w × HGB_POISSON + (1 − w) × TSB_90D, w는 검증 구간에서
    scores = [(p, metrics(p["w"] * hgb_val + (1 - p["w"]) * tsb90[s.val], s.y[s.val])) for p in BLEND_GRID]
    params, val = choose(scores)
    pred = params["w"] * hgb_test + (1 - params["w"]) * tsb90[s.test]
    results["BLEND_HGB_TSB"] = {"kind": "model", "params": params, "val": val, "test": metrics(pred, s.y[s.test]),
                                "pred": pred}

    # H30_SCALED: 30일 정답으로 학습한 HGB_POISSON × horizon/30. core 특징은 horizon과 상관없다
    if horizon == 30:
        pred = hgb_test
    else:
        if feature_set != "core":
            raise ValueError("H30_SCALED는 core 특징에서만 쓴다")
        s30 = split(panel, 30, val_from, test_from, test_to, bucket)
        X30 = make_features(panel, s30.card, s30.t, 30, feature_set)
        model = models.build("HGB_POISSON", {}).fit(X30[s30.train], s30.y[s30.train].astype(np.float64))
        pred = model.predict(X[s.test]) * horizon / 30
    results["H30_SCALED"] = {"kind": "model", "params": {}, "test": metrics(pred, s.y[s.test]), "pred": pred}

    gate_mae = results[GATE_BASELINE]["test"]["MAE"]
    for r in results.values():
        r["passes_gate"] = passes(r["test"], gate_mae)
        if not keep_predictions:
            del r["pred"]
    sizes = {k: int(getattr(s, k).sum()) for k in ("fit", "val", "train", "test")}
    block = {"rows": sizes, "features": list(X.columns), "gate_mae": gate_mae, "results": results}
    if keep_predictions:
        block["test_card"] = s.card[s.test]
        block["test_y"] = s.y[s.test]
    return block


def to_markdown(all_results: dict) -> str:
    """제외 규칙 × horizon마다 평가 구간 표."""
    out = []
    for variant, by_h in all_results.items():
        for h, block in by_h.items():
            out.append(f"\n### {variant} · {h} (평가 {block['rows']['test']}건, LAST_7D MAE {block['gate_mae']:,.0f})\n")
            out.append("| 방식 | 종류 | 설정값 | MAE | 편향(%) | 과소 비율 | MAE(실제 0) | MAE(실제>0) | 통과 |")
            out.append("|---|---|---|---|---|---|---|---|---|")
            for name, r in sorted(block["results"].items(), key=lambda kv: kv[1]["test"]["MAE"]):
                m = r["test"]
                params = ", ".join(f"{k}={v}" for k, v in r["params"].items()) or "-"
                out.append(f"| {name} | {r['kind']} | {params} | {m['MAE']:,.0f} | {m['rel_bias'] * 100:+.1f} | "
                           f"{m['under_rate']:.1%} | {m['MAE_zero']:,.0f} | {m['MAE_pos']:,.0f} | "
                           f"{'O' if r['passes_gate'] else '-'} |")
    return "\n".join(out) + "\n"
