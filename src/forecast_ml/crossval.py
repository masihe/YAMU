"""평가 보강: 사람 5묶음 × 평가 연도(2023, 2024)를 돌아가며 평가하고 예측을 합쳐 판정한다.

같은 평가 연도 안에서 5묶음의 평가 구간은 서로 겹치지 않으므로, 합치면 그해 모든 사람을 한 번씩 평가한 셈이다.
판정은 LAST_7D 기준과 TSB 기준을 둘 다 낸다. 차이는 사람 단위 부트스트랩 95% 구간으로 본다.
"""

import numpy as np

from forecast_ml.backtest import metrics
from forecast_ml.experiment import BIAS_LIMIT

FOLDS = 5
REFERENCES = ("LAST_7D", "TSB", "TSB_90D")


def pool(blocks: list[dict]) -> dict:
    """묶음별 결과를 이어 붙인다. 방식마다 pred, 공통 card·y, 묶음별 설정값."""
    names = list(blocks[0]["results"])
    return {
        "card": np.concatenate([b["test_card"] for b in blocks]),
        "y": np.concatenate([b["test_y"] for b in blocks]),
        "pred": {n: np.concatenate([np.asarray(b["results"][n]["pred"], dtype=np.float64) for b in blocks]) for n in names},
        "params": {n: [b["results"][n]["params"] for b in blocks] for n in names},
        "kind": {n: blocks[0]["results"][n]["kind"] for n in names},
    }


def bootstrap(pooled: dict, names: list[str], references: tuple[str, ...] = REFERENCES, rounds: int = 1000,
              seed: int = 0) -> dict:
    """사람을 복원 추출해 (방식 MAE ÷ 기준 MAE − 1)과 편향 비율의 평균·95% 구간을 낸다."""
    rng = np.random.default_rng(seed)
    card, y = pooled["card"], pooled["y"]
    people = np.unique(card)
    rows = {c: np.where(card == c)[0] for c in people}
    ratios = {(n, r): [] for n in names for r in references if n != r}
    biases = {n: [] for n in names}
    for _ in range(rounds):
        pick = np.concatenate([rows[c] for c in rng.choice(people, len(people))])
        yy = y[pick]
        mae = {n: np.abs(pooled["pred"][n][pick] - yy).mean() for n in set(names) | set(references)}
        for n, r in ratios:
            ratios[(n, r)].append(mae[n] / mae[r] - 1)
        for n in names:
            biases[n].append((pooled["pred"][n][pick] - yy).mean() / yy.mean())

    def summary(values: list[float]) -> dict:
        return {"mean": round(float(np.mean(values)), 4), "low": round(float(np.percentile(values, 2.5)), 4),
                "high": round(float(np.percentile(values, 97.5)), 4)}

    return {"people": int(len(people)),
            "mae_vs": {f"{n}|{r}": summary(v) for (n, r), v in ratios.items()},
            "bias": {n: summary(v) for n, v in biases.items()}}


def verdict(pooled: dict, boot: dict) -> dict:
    """방식마다 합친 지표와, 기준별 판정. 점추정 통과(편향 ±10%, MAE 낮음)와 구간 통과(MAE 비율 구간 상한 < 0)를 따로 둔다."""
    out = {}
    y = pooled["y"]
    ref_mae = {r: float(np.abs(pooled["pred"][r] - y).mean()) for r in REFERENCES}
    for n, pred in pooled["pred"].items():
        m = metrics(pred, y)
        judge = {}
        for r in REFERENCES:
            if n == r:
                continue
            ci = boot["mae_vs"].get(f"{n}|{r}")
            judge[r] = {"point": abs(m["rel_bias"]) <= BIAS_LIMIT and m["MAE"] < ref_mae[r],
                        "interval": ci is not None and ci["high"] < 0 and abs(m["rel_bias"]) <= BIAS_LIMIT}
        out[n] = {"kind": pooled["kind"][n], "metrics": m, "params_by_fold": pooled["params"][n], "judge": judge}
    return out
