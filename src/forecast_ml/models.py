"""비교할 모델 6종과 설정값 후보. 모두 7·14일 체크 소비 합계의 평균을 겨냥하는 손실을 쓴다(MAE 손실은 쓰지 않는다)."""

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import GammaRegressor, LogisticRegression, PoissonRegressor, TweedieRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

POWERS = (1.1, 1.3, 1.5, 1.7, 1.9)
GRIDS: dict[str, list[dict]] = {
    "TWEEDIE_GLM": [{"power": p} for p in POWERS],
    "POISSON_GLM": [{}],
    "TWO_STAGE_LINEAR": [{}],
    "TWO_STAGE_TREE": [{}],
    "HGB_POISSON": [{}],
    "LGBM_TWEEDIE": [{"power": p} for p in POWERS],
}


class TwoStage:
    """1단계: 다음 H일에 쓸 확률. 2단계: 쓴 경우의 합계. 예측 = 확률 × 합계."""

    def __init__(self, classifier, regressor):
        self.classifier = classifier
        self.regressor = regressor

    def fit(self, X, y):
        """양수 표본이 없으면 학습할 수 없으므로 거절한다."""
        positive = y > 0
        if not positive.any():
            raise ValueError("양수 정답이 없어 두 단계 모델을 학습할 수 없습니다")
        self.classifier.fit(X, positive)
        self.regressor.fit(X[positive], y[positive])
        return self

    def predict(self, X):
        return self.classifier.predict_proba(X)[:, 1] * self.regressor.predict(X)


def _linear(estimator) -> Pipeline:
    """선형 모델용: 금액·일수 특징을 log1p로 줄이고 표준화한다."""
    return Pipeline([("log", FunctionTransformer(np.log1p)), ("scale", StandardScaler()), ("model", estimator)])


def build(name: str, params: dict):
    """이름과 설정값으로 학습 전 모델을 만든다."""
    tree = {"max_iter": 300, "learning_rate": 0.05, "random_state": 0}
    if name == "TWEEDIE_GLM":
        return _linear(TweedieRegressor(power=params["power"], alpha=1e-3, link="log", max_iter=2000))
    if name == "POISSON_GLM":
        return _linear(PoissonRegressor(alpha=1e-3, max_iter=2000))
    if name == "TWO_STAGE_LINEAR":
        return TwoStage(_linear(LogisticRegression(max_iter=2000)), _linear(GammaRegressor(alpha=1e-3, max_iter=2000)))
    if name == "TWO_STAGE_TREE":
        return TwoStage(HistGradientBoostingClassifier(**tree), HistGradientBoostingRegressor(loss="gamma", **tree))
    if name == "HGB_POISSON":
        return HistGradientBoostingRegressor(loss="poisson", **tree)
    if name == "LGBM_TWEEDIE":
        return LGBMRegressor(objective="tweedie", tweedie_variance_power=params["power"], n_estimators=300,
                             learning_rate=0.05, num_leaves=31, min_child_samples=50, random_state=0, verbose=-1)
    raise ValueError(f"모르는 모델: {name}")
