"""예측 API: 정상 응답이 학습 때 특징 계산과 같은 값인지, 잘못된 입력은 400인지, 모델이 없으면 503인지."""

import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from forecast_ml import artifacts, models, serving
from forecast_ml.api import create_app
from forecast_ml.features import make_features
from tests.test_baselines import panel_from


def tiny_model_dir(root: Path) -> Path:
    """가짜 사람 30명의 기록으로 작은 7일 모델을 학습해 저장한다."""
    rng = np.random.default_rng(0)
    spend = {f"P{i}": (rng.random(200) < 0.3) * rng.integers(1_000, 30_000, 200) for i in range(30)}
    panel = panel_from({k: v.tolist() for k, v in spend.items()}, "2023-01-01")
    card, t = np.meshgrid(np.arange(30), np.arange(90, 190, 3), indexing="ij")
    card, t = card.ravel(), t.ravel()
    X = make_features(panel, card, t, 7)
    from forecast_ml.baselines import actual
    y = actual(panel, card, t, 7).astype(float)
    model = models.build("HGB_POISSON", {}).fit(X, y)
    out = root / "m"
    artifacts.save(out, 7, model, list(X.columns), {"model_version": "test-v1"})
    return out


class Api(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.model_dir = tiny_model_dir(Path(cls.tmp.name))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def body(self, **overrides):
        base = {"asOf": "2024-03-31", "horizonDays": 7,
                "accounts": [{"key": "a", "dailySpending": [0] * 80 + [5_000, 0, 0, 12_000, 0, 0, 0, 3_000, 0, 0]},
                             {"key": "b", "dailySpending": [0] * 90}]}
        base.update(overrides)
        return base

    def test_health_and_prediction_match_direct_computation(self):
        with TestClient(create_app(self.model_dir)) as client:
            self.assertEqual(client.get("/health").json()["horizons"], [7])
            res = client.post("/forecast/living-cost", json=self.body())
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["modelVersion"], "test-v1")
            loaded = artifacts.load_all(self.model_dir)[7]
            daily = np.array([a["dailySpending"] for a in self.body()["accounts"]])
            expected = serving.predict(loaded, daily, date(2024, 3, 31))
            self.assertEqual([p["amount"] for p in data["predictions"]], expected.tolist())
            self.assertEqual([p["key"] for p in data["predictions"]], ["a", "b"])

    def test_invalid_inputs_are_400_without_echoing_values(self):
        bad = [
            self.body(accounts=[{"key": "a", "dailySpending": [0] * 89}]),
            self.body(accounts=[{"key": "a", "dailySpending": [-1] + [0] * 89}]),
            self.body(accounts=[{"key": "a", "dailySpending": [0] * 90}, {"key": "a", "dailySpending": [0] * 90}]),
            self.body(horizonDays=30),
            self.body(accounts=[]),
            {"asOf": "2024-03-31", "accounts": [{"key": "a", "dailySpending": [0] * 90}]},
        ]
        with TestClient(create_app(self.model_dir)) as client:
            for body in bad:
                res = client.post("/forecast/living-cost", json=body)
                self.assertEqual(res.status_code, 400, body.get("horizonDays"))
                self.assertNotIn("-1", res.text.replace("\"loc\"", ""))

    def test_missing_models_are_503(self):
        with TestClient(create_app(Path(self.tmp.name) / "none")) as client:
            self.assertEqual(client.get("/health").status_code, 503)
            self.assertEqual(client.post("/forecast/living-cost", json=self.body()).status_code, 503)


if __name__ == "__main__":
    unittest.main()
