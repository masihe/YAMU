"""두 단계 모델을 저장했다가 읽으면 같은 예측을 내고, 파일에는 이 저장소의 클래스가 들어가지 않는지."""

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from forecast_ml import artifacts, models


class TwoStageArtifact(unittest.TestCase):

    def test_saved_as_plain_sklearn_dict_and_loaded_back(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"a": rng.random(300), "b": rng.random(300)})
        y = np.where(X["a"] > 0.5, rng.integers(1_000, 9_000, 300), 0).astype(np.float64)
        model = models.build("TWO_STAGE_TREE", {}).fit(X, y)

        with tempfile.TemporaryDirectory() as tmp:
            artifacts.save(Path(tmp), 14, model, list(X.columns), {"model_version": "t"})
            raw = joblib.load(Path(tmp) / "h14.joblib")
            loaded = artifacts.load_all(Path(tmp))[14]

        self.assertEqual(sorted(raw), ["classifier", "regressor"])
        self.assertNotIn("forecast_ml", type(raw["classifier"]).__module__ + type(raw["regressor"]).__module__)
        np.testing.assert_allclose(loaded.model.predict(X), model.predict(X))


if __name__ == "__main__":
    unittest.main()
