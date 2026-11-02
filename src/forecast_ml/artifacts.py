"""최종 모델 파일과 설명(meta.json)을 저장하고 읽는다.

모델 파일은 joblib(pickle) 형식이라 읽을 때 코드가 실행될 수 있다. 이 저장소가 만든 파일만 읽는다.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import sklearn

HISTORY_DAYS = 90  # 서비스가 받는 일별 기록 길이. 특징은 최근 84일까지 쓴다


@dataclass(frozen=True)
class LoadedModel:
    """horizon 하나의 모델과 설명."""

    horizon: int
    model: object
    features: list[str]
    meta: dict


def save(model_dir: Path, horizon: int, model, features: list[str], meta: dict) -> None:
    """model_dir/h{horizon}.joblib과 h{horizon}.meta.json을 쓴다."""
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / f"h{horizon}.joblib")
    record = {**meta, "horizon": horizon, "features": features, "sklearn_version": sklearn.__version__,
              "history_days": HISTORY_DAYS}
    (model_dir / f"h{horizon}.meta.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def load_all(model_dir: Path) -> dict[int, LoadedModel]:
    """model_dir의 모든 h*.joblib을 읽는다. 설명 파일이 없거나 horizon이 맞지 않으면 거절한다."""
    loaded = {}
    for path in sorted(model_dir.glob("h*.joblib")):
        meta_path = path.with_name(path.stem + ".meta.json")
        if not meta_path.exists():
            raise ValueError(f"설명 파일이 없습니다: {meta_path.name}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        horizon = int(path.stem[1:])
        if meta.get("horizon") != horizon:
            raise ValueError(f"{path.name}의 horizon이 설명과 다릅니다")
        loaded[horizon] = LoadedModel(horizon, joblib.load(path), list(meta["features"]), meta)
    return loaded
