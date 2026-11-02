"""생활비 예측 API (Task 7.3.4 시제품). BE만 호출하는 내부 API다.

실행: YAMU_AI_MODEL_DIR=<모델 폴더> uvicorn --factory forecast_ml.api:create_app --port 8001
- POST /forecast/living-cost: 계좌별 최근 90일 일별 체크 생활 소비 → 다음 horizonDays일 합계
- GET /health: 모델을 읽었는지
요청 본문은 로그에 남기지 않는다. 사용자 ID·계좌번호는 받지 않고, 계좌는 BE가 정한 임의 key로만 구분한다.
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from forecast_ml import artifacts, serving

log = logging.getLogger("forecast_ml.api")
MAX_ACCOUNTS = 20
MAX_DAILY_AMOUNT = 1_000_000_000_000  # 하루 1조 원. 이보다 크면 잘못된 입력으로 본다


class AccountHistory(BaseModel):
    key: str = Field(min_length=1, max_length=64)
    dailySpending: list[int] = Field(min_length=artifacts.HISTORY_DAYS, max_length=artifacts.HISTORY_DAYS)

    @field_validator("dailySpending")
    @classmethod
    def non_negative(cls, values: list[int]) -> list[int]:
        if any(v < 0 or v > MAX_DAILY_AMOUNT for v in values):
            raise ValueError("일별 소비는 0 이상이고 상한 이하여야 합니다")
        return values


class ForecastRequest(BaseModel):
    asOf: date
    horizonDays: int
    accounts: list[AccountHistory] = Field(min_length=1, max_length=MAX_ACCOUNTS)

    @field_validator("accounts")
    @classmethod
    def unique_keys(cls, accounts: list[AccountHistory]) -> list[AccountHistory]:
        if len({a.key for a in accounts}) != len(accounts):
            raise ValueError("계좌 key가 겹칩니다")
        return accounts


def create_app(model_dir: Path | None = None) -> FastAPI:
    """모델 폴더(없으면 환경변수 YAMU_AI_MODEL_DIR)를 앱 시작 때 한 번 읽는 앱을 만든다."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        path = model_dir or (Path(os.environ["YAMU_AI_MODEL_DIR"]) if os.environ.get("YAMU_AI_MODEL_DIR") else None)
        app.state.models = {}
        app.state.load_error = None
        if path is None:
            app.state.load_error = "YAMU_AI_MODEL_DIR가 없습니다"
        elif not path.is_dir():
            app.state.load_error = "모델 폴더가 없습니다"
        else:
            try:
                app.state.models = artifacts.load_all(path)
            except (OSError, ValueError) as e:
                app.state.load_error = f"모델을 읽지 못했습니다: {e}"
            if not app.state.models and app.state.load_error is None:
                app.state.load_error = "모델 파일이 없습니다"
        if app.state.load_error:
            log.error(app.state.load_error)
        yield

    app = FastAPI(title="YAMU AI living cost", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def bad_request(request: Request, exc: RequestValidationError):
        # 명세대로 400. 입력값은 응답에 되돌려 주지 않는다(위치와 이유만)
        errors = [{"loc": list(e.get("loc", [])), "msg": e.get("msg", "")} for e in exc.errors()]
        return JSONResponse(status_code=400, content={"error": "INVALID_REQUEST", "fieldErrors": errors})

    @app.get("/health")
    def health():
        models = app.state.models
        if not models:
            return JSONResponse(status_code=503, content={"status": "model_missing", "reason": app.state.load_error})
        return {"status": "ok", "horizons": sorted(models),
                "modelVersion": {h: m.meta.get("model_version") for h, m in models.items()}}

    @app.post("/forecast/living-cost")
    def living_cost(body: ForecastRequest):
        models = app.state.models
        if not models:
            return JSONResponse(status_code=503, content={"error": "MODEL_MISSING"})
        loaded = models.get(body.horizonDays)
        if loaded is None:
            return JSONResponse(status_code=400, content={"error": "INVALID_REQUEST", "fieldErrors": [
                {"loc": ["body", "horizonDays"], "msg": f"지원하는 일수: {sorted(models)}"}]})
        started = time.perf_counter()
        daily = np.array([a.dailySpending for a in body.accounts], dtype=np.int64)
        amounts = serving.predict(loaded, daily, body.asOf)
        log.info("living-cost horizon=%s accounts=%s ms=%.1f", body.horizonDays, len(body.accounts),
                 (time.perf_counter() - started) * 1000)
        return {"modelVersion": loaded.meta.get("model_version"), "horizonDays": body.horizonDays,
                "predictions": [{"key": a.key, "amount": int(v)} for a, v in zip(body.accounts, amounts)]}

    return app
