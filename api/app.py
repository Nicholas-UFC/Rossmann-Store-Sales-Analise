"""API FastAPI para predição de vendas Rossmann."""

import logging
import time
from contextlib import asynccontextmanager
from json import loads
from pickle import load

import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from api.config.config import MAX_BATCH_SIZE, MODEL_PATH
from api.models.schemas import RossmannPrediction, RossmannStore
from api.services.rossman import Rossman

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("rossmann.api")


def _carregar_modelo():
    """Carrega o modelo salvo em disco."""
    if not MODEL_PATH.is_file():
        raise RuntimeError(
            f"Modelo não encontrado em {MODEL_PATH}. "
            "Execute o main.ipynb para gerar o artefato antes de subir a API."
        )

    with MODEL_PATH.open("rb") as file:
        return load(file)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializa modelo e pipeline uma única vez."""
    app.state.model = _carregar_modelo()
    app.state.pipeline = Rossman()
    logger.info("Modelo carregado: %s", type(app.state.model).__name__)
    yield


app = FastAPI(
    title="Rossmann Predictions API",
    description="API de previsão de vendas para lojas Rossmann.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log básico de requisições HTTP."""
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s -> %s (%.1f ms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.get("/health")
def health(request: Request):
    """Health check simples."""
    return {
        "status": "ok",
        "model": type(request.app.state.model).__name__,
    }


@app.get("/model-info")
def model_info(request: Request):
    """Metadados do modelo carregado."""
    model = request.app.state.model
    pipeline: Rossman = request.app.state.pipeline

    model_features = pipeline.obter_features_modelo(model) or pipeline.feature_names or []

    return {
        "model": type(model).__name__,
        "feature_count": len(model_features),
        "features": model_features,
        "artifact": str(MODEL_PATH),
    }


@app.post("/rossman/predict", response_model=list[RossmannPrediction])
def rossman_predict(data: list[RossmannStore], request: Request):
    if not data:
        raise HTTPException(status_code=400, detail="Payload não pode ser vazio.")

    if len(data) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Lote muito grande. Limite máximo: {MAX_BATCH_SIZE} registros.",
        )

    # O modelo foi treinado apenas com lojas abertas e vendas positivas.
    if any(item.Open != 1 for item in data):
        raise HTTPException(
            status_code=422,
            detail="O modelo só atende lojas abertas (Open=1).",
        )

    model = request.app.state.model
    pipeline: Rossman = request.app.state.pipeline
    test_raw = pd.DataFrame([item.model_dump() for item in data])

    try:
        df_formatado = pipeline.formatando_dados(test_raw)
        response_json = pipeline.get_prediction(model, test_raw, df_formatado)
    except (ValueError, KeyError, TypeError) as exc:
        logger.exception("Erro ao processar predição.")
        raise HTTPException(
            status_code=422,
            detail=f"Não foi possível processar a predição: {exc}",
        ) from exc

    return loads(response_json)
