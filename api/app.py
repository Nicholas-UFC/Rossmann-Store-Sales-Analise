from json import loads
from pathlib import Path
from pickle import load

import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import RossmannPrediction, RossmannStore
from api.services.rossman import Rossman

# Define o caminho base de forma dinâmica
home_path = Path(__file__).parent.parent if "__file__" in globals() else Path()
model_path = home_path / "models" / "model_rossmann.pkl"

# Carrega o modelo ML uma única vez na inicialização
with Path.open(model_path, "rb") as f:
    model = load(f)

# Pipeline carregado uma única vez (evita ler os scalers a cada request)
pipeline = Rossman()

app = FastAPI(title="Rossman Predictions")


@app.post("/rossman/predict", response_model=list[RossmannPrediction])
def rossman_predict(data: list[RossmannStore]):
    if not data:
        raise HTTPException(status_code=400, detail="Payload não pode ser vazio.")

    # O modelo foi treinado apenas com lojas abertas e vendas positivas.
    if any(item.Open != 1 for item in data):
        raise HTTPException(
            status_code=422,
            detail="O modelo só atende lojas abertas (Open=1).",
        )

    test_raw = pd.DataFrame([item.model_dump() for item in data])

    try:
        df_formatado = pipeline.formatando_dados(test_raw)
        response_json = pipeline.get_prediction(model, test_raw, df_formatado)
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Não foi possível processar a predição: {exc}",
        ) from exc

    return loads(response_json)
