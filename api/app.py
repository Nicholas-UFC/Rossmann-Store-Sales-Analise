from json import loads
from pathlib import Path
from pickle import load

import pandas as pd
from fastapi import FastAPI

from api.services.rossman import Rossman

# Define o caminho base de forma dinâmica
home_path = Path(__file__).parent.parent if "__file__" in globals() else Path()
param_path = home_path / "models"

# Carrega o modelo ML
with Path.open(param_path / "model_rossmann.pkl", "rb") as f:
    model = load(f)

app = FastAPI(title="Rossman Predictions")


@app.post("/rossman/predict")
def rossman_predict(data: list[dict]):
    test_raw = pd.DataFrame(data)

    pipeline = Rossman()

    df_formatado = pipeline.formatando_dados(test_raw)

    response_json = pipeline.get_prediction(model, test_raw, df_formatado)

    return loads(response_json)
