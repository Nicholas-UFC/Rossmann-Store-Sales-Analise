"""Testes de integração do endpoint /rossman/predict (api/app.py).

Usa o TestClient do FastAPI: a aplicação roda em processo, sem servidor externo.
"""

import pandas as pd
from fastapi.testclient import TestClient


def payload_json(df: pd.DataFrame) -> list[dict]:
    """Converte o DataFrame em payload JSON válido (None no lugar de NaN)."""
    return df.astype(object).where(pd.notnull(df), None).to_dict(orient="records")


def test_predict_retorna_predicao_para_lote(client: TestClient, df_raw: pd.DataFrame) -> None:
    """POST /rossman/predict deve responder 200 com uma predição por registro."""
    response = client.post("/rossman/predict", json=payload_json(df_raw))

    assert response.status_code == 200

    preds = response.json()
    assert isinstance(preds, list) and len(preds) == len(df_raw), (
        "Resposta JSON com número de registros incorreto"
    )
    assert all("prediction" in item for item in preds), "Registros sem o campo 'prediction'"

    valores = [item["prediction"] for item in preds]
    assert all(v > 0 for v in valores), "Predições devem ser valores positivos de vendas"


def test_predict_aceita_registro_unico(client: TestClient, df_raw: pd.DataFrame) -> None:
    """O endpoint também deve funcionar com um único registro no payload."""
    response = client.post("/rossman/predict", json=payload_json(df_raw.head(1)))

    assert response.status_code == 200

    preds = response.json()
    assert isinstance(preds, list) and len(preds) == 1, (
        "Payload com 1 registro deve retornar 1 predição"
    )
    assert preds[0]["prediction"] > 0
