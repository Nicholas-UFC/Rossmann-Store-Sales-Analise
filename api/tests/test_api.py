"""Testes de integração do endpoint /rossman/predict e endpoints auxiliares."""

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.config.config import PARAMETERS_PATH

pytestmark = pytest.mark.integration


def feature_names() -> list[str]:
    """Lê o contrato real salvo pelo treino."""
    return json.loads((PARAMETERS_PATH / "feature_names.json").read_text(encoding="utf-8"))


def payload_json(df: pd.DataFrame) -> list[dict]:
    """Converte o DataFrame em payload JSON válido (None no lugar de NaN)."""
    return df.astype(object).where(pd.notnull(df), None).to_dict(orient="records")


def test_health_retorna_modelo_carregado(client: TestClient) -> None:
    """GET /health deve indicar que a API está ok e qual modelo está ativo."""
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model"] == "LGBMRegressor"


def test_model_info_retorna_contrato_de_features(client: TestClient) -> None:
    """GET /model-info deve expor o modelo e o contrato de features."""
    response = client.get("/model-info")

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "LGBMRegressor"
    assert body["feature_count"] == 16
    assert body["features"] == feature_names()
    assert "sales" not in body["features"]
    assert "date" not in body["features"]
    assert body["artifact"].endswith("model_rossmann.pkl")


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


def test_predict_rejeita_payload_vazio(client: TestClient) -> None:
    """Payload vazio deve retornar 400."""
    response = client.post("/rossman/predict", json=[])

    assert response.status_code == 400
    assert "vazio" in response.json()["detail"].lower()


def test_predict_rejeita_loja_fechada(client: TestClient, df_raw: pd.DataFrame) -> None:
    """O modelo só atende lojas abertas (Open=1)."""
    payload = payload_json(df_raw.head(1))
    payload[0]["Open"] = 0

    response = client.post("/rossman/predict", json=payload)

    assert response.status_code == 422
    assert "abertas" in response.json()["detail"]


def test_predict_rejeita_store_type_desconhecido(client: TestClient, df_raw: pd.DataFrame) -> None:
    """Categorias fora do treino devem ser rejeitadas pelo schema."""
    payload = payload_json(df_raw.head(1))
    payload[0]["StoreType"] = "z"

    response = client.post("/rossman/predict", json=payload)

    assert response.status_code == 422


def test_predict_rejeita_data_fora_do_periodo_suportado(
    client: TestClient, df_raw: pd.DataFrame
) -> None:
    """Datas muito distantes do treino devem ser rejeitadas."""
    payload = payload_json(df_raw.head(1))
    payload[0]["Date"] = "2016-01-01"

    response = client.post("/rossman/predict", json=payload)

    assert response.status_code == 422
    assert "período suportado" in response.json()["detail"][0]["msg"]


def test_predict_rejeita_promo2_sem_campos_obrigatorios(
    client: TestClient, df_raw: pd.DataFrame
) -> None:
    """Promo2=1 exige Promo2SinceWeek, Promo2SinceYear e PromoInterval."""
    payload = payload_json(df_raw.head(1))
    payload[0]["Promo2"] = 1
    payload[0]["Promo2SinceWeek"] = None
    payload[0]["Promo2SinceYear"] = None
    payload[0]["PromoInterval"] = None

    response = client.post("/rossman/predict", json=payload)

    assert response.status_code == 422
    assert any("Promo2" in error["msg"] for error in response.json()["detail"])


def test_predict_rejeita_promo_interval_invalido(client: TestClient, df_raw: pd.DataFrame) -> None:
    """PromoInterval com formato desconhecido deve ser rejeitado."""
    payload = payload_json(df_raw.head(1))
    payload[0]["Promo2"] = 1
    payload[0]["Promo2SinceWeek"] = 22.0
    payload[0]["Promo2SinceYear"] = 2012.0
    payload[0]["PromoInterval"] = "invalid_month"

    response = client.post("/rossman/predict", json=payload)

    assert response.status_code == 422
    assert any("PromoInterval" in error["msg"] for error in response.json()["detail"])


def test_predict_rejeita_campo_extra(client: TestClient, df_raw: pd.DataFrame) -> None:
    """Campos fora do contrato não devem ser aceitos."""
    payload = payload_json(df_raw.head(1))
    payload[0]["Sales"] = 9999

    response = client.post("/rossman/predict", json=payload)

    assert response.status_code == 422


def test_predict_rejeita_lote_acima_do_limite(client: TestClient, df_raw: pd.DataFrame) -> None:
    """Lotes acima do limite configurado devem ser rejeitados."""
    one = payload_json(df_raw.head(1))[0]
    payload = [one] * 10_001

    response = client.post("/rossman/predict", json=payload)

    assert response.status_code == 413
    assert "limite" in response.json()["detail"].lower()


def test_predict_e_deterministico(client: TestClient, df_raw: pd.DataFrame) -> None:
    """O mesmo payload deve gerar a mesma predição."""
    payload = payload_json(df_raw.head(1))

    first = client.post("/rossman/predict", json=payload)
    second = client.post("/rossman/predict", json=payload)

    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()


def test_predict_preserva_campos_e_ordem_do_lote(client: TestClient, df_raw: pd.DataFrame) -> None:
    """A resposta deve manter os campos originais e a ordem dos registros."""
    payload = payload_json(df_raw.head(3))

    response = client.post("/rossman/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert [item["Store"] for item in body] == [item["Store"] for item in payload]
    assert all(item["Date"] == payload[index]["Date"] for index, item in enumerate(body))
    assert all(isinstance(item["prediction"], float) for item in body)
