"""Testes unitários da camada HTTP, com modelo e pipeline falsos."""

import pickle
from copy import deepcopy

import pytest
from fastapi.testclient import TestClient

from api import app as app_module

VALID_PAYLOAD = {
    "Store": 22,
    "DayOfWeek": 4,
    "Date": "2015-09-17",
    "Open": 1,
    "Promo": 1,
    "StateHoliday": "0",
    "SchoolHoliday": 0,
    "StoreType": "a",
    "Assortment": "a",
    "CompetitionDistance": 1040.0,
    "CompetitionOpenSinceMonth": None,
    "CompetitionOpenSinceYear": None,
    "Promo2": 1,
    "Promo2SinceWeek": 22.0,
    "Promo2SinceYear": 2012.0,
    "PromoInterval": "Jan,Apr,Jul,Oct",
}


class FakeModel:
    """Modelo falso para os testes de HTTP."""


class FakePipeline:
    """Pipeline falso, sem scalers nem processamento real."""

    def __init__(self) -> None:
        self.features = ["store", "promo"]
        self.feature_names = ["store", "promo"]
        self.error = None
        self.format_calls = 0
        self.prediction_calls = 0

    def formatando_dados(self, data):
        self.format_calls += 1
        return data

    def get_prediction(self, model, original_data, test_data):
        self.prediction_calls += 1
        if self.error is not None:
            raise self.error
        return original_data.assign(prediction=123.45).to_json(
            orient="records",
            date_format="iso",
        )

    def obter_features_modelo(self, model):
        return self.features


@pytest.fixture
def fake_client(monkeypatch):
    """Cliente com lifespan falso; evita carregar o artefato real."""
    pipeline = FakePipeline()

    def fake_load_model():
        return FakeModel()

    def fake_rossman():
        return pipeline

    monkeypatch.setattr(app_module, "_carregar_modelo", fake_load_model)
    monkeypatch.setattr(app_module, "Rossman", fake_rossman)

    with TestClient(app_module.app, raise_server_exceptions=False) as client:
        yield client, pipeline


def payload(**overrides) -> list[dict]:
    item = deepcopy(VALID_PAYLOAD)
    item.update(overrides)
    return [item]


def test_carregar_modelo_falha_quando_arquivo_nao_existe(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(app_module, "MODEL_PATH", tmp_path / "missing.pkl")

    with pytest.raises(RuntimeError, match="Modelo não encontrado"):
        app_module._carregar_modelo()


def test_carregar_modelo_le_arquivo_pickle(tmp_path, monkeypatch) -> None:
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(pickle.dumps({"ok": True}))
    monkeypatch.setattr(app_module, "MODEL_PATH", model_path)

    assert app_module._carregar_modelo() == {"ok": True}


def test_health_retorna_modelo_falso(fake_client) -> None:
    client, _ = fake_client

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model": "FakeModel"}


def test_model_info_usa_features_do_modelo(fake_client) -> None:
    client, pipeline = fake_client
    pipeline.features = ["a", "b"]

    response = client.get("/model-info")

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "FakeModel"
    assert body["feature_count"] == 2
    assert body["features"] == ["a", "b"]
    assert body["artifact"].endswith("model_rossmann.pkl")


def test_model_info_usa_contrato_quando_modelo_nao_informa_features(fake_client) -> None:
    client, pipeline = fake_client
    pipeline.features = None
    pipeline.feature_names = ["c"]

    body = client.get("/model-info").json()

    assert body["features"] == ["c"]


def test_model_info_retorna_lista_vazia_sem_features(fake_client) -> None:
    client, pipeline = fake_client
    pipeline.features = None
    pipeline.feature_names = None

    body = client.get("/model-info").json()

    assert body["feature_count"] == 0
    assert body["features"] == []


def test_predict_rejeita_payload_vazio(fake_client) -> None:
    client, pipeline = fake_client

    response = client.post("/rossman/predict", json=[])

    assert response.status_code == 400
    assert pipeline.format_calls == 0


def test_predict_rejeita_lote_acima_do_limite(fake_client, monkeypatch) -> None:
    client, pipeline = fake_client
    monkeypatch.setattr(app_module, "MAX_BATCH_SIZE", 2)

    response = client.post("/rossman/predict", json=payload() * 3)

    assert response.status_code == 413
    assert pipeline.format_calls == 0


def test_predict_aceita_lote_no_limite(fake_client, monkeypatch) -> None:
    client, pipeline = fake_client
    monkeypatch.setattr(app_module, "MAX_BATCH_SIZE", 2)

    response = client.post("/rossman/predict", json=payload() * 2)

    assert response.status_code == 200
    assert len(response.json()) == 2
    assert pipeline.format_calls == 1


def test_predict_rejeita_loja_fechada(fake_client) -> None:
    client, pipeline = fake_client

    response = client.post("/rossman/predict", json=payload(Open=0))

    assert response.status_code == 422
    assert "abertas" in response.json()["detail"]
    assert pipeline.format_calls == 0


def test_predict_retorna_sucesso(fake_client) -> None:
    client, pipeline = fake_client

    response = client.post("/rossman/predict", json=payload())

    assert response.status_code == 200
    assert response.json()[0]["prediction"] == 123.45
    assert response.json()[0]["Store"] == 22
    assert pipeline.format_calls == 1
    assert pipeline.prediction_calls == 1


@pytest.mark.parametrize(
    "error",
    [
        ValueError("erro de valor"),
        KeyError("chave ausente"),
        TypeError("tipo inválido"),
    ],
)
def test_predict_converte_erro_de_processamento_em_422(fake_client, error) -> None:
    client, pipeline = fake_client
    pipeline.error = error

    response = client.post("/rossman/predict", json=payload())

    assert response.status_code == 422
    assert "Não foi possível processar" in response.json()["detail"]


def test_predict_erro_inesperado_retorna_500_generico(fake_client) -> None:
    client, pipeline = fake_client
    pipeline.error = RuntimeError("segredo interno")

    response = client.post("/rossman/predict", json=payload())

    assert response.status_code == 500
    assert response.json() == {"detail": "Erro interno ao processar a requisição."}
    assert "segredo interno" not in response.text


def test_limit_body_size_rejeita_corpo_acima_do_limite(fake_client, monkeypatch) -> None:
    client, pipeline = fake_client
    monkeypatch.setattr(app_module, "MAX_BODY_BYTES", 10)

    response = client.post(
        "/rossman/predict",
        content=b"x" * 11,
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 413
    assert "excede o limite" in response.json()["detail"]
    assert pipeline.format_calls == 0


def test_limit_body_size_aceita_corpo_dentro_do_limite(fake_client, monkeypatch) -> None:
    client, _ = fake_client
    monkeypatch.setattr(app_module, "MAX_BODY_BYTES", 10)

    response = client.post(
        "/rossman/predict",
        content=b"[]",
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 400


def test_limit_body_size_rejeita_content_length_invalido(fake_client) -> None:
    client, _ = fake_client

    response = client.post(
        "/rossman/predict",
        content=b"{}",
        headers={"content-length": "abc", "content-type": "application/json"},
    )

    assert response.status_code == 400
    assert "Content-Length" in response.json()["detail"]
