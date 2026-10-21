"""Testes do pipeline Rossman (api/services/rossman.py).

Valida o fluxo completo usado pela API:
    limpando_dados -> engenharia_recursos -> preparando_dados -> get_prediction
"""

import json
from json import loads

import pandas as pd
import pytest

from api.config.config import PARAMETERS_PATH
from api.services.rossman import Rossman

pytestmark = pytest.mark.integration


def contract() -> list[str]:
    """Lê o contrato de features produzido pelo treino."""
    return json.loads((PARAMETERS_PATH / "feature_names.json").read_text(encoding="utf-8"))


def test_pipeline_carrega_todos_os_scalers(pipeline: Rossman) -> None:
    """O construtor deve carregar os 5 scalers de parameters/."""
    scalers = {
        "rs_competition_distance": pipeline.rs_competition_distance,
        "mms_year": pipeline.mms_year,
        "rs_competition_time_month": pipeline.rs_competition_time_month,
        "mms_promo_time_week": pipeline.mms_promo_time_week,
        "le": pipeline.le,
    }

    for nome, scaler in scalers.items():
        assert scaler is not None, f"Scaler '{nome}' não foi carregado"


def test_pipeline_carrega_contrato_de_features(pipeline: Rossman) -> None:
    """O contrato deve ser exatamente o salvo em parameters/feature_names.json."""
    assert pipeline.feature_names == contract()
    assert len(pipeline.feature_names) == 16
    assert "date" not in pipeline.feature_names
    assert "sales" not in pipeline.feature_names


def test_modelo_real_usa_o_contrato_de_features(pipeline: Rossman, model) -> None:
    """O modelo carregado deve esperar exatamente as features do contrato."""
    assert pipeline.obter_features_modelo(model) == pipeline.feature_names


def test_formatando_dados_gera_features_esperadas(pipeline: Rossman, df_raw: pd.DataFrame) -> None:
    """formatando_dados deve produzir todas as features do contrato do modelo."""
    df_formatado = pipeline.formatando_dados(df_raw.copy())

    assert not df_formatado.empty, "formatando_dados retornou DataFrame vazio"
    for col in pipeline.feature_names:
        assert col in df_formatado.columns, f"Coluna '{col}' ausente após formatando_dados"


def test_get_prediction_retorna_predicoes_validas(
    pipeline: Rossman, model, df_raw: pd.DataFrame
) -> None:
    """get_prediction deve retornar JSON com uma predição positiva por registro."""
    df_formatado = pipeline.formatando_dados(df_raw.copy())
    response = pipeline.get_prediction(model, df_raw.copy(), df_formatado)

    preds = loads(response)
    assert isinstance(preds, list) and len(preds) == len(df_raw), (
        "Resposta JSON com número de registros incorreto"
    )
    assert all("prediction" in item for item in preds), "Registros sem o campo 'prediction'"

    valores = [item["prediction"] for item in preds]
    assert all(v > 0 for v in valores), "Predições devem ser valores positivos de vendas"
