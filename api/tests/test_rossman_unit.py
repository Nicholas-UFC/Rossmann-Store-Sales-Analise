"""Testes unitários do pipeline Rossman, sem depender do Kaggle ou do modelo real."""

import json
import pickle

import numpy as np
import pandas as pd
import pytest

from api.services import rossman as rossman_module
from api.services.rossman import Rossman


class IdentityScaler:
    """Scaler mínimo para testes: devolve os valores recebidos."""

    def __init__(self) -> None:
        self.calls = 0

    def transform(self, values):
        self.calls += 1
        return values


class FakeLabelEncoder:
    """Label encoder mínimo para testes."""

    def transform(self, values):
        return [1 for _ in values]


class ModelWithFeatureNames:
    feature_names_in_ = ("b", "a")


class Booster:
    feature_names = ("x", "y")


class ModelWithBooster:
    def get_booster(self):
        return Booster()


class ModelWithoutFeatureNames:
    pass


class ModelWithEmptyBooster:
    def get_booster(self):
        return None


class RecordingModel:
    feature_names_in_ = ("b", "a")

    def __init__(self) -> None:
        self.columns = None

    def predict(self, data):
        self.columns = list(data.columns)
        return np.log1p(np.array([1.0, 2.0]))


class ContractOnlyModel:
    def __init__(self) -> None:
        self.columns = None

    def predict(self, data):
        self.columns = list(data.columns)
        return np.log1p(np.array([1.0, 2.0]))


def make_pipeline(feature_names=None) -> Rossman:
    """Cria um Rossman sem carregar arquivos, usando scalers controlados."""
    pipeline = object.__new__(Rossman)
    pipeline.rs_competition_distance = IdentityScaler()
    pipeline.mms_year = IdentityScaler()
    pipeline.rs_competition_time_month = IdentityScaler()
    pipeline.mms_promo_time_week = IdentityScaler()
    pipeline.le = FakeLabelEncoder()
    pipeline.feature_names = feature_names
    return pipeline


def raw_data() -> pd.DataFrame:
    """Dados brutos sintéticos com os casos de limpeza e engenharia."""
    return pd.DataFrame(
        {
            "Store": [22, 23, 24],
            "DayOfWeek": [4, 5, 6],
            "Date": ["2015-01-15", "2015-09-17", "2015-12-25"],
            "Sales": [100.0, 200.0, 300.0],
            "Customers": [10, 20, 30],
            "Open": [1, 1, 1],
            "Promo": [1, 0, 1],
            "StateHoliday": ["0", "a", "c"],
            "SchoolHoliday": [0, 1, 0],
            "StoreType": ["a", "b", "c"],
            "Assortment": ["a", "b", "c"],
            "CompetitionDistance": [1000.0, None, 2000.0],
            "CompetitionOpenSinceMonth": [None, 5.0, 3.0],
            "CompetitionOpenSinceYear": [None, 2010.0, 2012.0],
            "Promo2": [1, 0, 0],
            "Promo2SinceWeek": [22.0, None, None],
            "Promo2SinceYear": [2012.0, None, None],
            "PromoInterval": ["Jan,Apr,Jul,Oct", None, None],
        }
    )


def test_init_carrega_scalers_e_contrato(tmp_path, monkeypatch) -> None:
    for name in [
        "competition_distance_scaler.pkl",
        "year_scaler.pkl",
        "competition_time_month_scaler.pkl",
        "promo_time_week_scaler.pkl",
        "store_type_scaler.pkl",
    ]:
        (tmp_path / name).write_bytes(pickle.dumps(IdentityScaler()))
    (tmp_path / "feature_names.json").write_text(json.dumps(["store"]), encoding="utf-8")
    monkeypatch.setattr(rossman_module, "PARAMETERS_PATH", tmp_path)

    pipeline = Rossman()

    assert pipeline.feature_names == ["store"]
    assert isinstance(pipeline.rs_competition_distance, IdentityScaler)
    assert isinstance(pipeline.le, IdentityScaler)


def test_init_sem_contrato_de_features(tmp_path, monkeypatch) -> None:
    for name in [
        "competition_distance_scaler.pkl",
        "year_scaler.pkl",
        "competition_time_month_scaler.pkl",
        "promo_time_week_scaler.pkl",
        "store_type_scaler.pkl",
    ]:
        (tmp_path / name).write_bytes(pickle.dumps(IdentityScaler()))
    monkeypatch.setattr(rossman_module, "PARAMETERS_PATH", tmp_path)

    pipeline = Rossman()

    assert pipeline.feature_names is None


def test_init_falha_sem_scaler(tmp_path, monkeypatch) -> None:
    (tmp_path / "feature_names.json").write_text(json.dumps(["store"]), encoding="utf-8")
    monkeypatch.setattr(rossman_module, "PARAMETERS_PATH", tmp_path)

    with pytest.raises(FileNotFoundError):
        Rossman()


def test_limpando_dados_preenche_nulos_e_cria_is_promo() -> None:
    pipeline = make_pipeline()

    df = pipeline.limpando_dados(raw_data())

    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df.loc[0, "is_promo"] == 1
    assert df.loc[1, "is_promo"] == 0
    assert df.loc[1, "competition_distance"] == 200000
    assert df.loc[0, "competition_open_since_month"] == 1
    assert df.loc[0, "competition_open_since_year"] == 2015
    assert df.loc[1, "promo2_since_week"] == 38
    assert df.loc[1, "promo2_since_year"] == 2015
    assert df.loc[1, "promo_interval"] == 0
    assert df["competition_open_since_month"].dtype.kind in "iu"


def test_engenharia_recursos_cria_features_de_tempo_e_mapeia_categorias() -> None:
    pipeline = make_pipeline()
    df = pipeline.limpando_dados(raw_data())

    df = pipeline.engenharia_recursos(df)

    assert df.loc[0, "year"] == 2015
    assert df.loc[0, "month"] == 1
    assert df.loc[0, "day"] == 15
    assert df.loc[0, "week_of_year"] == 3
    assert df.loc[0, "year_week"] == "2015-02"
    assert df.loc[0, "competition_time_month"] == 0
    assert df.loc[0, "promo_time_week"] > 0
    assert df.loc[0, "assortment"] == "basic"
    assert df.loc[1, "assortment"] == "extra"
    assert df.loc[2, "assortment"] == "extended"
    assert df.loc[0, "state_holiday"] == "regular_day"
    assert df.loc[1, "state_holiday"] == "public_holiday"
    assert df.loc[2, "state_holiday"] == "christmas"


def test_filtrando_variaveis_remove_lojas_fechadas_e_colunas_auxiliares() -> None:
    pipeline = make_pipeline()
    df = pipeline.engenharia_recursos(pipeline.limpando_dados(raw_data()))
    df.loc[0, "open"] = 0
    df.loc[1, "sales"] = 0

    filtered = pipeline.filtrando_variaveis(df)

    assert len(filtered) == 1
    assert filtered.iloc[0]["store"] == 24
    assert "customers" not in filtered.columns
    assert "open" not in filtered.columns
    assert "promo_interval" not in filtered.columns
    assert "month_map" not in filtered.columns


def test_preparando_dados_aplica_scalers_dummies_e_ciclos() -> None:
    feature_names = [
        "store",
        "state_holiday_christmas",
        "state_holiday_easter_holiday",
        "store_type",
        "assortment",
        "sales",
        "day_of_week_sin",
        "day_of_week_cos",
        "month_sin",
        "month_cos",
        "day_sin",
        "day_cos",
        "week_of_year_sin",
        "week_of_year_cos",
    ]
    pipeline = make_pipeline(feature_names)
    df = pipeline.engenharia_recursos(pipeline.limpando_dados(raw_data()))

    df = pipeline.preparando_dados(df)

    assert pipeline.rs_competition_distance.calls == 1
    assert pipeline.mms_year.calls == 1
    assert pipeline.rs_competition_time_month.calls == 1
    assert pipeline.mms_promo_time_week.calls == 1
    assert df.loc[0, "store_type"] == 1
    assert df.loc[0, "assortment"] == 1
    assert df.loc[1, "assortment"] == 2
    assert df.loc[2, "assortment"] == 3
    assert np.isclose(df.loc[0, "sales"], np.log1p(100.0))
    assert "state_holiday_christmas" in df.columns
    assert "state_holiday_easter_holiday" in df.columns
    assert df["state_holiday_easter_holiday"].eq(0).all()
    assert "day_of_week_sin" in df.columns
    assert "month_cos" in df.columns
    assert "day_cos" in df.columns
    assert "week_of_year_sin" in df.columns


def test_preparando_dados_sem_sales_nao_cria_target() -> None:
    pipeline = make_pipeline(["store"])
    df = pipeline.engenharia_recursos(pipeline.limpando_dados(raw_data()))
    df = df.drop(columns=["sales"])

    prepared = pipeline.preparando_dados(df)

    assert "sales" not in prepared.columns


def test_selecao_variaveis_usa_ordem_do_contrato() -> None:
    pipeline = make_pipeline(["b", "a"])
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})

    selected = pipeline.selecao_variaveis(df)

    assert list(selected.columns) == ["b", "a"]


def test_selecao_variaveis_falha_sem_contrato() -> None:
    pipeline = make_pipeline(None)

    with pytest.raises(ValueError, match=r"feature_names\.json"):
        pipeline.selecao_variaveis(pd.DataFrame({"a": [1]}))


def test_formatando_dados_entrega_features_do_contrato() -> None:
    feature_names = [
        "store",
        "promo",
        "store_type",
        "assortment",
        "competition_distance",
        "competition_open_since_month",
        "competition_open_since_year",
        "promo2",
        "promo2_since_week",
        "promo2_since_year",
        "competition_time_month",
        "promo_time_week",
        "day_of_week_sin",
        "day_of_week_cos",
        "month_cos",
        "day_cos",
    ]
    pipeline = make_pipeline(feature_names)

    df = pipeline.formatando_dados(raw_data())

    assert not df.empty
    assert all(column in df.columns for column in feature_names)


def test_model_feature_names_usa_feature_names_in() -> None:
    assert Rossman._model_feature_names(ModelWithFeatureNames()) == ["b", "a"]


def test_model_feature_names_usa_booster_quando_nao_ha_feature_names_in() -> None:
    assert Rossman._model_feature_names(ModelWithBooster()) == ["x", "y"]


def test_model_feature_names_retorna_none_sem_metadados() -> None:
    assert Rossman._model_feature_names(ModelWithoutFeatureNames()) is None
    assert Rossman._model_feature_names(ModelWithEmptyBooster()) is None


def test_obter_features_modelo_delega_para_metadados() -> None:
    pipeline = make_pipeline()

    assert pipeline.obter_features_modelo(ModelWithFeatureNames()) == ["b", "a"]


def test_get_prediction_usa_ordem_do_modelo_e_desfaz_log1p() -> None:
    pipeline = make_pipeline(["a", "b"])
    model = RecordingModel()
    original = pd.DataFrame({"Store": [1, 2]})
    formatted = pd.DataFrame({"a": [10, 20], "b": [30, 40]})

    response = pipeline.get_prediction(model, original, formatted)
    body = json.loads(response)

    assert model.columns == ["b", "a"]
    assert body[0]["prediction"] == pytest.approx(1.0)
    assert body[1]["prediction"] == pytest.approx(2.0)
    assert original["prediction"].tolist() == pytest.approx([1.0, 2.0])


def test_get_prediction_usa_contrato_quando_modelo_nao_tem_features() -> None:
    pipeline = make_pipeline(["a", "b"])
    model = ContractOnlyModel()
    original = pd.DataFrame({"Store": [1, 2]})
    formatted = pd.DataFrame({"a": [10, 20], "b": [30, 40]})

    pipeline.get_prediction(model, original, formatted)

    assert model.columns == ["a", "b"]


def test_get_prediction_falha_sem_features_do_modelo_ou_contrato() -> None:
    pipeline = make_pipeline(None)

    with pytest.raises(ValueError, match="Não foi possível identificar"):
        pipeline.get_prediction(
            ModelWithoutFeatureNames(),
            pd.DataFrame({"Store": [1]}),
            pd.DataFrame({"a": [1]}),
        )


def test_get_prediction_falha_com_contrato_divergente() -> None:
    pipeline = make_pipeline(["a", "c"])

    with pytest.raises(ValueError, match="divergente"):
        pipeline.get_prediction(
            ModelWithFeatureNames(),
            pd.DataFrame({"Store": [1]}),
            pd.DataFrame({"a": [1], "b": [2], "c": [3]}),
        )


@pytest.mark.parametrize("leaked", ["date", "sales"])
def test_get_prediction_rejeita_vazamento(leaked: str) -> None:
    pipeline = make_pipeline([leaked, "a"])
    model = ContractOnlyModel()

    with pytest.raises(ValueError, match="vazamento"):
        pipeline.get_prediction(
            model,
            pd.DataFrame({"Store": [1]}),
            pd.DataFrame({leaked: [1], "a": [2]}),
        )


def test_get_prediction_rejeita_feature_ausente() -> None:
    pipeline = make_pipeline(["a", "b"])

    with pytest.raises(ValueError, match="ausentes"):
        pipeline.get_prediction(
            ContractOnlyModel(),
            pd.DataFrame({"Store": [1]}),
            pd.DataFrame({"a": [1]}),
        )


def test_get_prediction_propaga_erro_do_modelo() -> None:
    class BrokenModel:
        feature_names_in_ = ("a",)

        def predict(self, data):
            raise RuntimeError("falha interna")

    pipeline = make_pipeline(["a"])

    with pytest.raises(RuntimeError, match="falha interna"):
        pipeline.get_prediction(
            BrokenModel(),
            pd.DataFrame({"Store": [1]}),
            pd.DataFrame({"a": [1]}),
        )
