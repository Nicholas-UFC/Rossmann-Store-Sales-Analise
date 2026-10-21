"""Testes unitários dos schemas Pydantic da API Rossmann."""

from copy import deepcopy
from datetime import date

import pytest
from pydantic import ValidationError

from api.models.schemas import RossmannPrediction, RossmannStore

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


def make_payload(**overrides) -> dict:
    """Cria um payload válido com alterações pontuais."""
    payload = deepcopy(VALID_PAYLOAD)
    payload.update(overrides)
    return payload


def test_payload_completo_e_aceito() -> None:
    store = RossmannStore(**make_payload())

    assert store.Store == 22
    assert store.Date == date(2015, 9, 17)
    assert store.Promo2 == 1


def test_promo2_desativada_aceita_campos_ausentes() -> None:
    store = RossmannStore(
        **make_payload(
            Promo2=0,
            Promo2SinceWeek=None,
            Promo2SinceYear=None,
            PromoInterval=None,
        )
    )

    assert store.Promo2 == 0
    assert store.Promo2SinceWeek is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("Store", 0),
        ("Store", 1116),
        ("Store", "invalida"),
        ("DayOfWeek", 0),
        ("DayOfWeek", 8),
        ("Open", 2),
        ("Promo", 2),
        ("StateHoliday", "x"),
        ("SchoolHoliday", 2),
        ("StoreType", "z"),
        ("StoreType", "A"),
        ("Assortment", "z"),
        ("CompetitionDistance", -1),
        ("CompetitionOpenSinceMonth", 0),
        ("CompetitionOpenSinceMonth", 13),
        ("CompetitionOpenSinceYear", 1899),
        ("CompetitionOpenSinceYear", 2101),
        ("Promo2", 2),
        ("Promo2SinceWeek", 0),
        ("Promo2SinceWeek", 54),
        ("Promo2SinceYear", 1899),
        ("Promo2SinceYear", 2101),
        ("PromoInterval", "invalid_month"),
        ("PromoInterval", "jan"),
        ("PromoInterval", "January"),
        ("PromoInterval", "Jan, Apr"),
        ("PromoInterval", "Jan,Apr,Jul,Oct,Dec"),
        ("PromoInterval", ""),
        ("Date", "2012-12-31"),
        ("Date", "2016-01-01"),
        ("Date", "data-invalida"),
    ],
)
def test_rejeita_valores_invalidos(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        RossmannStore(**make_payload(**{field: value}))


@pytest.mark.parametrize(
    "field",
    [
        "Store",
        "DayOfWeek",
        "Date",
        "Open",
        "Promo",
        "StateHoliday",
        "SchoolHoliday",
        "StoreType",
        "Assortment",
        "Promo2",
    ],
)
def test_rejeita_campo_obrigatorio_ausente(field: str) -> None:
    payload = make_payload()
    payload.pop(field)

    with pytest.raises(ValidationError):
        RossmannStore(**payload)


def test_rejeita_campo_extra() -> None:
    with pytest.raises(ValidationError):
        RossmannStore(**make_payload(Sales=9999))


@pytest.mark.parametrize(
    "field",
    ["Promo2SinceWeek", "Promo2SinceYear", "PromoInterval"],
)
def test_promo2_ativa_exige_campos_da_promocao(field: str) -> None:
    with pytest.raises(ValidationError, match="Promo2SinceWeek"):
        RossmannStore(**make_payload(Promo2=1, **{field: None}))


@pytest.mark.parametrize(
    "interval",
    ["Jan", "Sep", "Sept", "Jan,Apr,Jul,Oct", "Sept,Dec", "Jan,Jan"],
)
def test_aceita_promo_interval_valido(interval: str) -> None:
    store = RossmannStore(**make_payload(PromoInterval=interval))

    assert store.PromoInterval == interval


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("Store", 1),
        ("Store", 1115),
        ("DayOfWeek", 1),
        ("DayOfWeek", 7),
        ("Date", "2013-01-01"),
        ("Date", "2015-12-31"),
        ("CompetitionDistance", 0),
        ("CompetitionOpenSinceMonth", 1),
        ("CompetitionOpenSinceMonth", 12),
        ("CompetitionOpenSinceYear", 1900),
        ("CompetitionOpenSinceYear", 2100),
        ("Promo2SinceWeek", 1),
        ("Promo2SinceWeek", 53),
        ("Promo2SinceYear", 1900),
        ("Promo2SinceYear", 2100),
    ],
)
def test_aceita_limites_validos(field: str, value: object) -> None:
    store = RossmannStore(**make_payload(**{field: value}))

    if field == "Date":
        assert store.Date.isoformat() == value
    else:
        assert getattr(store, field) == value


def test_prediction_inclui_campo_de_saida() -> None:
    prediction = RossmannPrediction(**make_payload(prediction=123.45))

    assert prediction.prediction == 123.45
