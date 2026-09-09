"""Schemas Pydantic usados pela API Rossmann."""

import re
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from api.config.config import MAX_SUPPORTED_DATE, MIN_SUPPORTED_DATE

_PROMO_INTERVAL_REGEX = re.compile(
    r"^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec)"
    r"(?:,(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec)){0,3}$"
)


class RossmannStore(BaseModel):
    """Payload de entrada esperado pela API.

    O modelo foi treinado apenas para lojas abertas com vendas positivas,
    por isso o schema já restringe valores inválidos como dias da semana
    fora de 1..7, loja fora do range do dataset, etc.
    """

    model_config = ConfigDict(extra="forbid")

    Store: int = Field(ge=1, le=1115)
    DayOfWeek: int = Field(ge=1, le=7)
    Date: date
    Open: Literal[0, 1]
    Promo: Literal[0, 1]
    StateHoliday: Literal["0", "a", "b", "c"]
    SchoolHoliday: Literal[0, 1]
    StoreType: Literal["a", "b", "c", "d"]
    Assortment: Literal["a", "b", "c"]
    CompetitionDistance: float | None = Field(default=None, ge=0)
    CompetitionOpenSinceMonth: int | None = Field(default=None, ge=1, le=12)
    CompetitionOpenSinceYear: int | None = Field(default=None, ge=1900, le=2100)
    Promo2: Literal[0, 1]
    Promo2SinceWeek: float | None = Field(default=None, ge=1, le=53)
    Promo2SinceYear: float | None = Field(default=None, ge=1900, le=2100)
    PromoInterval: str | None = None

    @field_validator("Date")
    @classmethod
    def validar_data_suportada(cls, value: date) -> date:
        """Rejeita datas fora do período coberto pelo modelo atual."""
        if not MIN_SUPPORTED_DATE <= value <= MAX_SUPPORTED_DATE:
            raise ValueError(
                f"Data fora do período suportado "
                f"({MIN_SUPPORTED_DATE.isoformat()} a {MAX_SUPPORTED_DATE.isoformat()})."
            )
        return value

    @field_validator("PromoInterval")
    @classmethod
    def validar_promo_interval(cls, value: str | None) -> str | None:
        """Valida o formato usado pelo dataset original da Rossmann."""
        if value is None:
            return value
        if not _PROMO_INTERVAL_REGEX.fullmatch(value):
            raise ValueError(
                "PromoInterval inválido. Use meses separados por vírgula, "
                'ex.: "Jan,Apr,Jul,Oct".'
            )
        return value

    @model_validator(mode="after")
    def validar_promo2(self) -> "RossmannStore":
        """Quando Promo2=1, os campos da promoção contínua são obrigatórios."""
        if self.Promo2 == 1 and (
            self.Promo2SinceWeek is None
            or self.Promo2SinceYear is None
            or self.PromoInterval is None
        ):
            raise ValueError(
                "Promo2SinceWeek, Promo2SinceYear e PromoInterval são obrigatórios "
                "quando Promo2=1."
            )
        return self


class RossmannPrediction(RossmannStore):
    """Payload de saída: dados originais + predição de vendas."""

    prediction: float
