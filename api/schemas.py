"""Schemas Pydantic usados pela API Rossmann."""

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


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
    CompetitionDistance: Optional[float] = Field(default=None, ge=0)
    CompetitionOpenSinceMonth: Optional[int] = Field(default=None, ge=1, le=12)
    CompetitionOpenSinceYear: Optional[int] = Field(default=None, ge=1900, le=2100)
    Promo2: Literal[0, 1]
    Promo2SinceWeek: Optional[float] = Field(default=None, ge=1, le=53)
    Promo2SinceYear: Optional[float] = Field(default=None, ge=1900, le=2100)
    PromoInterval: Optional[str] = None


class RossmannPrediction(RossmannStore):
    """Payload de saída: dados originais + predição de vendas."""

    prediction: float
