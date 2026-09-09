"""Fixtures compartilhadas (pytest).

Os dados do Kaggle são localizados de forma portátil:

1. Se a variável de ambiente ``ROSSMANN_DATA_DIR`` estiver definida, usa esse diretório.
2. Caso contrário, baixa/localiza os dados através do ``kagglehub``.
"""

import os
from pathlib import Path
from pickle import load

import kagglehub
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.config.config import MODEL_PATH
from api.services.rossman import Rossman


def get_data_dir() -> Path:
    """Resolve o diretório com os CSVs do Kaggle."""
    env_dir = os.getenv("ROSSMANN_DATA_DIR")
    if env_dir:
        data_dir = Path(env_dir).expanduser().resolve()
    else:
        try:
            data_dir = Path(kagglehub.competition_download("rossmann-store-sales")).resolve()
        except Exception as exc:
            raise RuntimeError(
                "Não foi possível baixar/localizar os dados da Rossmann pelo kagglehub. "
                "Defina ROSSMANN_DATA_DIR apontando para a pasta que contém os arquivos "
                "train.csv, test.csv e store.csv, ou execute `uv run python db/db.py`."
            ) from exc

    required = ["test.csv", "store.csv"]
    missing = [name for name in required if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Arquivos obrigatórios ausentes em {data_dir}: {missing}. "
            "Execute `uv run python db/db.py` ou defina ROSSMANN_DATA_DIR."
        )
    return data_dir


def require_model() -> Path:
    """Garante que o artefato do modelo exista, com mensagem orientativa."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Modelo não encontrado em {MODEL_PATH}. "
            "Os artefatos não são versionados no Git; execute o notebook main.ipynb "
            "para gerar models/model_rossmann.pkl e os scalers em parameters/."
        )
    return MODEL_PATH


@pytest.fixture
def df_raw() -> pd.DataFrame:
    """Dados reais de teste (loja 22, apenas lojas abertas), no formato bruto da API."""
    data_dir = get_data_dir()
    df_test = pd.read_csv(data_dir / "test.csv")
    df_store_raw = pd.read_csv(data_dir / "store.csv")

    df_test = pd.merge(df_test, df_store_raw, how="left", on="Store")
    df_test = df_test[df_test["Store"] == 22]
    df_test = df_test[df_test["Open"] != 0]
    df_test = df_test[~df_test["Open"].isnull()]
    return df_test.drop("Id", axis=1)


@pytest.fixture(scope="session")
def pipeline() -> Rossman:
    """Instância do pipeline com os scalers carregados."""
    return Rossman()


@pytest.fixture(scope="session")
def model():
    """Modelo treinado carregado do disco."""
    with MODEL_PATH.open("rb") as file:
        return load(file)


@pytest.fixture(scope="session")
def client() -> TestClient:
    """Cliente de teste do FastAPI com lifespan executado."""
    require_model()

    # Import tardio: api.app registra a aplicação sem carregar o modelo no import.
    from api.app import app  # noqa: PLC0415

    with TestClient(app) as test_client:
        yield test_client
