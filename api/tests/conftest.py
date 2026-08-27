from pathlib import Path
from pickle import load

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.services.rossman import Rossman

DATA_DIR = Path(r"C:\Users\Admin\.cache\kagglehub\competitions\rossmann-store-sales")
MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "model_rossmann.pkl"


@pytest.fixture
def df_raw() -> pd.DataFrame:
    """Dados reais de teste (loja 22, apenas lojas abertas), no formato bruto da API."""
    df_test = pd.read_csv(DATA_DIR / "test.csv")
    df_store_raw = pd.read_csv(DATA_DIR / "store.csv")

    df_test = pd.merge(df_test, df_store_raw, how="left", on="Store")
    df_test = df_test[df_test["Store"] == 22]
    df_test = df_test[df_test["Open"] != 0]
    df_test = df_test[~df_test["Open"].isnull()]
    return df_test.drop("Id", axis=1)


@pytest.fixture(scope="session")
def pipeline() -> Rossman:
    """Instância do pipeline com os 5 scalers carregados."""
    return Rossman()


@pytest.fixture(scope="session")
def model():
    """Modelo treinado carregado do disco."""
    with Path.open(MODEL_PATH, "rb") as f:
        return load(f)


@pytest.fixture(scope="session")
def client() -> TestClient:
    """Cliente de teste do FastAPI (roda a aplicação em processo, sem servidor)."""
    return TestClient(app)
