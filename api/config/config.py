"""Configurações da API Rossmann."""

from datetime import date
from pathlib import Path

# Caminhos
CONFIG_DIR = Path(__file__).resolve().parent
API_DIR = CONFIG_DIR.parent
PROJECT_ROOT = API_DIR.parent
MODEL_PATH = PROJECT_ROOT / "models" / "model_rossmann.pkl"
PARAMETERS_PATH = PROJECT_ROOT / "parameters"

# Validação de payload
MAX_BATCH_SIZE = 10_000

# Período suportado pelo modelo atual.
# O treino usa dados até jul/2015; o período de competição vai até set/2015.
MIN_SUPPORTED_DATE = date(2013, 1, 1)
MAX_SUPPORTED_DATE = date(2015, 12, 31)
