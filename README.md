# 🏪 Rossmann Store Sales — Previsão de Vendas em Produção

> Previsão diária de vendas de **1.115 lojas** da Rossmann usando **XGBoost**, servida por uma **API FastAPI** pronta para produção.

Projeto completo de **Data Science em Produção**: da análise exploratória ao deploy do modelo em uma API REST.

---

## ✨ Destaques

- 🏆 **MAPE de 13,3%** no teste — ~3,4x melhor que o modelo baseline
- 🌲 **Melhor modelo: XGBoost Regressor** com seleção de variáveis via **Boruta**
- 🚀 **API em produção** com FastAPI que serve o modelo (endpoint `POST /rossman/predict`)
- 🧪 **5 testes automatizados** com pytest — pipeline e endpoint cobertos

## 📊 Sobre o Projeto

- **Dataset:** [Rossmann Store Sales — Kaggle](https://www.kaggle.com/c/rossmann-store-sales)
- **Volume:** ~1 milhão de registros de vendas (jan/2013 – jul/2015)
- **Objetivo:** prever as vendas diárias de cada loja
- **Metodologia:** pipeline CRISP-DS — limpeza → engenharia de features → seleção de variáveis → modelagem → validação cruzada → deploy

## 🏆 Resultados

| Modelo | MAE | MAPE | RMSE |
|---|---|---|---|
| Baseline (média) | 1.354,80 | 45,5% | 1.835,14 |
| **XGBoost Regressor (tuned)** | **949,74** | **13,3%** | **1.496,36** |

Outros modelos avaliados no processo: Linear Regression, Lasso e Random Forest.

## 🧠 Modelo em Produção

- **XGBoost Regressor** com ajuste fino de hiperparâmetros (Random Search + validação cruzada 5-fold)
- **Seleção de variáveis** com Boruta
- Scalers e modelo serializados em `parameters/` e `models/`
- O arquivo `models/model_rossmann.pkl` é o que a API carrega na inicialização

## 🚀 API em Produção

API REST construída com **FastAPI** + **Uvicorn** que recebe dados de lojas e retorna a previsão de vendas.

```
POST /rossman/predict
```

**Request:**
```json
[
  {
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
    "CompetitionOpenSinceMonth": null,
    "CompetitionOpenSinceYear": null,
    "Promo2": 1,
    "Promo2SinceWeek": 22.0,
    "Promo2SinceYear": 2012.0,
    "PromoInterval": "Jan,Apr,Jul,Oct"
  }
]
```

**Response:**
```json
[
  {
    "Store": 22,
    "DayOfWeek": 4,
    "Date": "2015-09-17",
    "Open": 1.0,
    "Promo": 1,
    "StateHoliday": "0",
    "SchoolHoliday": 0,
    "StoreType": "a",
    "Assortment": "a",
    "CompetitionDistance": 1040.0,
    "CompetitionOpenSinceMonth": null,
    "CompetitionOpenSinceYear": null,
    "Promo2": 1,
    "Promo2SinceWeek": 22.0,
    "Promo2SinceYear": 2012.0,
    "PromoInterval": "Jan,Apr,Jul,Oct",
    "prediction": 2805.94
  }
]
```

📘 Documentação interativa (Swagger): `http://localhost:8000/docs`

## 📁 Estrutura do Projeto

```
.
├── api/
│   ├── app.py                  # API FastAPI
│   ├── services/
│   │   └── rossman.py          # Pipeline de transformação + predição
│   └── tests/
│       ├── conftest.py         # Fixtures compartilhadas (pytest)
│       ├── test_api_tester.py  # Testes do endpoint (TestClient)
│       └── test_rossman.py     # Testes do pipeline
├── models/
│   └── model_rossmann.pkl      # Modelo treinado (XGBoost)
├── parameters/                 # Scalers serializados
├── utils/                      # Métricas e validação cruzada
└── main.ipynb                  # Análise e modelagem (CRISP-DS)
```

## ▶️ Como Rodar

```bash
# 1. Instalar dependências
uv sync

# 2. Subir a API
uv run uvicorn api.app:app --reload

# 3. Executar os testes
uv run pytest
```

## 🛠️ Stack

🐍 Python · 📊 Pandas · 🌲 XGBoost · 🔬 Boruta · ⚙️ Scikit-learn · ⚡ FastAPI · 🚀 Uvicorn · 🧪 pytest · 🧹 Ruff · 📦 uv
