# 🏪 Rossmann Store Sales — Previsão de Vendas em Produção

> Previsão diária de vendas de **1.115 lojas** da Rossmann usando **Random Forest Regressor**, servida por uma **API FastAPI**.

Projeto completo de **Data Science em Produção**: da análise exploratória ao deploy do modelo em uma API REST.

---

## ✨ Destaques

- 🏆 **Random Forest Regressor** no holdout: **MAPE 11,0%** e **MAE 727,10**
- 📊 Comparação de **12 modelos** com holdout + validação temporal
- 🔬 Seleção de variáveis com **Boruta**
- ⏱️ Validação temporal com **walk-forward de 42 dias**
- 🚀 **API FastAPI** servindo o modelo (`POST /rossman/predict`)
- 🧪 **6 testes automatizados** com pytest

## 📊 Sobre o Projeto

- **Dataset:** [Rossmann Store Sales — Kaggle](https://www.kaggle.com/c/rossmann-store-sales)
- **Volume:** ~1 milhão de registros de vendas (jan/2013 – jul/2015)
- **Objetivo:** prever as vendas diárias de cada loja
- **Metodologia:** pipeline CRISP-DS — limpeza → engenharia de features → seleção de variáveis → modelagem → validação temporal → deploy
- **Split:** treino até 18/06/2015 | holdout de 19/06/2015 a 31/07/2015

---

## 🏆 Resultados — Holdout

> O holdout é o período **não visto pelo modelo** durante o treino.

**Como ler:**
- **MAE:** erro médio absoluto em unidades de venda por loja/dia
- **MAPE:** erro percentual médio — principal métrica de negócio
- **RMSE:** penaliza erros grandes

| # | Modelo | MAE | MAPE | RMSE |
|---:|---|---:|---:|---:|
| 1 | **Random Forest Regressor** | **727,10** | **11,0%** | **1.049,61** |
| 2 | Extra Trees Regressor | 740,67 | 11,3% | 1.060,56 |
| 3 | Decision Tree Regressor | 1.014,90 | 14,9% | 1.517,56 |
| 4 | Hist Gradient Boosting | 1.318,06 | 19,7% | 1.900,94 |
| 5 | LightGBM Regressor | 1.541,63 | 23,2% | 2.198,61 |
| 6 | XGBoost Regressor | 1.668,35 | 24,9% | 2.427,98 |
| 7 | CatBoost Regressor | 1.701,92 | 25,6% | 2.446,45 |
| 8 | Elastic Net Regressor | 1.880,70 | 28,8% | 2.721,73 |
| 9 | Lasso (Linear Regularized) | 1.891,85 | 29,0% | 2.740,80 |
| 10 | Linear Regression | 1.876,96 | 29,9% | 2.666,29 |
| 11 | Ridge Regression | 1.876,87 | 29,9% | 2.666,01 |
| 12 | Baseline (média por loja) | 1.354,80 | 45,5% | 1.835,14 |

> Os modelos de boosting (HistGradientBoosting, LightGBM, CatBoost e XGBoost) foram avaliados com hiperparâmetros básicos, **sem ajuste fino**. O Random Forest passou por `GridSearchCV`.

---

## 🏆 Resultados — Validação Temporal (CV)

Validação walk-forward com 5 janelas de **42 dias**, usando apenas informações do passado para prever o futuro.

| Modelo | MAE CV | MAPE CV | RMSE CV |
|---|---:|---:|---:|
| Extra Trees Regressor | 824,64 ± 206,07 | 11% ± 2% | 1.247,62 ± 356,17 |
| **Random Forest Regressor** | **861,60 ± 256,02** | **12% ± 3%** | **1.315,20 ± 415,12** |
| Decision Tree Regressor | 1.223,73 ± 327,71 | 17% ± 4% | 1.872,59 ± 537,63 |
| Hist Gradient Boosting | 1.501,42 ± 231,40 | 20% ± 2% | 2.166,15 ± 329,81 |
| LightGBM Regressor | 1.699,42 ± 230,70 | 23% ± 1% | 2.436,85 ± 337,70 |
| XGBoost Regressor | 1.861,78 ± 292,89 | 25% ± 1% | 2.690,66 ± 438,33 |
| CatBoost Regressor | 1.869,17 ± 263,54 | 26% ± 1% | 2.692,90 ± 393,83 |
| Elastic Net Regressor | 2.106,51 ± 339,20 | 29% ± 1% | 3.040,89 ± 501,12 |
| Lasso (Linear Regularized) | 2.116,91 ± 340,84 | 29% ± 1% | 3.059,56 ± 501,88 |
| Ridge Regression | 2.079,16 ± 310,18 | 30% ± 1% | 2.969,55 ± 477,49 |
| Linear Regression | 2.079,38 ± 310,44 | 30% ± 1% | 2.969,90 ± 477,73 |

> O Extra Trees teve CV levemente melhor que o Random Forest, mas o Random Forest venceu no holdout e foi o modelo escolhido para produção.

---

## 💰 Resultado de Negócio

Projeção para o período de teste (19/06/2015 a 31/07/2015), somando todas as lojas:

| Cenário | Valor previsto |
|---|---:|
| 📈 Previsão | R$ 298.678.591,58 |
| ⚠️ Cenário pessimista | R$ 297.863.641,65 |
| ✅ Cenário otimista | R$ 299.493.541,51 |

---

## 🧠 Modelo em Produção

**Modelo final:** Random Forest Regressor

Hiperparâmetros escolhidos via `GridSearchCV` + validação temporal:

```python
{
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
}
```

Detalhes:

- **Seleção de variáveis:** Boruta em amostra de 100.000 registros com `random_state=42`
- **Features finais:** 16 variáveis (sem `date` e sem `sales` para evitar vazamento)
- **Target:** vendas transformadas com `log1p`
- **Artefatos:** modelo e scalers serializados em `models/` e `parameters/`
- **Arquivo servido pela API:** `models/model_rossmann.pkl`

---

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

---

## 📁 Estrutura do Projeto

```
.
├── api/
│   ├── app.py                  # API FastAPI
│   ├── services/
│   │   └── rossman.py          # Pipeline de transformação + predição
│   └── tests/
│       ├── conftest.py         # Fixtures compartilhadas (pytest)
│       ├── test_api.py         # Testes do endpoint (TestClient)
│       └── test_rossman.py     # Testes do pipeline
├── models/
│   └── model_rossmann.pkl      # Modelo treinado (Random Forest)
├── parameters/                 # Scalers + contrato de features
├── utils/                      # Métricas e validação cruzada
└── main.ipynb                  # Análise e modelagem (CRISP-DS)
```

---

## 📥 Artefatos do Modelo e Dados

Os arquivos `models/model_rossmann.pkl` e `parameters/*.pkl` **não são versionados no Git** para manter o repositório leve. Para rodar a API ou os testes, é preciso gerá-los localmente antes:

1. Baixe os dados do Kaggle:
   ```bash
   uv run python db/db.py
   ```
   O download usa o `kagglehub`. Se você já tiver os dados em outro diretório, pule esta etapa e aponte a variável de ambiente `ROSSMANN_DATA_DIR` para a pasta que contém `train.csv`, `test.csv` e `store.csv`.

2. Execute o notebook `main.ipynb` até o final. As células das seções 5 e 8 salvam os arquivos de transformação em `parameters/` e o modelo em `models/model_rossmann.pkl`.

3. Confira se os arquivos foram criados:
   ```bash
   ls models/model_rossmann.pkl parameters/*.pkl
   ```

> **Dica:** os testes também usam `ROSSMANN_DATA_DIR` quando definida. Se a variável não existir, eles tentam localizar/baixar os dados automaticamente pelo `kagglehub`.

---

## ▶️ Como Rodar

```bash
# 1. Instalar dependências
uv sync

# 2. Preparar dados e artefatos (necessário apenas na primeira vez)
uv run python db/db.py
# depois execute o main.ipynb até o final para gerar os .pkl

# 3. Subir a API
uv run uvicorn api.app:app --reload

# 4. Executar os testes
uv run pytest
```

Caso os dados do Kaggle estejam em outro local, informe o diretório via variável de ambiente:

```bash
# Linux/macOS
ROSSMANN_DATA_DIR="/caminho/para/rossmann-store-sales" uv run pytest

# Windows (PowerShell)
$env:ROSSMANN_DATA_DIR="C:\caminho\para\rossmann-store-sales"; uv run pytest
```

---

## 🛠️ Stack

🐍 Python · 📊 Pandas · 🌲 Random Forest · ⚙️ Scikit-learn · 🌿 LightGBM · 🐱 CatBoost · 🔬 Boruta · ⚡ FastAPI · 🚀 Uvicorn · 🧪 pytest · 🧹 Ruff · 📦 uv
