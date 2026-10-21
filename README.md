# 🏪 Rossmann Store Sales — Previsão de Vendas em Produção

> Previsão diária de vendas de **1.115 lojas** da Rossmann. O **LightGBM tunado** foi escolhido para substituir o Random Forest de **7,3 GB**, reduzindo o artefato para **20,45 MB** e melhorando o MAPE no holdout.

Projeto completo de **Data Science em Produção**: da análise exploratória ao deploy do modelo em uma API REST.

---

## ✨ Destaques

- 🏆 **LightGBM tunado** no holdout: **MAPE 10,14%**, **MAE 674,09** e artefato de **20,45 MB**
- 💰 Projeção no holdout: **R$ 293.910.700,44**
- 📦 **Random Forest anterior:** MAPE 11,0% e MAE 727,10, mas artefato de **7,3 GB**
- 📊 Comparação de **12 modelos base** + tuning do LightGBM
- 🔬 Seleção de variáveis com **Boruta**
- ⏱️ Validação temporal com **walk-forward de 42 dias**
- 🚀 **API FastAPI** servindo o modelo (`POST /rossman/predict`)
- 🧪 **129 testes automatizados** com pytest, cobrindo unidade, contrato e integração

---

## 🔁 Atualização do Modelo — Random Forest → LightGBM

O Random Forest foi a primeira escolha: era o melhor no holdout (**MAPE 11,0%**, **MAE 727,10**) e estava à frente do LightGBM baseline na validação temporal. Porém, o artefato serializado tinha **7,3 GB**, o que tornava o deploy caro: exigia muita RAM e aumentava o tempo de carregamento. Em ambiente de 16 GB, com ~2,2 GB livres, o modelo não carregava. Antes de fazer o deploy, era necessário testar outros modelos da lista para buscar uma resposta parecida com um artefato menor.

Para resolver isso, testei outro modelo da lista: o **LightGBM**. Usei o mesmo split temporal, as mesmas 16 features e as mesmas 5 janelas walk-forward de 42 dias, com **grid search** e scorer **MAPE na escala original**.

| Modelo | MAPE | MAE | RMSE | Tamanho do artefato |
|---|---:|---:|---:|---:|
| Random Forest (anterior) | 11,0% | 727,10 | 1.049,61 | 7,3 GB |
| **LightGBM tunado** | **10,14%** | **674,09** | **970,52** | **20,45 MB** |

O LightGBM tunado ficou **0,86 ponto percentual melhor no MAPE** e **~357× menor**. O gate de 1 GB passou com folga. A configuração escolhida foi `900` árvores, `255` folhas, `learning_rate=0.1` e `min_child_samples=50`. O grid search foi usado para ajustar a qualidade; a redução do artefato vem da troca de modelo e dos limites de folhas/árvores, não do grid sozinho.

> O baseline LightGBM de 23,2% estava underfit; após tuning, o modelo passou a superar o Random Forest. O melhor ponto ficou na borda do grid, então ainda existe espaço para novos testes, mas o resultado atual já atende ao critério de qualidade e tamanho.

Projeção de negócio no holdout: **R$ 293.910.700,44**.

O LightGBM é o modelo escolhido para substituir o Random Forest. A API carrega `models/model_rossmann.pkl`, que já contém o LightGBM. O RF antigo deve ser mantido como backup para rollback.

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
| 1 | **LightGBM Regressor (tunado)** | **674,09** | **10,14%** | **970,52** |
| 2 | Random Forest Regressor | 727,10 | 11,0% | 1.049,61 |
| 3 | Extra Trees Regressor | 740,67 | 11,3% | 1.060,56 |
| 4 | Decision Tree Regressor | 1.014,90 | 14,9% | 1.517,56 |
| 5 | Hist Gradient Boosting | 1.318,06 | 19,7% | 1.900,94 |
| 6 | LightGBM Regressor (baseline) | 1.541,63 | 23,2% | 2.198,61 |
| 7 | XGBoost Regressor | 1.668,35 | 24,9% | 2.427,98 |
| 8 | CatBoost Regressor | 1.701,92 | 25,6% | 2.446,45 |
| 9 | Elastic Net Regressor | 1.880,70 | 28,8% | 2.721,73 |
| 10 | Lasso (Linear Regularized) | 1.891,85 | 29,0% | 2.740,80 |
| 11 | Linear Regression | 1.876,96 | 29,9% | 2.666,29 |
| 12 | Ridge Regression | 1.876,87 | 29,9% | 2.666,01 |
| 13 | Baseline (média por loja) | 1.354,80 | 45,5% | 1.835,14 |

> O LightGBM tunado passou por `GridSearchCV` com scorer MAPE. O Random Forest também passou por `GridSearchCV`, mas foi limitado pelo tamanho do artefato. Os demais modelos de boosting foram avaliados com hiperparâmetros básicos, **sem ajuste fino**.

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

> Na comparação inicial, o Random Forest estava à frente do LightGBM baseline na CV (**12% ± 3%** vs **23% ± 1%** de MAPE). Após o tuning, o LightGBM tunado chegou a **11,36% ± 1,62** de MAPE CV e superou o RF no holdout. MAE e RMSE CV do LightGBM tunado não foram registrados nessa execução do grid.

> O Extra Trees teve CV levemente melhor que o Random Forest, mas o Random Forest venceu no holdout inicial. Depois, o LightGBM tunado superou ambos no holdout e foi escolhido para produção.

---

## 💰 Resultado de Negócio — LightGBM tunado

Projeção para o período de teste (19/06/2015 a 31/07/2015), somando todas as lojas:

| Cenário | Valor previsto |
|---|---:|
| 📈 Previsão | R$ 293.910.700,44 |
| ⚠️ Cenário pessimista | R$ 293.154.958,54 |
| ✅ Cenário otimista | R$ 294.666.442,33 |

> Projeção calculada com o LightGBM tunado no holdout.

---

## 🧠 Modelo Escolhido

**Modelo escolhido:** LightGBM Regressor tunado

Hiperparâmetros escolhidos via `GridSearchCV` + validação temporal:

```python
{
    "n_estimators": 900,
    "learning_rate": 0.1,
    "num_leaves": 255,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
}
```

Detalhes:

- **Seleção de variáveis:** Boruta em amostra de 100.000 registros com `random_state=42`
- **Features finais:** 16 variáveis (sem `date` e sem `sales` para evitar vazamento)
- **Target:** vendas transformadas com `log1p`
- **Artefato:** ~20,45 MB (vs 7,3 GB do Random Forest)
- **Projeção no holdout:** R$ 293.910.700,44
- **Arquivo servido pela API:** `models/model_rossmann.pkl` (LightGBM)
- **Rollback:** manter uma cópia do Random Forest antigo, se necessário

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

### Endpoints auxiliares

```text
GET /health        → status da API e modelo carregado
GET /model-info    → tipo do modelo, quantidade e nomes das features
```

### Validações da API

- Rejeita payload vazio (`400`)
- Rejeita lojas fechadas (`Open != 1`) (`422`)
- Rejeita categorias desconhecidas e campos extras (`422`)
- Rejeita datas fora do período suportado (`422`)
- Limite de **10.000 registros** por requisição (`413`)
- Limite de **10 MB** para o corpo JSON, validado antes do parse (`413`)

---

## 📁 Estrutura do Projeto

```
.
├── api/
│   ├── __init__.py
│   ├── app.py                  # API FastAPI
│   ├── config/
│   │   └── config.py           # Caminhos, limites e período suportado
│   ├── models/
│   │   └── schemas.py          # Schemas Pydantic de entrada/saída
│   ├── services/
│   │   ├── __init__.py
│   │   └── rossman.py          # Pipeline de transformação + predição
│   └── tests/
│       ├── conftest.py         # Fixtures compartilhadas (pytest)
│       ├── test_api.py         # Integração do endpoint (TestClient)
│       ├── test_app_unit.py    # Unidade da camada HTTP
│       ├── test_rossman.py     # Integração do pipeline
│       ├── test_rossman_unit.py # Unidade do pipeline
│       └── test_schemas.py     # Unidade dos schemas Pydantic
├── models/
│   └── model_rossmann.pkl      # Modelo final (LightGBM tunado)
├── parameters/                 # Scalers + contrato de features
├── utils/                      # Métricas e validação cruzada
└── main.ipynb                  # Análise, modelagem e salvamento do modelo
```

---

## 📥 Artefatos do Modelo e Dados

Os arquivos `models/model_rossmann.pkl` e `parameters/*.pkl` **não são versionados no Git** para manter o repositório leve. Para rodar a API ou os testes, é preciso gerá-los localmente antes:

1. Baixe os dados do Kaggle:
   ```bash
   uv run python db/db.py
   ```
   O download usa o `kagglehub`. Se você já tiver os dados em outro diretório, pule esta etapa e aponte a variável de ambiente `ROSSMANN_DATA_DIR` para a pasta que contém `train.csv`, `test.csv` e `store.csv`.

2. Execute o notebook `main.ipynb` até o final. As células das seções 5 e 8 salvam os arquivos de transformação em `parameters/` e o modelo final LightGBM em `models/model_rossmann.pkl`.

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

# 4. Executar os testes (unidade + integração)
uv run pytest

# 5. Executar apenas o gate de integração com modelo e dados reais
uv run pytest -m integration --no-cov
```

> A suíte roda com cobertura de branches e exige no mínimo **90%** em `api/`. Os testes de integração dependem do artefato `models/model_rossmann.pkl` e dos CSVs do Kaggle.

Caso os dados do Kaggle estejam em outro local, informe o diretório via variável de ambiente:

```bash
# Linux/macOS
ROSSMANN_DATA_DIR="/caminho/para/rossmann-store-sales" uv run pytest

# Windows (PowerShell)
$env:ROSSMANN_DATA_DIR="C:\caminho\para\rossmann-store-sales"; uv run pytest
```

---

## 🛠️ Stack

🐍 Python · 📊 Pandas · 🌿 LightGBM · ⚙️ Scikit-learn · 🐱 CatBoost · 🔬 Boruta · ⚡ FastAPI · 🚀 Uvicorn · 🧪 pytest · 🧹 Ruff · 📦 uv
