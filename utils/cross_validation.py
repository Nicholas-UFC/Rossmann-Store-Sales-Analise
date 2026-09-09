import numpy as np
import pandas as pd

from utils.ml_error_regression import ml_error_regression


def criar_folds_validacao_temporal(x_training, n_splits=5, window_days=42):
    """Cria folds posicionais para validação temporal por blocos de datas.

    Cada fold usa:
      - treino: todas as linhas com data anterior ao início da janela;
      - validação: um bloco contínuo de `window_days` dias.

    Retorna uma lista de tuplas (train_idx, validation_idx) no formato aceito
    pelo GridSearchCV/GridSearchCV-like.
    """
    max_date = x_training["date"].max()
    folds = []

    for k in reversed(range(1, n_splits + 1)):
        validation_start_date = max_date - (k * window_days)
        validation_end_date = max_date - ((k - 1) * window_days)

        training_mask = x_training["date"] < validation_start_date
        validation_mask = (x_training["date"] >= validation_start_date) & (
            x_training["date"] <= validation_end_date
        )

        folds.append(
            (
                np.flatnonzero(training_mask.to_numpy()),
                np.flatnonzero(validation_mask.to_numpy()),
            )
        )

    return folds


def cross_validation(x_training, kfold, model_name, model):
    mae_list = []
    mape_list = []
    rmse_list = []

    folds = criar_folds_validacao_temporal(x_training, n_splits=kfold)

    for training_idx, validation_idx in folds:
        # Filtrando o Dataset
        training = x_training.iloc[training_idx]
        validation = x_training.iloc[validation_idx]

        # Datasets de treino e validação
        xtraining = training.drop(columns=["date", "sales"], errors="ignore")
        ytraining = training["sales"]

        xvalidation = validation.drop(columns=["date", "sales"], errors="ignore")
        yvalidation = validation["sales"]

        # Treino e Predição
        m = model.fit(xtraining, ytraining)
        yhat = m.predict(xvalidation)

        # Performance
        result = ml_error_regression(model_name, np.expm1(yvalidation), np.expm1(yhat))

        mae_list.append(result["MAE"])
        mape_list.append(result["MAPE"])
        rmse_list.append(result["RMSE"])

    # Retorno fora do laço FOR
    return pd.DataFrame(
        {
            "Model Name": model_name,
            "MAE CV": f"{np.round(np.mean(mae_list), 2)} +/- {np.round(np.std(mae_list), 2)}",
            "MAPE CV": f"{np.round(np.mean(mape_list), 2)} +/- {np.round(np.std(mape_list), 2)}",
            "RMSE CV": f"{np.round(np.mean(rmse_list), 2)} +/- {np.round(np.std(rmse_list), 2)}",
        },
        index=[0],
    )
