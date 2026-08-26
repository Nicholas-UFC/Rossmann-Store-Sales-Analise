import numpy as np
import pandas as pd

from utils.ml_error_regression import ml_error_regression


def cross_validation(x_training, kfold, model_name, model):
    mae_list = []
    mape_list = []
    rmse_list = []

    for k in reversed(range(1, kfold + 1)):
        # Cálculos com inteiros (6 semanas = 42 dias)
        validation_start_date = x_training["date"].max() - (k * 42)
        validation_end_date = x_training["date"].max() - ((k - 1) * 42)

        # Filtrando o Dataset
        training = x_training[x_training["date"] < validation_start_date]
        validation = x_training[
            (x_training["date"] >= validation_start_date)
            & (x_training["date"] <= validation_end_date)
        ]

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
