import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from utils.mean_absolute_percentage_error import mean_absolute_percentage_error


def ml_error_regression(model_name, y, yhat):
    mae = mean_absolute_error(y, yhat)
    mape = mean_absolute_percentage_error(y, yhat)
    rmse = np.sqrt(mean_squared_error(y, yhat))

    return pd.DataFrame(
        {"Model Name": model_name, "MAE": mae, "MAPE": mape, "RMSE": rmse}, index=[0]
    )
