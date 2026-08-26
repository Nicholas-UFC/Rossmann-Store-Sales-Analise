import numpy as np


def mean_absolute_percentage_error(y, yhat):
    return np.mean(np.abs((y - yhat) / y))
