import numpy as np
import pandas as pd
from scipy import stats


def cramer_v(x, y):
    cm = pd.crosstab(x, y).to_numpy()
    n = cm.sum()
    r, k = cm.shape

    chi2 = stats.chi2_contingency(cm)[0]
    chi2corr = max(0, chi2 - (k - 1) * (r - 1) / (n - 1))

    kcorr = k - ((k - 1) ** 2) / (n - 1)
    rcorr = r - ((r - 1) ** 2) / (n - 1)

    return np.sqrt((chi2corr / n) / (min(kcorr - 1, rcorr - 1)))
