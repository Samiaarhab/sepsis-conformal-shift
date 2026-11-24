"""Small helpers shared by notebooks 04 (baseline) and 05 (evaluation).

They live in a .py file, not in a notebook, so that the model saved in notebook 04
can be loaded again in notebook 05 (loading a saved model needs the code of its steps).
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip each column to its [lower, upper] percentiles.

    The limits are learned in fit(), i.e. on the training rows only (inside the Pipeline,
    so also inside each cross-validation fold). Columns that only contain 0 and 1 (the
    "never measured" indicators) are not clipped: clipping a rare indicator at its 99.5th
    percentile would turn it into a column of zeros. Missing values stay missing: the
    imputer is the next step of the Pipeline.
    """

    def __init__(self, lower=0.5, upper=99.5):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # a column with no value at all
            self.low_ = np.nanpercentile(X, self.lower, axis=0)
            self.high_ = np.nanpercentile(X, self.upper, axis=0)
        self.binary_ = np.all(np.isnan(X) | (X == 0) | (X == 1), axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.array(X, dtype=float)  # a copy: the input table is not modified
        clip = ~self.binary_
        X[:, clip] = np.clip(X[:, clip], self.low_[clip], self.high_[clip])
        return X


def qsofa_score(df):
    """qSOFA-style score from the last observed values: 1 point if Resp >= 22 breaths/min,
    1 point if SBP <= 100 mmHg. The third qSOFA criterion (altered mentation) is not in the
    data. A missing value counts as "criterion not met"."""
    return (df["Resp"] >= 22).astype(int) + (df["SBP"] <= 100).astype(int)


def logit(p, eps=1e-6):
    """log(p / (1 - p)), with p kept away from 0 and 1."""
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def expected_calibration_error(y, p, n_bins=10):
    """ECE: gap between predicted risk and observed frequency, averaged over 10 equal-width
    bins of predicted risk and weighted by the share of hours in each bin. 0 = perfect."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.any():
            ece += in_bin.mean() * abs(y[in_bin].mean() - p[in_bin].mean())
    return ece


def reliability_table(y, p, n_bins=10):
    """Per bin of predicted risk: number of hours, mean predicted risk, observed share of sepsis."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    df = pd.DataFrame({"bin": bins, "y": y, "p": p})
    return (df.groupby("bin")
              .agg(n_hours=("y", "size"), mean_predicted=("p", "mean"), observed=("y", "mean"))
              .reset_index())


def predict_risk(bundle, X):
    """Risk predicted by the saved baseline: raw logistic probability and Platt-recalibrated one."""
    p_raw = bundle["pipeline"].predict_proba(X[bundle["features"]])[:, 1]
    p_cal = bundle["platt"].predict_proba(logit(p_raw).reshape(-1, 1))[:, 1]
    return p_raw, p_cal
