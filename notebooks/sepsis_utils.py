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


def calibration_bins(p, n_bins=10, strategy="quantile"):
    """Bin number (0 to n_bins - 1) of each predicted risk.

    - "quantile" (default): equal-count bins, each holds about the same number of hours.
      Needed here because almost all predicted risks are small: with equal-width bins,
      nearly every hour falls in the first bin (0-10 %) and the other bins are almost empty.
    - "uniform": equal-width bins of predicted risk (0-10 %, 10-20 %, ...).
    """
    p = np.asarray(p, dtype=float)
    if strategy == "uniform":
        return np.minimum((p * n_bins).astype(int), n_bins - 1)
    inner_edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))[1:-1]
    return np.searchsorted(inner_edges, p, side="right")


def expected_calibration_error(y, p, n_bins=10, strategy="quantile"):
    """ECE: gap between predicted risk and observed frequency in each bin, averaged over the
    bins and weighted by the share of hours in each bin. 0 = perfectly calibrated."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    bins = calibration_bins(p, n_bins, strategy)
    ece = 0.0
    for b in np.unique(bins):
        in_bin = bins == b
        ece += in_bin.mean() * abs(y[in_bin].mean() - p[in_bin].mean())
    return ece


def reliability_table(y, p, n_bins=10, strategy="quantile"):
    """Per bin of predicted risk: number of hours, mean predicted risk, observed share of sepsis."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    df = pd.DataFrame({"bin": calibration_bins(p, n_bins, strategy), "y": y, "p": p})
    return (df.groupby("bin")
              .agg(n_hours=("y", "size"), mean_predicted=("p", "mean"), observed=("y", "mean"))
              .reset_index())


def predict_risk(bundle, X):
    """Risk predicted by the saved baseline: raw logistic probability and Platt-recalibrated one."""
    p_raw = bundle["pipeline"].predict_proba(X[bundle["features"]])[:, 1]
    p_cal = bundle["platt"].predict_proba(logit(p_raw).reshape(-1, 1))[:, 1]
    return p_raw, p_cal


# --------------------------------------------------------------------------- evaluation (notebook 05)

class FastMetrics:
    """AUPRC, AUROC, Brier and ECE of one score vector, computed for many weight vectors.

    In a patient bootstrap, each row gets a weight = number of times its patient was drawn.
    Re-sorting 760 000 scores 1 000 times would be slow, so the sort is done once here; each
    call with new weights is then a few cumulative sums. With all weights equal to 1 the results
    equal scikit-learn's average_precision_score, roc_auc_score and brier_score_loss.
    """

    def __init__(self, y, score, is_probability=True, n_bins=10):
        self.y = np.asarray(y, dtype=float)
        self.score = np.asarray(score, dtype=float)
        self.order = np.argsort(-self.score, kind="mergesort")          # highest score first
        sorted_score = self.score[self.order]
        self.y_sorted = self.y[self.order]
        # last row of each group of tied scores (the curves only change between different scores)
        self.ends = np.r_[np.flatnonzero(np.diff(sorted_score) != 0), len(sorted_score) - 1]
        self.is_probability = is_probability
        self.n_bins = n_bins
        self.bins = calibration_bins(self.score, n_bins) if is_probability else None

    def compute(self, w=None, ranking=True, calibration=True):
        w = np.ones(len(self.y)) if w is None else np.asarray(w, dtype=float)
        out = {}
        if ranking:
            ws = w[self.order]
            tp = np.cumsum(ws * self.y_sorted)[self.ends]          # weighted true positives above each threshold
            fp = np.cumsum(ws * (1 - self.y_sorted))[self.ends]    # weighted false positives above each threshold
            recall, fpr = tp / tp[-1], fp / fp[-1]
            out["AUROC"] = np.trapezoid(np.r_[0, recall], np.r_[0, fpr])
            # precision at each threshold; 0 where no drawn row is above it (recall does not move there)
            precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
            out["AUPRC"] = np.sum(np.diff(np.r_[0, recall]) * precision)
            out["prevalence"] = np.sum(w * self.y) / np.sum(w)
            out["AUPRC_lift"] = out["AUPRC"] / out["prevalence"]   # how many times better than random
        if calibration and self.is_probability:
            out["Brier"] = np.sum(w * (self.score - self.y) ** 2) / np.sum(w)
            n_in_bin = np.bincount(self.bins, weights=w, minlength=self.n_bins)
            sum_y = np.bincount(self.bins, weights=w * self.y, minlength=self.n_bins)
            sum_p = np.bincount(self.bins, weights=w * self.score, minlength=self.n_bins)
            out["ECE"] = np.sum(np.abs(sum_y - sum_p)) / np.sum(n_in_bin)
        return out


def alert_metrics(y, score, threshold, w=None):
    """What happens if an alert is raised for every hour with score >= threshold.
    One row = one patient-hour, so 24 rows = one patient-day."""
    y = np.asarray(y).astype(bool)
    alert = np.asarray(score) >= threshold
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=float)
    tp, fp = np.sum(w * (alert & y)), np.sum(w * (alert & ~y))
    n_pos, n = np.sum(w * y), np.sum(w)
    return {
        "alert_rate": (tp + fp) / n,
        "sensitivity": tp / n_pos,
        "PPV": tp / (tp + fp) if tp + fp > 0 else np.nan,
        "false_alarms_per_patient_day": fp / (n / 24),
    }


def patient_bootstrap(patient_ids, compute, n_boot=1000, seed=42):
    """Call compute(w) on n_boot resamples of the patients (drawn with replacement).
    All hours of a drawn patient are kept: the weight of a row = number of times its patient
    was drawn. Returns one row per resample."""
    codes, uniques = pd.factorize(np.asarray(patient_ids))
    n = len(uniques)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, n, n), minlength=n)
        rows.append(compute(counts[codes].astype(float)))
    return pd.DataFrame(rows)


def percentile_ci(replicates, level=0.95):
    """95 % percentile interval of each column of a table of bootstrap replicates."""
    a = (1 - level) / 2
    return replicates.quantile(a), replicates.quantile(1 - a)


def bootstrap_p_value(differences):
    """Two-sided p-value for 'the difference is 0', from bootstrap replicates of a difference.
    The +1 keeps it above 0: with 1 000 replicates the smallest possible value is about 0.002."""
    d = np.asarray(differences, dtype=float)
    return min(1.0, 2 * min(np.sum(d <= 0) + 1, np.sum(d >= 0) + 1) / (len(d) + 1))


def holm(p_values):
    """Holm correction for several tests: adjusted p-values, same order as the input."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(running, 1.0)
    return adjusted
