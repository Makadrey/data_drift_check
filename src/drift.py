"""Drift metrics: PSI and KS, per-feature report, and drift over time."""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def psi(expected, actual, bins=10):
    """Population Stability Index using quantile bins from the expected (train) data."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_label(v):
    return "stable" if v < 0.1 else "moderate" if v < 0.25 else "significant"


def drift_report(train, live, features):
    rows = []
    for f in features:
        stat, p = ks_2samp(train[f], live[f])
        v = psi(train[f], live[f])
        rows.append({"feature": f, "psi": round(v, 4), "psi_status": psi_label(v),
                     "ks_stat": round(stat, 4), "ks_pvalue": round(p, 6),
                     "ks_drift": p < 0.05})
    return pd.DataFrame(rows).sort_values("psi", ascending=False)


def drift_over_time(train, live, features, date_col="date", freq="W"):
    """PSI and KS per feature for each time window of live data."""
    live = live.assign(**{date_col: pd.to_datetime(live[date_col])})
    rows = []
    for period, g in live.groupby(pd.Grouper(key=date_col, freq=freq)):
        if len(g) < 50:
            continue
        for f in features:
            rows.append({"period": period, "feature": f,
                         "psi": psi(train[f], g[f]),
                         "ks_stat": ks_2samp(train[f], g[f])[0]})
    return pd.DataFrame(rows)
