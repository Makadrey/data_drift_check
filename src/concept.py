"""Concept drift and label drift checks (need a target column)."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fit_baseline(train, features, target):
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    return model.fit(train[features], train[target])


def performance_over_time(model, live, features, target, date_col="date", freq="W"):
    """Weekly AUC of the train-time model + weekly target rate (label drift)."""
    live = live.assign(**{date_col: pd.to_datetime(live[date_col])})
    rows = []
    for period, g in live.groupby(pd.Grouper(key=date_col, freq=freq)):
        if len(g) < 50 or g[target].nunique() < 2:
            continue
        rows.append({"period": period,
                     "auc": roc_auc_score(g[target], model.predict_proba(g[features])[:, 1]),
                     "target_rate": g[target].mean()})
    return pd.DataFrame(rows)


def concept_drift_test(model, train, live, features, target, recent_frac=0.2, gap=0.02):
    """Retrain-gap test. On recent live data, compare the old model with a model
    retrained on live data. If retraining clearly helps, P(y|X) has changed."""
    recent = live.sort_values("date").tail(int(len(live) * recent_frac))
    fit_part, test_part = train_test_split(recent, test_size=0.5, random_state=0,
                                           stratify=recent[target])
    new = fit_baseline(fit_part, features, target)
    old_auc = roc_auc_score(test_part[target], model.predict_proba(test_part[features])[:, 1])
    new_auc = roc_auc_score(test_part[target], new.predict_proba(test_part[features])[:, 1])
    return {"old_model_auc": round(old_auc, 4), "retrained_auc": round(new_auc, 4),
            "gap": round(new_auc - old_auc, 4), "concept_drift": (new_auc - old_auc) > gap}
