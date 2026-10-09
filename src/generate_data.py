"""Synthetic train/live data. Live data has BOTH data drift and concept drift."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)


def _target(df, w_credit):
    """Default probability depends on standardized features. w_credit can change over time."""
    zc = (df["credit_score"] - 650) / 50
    zi = (np.log(df["income"]) - 10.5) / 0.4
    zt = (df["tenure_months"] - 24) / 24
    logit = -1 + w_credit * zc - 0.5 * zi + 0.3 * zt
    return (rng.random(len(df)) < 1 / (1 + np.exp(-logit))).astype(int)


def make_train(n=5000):
    df = pd.DataFrame({
        "age": rng.normal(35, 8, n).clip(18, 80),
        "income": rng.lognormal(10.5, 0.4, n),
        "credit_score": rng.normal(650, 50, n).clip(300, 850),
        "tenure_months": rng.exponential(24, n),
    })
    df["default"] = _target(df, w_credit=-1.0)
    return df


def make_live(days=90, per_day=200):
    frames = []
    for d, date in enumerate(pd.date_range("2025-01-01", periods=days)):
        t = d / days  # drift strength grows 0 -> 1
        df = pd.DataFrame({
            "date": date,
            "age": rng.normal(35 + 10 * t, 8, per_day).clip(18, 80),          # data drift
            "income": rng.lognormal(10.5 + 0.3 * t, 0.4 + 0.2 * t, per_day),  # data drift
            "credit_score": rng.normal(650, 50, per_day).clip(300, 850),       # NO data drift
            "tenure_months": rng.exponential(24 - 10 * t, per_day),            # data drift
        })
        # CONCEPT drift: credit_score slowly stops predicting default
        df["default"] = _target(df, w_credit=-1.0 * (1 - t))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    make_train().to_csv("data/train.csv", index=False)
    make_live().to_csv("data/live.csv", index=False)
    print("Saved data/train.csv and data/live.csv")
