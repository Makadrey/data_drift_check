import pandas as pd
from sklearn.metrics import roc_auc_score
from src.drift import drift_report, drift_over_time
from src.concept import fit_baseline, performance_over_time, concept_drift_test
from src.plots import (plot_distributions, plot_drift_over_time,
                       plot_psi_heatmap, plot_performance)

TARGET = "default"
train = pd.read_csv("data/train.csv")
live = pd.read_csv("data/live.csv")
features = [c for c in train.columns if c != TARGET]

# 1) Data drift (features only): PSI + KS
report = drift_report(train, live, features)
report.to_csv("data/drift_report.csv", index=False)
ts = drift_over_time(train, live, features, freq="W")
ts.to_csv("data/drift_over_time.csv", index=False)

# 2) Label drift + concept drift (needs target)
model = fit_baseline(train, features, TARGET)
base_auc = roc_auc_score(train[TARGET], model.predict_proba(train[features])[:, 1])
perf = performance_over_time(model, live, features, TARGET)
perf.to_csv("data/performance_over_time.csv", index=False)
cd = concept_drift_test(model, train, live, features, TARGET)

plot_distributions(train, live, features, "figures/distributions.png")
plot_drift_over_time(ts, "figures/drift_over_time.png")
plot_psi_heatmap(ts, "figures/psi_heatmap.png")
plot_performance(perf, base_auc, "figures/performance_over_time.png")

# 3) Verdict
print("\n=== DATA DRIFT (PSI / KS per feature) ===")
print(report.to_string(index=False))
drifted = report.loc[report.psi >= 0.1, "feature"].tolist()
print(f"\nData drift detected: {bool(drifted)}  -> {drifted}")

rate_shift = perf.target_rate.iloc[-1] - train[TARGET].mean()
print(f"\n=== LABEL DRIFT ===\nTrain default rate {train[TARGET].mean():.3f} | "
      f"last live week {perf.target_rate.iloc[-1]:.3f} (change {rate_shift:+.3f})")

print("\n=== CONCEPT DRIFT (retrain-gap test on recent live data) ===")
print(cd)
print(f"AUC: train {base_auc:.3f} -> first live week {perf.auc.iloc[0]:.3f} "
      f"-> last live week {perf.auc.iloc[-1]:.3f}")
print(f"\nConcept drift detected: {cd['concept_drift']}")
