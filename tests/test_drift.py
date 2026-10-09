import numpy as np
from src.drift import psi


def test_psi_same_distribution_is_low():
    x = np.random.default_rng(0).normal(0, 1, 5000)
    y = np.random.default_rng(1).normal(0, 1, 5000)
    assert psi(x, y) < 0.1


def test_psi_shifted_distribution_is_high():
    x = np.random.default_rng(0).normal(0, 1, 5000)
    y = np.random.default_rng(1).normal(2, 1, 5000)
    assert psi(x, y) > 0.25


def test_concept_drift_detected_on_synthetic_data():
    from src.generate_data import make_train, make_live
    from src.concept import fit_baseline, concept_drift_test
    train, live = make_train(), make_live()
    feats = ["age", "income", "credit_score", "tenure_months"]
    model = fit_baseline(train, feats, "default")
    assert concept_drift_test(model, train, live, feats, "default")["concept_drift"]
