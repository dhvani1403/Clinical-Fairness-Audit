import numpy as np
import pytest

from subgroup_audit.metrics import rate, summary


def test_rates_on_hand_computed_example():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([1, 0, 0, 0, 1, 1, 0, 0])
    assert rate(y, p, "fpr") == pytest.approx(0.25)
    assert rate(y, p, "fnr") == pytest.approx(0.5)
    assert rate(y, p, "positive_rate") == pytest.approx(3 / 8)
    s = summary(y, p)
    assert s["accuracy"] == pytest.approx(5 / 8)
    assert s["balanced_accuracy"] == pytest.approx(1 - (0.25 + 0.5) / 2)


def test_probabilistic_predictions_give_expected_rates():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.5, 0.1, 0.9, 0.3])
    assert rate(y, p, "fpr") == pytest.approx(0.3)
    assert rate(y, p, "fnr") == pytest.approx(0.4)


def test_rate_with_mask_and_empty_mask():
    y = np.array([0, 0, 1])
    p = np.array([1, 0, 1])
    assert rate(y, p, "fpr", mask=np.array([True, False, True])) == 1.0
    assert np.isnan(rate(y, p, "fpr", mask=np.array([False, False, True])))


def test_unknown_metric_rejected():
    with pytest.raises(ValueError):
        rate(np.array([0]), np.array([0]), "precision")
