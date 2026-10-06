import numpy as np
import pytest

from src.common.evaluation.bootstrap import safe_auroc
from src.supervised.models import (
    MODEL_NAMES,
    balanced_sample_weights,
    fit_predict,
    impute_with_train_medians,
    signed_log,
)


def _toy(n=600, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.15).astype(int)
    x = rng.normal(size=(n, 4))
    x[:, 0] += 2.0 * y          # one informative column
    x[:, 3] = np.abs(x[:, 3]) * 1000  # a heavy-scale column
    return x, y


def test_balanced_weights_give_both_classes_equal_total_weight():
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    w = balanced_sample_weights(y)
    assert w[y == 1].sum() == pytest.approx(w[y == 0].sum())
    assert w.sum() == pytest.approx(len(y))


def test_balanced_weights_for_a_single_class_are_uniform():
    assert (balanced_sample_weights(np.zeros(5)) == 1.0).all()
    assert (balanced_sample_weights(np.ones(5)) == 1.0).all()


def test_imputation_uses_training_medians_only():
    train = np.array([[1.0, np.nan], [3.0, np.nan], [np.inf, np.nan]])
    test = np.array([[np.nan, 5.0], [10.0, np.nan]])
    a, b = impute_with_train_medians(train, test)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert a[2, 0] == 2.0           # inf replaced by the median of the finite training values (1, 3)
    assert b[0, 0] == 2.0           # test NaN filled from TRAINING, not from the test column
    assert b[0, 1] == 5.0 and b[1, 1] == 0.0  # an all-NaN training column falls back to 0


def test_imputation_does_not_modify_its_inputs():
    train = np.array([[1.0, np.nan]])
    test = np.array([[np.nan, 2.0]])
    impute_with_train_medians(train, test)
    assert np.isnan(train[0, 1]) and np.isnan(test[0, 0])


def test_signed_log_is_odd_and_monotone():
    x = np.array([-100.0, -1.0, 0.0, 1.0, 100.0])
    y = signed_log(x)
    assert (np.diff(y) > 0).all() and y[2] == 0 and np.allclose(y, -y[::-1])


def test_unknown_model_name_is_rejected():
    with pytest.raises(ValueError):
        fit_predict("random_forest", np.zeros((3, 1)), np.array([0, 1, 0]), np.zeros((1, 1)))


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_models_learn_a_planted_signal_on_held_out_rows(model_name):
    x, y = _toy()
    x_test, y_test = _toy(seed=1)
    scores = fit_predict(model_name, x, y, x_test, seed=0)
    assert scores.shape == (len(x_test),) and np.isfinite(scores).all()
    assert safe_auroc(y_test, scores) > 0.8


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_models_handle_nan_features(model_name):
    x, y = _toy()
    x_test, _ = _toy(seed=1)
    x[::7, 1] = np.nan
    x_test[::5, 2] = np.nan
    assert np.isfinite(fit_predict(model_name, x, y, x_test)).all()


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_a_single_class_in_training_gives_zero_scores(model_name):
    x, _ = _toy(n=50)
    scores = fit_predict(model_name, x, np.zeros(50, dtype=int), x[:10])
    assert (scores == 0).all()


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_same_seed_gives_the_same_scores(model_name):
    x, y = _toy()
    x_test, _ = _toy(seed=1)
    a = fit_predict(model_name, x, y, x_test, seed=3)
    b = fit_predict(model_name, x, y, x_test, seed=3)
    assert np.allclose(a, b)
