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


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_fit_model_predictor_matches_fit_predict_and_scores_new_rows_consistently(model_name):
    from src.supervised.models import fit_model
    x, y = _toy()
    x_test, _ = _toy(seed=1)
    predict = fit_model(model_name, x, y, seed=2)
    assert np.allclose(predict(x_test), fit_predict(model_name, x, y, x_test, seed=2))
    assert np.allclose(predict(x_test[:7]), predict(x_test)[:7])   # row-wise: no dependence on the batch


def test_fit_model_with_a_single_class_predicts_zeros_for_any_input():
    from src.supervised.models import fit_model
    predict = fit_model("logistic", np.zeros((5, 2)), np.zeros(5, dtype=int))
    assert (predict(np.ones((3, 2))) == 0).all()


def test_train_medians_and_apply_medians():
    from src.supervised.models import apply_medians, train_medians
    medians = train_medians(np.array([[1.0, np.nan], [3.0, np.nan], [np.inf, np.nan]]))
    assert medians.tolist() == [2.0, 0.0]
    filled = apply_medians(np.array([[np.nan, 4.0]]), medians)
    assert filled.tolist() == [[2.0, 4.0]]


def test_date_matched_weights_balance_every_usable_stratum():
    from src.supervised.models import date_matched_weights
    labels = np.array([1, 0, 0, 0,   1, 1, 0,   0, 0,   1, 1])
    strata = np.array([0, 0, 0, 0,   1, 1, 1,   2, 2,   3, 3])   # stratum 2 has no positive, stratum 3 no negative
    w = date_matched_weights(labels, strata)
    for s in (0, 1):
        rows = strata == s
        assert w[rows & (labels == 1)].sum() == pytest.approx(w[rows & (labels == 0)].sum())
    assert (w[strata == 2] == 0).all() and (w[strata == 3] == 0).all()
    assert w[labels == 1][:1].tolist() == [1.0]
    assert w[(strata == 0) & (labels == 0)].tolist() == pytest.approx([1 / 3] * 3)


def test_date_matched_weights_make_the_label_independent_of_the_stratum():
    from src.supervised.models import date_matched_weights
    rng = np.random.default_rng(0)
    strata = rng.integers(0, 6, 500)
    labels = (rng.random(500) < (0.05 + 0.15 * strata)).astype(int)    # the label strongly depends on the stratum
    w = date_matched_weights(labels, strata)
    shares = [w[(strata == s) & (labels == 1)].sum() / w[strata == s].sum() for s in range(6) if w[strata == s].sum() > 0]
    assert shares == pytest.approx([0.5] * len(shares))


def test_date_matched_weights_fall_back_to_class_balanced_when_nothing_matches():
    from src.supervised.models import date_matched_weights
    labels = np.array([1, 1, 0, 0, 0, 0])
    strata = np.array([0, 0, 1, 1, 1, 1])
    assert date_matched_weights(labels, strata).tolist() == pytest.approx(balanced_sample_weights(labels).tolist())


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_explicit_sample_weights_change_what_is_learned(model_name):
    from src.supervised.models import fit_model
    x, y = _toy()
    x_test, y_test = _toy(seed=1)
    zero_on_positives = np.where(y == 1, 1e-9, 1.0)
    default = fit_model(model_name, x, y, seed=0)(x_test)
    weighted = fit_model(model_name, x, y, seed=0, sample_weight=zero_on_positives)(x_test)
    assert not np.allclose(default, weighted)
