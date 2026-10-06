import numpy as np
import pytest

import src.supervised.cross_validation as cv
from src.supervised.cross_validation import make_user_folds, out_of_fold_scores, repeated_out_of_fold_scores


def _users():
    insiders = {f"I{i:02d}": (1 if i < 4 else 2 if i < 9 else 3) for i in range(12)}
    benign = [f"B{i:02d}" for i in range(8)]
    return insiders, sorted(insiders) + benign


def test_every_user_gets_exactly_one_fold_and_folds_are_balanced():
    insiders, users = _users()
    folds = make_user_folds(users, insiders, n_folds=5, seed=0)
    assert set(folds) == set(users)
    sizes = np.bincount(list(folds.values()), minlength=5)
    assert sizes.max() - sizes.min() <= 1


def test_insiders_of_each_scenario_are_spread_over_the_folds():
    insiders, users = _users()
    folds = make_user_folds(users, insiders, n_folds=4, seed=1)
    for scenario in (1, 2, 3):
        members = [u for u, s in insiders.items() if s == scenario]
        counts = np.bincount([folds[u] for u in members], minlength=4)
        assert counts.max() - counts.min() <= 2
    for fold in range(4):  # every fold contains at least one insider and one benign user
        assert any(folds[u] == fold for u in insiders)
        assert any(folds[u] == fold for u in users if u not in insiders)


def test_folds_are_deterministic_and_depend_on_the_seed():
    insiders, users = _users()
    assert make_user_folds(users, insiders, 5, seed=3) == make_user_folds(users, insiders, 5, seed=3)
    assert make_user_folds(users, insiders, 5, seed=3) != make_user_folds(users, insiders, 5, seed=4)


def test_input_order_does_not_change_the_folds():
    insiders, users = _users()
    assert make_user_folds(users, insiders, 5, seed=0) == make_user_folds(list(reversed(users)), insiders, 5, seed=0)


def test_too_few_users_or_folds_is_an_error():
    with pytest.raises(ValueError):
        make_user_folds(["a", "b"], {}, n_folds=5)
    with pytest.raises(ValueError):
        make_user_folds(["a", "b", "c"], {}, n_folds=1)


def test_a_users_days_never_straddle_training_and_test(monkeypatch):
    seen = []

    def fake_fit_predict(model_name, x_train, y_train, x_test, seed=0):
        seen.append((len(x_train), len(x_test)))
        return np.full(len(x_test), 7.0)

    monkeypatch.setattr(cv, "fit_predict", fake_fit_predict)
    insiders, users = _users()
    day_users = np.repeat(users, 6)  # six days per user
    x = np.arange(len(day_users), dtype=float).reshape(-1, 1)
    y = np.zeros(len(day_users), dtype=int)
    folds = make_user_folds(users, insiders, 5, seed=0)
    scores = out_of_fold_scores("logistic", day_users, x, y, folds)
    assert (scores == 7.0).all()          # every row was scored by exactly one fold's model
    assert len(seen) == 5
    assert all(a + b == len(day_users) for a, b in seen)


def test_out_of_fold_scores_come_from_a_model_that_never_saw_the_user(monkeypatch):
    # The fake "model" memorises the training rows' x values; a test row's score is 1
    # if its own x was in training. Out-of-fold means that must never happen.
    def memorising_fit_predict(model_name, x_train, y_train, x_test, seed=0):
        train_values = set(x_train[:, 0].tolist())
        return np.array([1.0 if v in train_values else 0.0 for v in x_test[:, 0]])

    monkeypatch.setattr(cv, "fit_predict", memorising_fit_predict)
    insiders, users = _users()
    day_users = np.repeat(users, 4)
    x = np.arange(len(day_users), dtype=float).reshape(-1, 1)
    y = np.zeros(len(day_users), dtype=int)
    scores = out_of_fold_scores("logistic", day_users, x, y, make_user_folds(users, insiders, 5, seed=2))
    assert scores.sum() == 0


def test_repeats_use_different_fold_assignments(monkeypatch):
    def echo_fit_predict(model_name, x_train, y_train, x_test, seed=0):
        return np.full(len(x_test), float(seed))

    monkeypatch.setattr(cv, "fit_predict", echo_fit_predict)
    insiders, users = _users()
    day_users = np.repeat(users, 2)
    x = np.zeros((len(day_users), 1))
    y = np.zeros(len(day_users), dtype=int)
    matrix = repeated_out_of_fold_scores("logistic", day_users, x, y, insiders, n_folds=4, n_repeats=3, seed=10)
    assert matrix.shape == (3, len(day_users))
    # repeat r passes seed 10 + r to the model fits; each fold of repeat r got that seed
    assert set(np.unique(matrix[0])) == {10.0} and set(np.unique(matrix[2])) == {12.0}
