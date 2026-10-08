import numpy as np
import pandas as pd
import pytest

from src.common.preprocessing.feature_groups import available_groups
from src.supervised.dataset import WINDOW_LABEL, day_feature_columns
from src.supervised.models import SEQ_LEN, SEQUENCE_MODEL_NAMES, fit_model
from src.supervised.sequences import build_sequence_matrix


def _small_table():
    rows = []
    for user, days in (("A", [1, 2, 3, 6]), ("B", [10, 11])):
        for k, day in enumerate(days):
            rows.append({"user": user, "day": pd.Timestamp("2010-01-01") + pd.Timedelta(days=day),
                         "f1": 10.0 * (1 if user == "A" else 2) + k, "f2": float(day)})
    return pd.DataFrame(rows)


def test_each_row_sees_its_own_users_previous_days_with_itself_last():
    table = _small_table()
    out = build_sequence_matrix(table, ["f1", "f2"], seq_len=3).reshape(len(table), 3, 3)
    # user A's 4th day (row 3): steps = A's days 2, 3 and 6 (oldest first), all real
    assert out[3, :, 1].tolist() == [2.0, 3.0, 6.0]
    assert out[3, :, 2].tolist() == [1.0, 1.0, 1.0]
    # user A's first day (row 0): two padding steps, then itself
    assert np.isnan(out[0, :2, :2]).all() and out[0, :2, 2].tolist() == [0.0, 0.0]
    assert out[0, 2, :].tolist() == [10.0, 1.0, 1.0]
    # user B's 2nd day (row 5) never sees user A, and has exactly one earlier day
    assert out[5, :, 2].tolist() == [0.0, 1.0, 1.0]
    assert out[5, 1, 1] == 10.0


def test_windows_are_causal_and_independent_of_row_order():
    table = _small_table()
    base = build_sequence_matrix(table, ["f1", "f2"], seq_len=3)

    changed = table.copy()
    changed.loc[3, ["f1", "f2"]] = [999.0, 999.0]            # alter A's LAST day
    after = build_sequence_matrix(changed, ["f1", "f2"], seq_len=3)
    assert np.array_equal(np.nan_to_num(base[:3]), np.nan_to_num(after[:3]))   # earlier days' windows unchanged
    assert np.array_equal(np.nan_to_num(base[4:]), np.nan_to_num(after[4:]))   # the other user unchanged

    shuffled = table.sample(frac=1.0, random_state=1)
    out = build_sequence_matrix(shuffled, ["f1", "f2"], seq_len=3)
    expected = pd.DataFrame(base, index=table.index).loc[shuffled.index].to_numpy()
    assert np.array_equal(np.nan_to_num(out), np.nan_to_num(expected))


def _window_data(n=500, seed=0, d=3):
    """Windows whose LAST step carries the signal (feature 0 is raised on malicious rows)."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.2).astype(int)
    x = rng.normal(size=(n, SEQ_LEN, d + 1))
    x[:, :, d] = 1.0                                         # mask: every step real
    x[:, -1, 0] += 3.0 * y
    x[:, 0, 1] = np.nan                                      # a missing value
    return x.reshape(n, -1), y


def test_gru_learns_a_planted_signal_on_held_out_windows():
    from sklearn.metrics import roc_auc_score
    x, y = _window_data()
    x_test, y_test = _window_data(seed=1)
    scores = fit_model("gru", x, y, seed=0)(x_test)
    assert scores.shape == (len(y_test),)
    assert roc_auc_score(y_test, scores) > 0.85


def test_gru_is_deterministic_and_scores_rows_independently():
    x, y = _window_data(n=300)
    x_test, _ = _window_data(n=40, seed=2)
    a = fit_model("gru", x, y, seed=3)(x_test)
    b = fit_model("gru", x, y, seed=3)(x_test)
    assert np.allclose(a, b)
    predict = fit_model("gru", x, y, seed=3)
    assert np.allclose(predict(x_test[:7]), predict(x_test)[:7], atol=1e-5)


def test_gru_ignores_zero_weight_rows_and_handles_a_single_class():
    x, y = _window_data(n=300)
    weights = np.ones(len(y))
    weights[:100] = 0.0
    garbled = x.copy()
    garbled[:100] = 1e6                                      # rows with weight 0 must not matter
    a = fit_model("gru", x, y, seed=0, sample_weight=weights)(x[100:120])
    b = fit_model("gru", garbled, y, seed=0, sample_weight=weights)(x[100:120])
    assert np.allclose(a, b)
    assert (fit_model("gru", x, np.zeros(len(y), dtype=int))(x[:5]) == 0).all()


def test_gru_rejects_a_matrix_that_is_not_a_window():
    with pytest.raises(ValueError):
        fit_model("gru", np.zeros((10, 5)), np.array([0, 1] * 5))


def test_experiment_runs_with_the_gru_and_skips_importance_for_it():
    from src.supervised.experiment import model_key, run_experiment

    rng = np.random.default_rng(0)
    columns = day_feature_columns(available_groups())
    rows, scenarios = [], {}
    for u in range(12):
        user = f"U{u:02d}"
        insider = u < 6
        if insider:
            scenarios[user] = 1 + u % 3
        for d in range(30):
            malicious = int(insider and d >= 15)
            row = {c: float(rng.normal()) for c in columns}
            row["day_n_sessions"] = 1.0
            if malicious:
                row["http_offbaseline_share_7d__mean"] += 3.0
            row.update({"user": user, "day": pd.Timestamp("2010-06-01") + pd.Timedelta(days=d),
                        WINDOW_LABEL: malicious, "answer_key_label": malicious})
            rows.append(row)
    results, markdown = run_experiment(
        pd.DataFrame(rows), scenarios, model_names=SEQUENCE_MODEL_NAMES, config_names=("file+device+logon", "all"),
        n_folds=3, n_repeats=1, n_boot=20, seed=0,
    )
    models = results["labelings"]["window"]["models"]
    for config in ("file+device+logon", "all", "calendar_only", "user_clock_only"):
        assert model_key("gru", config) in models
    assert results["permutation_importance"] == {} and results["feature_importance"] == {}
    assert "GRU over the previous 14 days" in markdown
