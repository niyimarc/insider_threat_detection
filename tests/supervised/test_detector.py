import numpy as np
import pandas as pd
import pytest

from src.supervised.dataset import WINDOW_LABEL, add_calendar_controls
from src.supervised.detector import out_of_fold_day_scores
from src.supervised.user_level import alert_analysis, user_level_auroc, user_scores, windows_from_labels


def _table(seed=0, n_users=20, n_days=40):
    """10 insiders (window from day 20; feature x1 raised inside it) and 10 benign users."""
    rng = np.random.default_rng(seed)
    rows, scenarios = [], {}
    for u in range(n_users):
        user = f"U{u:02d}"
        insider = u < n_users // 2
        if insider:
            scenarios[user] = 1 + u % 3
        for d in range(n_days):
            malicious = int(insider and d >= 20)
            rows.append({
                "user": user, "day": pd.Timestamp("2010-06-01") + pd.Timedelta(days=d),
                "x1": float(rng.normal() + 3.0 * malicious), "x2": float(rng.normal()),
                WINDOW_LABEL: malicious,
            })
    return add_calendar_controls(pd.DataFrame(rows)), scenarios


@pytest.mark.parametrize("model_name", ["logistic", "gru"])
def test_out_of_fold_scores_cover_every_row_and_rank_insiders_above_benign_users(model_name):
    table, scenarios = _table()
    scores = out_of_fold_day_scores(table, scenarios, model_name, ["x1", "x2"], n_folds=4, n_repeats=2)
    assert scores.shape == (len(table),) and np.isfinite(scores).all()

    insiders = set(scenarios)
    by_user = user_scores(table["user"], scores)
    res = user_level_auroc({"m": by_user}, insiders, n_boot=30)
    assert res["m"]["auroc"] > 0.9

    windows = windows_from_labels(table["user"], table["day"], table[WINDOW_LABEL])
    a = alert_analysis(table["user"], table["day"], scores, windows)
    assert a["summary"]["detection_rate"] >= 0.8


def test_date_matched_scoring_runs_and_keeps_the_signal():
    table, scenarios = _table()
    scores = out_of_fold_day_scores(
        table, scenarios, "logistic", ["x1", "x2"], n_folds=4, n_repeats=1, date_matched=True)
    assert scores.shape == (len(table),)
    malicious = table[WINDOW_LABEL].to_numpy() == 1
    assert scores[malicious].mean() > scores[~malicious].mean()


def test_align_calendar_origin_counts_days_from_the_reference_tables_first_day():
    from src.supervised.detector import align_calendar_origin
    reference, _ = _table(n_days=10)
    later = reference.copy()
    later["day"] = later["day"] + pd.Timedelta(days=30)
    later = add_calendar_controls(later)                       # counts from its OWN first day (0 ...)
    assert later["calendar_day_index"].min() == 0
    aligned = align_calendar_origin(later, reference)
    assert aligned["calendar_day_index"].min() == 30
    assert later["calendar_day_index"].min() == 0              # the input is not modified


@pytest.mark.parametrize("model_name", ["logistic", "gru"])
def test_fit_and_score_scores_users_the_model_never_saw(model_name):
    from src.supervised.detector import align_calendar_origin, fit_and_score
    train, _ = _table(seed=0)
    other, other_scenarios = _table(seed=5)
    other["user"] = other["user"].str.replace("U", "V")        # different people
    other_scenarios = {u.replace("U", "V"): s for u, s in other_scenarios.items()}
    assert not set(train["user"]) & set(other["user"])
    other = align_calendar_origin(other, train)
    scores = fit_and_score(train, other, model_name, ["x1", "x2"], seeds=(0, 1))
    assert scores.shape == (len(other),) and np.isfinite(scores).all()
    by_user = user_scores(other["user"], scores)
    assert user_level_auroc({"m": by_user}, set(other_scenarios), n_boot=20)["m"]["auroc"] > 0.9


def test_scores_are_deterministic():
    table, scenarios = _table()
    a = out_of_fold_day_scores(table, scenarios, "logistic", ["x1", "x2"], n_folds=4, n_repeats=1, seed=3)
    b = out_of_fold_day_scores(table, scenarios, "logistic", ["x1", "x2"], n_folds=4, n_repeats=1, seed=3)
    assert np.allclose(a, b)
