"""
Day scores for the insider DETECTOR, produced so that no user is ever scored
by a model that saw that user (out-of-fold, user-level folds).

`out_of_fold_day_scores` is the single entry point used by the detector
script: it builds the model's input (a feature matrix, or the user's window of
previous days for the GRU), trains with the window label as in every other
supervised experiment (optionally with DATE-MATCHED weights, which keeps the
model from using the calendar), and returns one score per user-day, averaged
over `n_repeats` independent fold assignments.
"""

import numpy as np

from src.supervised.cross_validation import repeated_out_of_fold_scores
from src.supervised.dataset import WINDOW_LABEL
from src.supervised.experiment import _matrix


def out_of_fold_day_scores(day_table, scenario_by_user, model_name, columns, n_folds=5, n_repeats=3, seed=0,
                           date_matched=False, strata_days=7):
    """
    day_table must carry `user`, `day`, the WINDOW_LABEL column, `calendar_day_index` (see
    dataset.add_calendar_controls) and the input `columns`. Returns an (n,) array of scores,
    higher = more suspicious.
    """
    users = day_table["user"].to_numpy()
    labels = day_table[WINDOW_LABEL].to_numpy().astype(int)
    strata = None
    if date_matched:
        strata = (day_table["calendar_day_index"].to_numpy() // strata_days).astype(int)
    x = _matrix(model_name, day_table, columns)
    scores = repeated_out_of_fold_scores(
        model_name, users, x, labels, scenario_by_user,
        n_folds=n_folds, n_repeats=n_repeats, seed=seed, strata=strata,
    )
    return np.asarray(scores, dtype=float).mean(axis=0)
