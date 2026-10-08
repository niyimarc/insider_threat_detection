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
import pandas as pd

from src.supervised.cross_validation import repeated_out_of_fold_scores
from src.supervised.dataset import WINDOW_LABEL
from src.supervised.experiment import _matrix
from src.supervised.models import date_matched_weights, fit_model


def align_calendar_origin(score_table, reference_table):
    """
    A copy of `score_table` whose `calendar_day_index` counts days from the EARLIEST DAY OF
    `reference_table` (add_calendar_controls counts from each table's own earliest day, so a model
    trained on one table and applied to another would otherwise see shifted dates).
    """
    out = score_table.copy()
    shift = (pd.to_datetime(score_table["day"]).min() - pd.to_datetime(reference_table["day"]).min()).days
    out["calendar_day_index"] = out["calendar_day_index"] + float(shift)
    return out


def fit_and_score(train_table, score_table, model_name, columns, seeds=(0, 1, 2), date_matched=True, strata_days=7):
    """
    Trains the model on EVERY row of `train_table` (window label; DATE-MATCHED weights if asked)
    once per seed and returns the seed-mean score of every row of `score_table`. The score table's
    users are never seen in training: this is the frozen-detector path used on new users.
    """
    labels = train_table[WINDOW_LABEL].to_numpy().astype(int)
    weights = None
    if date_matched:
        strata = (train_table["calendar_day_index"].to_numpy() // strata_days).astype(int)
        weights = date_matched_weights(labels, strata)
    x_train = _matrix(model_name, train_table, columns)
    x_score = _matrix(model_name, score_table, columns)
    scores = [fit_model(model_name, x_train, labels, seed=s, sample_weight=weights)(x_score) for s in seeds]
    return np.mean(np.asarray(scores, dtype=float), axis=0)


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
