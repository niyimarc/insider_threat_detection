"""
User-level cross-validation.

The folds split USERS, never days: a user's days all sit in the same
fold. Splitting days would put one person's habits in both training and
test, and the model would be rewarded for recognising the person rather
than the behaviour. Users are stratified by scenario (insiders of each
scenario, and benign users, are spread evenly over the folds), and the
split is deterministic given the seed.

Every row's score is out-of-fold: it comes from a model that never saw
that row's user. Repeating with different fold seeds gives several
independent score vectors, which are averaged into a seed-mean in the
report exactly like the training seeds of the unsupervised models.
"""

import numpy as np

from src.supervised.models import date_matched_weights, fit_model, fit_predict


def make_user_folds(users, scenario_by_user, n_folds=5, seed=0):
    """
    Returns {user: fold_index}. Insiders are stratified by scenario,
    benign users form their own stratum, and strata are dealt round-robin
    (continuing the count across strata) so fold sizes stay balanced.
    """
    users = sorted(set(users))
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if len(users) < n_folds:
        raise ValueError(f"Need at least {n_folds} users for {n_folds} folds, got {len(users)}.")

    rng = np.random.default_rng(seed)
    strata = {}
    for u in users:
        strata.setdefault(int(scenario_by_user.get(u, 0)), []).append(u)

    assignment, counter = {}, 0
    for scenario in sorted(strata):
        members = list(strata[scenario])
        order = rng.permutation(len(members))
        for i in order:
            assignment[members[i]] = counter % n_folds
            counter += 1
    return assignment


def out_of_fold_scores(model_name, day_users, x, y, folds, seed=0, strata=None):
    """
    day_users: (n,) user of each row; x: (n, d) features; y: (n,) 0/1
    training label; folds: {user: fold}. Returns (n,) out-of-fold scores.

    strata: optional (n,) calendar stratum of each row. If given, every
    fold's model is trained with DATE-MATCHED weights computed from that
    fold's training rows only (src.supervised.models.date_matched_weights),
    so the model cannot use the date to predict the label.
    """
    day_users = np.asarray(day_users)
    y = np.asarray(y).astype(int)
    if strata is not None:
        strata = np.asarray(strata)
    fold_of_row = np.array([folds[u] for u in day_users])
    scores = np.zeros(len(day_users), dtype=float)
    for fold in sorted(set(fold_of_row.tolist())):
        test = fold_of_row == fold
        train = ~test
        if set(day_users[train].tolist()) & set(day_users[test].tolist()):
            raise AssertionError("a user appears in both the training and the test side of a fold")
        if strata is None:
            scores[test] = fit_predict(model_name, x[train], y[train], x[test], seed=seed)
        else:
            weights = date_matched_weights(y[train], strata[train])
            scores[test] = fit_predict(model_name, x[train], y[train], x[test], seed=seed, sample_weight=weights)
    return scores


def grouped_permutation_scores(model_name, day_users, x, y, folds, group_to_columns, seed=0, n_permutations=5):
    """
    Permutation importance by SIGNAL GROUP. For each fold the model is
    trained on the other folds' users; its held-out rows are scored once
    as they are (`base`) and again with one group's columns shuffled
    across the held-out rows (all of a group's columns are shuffled
    together by the same row permutation, which keeps the group's
    internal structure and breaks only its link to the label).

    group_to_columns: {group_name: [column indices of x]}
    Returns (base, permuted): base (n,) out-of-fold scores, and
    permuted {group: (n_permutations, n)} out-of-fold scores. A large
    drop in AUROC from base to a group's permuted scores means the model
    relies on that group.
    """
    day_users = np.asarray(day_users)
    y = np.asarray(y).astype(int)
    x = np.asarray(x, dtype=float)
    fold_of_row = np.array([folds[u] for u in day_users])
    rng = np.random.default_rng(seed)
    n = len(day_users)
    base = np.zeros(n, dtype=float)
    permuted = {g: np.zeros((n_permutations, n), dtype=float) for g in group_to_columns}
    for fold in sorted(set(fold_of_row.tolist())):
        test = fold_of_row == fold
        train = ~test
        if set(day_users[train].tolist()) & set(day_users[test].tolist()):
            raise AssertionError("a user appears in both the training and the test side of a fold")
        predict = fit_model(model_name, x[train], y[train], seed=seed)
        x_test = x[test]
        base[test] = predict(x_test)
        for group, columns in group_to_columns.items():
            columns = list(columns)
            for r in range(n_permutations):
                shuffled = x_test.copy()
                order = rng.permutation(len(x_test))
                shuffled[:, columns] = x_test[order][:, columns]
                permuted[group][r, test] = predict(shuffled)
    return base, permuted


def repeated_out_of_fold_scores(model_name, day_users, x, y, scenario_by_user, n_folds=5, n_repeats=3, seed=0,
                                strata=None):
    """
    (n_repeats, n) matrix: repeat r uses fold seed `seed + r`. `strata`
    switches on date-matched training (see out_of_fold_scores).
    """
    day_users = np.asarray(day_users)
    rows = []
    for r in range(n_repeats):
        folds = make_user_folds(day_users, scenario_by_user, n_folds=n_folds, seed=seed + r)
        rows.append(out_of_fold_scores(model_name, day_users, x, y, folds, seed=seed + r, strata=strata))
    return np.vstack(rows)
