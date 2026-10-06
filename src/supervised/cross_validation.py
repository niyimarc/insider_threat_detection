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

from src.supervised.models import fit_predict


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


def out_of_fold_scores(model_name, day_users, x, y, folds, seed=0):
    """
    day_users: (n,) user of each row; x: (n, d) features; y: (n,) 0/1
    training label; folds: {user: fold}. Returns (n,) out-of-fold scores.
    """
    day_users = np.asarray(day_users)
    y = np.asarray(y).astype(int)
    fold_of_row = np.array([folds[u] for u in day_users])
    scores = np.zeros(len(day_users), dtype=float)
    for fold in sorted(set(fold_of_row.tolist())):
        test = fold_of_row == fold
        train = ~test
        if set(day_users[train].tolist()) & set(day_users[test].tolist()):
            raise AssertionError("a user appears in both the training and the test side of a fold")
        scores[test] =fit_predict(model_name, x[train], y[train], x[test], seed=seed)
    return scores


def repeated_out_of_fold_scores(model_name, day_users, x, y, scenario_by_user, n_folds=5, n_repeats=3, seed=0):
    """
    (n_repeats, n) matrix: repeat r uses fold seed `seed + r`.
    """
    day_users = np.asarray(day_users)
    rows = []
    for r in range(n_repeats):
        folds = make_user_folds(day_users, scenario_by_user, n_folds=n_folds, seed=seed + r)
        rows.append(out_of_fold_scores(model_name, day_users, x, y, folds, seed=seed + r))
    return np.vstack(rows)
