"""
The supervised classifiers, with FIXED hyperparameters.

Nothing here is tuned. With only ~35 insiders on the development
partition, tuning hyperparameters on the same folds that are then
reported would inflate the result. The settings below are conventional
defaults for small tabular data, chosen before looking at any score.

  logistic            standardised, sign-log-compressed features, L2 logistic
                      regression: the simple, interpretable baseline
  gradient_boosting   small histogram gradient-boosted trees (depth 3, 200
                      rounds, learning rate 0.05, strong regularisation)

Both are trained with BALANCED sample weights (positives are rare), and
both return a score where higher = more likely malicious.
scikit-learn is imported lazily.
"""

import warnings

import numpy as np

MODEL_NAMES = ("logistic", "gradient_boosting")

# A SEQUENCE model scores a day from the user's previous SEQ_LEN active days
# (that day last). It takes the flattened window matrix of
# src/supervised/sequences.py, not a plain feature matrix.
#
#   gru   one-layer GRU (32 hidden units) over the window, a linear scoring
#         head, trained with the same balanced / date-matched weights as the
#         other models; fixed hyperparameters, nothing tuned.
SEQUENCE_MODEL_NAMES = ("gru",)
ALL_MODEL_NAMES = MODEL_NAMES + SEQUENCE_MODEL_NAMES
SEQ_LEN = 14
GRU_HIDDEN = 32
GRU_EPOCHS = 10
GRU_BATCH = 256
GRU_LEARNING_RATE = 2e-3
GRU_WEIGHT_DECAY = 1e-4

DISPLAY_NAMES = {
    "logistic": "Logistic regression (supervised)",
    "gradient_boosting": "Gradient boosting (supervised)",
    "gru": "GRU over the previous 14 days (supervised)",
}


def balanced_sample_weights(labels):
    """Weight each class so both contribute equally in total."""
    labels = np.asarray(labels).astype(int)
    n = len(labels)
    n_pos = int(labels.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.ones(n)
    weights = np.where(labels == 1, n / (2.0 * n_pos), n / (2.0 * n_neg))
    return weights.astype(float)


def date_matched_weights(labels, strata):
    """
    Sample weights that remove the calendar from the label. Within every
    calendar stratum (e.g. week) that has both classes, malicious rows get
    weight 1 and non-malicious rows get n_malicious / n_non_malicious, so
    each such stratum contributes equally many malicious and
    non-malicious weight. Strata with only one class get weight 0: there
    is nothing to match them against. A model trained with these weights
    sees a label that is independent of the date, so it cannot profit from
    knowing when a day happened.

    If no stratum contains both classes, falls back to the ordinary
    class-balanced weights.
    """
    labels = np.asarray(labels).astype(int)
    _, ids = np.unique(np.asarray(strata), return_inverse=True)
    weights = np.zeros(len(labels), dtype=float)
    for stratum in np.unique(ids):
        rows = ids == stratum
        n_pos = int(labels[rows].sum())
        n_neg = int(rows.sum()) - n_pos
        if n_pos == 0 or n_neg == 0:
            continue
        weights[rows & (labels == 1)] = 1.0
        weights[rows & (labels == 0)] = n_pos / n_neg
    if weights.sum() == 0:
        return balanced_sample_weights(labels)
    return weights


def train_medians(x_train):
    """Per-column medians of the finite training values (0 for a column with none)."""
    x = np.array(x_train, dtype=float, copy=True)
    x[~np.isfinite(x)] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # "All-NaN slice" is handled just below
        medians = np.nanmedian(x, axis=0) if len(x) else np.zeros(x.shape[1])
    return np.where(np.isfinite(medians), medians, 0.0)


def apply_medians(x, medians):
    """A copy of `x` with every NaN/inf replaced by the given (training) column medians."""
    x = np.array(x, dtype=float, copy=True)
    x[~np.isfinite(x)] = np.nan
    rows, cols = np.where(np.isnan(x))
    x[rows, cols] = medians[cols]
    return x


def impute_with_train_medians(x_train, x_test):
    """Replaces NaN/inf in both matrices with the TRAINING columns' medians (0 if a column is all NaN)."""
    medians = train_medians(x_train)
    return apply_medians(x_train, medians), apply_medians(x_test, medians)


def signed_log(x):
    """sign(x) * log1p(|x|): tames the heavy right tails of seconds and counts."""
    return np.sign(x) * np.log1p(np.abs(x))


def _fit_gru(x_train, y_train, weights, seed):
    """
    x_train: (n, SEQ_LEN * (d + 1)) window matrix (see sequences.py): per step the d
    features (NaN where the step is padding or the value is missing) and a 0/1 mask
    channel. Statistics (medians, mean, std) come from the TRAINING rows' real steps
    only. Returns predict(x) -> logit scores, row-wise (no dependence on the batch).
    Rows with weight 0 (date-matched training) are dropped, they carry no signal.
    """
    import torch
    from torch import nn

    steps = SEQ_LEN
    x_train = np.asarray(x_train, dtype=float)
    block = x_train.shape[1] // steps
    if block * steps != x_train.shape[1] or block < 2:
        raise ValueError(f"A GRU input must have {steps} * (features + 1) columns, got {x_train.shape[1]}.")
    d = block - 1

    keep = np.asarray(weights, dtype=float) > 0
    if not keep.any():
        return lambda x: np.zeros(len(x), dtype=float)
    y_train = np.asarray(y_train)[keep]
    row_weights = np.asarray(weights, dtype=float)[keep]
    x_train = x_train[keep]

    seq = x_train.reshape(len(x_train), steps, block)
    real_steps = seq[:, :, d] > 0.5
    valid = seq[:, :, :d][real_steps]
    medians = train_medians(valid)
    valid = signed_log(apply_medians(valid, medians))
    mean = valid.mean(axis=0) if len(valid) else np.zeros(d)
    std = valid.std(axis=0) if len(valid) else np.ones(d)
    std = np.where(std < 1e-8, 1.0, std)

    def prepare(x):
        s = np.asarray(x, dtype=float).reshape(len(x), steps, block)
        real = s[:, :, d] > 0.5
        flat = apply_medians(s[:, :, :d].reshape(-1, d), medians)
        flat = ((signed_log(flat) - mean) / std).reshape(len(x), steps, d) * real[:, :, None]
        return np.concatenate([flat, real[:, :, None].astype(float)], axis=2).astype(np.float32)

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(d + 1, GRU_HIDDEN, batch_first=True)
            self.head = nn.Linear(GRU_HIDDEN, 1)

        def forward(self, batch):
            _, hidden = self.gru(batch)
            return self.head(hidden[-1]).squeeze(-1)

    inputs = torch.from_numpy(prepare(x_train))
    targets = torch.from_numpy(y_train.astype(np.float32))
    row_weights = torch.from_numpy((row_weights / row_weights.mean()).astype(np.float32))

    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    net = _Net()
    optimiser = torch.optim.Adam(net.parameters(), lr=GRU_LEARNING_RATE, weight_decay=GRU_WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    net.train()
    for _ in range(GRU_EPOCHS):
        order = torch.randperm(len(inputs), generator=generator)
        for start in range(0, len(order), GRU_BATCH):
            idx = order[start:start + GRU_BATCH]
            optimiser.zero_grad()
            loss = (loss_fn(net(inputs[idx]), targets[idx]) * row_weights[idx]).mean()
            loss.backward()
            optimiser.step()
    net.eval()

    def predict(x):
        out = []
        with torch.no_grad():
            for start in range(0, len(x), 2048):
                out.append(net(torch.from_numpy(prepare(np.asarray(x)[start:start + 2048]))).numpy())
        return np.concatenate(out).astype(float) if out else np.zeros(0)

    return predict


def fit_model(model_name, x_train, y_train, seed=0, sample_weight=None):
    """
    `sample_weight` replaces the default class-balanced weights (for
    example with date_matched_weights).

    Trains `model_name` on the training rows and returns a function
    `predict(x) -> scores` (float array, higher = more likely malicious)
    that applies the same training-set imputation and scaling to any new
    rows. If the training labels contain a single class there is nothing
    to learn: the predictor returns zeros.
    """
    if model_name not in ALL_MODEL_NAMES:
        raise ValueError(f"Unknown supervised model {model_name!r}; choose from {ALL_MODEL_NAMES}.")
    y_train = np.asarray(y_train).astype(int)
    if y_train.sum() == 0 or y_train.sum() == len(y_train):
        return lambda x: np.zeros(len(x), dtype=float)

    weights = balanced_sample_weights(y_train) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if model_name in SEQUENCE_MODEL_NAMES:
        return _fit_gru(x_train, y_train, weights, seed)

    medians = train_medians(x_train)
    a = apply_medians(x_train, medians)

    if model_name == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        a = signed_log(a)
        scaler = StandardScaler().fit(a)
        model = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
        model.fit(scaler.transform(a), y_train, sample_weight=weights)

        def predict(x):
            b = scaler.transform(signed_log(apply_medians(x, medians)))
            return model.decision_function(b).astype(float)

        return predict

    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(
        max_depth=3,
        learning_rate=0.05,
        max_iter=200,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=seed,
    )
    model.fit(a, y_train, sample_weight=weights)

    def predict(x):
        return model.predict_proba(apply_medians(x, medians))[:, 1].astype(float)

    return predict


def fit_predict(model_name, x_train, y_train, x_test, seed=0, sample_weight=None):
    """Trains on the training rows and scores the test rows (see fit_model)."""
    return fit_model(model_name, x_train, y_train, seed=seed, sample_weight=sample_weight)(x_test)
