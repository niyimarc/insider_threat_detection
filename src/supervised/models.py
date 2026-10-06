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

DISPLAY_NAMES = {
    "logistic": "Logistic regression (supervised)",
    "gradient_boosting": "Gradient boosting (supervised)",
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


def fit_model(model_name, x_train, y_train, seed=0):
    """
    Trains `model_name` on the training rows and returns a function
    `predict(x) -> scores` (float array, higher = more likely malicious)
    that applies the same training-set imputation and scaling to any new
    rows. If the training labels contain a single class there is nothing
    to learn: the predictor returns zeros.
    """
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown supervised model {model_name!r}; choose from {MODEL_NAMES}.")
    y_train = np.asarray(y_train).astype(int)
    if y_train.sum() == 0 or y_train.sum() == len(y_train):
        return lambda x: np.zeros(len(x), dtype=float)

    medians = train_medians(x_train)
    a = apply_medians(x_train, medians)
    weights = balanced_sample_weights(y_train)

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


def fit_predict(model_name, x_train, y_train, x_test, seed=0):
    """Trains on the training rows and scores the test rows (see fit_model)."""
    return fit_model(model_name, x_train, y_train, seed=seed)(x_test)
