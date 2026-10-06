"""
The model registry: every model the project trains or evaluates is
declared ONCE, here. Training, evaluation, the ensemble and the final
evaluation all loop over this registry, so adding or swapping a model is
a change to this one file (plus the model's own module).

A model is one of three kinds:

  "torch"      a PyTorch sequence autoencoder, trained with the shared
               loop in src/unsupervised/training/train.py. Needs: build(), loss_fn(),
               score_fn() and a checkpoint_file.
  "static"     a classical detector fitted in one shot on per-session
               vectors (the Isolation Forest). Needs: build_static() and
               a checkpoint_file.
  "heuristic"  no training at all: the score is one raw feature of the
               session (or its length). These are the BASELINES any
               learned model has to beat -- a deep model that cannot
               beat "score each session by one raw feature" has not
               earned its complexity. The direction (higher = more
               anomalous) is fixed in advance and never flipped to flatter
               a result.

HOW TO ADD A MODEL (e.g. if a supervisor asks for another architecture)
  1. Write src/unsupervised/models/<name>.py with the model class, a
     reconstruction_loss(outputs, batch) -> {"total": ...} and a
     session_anomaly_score(outputs, batch) -> (B,) tensor, where HIGHER
     means more anomalous. The model is called as model(batch) with the
     dict produced by src/unsupervised/training/dataset.collate_sessions.
  2. Add two small lazy-import wrappers below (build / loss / score) and
     register(ModelSpec(...)) at the bottom of this file.
  Nothing else changes: train_models.py, evaluate_models.py,
  ensemble_scores.py and final_evaluation.py pick it up.

Heavy libraries (torch, scikit-learn) are imported lazily inside the
wrappers, so importing this module is cheap and works without them.

INSIDER_MODELS=name1,name2 restricts any script to a subset of the
registered models (default: all of them).
"""

import os
from dataclasses import dataclass
from typing import Callable, Optional

MODELS_ENV_VAR = "INSIDER_MODELS"
VALID_KINDS = ("torch", "static", "heuristic")


@dataclass(frozen=True)
class ModelSpec:
    name: str                       # stable id: used in results files, env var, tables
    display_name: str
    kind: str                       # "torch" | "static" | "heuristic"
    checkpoint_file: Optional[str] = None
    # kind == "torch"
    build: Optional[Callable] = None        # (file_type_vocab_size, file_extension_vocab_size) -> nn.Module
    loss_fn: Optional[Callable] = None      # (outputs, batch) -> {"total": scalar, ...}
    score_fn: Optional[Callable] = None     # (outputs, batch) -> (B,) tensor, higher = more anomalous
    # kind == "static"
    build_static: Optional[Callable] = None  # (seed) -> detector with .fit(vectors) / .anomaly_score(vectors)
    # kind == "heuristic"
    heuristic_column: Optional[str] = None   # dataset column, or None for session length
    heuristic_reduce: str = "mean"           # "mean" over the session's events, or "count"
    required_columns: tuple = ()             # columns the dataset must contain for this model

    def __post_init__(self):
        if self.kind not in VALID_KINDS:
            raise ValueError(f"{self.name}: kind must be one of {VALID_KINDS}, got {self.kind!r}")
        if self.kind == "torch" and not (self.build and self.loss_fn and self.score_fn and self.checkpoint_file):
            raise ValueError(f"{self.name}: a torch model needs build, loss_fn, score_fn and checkpoint_file")
        if self.kind == "static" and not (self.build_static and self.checkpoint_file):
            raise ValueError(f"{self.name}: a static model needs build_static and checkpoint_file")
        if self.kind == "heuristic" and self.heuristic_reduce not in ("mean", "count"):
            raise ValueError(f"{self.name}: heuristic_reduce must be 'mean' or 'count'")

    @property
    def is_trainable(self):
        return self.kind in ("torch", "static")


_REGISTRY = {}


def register(spec):
    if spec.name in _REGISTRY:
        raise ValueError(f"A model named {spec.name!r} is already registered.")
    _REGISTRY[spec.name] = spec
    return spec


def get_spec(name):
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown model {name!r}. Registered models: {list(_REGISTRY)}.")


def registered_names():
    return list(_REGISTRY)


def get_model_specs(names=None):
    """
    The registered models, in registration order. `names` may be an
    iterable of model names; if None, the INSIDER_MODELS environment
    variable is used, and if that is unset or empty, every model.
    An unknown name raises, so a typo never silently drops a model.
    """
    if names is None:
        raw = os.environ.get(MODELS_ENV_VAR, "").strip()
        names = [n.strip() for n in raw.split(",") if n.strip()] if raw else list(_REGISTRY)
    names = list(names)
    unknown = [n for n in names if n not in _REGISTRY]
    if unknown:
        raise ValueError(f"Unknown model(s) {unknown}. Registered models: {list(_REGISTRY)}.")
    return [_REGISTRY[n] for n in _REGISTRY if n in names]


# ---------------------------------------------------------------------
# Lazy wrappers: import the heavy model modules only when they are used.
# ---------------------------------------------------------------------

def _build_transformer(file_type_vocab_size, file_extension_vocab_size):
    from src.unsupervised.models.transformer_autoencoder import DualChannelTransformerAutoencoder
    return DualChannelTransformerAutoencoder(
        file_type_vocab_size=file_type_vocab_size,
        file_extension_vocab_size=file_extension_vocab_size,
    )


def _transformer_loss(outputs, batch):
    from src.unsupervised.models.transformer_autoencoder import reconstruction_loss
    return reconstruction_loss(outputs, batch)


def _transformer_score(outputs, batch):
    from src.unsupervised.models.transformer_autoencoder import session_anomaly_score
    return session_anomaly_score(outputs, batch)


def _build_lstm(file_type_vocab_size, file_extension_vocab_size):
    from src.unsupervised.models.lstm_autoencoder import LSTMAutoencoder
    # hidden_dim=128: see the comparison in scripts/unsupervised/tune_models.py.
    return LSTMAutoencoder(
        file_type_vocab_size=file_type_vocab_size,
        file_extension_vocab_size=file_extension_vocab_size,
        hidden_dim=128,
    )


def _lstm_loss(outputs, batch):
    from src.unsupervised.models.lstm_autoencoder import reconstruction_loss
    return reconstruction_loss(outputs, batch)


def _lstm_score(outputs, batch):
    from src.unsupervised.models.lstm_autoencoder import session_anomaly_score
    return session_anomaly_score(outputs, batch)


def _build_static_detector(seed):
    from src.unsupervised.models.static_vector_baseline import StaticVectorAnomalyDetector
    return StaticVectorAnomalyDetector(random_state=seed)


# ---------------------------------------------------------------------
# The registered models, in the order they are trained and reported.
# ---------------------------------------------------------------------

register(ModelSpec(
    name="transformer_ae",
    display_name="Dual-channel Transformer autoencoder (proposed)",
    kind="torch",
    checkpoint_file="transformer_autoencoder.pt",
    build=_build_transformer, loss_fn=_transformer_loss, score_fn=_transformer_score,
))

register(ModelSpec(
    name="lstm_ae",
    display_name="LSTM autoencoder (baseline 1)",
    kind="torch",
    checkpoint_file="lstm_autoencoder.pt",
    build=_build_lstm, loss_fn=_lstm_loss, score_fn=_lstm_score,
))

register(ModelSpec(
    name="static_iforest",
    display_name="Static-vector Isolation Forest (baseline 2)",
    kind="static",
    checkpoint_file="static_vector_detector.joblib",
    build_static=_build_static_detector,
))

register(ModelSpec(
    name="single_feature_http",
    display_name="Heuristic: session mean of http_offbaseline_share_7d",
    kind="heuristic",
    heuristic_column="http_offbaseline_share_7d", heuristic_reduce="mean",
    required_columns=("http_offbaseline_share_7d",),
))

register(ModelSpec(
    name="session_length_only",
    display_name="Heuristic: session length (more events = more anomalous)",
    kind="heuristic",
    heuristic_column=None, heuristic_reduce="count",
))
