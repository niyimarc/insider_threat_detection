"""
Trains ONE registered model (see src/models/registry.py) and saves its
checkpoint. scripts/train_models.py calls this in a loop over the
registry, so the training procedure is identical for every model of a
kind and a new model needs no new training code.
"""

import json
import os

from src.models.registry import ModelSpec  # noqa: F401  (documentation of the argument type)


def _cpu_state_dict(model):
    """Checkpoints are always saved with CPU tensors, so they load fine on a
    CPU-only machine regardless of where they were trained."""
    return {key: value.cpu() for key, value in model.state_dict().items()}


def train_registered_model(
    spec,
    *,
    train_df,
    train_loader,
    val_loader,
    file_type_vocab,
    file_extension_vocab,
    checkpoint_dir,
    seed,
    device="cpu",
    num_epochs=100,
    patience=10,
):
    """
    Trains and saves one model. Returns the training history dict for
    torch models, and None for models with no iterative training
    (static detectors, heuristics).
    """
    if spec.kind == "heuristic":
        print(f"  {spec.name}: heuristic baseline, nothing to train.")
        return None

    if spec.kind == "torch":
        import torch
        from src.training.train import train_model

        model = spec.build(len(file_type_vocab), len(file_extension_vocab))
        model, history = train_model(
            model, train_loader, val_loader, spec.loss_fn,
            num_epochs=num_epochs, patience=patience, device=device,
        )
        torch.save(_cpu_state_dict(model), os.path.join(checkpoint_dir, spec.checkpoint_file))
        history_name = os.path.splitext(spec.checkpoint_file)[0] + "_history.json"
        with open(os.path.join(checkpoint_dir, history_name), "w") as f:
            json.dump(history, f, indent=2)
        return history

    if spec.kind == "static":
        import joblib
        from src.models.static_vector_baseline import build_session_vectors

        vectors, _ = build_session_vectors(train_df, file_type_vocab, file_extension_vocab)
        detector = spec.build_static(seed)
        detector.fit(vectors)
        joblib.dump(detector, os.path.join(checkpoint_dir, spec.checkpoint_file))
        return None

    raise ValueError(f"Unsupported model kind {spec.kind!r}")
