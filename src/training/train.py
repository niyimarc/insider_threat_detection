"""
Generic training loop for the two PyTorch-based models (the proposed
dual-channel Transformer autoencoder and the LSTM baseline). Both models
share the same interface -- model(batch) -> outputs dict, and a matching
loss_fn(outputs, batch) -> {"total": scalar, ...} -- so one function
trains either, keeping the training PROCEDURE identical across both per
Chapter 3.6 ("Both baseline models ... are trained under this same
procedure ... so that any difference in reported performance can be put
down to architectural differences rather than to differences in how each
model was trained").

Baseline 2 (the static-vector Isolation Forest) doesn't go through this
function at all -- it's a one-shot .fit(), not an iterative training
loop -- see src/models/static_vector_baseline.py.
"""

import copy

import torch


def _move_batch_to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def train_model(
    model,
    train_loader,
    val_loader,
    loss_fn,
    num_epochs=100,
    patience=5,
    min_delta=1e-4,
    lr=1e-3,
    device="cpu",
):
    """
    Trains `model` on batches from `train_loader`, evaluating on
    `val_loader` after every epoch, and stops early once validation
    reconstruction loss stops meaningfully improving -- exactly the
    stopping rule Chapter 3.6 commits to, rather than a fixed epoch count.

    - patience: how many consecutive epochs of no improvement to
      tolerate before stopping.
    - min_delta: how much the validation loss must drop to count as
      "improved" -- guards against stopping being triggered by
      meaningless floating-point noise.

    Returns (trained_model, history), where trained_model has the
    weights from whichever epoch had the BEST validation loss (not
    necessarily the last epoch -- if validation loss got worse after
    epoch 12, we still want epoch 12's weights, not epoch 17's), and
    history is a dict of per-epoch train/validation losses for later
    inspection or plotting.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state_dict = None
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = _move_batch_to_device(batch, device)
            optimizer.zero_grad()
            outputs = model(batch)
            losses = loss_fn(outputs, batch)
            losses["total"].backward()
            optimizer.step()
            train_losses.append(losses["total"].item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = _move_batch_to_device(batch, device)
                outputs = model(batch)
                losses = loss_fn(outputs, batch)
                val_losses.append(losses["total"].item())

        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / len(val_losses)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"epoch {epoch + 1:3d} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f"Early stopping at epoch {epoch + 1} "
                    f"(no improvement for {patience} epochs, best val loss = {best_val_loss:.4f})"
                )
                break

    # Restore the best-seen weights, not whatever the last epoch left
    # behind -- the last epoch is only "the end of training", not
    # necessarily "the best model", once early stopping is in play.
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    history["best_val_loss"] = best_val_loss
    history["stopped_epoch"] = epoch + 1
    return model, history