import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from src.training.train import train_model
from src.training.dataset import SessionDataset, collate_sessions
from src.training.vocab import build_vocab
from src.models.lstm_autoencoder import LSTMAutoencoder, reconstruction_loss
import pytest

# ---------------------------------------------------------------------
# Smoke test: a real model on real (toy) session data, just to confirm
# train_model's plumbing actually works end-to-end with the models we
# built in task 25 -- not just with a synthetic dummy model.
# ---------------------------------------------------------------------

def _toy_events_df():
    session_start = pd.to_datetime(["2010-06-01 09:00:00"] * 2 + ["2010-06-02 09:00:00"] * 3)
    session_end = pd.to_datetime(["2010-06-01 17:00:00"] * 2 + ["2010-06-02 17:00:00"] * 3)
    return pd.DataFrame({
        "user": ["AAA0001", "AAA0001", "BBB0002", "BBB0002", "BBB0002"],
        "session_start": session_start,
        "session_end": session_end,
        "date": pd.to_datetime([
            "2010-06-01 10:00:00", "2010-06-01 10:05:00",
            "2010-06-02 10:00:00", "2010-06-02 10:05:00", "2010-06-02 10:10:00",
        ]),
        "file_type": ["pdf", "zip", "office_document", "pdf", "exe"],
        "file_extension": ["pdf", "zip", "doc", "pdf", "exe"],
        "seconds_since_last_event": [0.0, 5.0, 0.0, 5.0, 5.0],
        "seconds_since_device_connect": [10.0, 15.0, 20.0, 25.0, 30.0],
        "copies_since_device_connect": [0, 1, 0, 1, 2],
        "outside_working_hours": [False, False, True, True, True],
        "user_session_volume_deviation": [1.0, 1.0, 1.0, 1.0, 1.0],
        "user_session_frequency_deviation": [1.0, 1.0, 1.0, 1.0, 1.0],
        "http_new_domains_today": [0.0, 0.0, 0.0, 0.0, 0.0],
        "http_new_domains_7d": [0.0, 0.0, 0.0, 0.0, 0.0],
        "http_volume_deviation": [0.0, 0.0, 0.0, 0.0, 0.0],
        "session_is_malicious": [False, False, False, False, False],  # benign-only, like real training data
    })

def test_train_model_smoke_test_with_real_model_and_history_shape():
    events_df = _toy_events_df()
    file_type_vocab = build_vocab(events_df["file_type"])
    file_extension_vocab = build_vocab(events_df["file_extension"])
    dataset = SessionDataset(events_df, file_type_vocab, file_extension_vocab)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_sessions)

    model = LSTMAutoencoder(
        file_type_vocab_size=len(file_type_vocab),
        file_extension_vocab_size=len(file_extension_vocab),
        file_type_embed_dim=4,
        file_extension_embed_dim=4,
        hidden_dim=8,
        num_layers=1,
    )

    trained_model, history = train_model(
        model, loader, loader, reconstruction_loss, num_epochs=3, patience=10,
    )

    assert len(history["train_loss"]) == 3
    assert len(history["val_loss"]) == 3
    assert "best_val_loss" in history
    assert "stopped_epoch" in history
    assert isinstance(trained_model, LSTMAutoencoder)


# ---------------------------------------------------------------------
# Precise, deterministic tests of train_model's own logic (early
# stopping, best-checkpoint restoration) using a scripted dummy model
# whose "validation loss" follows an exact, pre-set sequence -- so the
# test doesn't depend on a real model happening to overfit/oscillate in
# a particular way, which would make the test flaky.
# ---------------------------------------------------------------------

class _ScriptedValLossModel(nn.Module):
    """
    A minimal model with one learnable parameter (needed so the
    optimizer has something to step). In train mode it computes a real,
    differentiable loss so gradients actually flow and the parameter
    actually changes epoch to epoch. In eval mode (validation) it
    ignores the data entirely and just returns the next value from a
    pre-scripted sequence, so we know EXACTLY what validation loss
    train_model will see at every epoch.
    """

    def __init__(self, val_loss_sequence):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))
        self.val_loss_sequence = val_loss_sequence
        self._val_index = 0
        self.train_forward_weight_snapshots = []  # weight value AT the start of each train-mode forward call

    def forward(self, batch):
        if self.training:
            self.train_forward_weight_snapshots.append(self.weight.item())
            # A real, learnable (if arbitrary) function of the parameter,
            # so backward()/optimizer.step() do something meaningful --
            # its exact trajectory doesn't matter for this test.
            return {"mode": "train", "prediction": batch["x"] * self.weight}
        else:
            scripted_loss = self.val_loss_sequence[self._val_index]
            self._val_index += 1
            # Still depends on self.weight (times zero) purely so it's a
            # valid tensor in the autograd graph -- its VALUE is fixed.
            return {"mode": "val", "scripted_loss": torch.tensor(scripted_loss) + 0.0 * self.weight}


def _scripted_loss_fn(outputs, batch):
    if outputs["mode"] == "val":
        return {"total": outputs["scripted_loss"]}
    loss = ((outputs["prediction"] - batch["target"]) ** 2).mean()
    return {"total": loss}


class _SingleBatchDataset(Dataset):
    """A dataset with exactly one sample, so a DataLoader over it
    produces exactly one batch per epoch -- this is what lets us line
    up "one entry in val_loss_sequence" with "one epoch" precisely."""

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {"x": torch.tensor(1.0), "target": torch.tensor(2.0)}


def _single_batch_collate(batch):
    return {
        "x": torch.stack([sample["x"] for sample in batch]),
        "target": torch.stack([sample["target"] for sample in batch]),
    }


def _single_batch_loader():
    return DataLoader(_SingleBatchDataset(), batch_size=1, collate_fn=_single_batch_collate)


def test_early_stopping_triggers_after_patience_epochs_without_improvement():
    # Best value (0.3) occurs at epoch 3; epochs 4, 5, 6 are all worse
    # (no improvement). With patience=3, training should stop right
    # after epoch 6 -- the 3rd consecutive non-improving epoch -- never
    # reaching epoch 7's scripted 0.5.
    val_loss_sequence = [1.0, 0.5, 0.3, 0.35, 0.4, 0.45, 0.5]
    model = _ScriptedValLossModel(val_loss_sequence)

    trained_model, history = train_model(
        model,
        _single_batch_loader(),
        _single_batch_loader(),
        _scripted_loss_fn,
        num_epochs=len(val_loss_sequence),
        patience=3,
        min_delta=1e-4,
        lr=0.1,
    )

    assert history["stopped_epoch"] == 6
    assert history["best_val_loss"] == pytest.approx(0.3)
    assert history["val_loss"] == pytest.approx([1.0, 0.5, 0.3, 0.35, 0.4, 0.45])  # never reached the 7th, 0.5


def test_best_checkpoint_is_restored_not_the_final_epoch():
    # Same scripted trajectory: best epoch is 3. The parameter keeps
    # changing every epoch regardless of the (scripted, fake) validation
    # loss, so if train_model just kept the LAST epoch's weights, the
    # restored weight would differ from what it was right after epoch 3.
    val_loss_sequence = [1.0, 0.5, 0.3, 0.35, 0.4, 0.45, 0.5]
    model = _ScriptedValLossModel(val_loss_sequence)

    trained_model, history = train_model(
        model,
        _single_batch_loader(),
        _single_batch_loader(),
        _scripted_loss_fn,
        num_epochs=len(val_loss_sequence),
        patience=3,
        min_delta=1e-4,
        lr=0.1,
    )

    # weight value AS IT ENTERED epoch 4's forward pass == weight value
    # immediately after epoch 3's optimizer step == what should have been
    # checkpointed as "best".
    weight_after_epoch_3 = trained_model.train_forward_weight_snapshots[3]
    final_weight = trained_model.weight.item()

    assert final_weight == weight_after_epoch_3
    # Sanity check the fixture itself isn't accidentally vacuous (i.e.
    # the weight actually was still changing epoch to epoch, so
    # "restoring the wrong epoch" would actually have been detectable).
    assert len(set(trained_model.train_forward_weight_snapshots)) > 1