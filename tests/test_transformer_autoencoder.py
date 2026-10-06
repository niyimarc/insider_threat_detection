import pandas as pd
import torch

from src.training.dataset import SessionDataset, collate_sessions, NUMERIC_FEATURE_COLUMNS
from src.training.vocab import build_vocab
from src.models.transformer_autoencoder import (
    DualChannelTransformerAutoencoder,
    reconstruction_loss,
    session_anomaly_score,
    MAGNITUDE_NUMERIC_INDICES,
)


def _toy_events_df():
    # Two sessions, different lengths, mirroring tests/test_dataset.py's fixture.
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
        "http_offbaseline_share_7d": [0.0, 0.0, 0.0, 0.0, 0.0],
        "session_is_malicious": [False, False, True, True, True],
    })

def _toy_batch():
    events_df = _toy_events_df()
    file_type_vocab = build_vocab(events_df["file_type"])
    file_extension_vocab = build_vocab(events_df["file_extension"])
    dataset = SessionDataset(events_df, file_type_vocab, file_extension_vocab)
    batch = collate_sessions([dataset[i] for i in range(len(dataset))])
    return batch, file_type_vocab, file_extension_vocab


def _toy_model(file_type_vocab, file_extension_vocab):
    return DualChannelTransformerAutoencoder(
        file_type_vocab_size=len(file_type_vocab),
        file_extension_vocab_size=len(file_extension_vocab),
        file_type_embed_dim=4,
        file_extension_embed_dim=4,
        presence_dim=8,
        magnitude_dim=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
    )


def test_forward_pass_shapes():
    batch, file_type_vocab, file_extension_vocab = _toy_batch()
    model = _toy_model(file_type_vocab, file_extension_vocab)

    outputs = model(batch)

    batch_size, max_len = batch["mask"].shape
    assert outputs["presence"].shape == (batch_size, max_len, 1)
    assert outputs["magnitude_numeric"].shape == (batch_size, max_len, len(MAGNITUDE_NUMERIC_INDICES))
    assert outputs["file_type_logits"].shape == (batch_size, max_len, len(file_type_vocab))
    assert outputs["file_extension_logits"].shape == (batch_size, max_len, len(file_extension_vocab))


def test_reconstruction_loss_is_finite_scalar():
    batch, file_type_vocab, file_extension_vocab = _toy_batch()
    model = _toy_model(file_type_vocab, file_extension_vocab)

    outputs = model(batch)
    losses = reconstruction_loss(outputs, batch)

    assert losses["total"].dim() == 0  # scalar
    assert torch.isfinite(losses["total"])
    for key in ["presence", "magnitude_numeric", "file_type", "file_extension"]:
        assert torch.isfinite(losses[key])


def test_model_can_learn_to_reduce_loss_on_a_fixed_batch():
    # Classic overfitting sanity check: if we can't drive the loss down
    # on a single small fixed batch after a bunch of gradient steps,
    # something in the forward/backward/loss wiring is broken.
    batch, file_type_vocab, file_extension_vocab = _toy_batch()
    model = _toy_model(file_type_vocab, file_extension_vocab)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    first_loss = None
    last_loss = None
    for step in range(50):
        optimizer.zero_grad()
        outputs = model(batch)
        losses = reconstruction_loss(outputs, batch)
        losses["total"].backward()
        optimizer.step()
        if step == 0:
            first_loss = losses["total"].item()
        last_loss = losses["total"].item()

    assert last_loss < first_loss


def test_padding_does_not_affect_loss():
    # Corrupting the PADDED positions of a batch's inputs should not
    # change the computed loss at all, since the mask must exclude them
    # from every term. This is the real test of the masking logic.
    batch, file_type_vocab, file_extension_vocab = _toy_batch()
    model = _toy_model(file_type_vocab, file_extension_vocab)
    model.eval()  # disable dropout so forward passes are deterministic

    with torch.no_grad():
        outputs_before = model(batch)
        loss_before = reconstruction_loss(outputs_before, batch)["total"].item()

    corrupted_batch = {k: v.clone() for k, v in batch.items()}
    pad_positions = ~corrupted_batch["mask"]
    assert pad_positions.any(), "fixture has no padding to corrupt -- test is vacuous"
    corrupted_batch["numeric"][pad_positions] = 9999.0
    corrupted_batch["file_type"][pad_positions] = 0
    corrupted_batch["file_extension"][pad_positions] = 0

    with torch.no_grad():
        outputs_after = model(corrupted_batch)
        loss_after = reconstruction_loss(outputs_after, corrupted_batch)["total"].item()

    assert abs(loss_before - loss_after) < 1e-5


def test_session_anomaly_score_returns_one_value_per_session():
    batch, file_type_vocab, file_extension_vocab = _toy_batch()
    model = _toy_model(file_type_vocab, file_extension_vocab)

    with torch.no_grad():
        outputs = model(batch)
        scores = session_anomaly_score(outputs, batch)

    batch_size = batch["mask"].shape[0]
    assert scores.shape == (batch_size,)
    assert torch.isfinite(scores).all()