"""
Proposed model: a dual-channel Transformer-encoder sequence autoencoder.

Adapted from Beadles and Cain (2026)'s dual-channel presence/magnitude
design (see Chapter 3.5). Their original design assumes a fixed time grid
where "presence" means "did any event happen in this time bin" -- most
bins empty. Our SessionDataset doesn't have that structure: every
timestep in a session IS a real file-copy event (sessions with zero file
activity never appear in the data at all), so there's no per-timestep
"nothing happened" signal to separate out within a session.

Instead, this implementation adapts the same presence/magnitude split to
what our event-level sequence actually has:
  - "presence" channel: seconds_since_last_event -- the gap before this
    event happened. This is the closest analogue to occurrence-pattern
    sparsity in our representation: large gaps mean sparse activity,
    small gaps mean dense activity, and it's the one feature that's
    purely about WHEN events happen rather than WHAT happens in them.
  - "magnitude" channel: everything describing WHAT happened once an
    event has occurred -- file_type, file_extension, time since device
    connect, copies since device connect, and whether it's outside
    working hours.

Both channels are embedded separately, concatenated into one per-event
vector, and passed through a *shared* Transformer encoder (this is what
lets presence and magnitude information inform each other via
self-attention) -- but reconstruction happens through two separate
output heads, one per channel, so the training loss can't be dominated
by one channel at the expense of the other.
"""

import math
import torch
import torch.nn as nn
from src.training.dataset import NUMERIC_FEATURE_COLUMNS

# Index of each numeric column within the "numeric" tensor produced by
# session_to_tensors / collate_sessions -- used to split it into the
# presence input (index 0) and the magnitude inputs (the rest).
PRESENCE_COLUMN_INDEX = NUMERIC_FEATURE_COLUMNS.index("seconds_since_last_event")
MAGNITUDE_NUMERIC_INDICES = [
    i for i in range(len(NUMERIC_FEATURE_COLUMNS)) if i != PRESENCE_COLUMN_INDEX
]


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al., 2017),
    added to each event's embedding so the transformer knows event
    order within a session -- self-attention has no built-in notion of
    sequence position on its own."""

    def __init__(self, d_model, max_len=200):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[: x.shape[1]]


class DualChannelTransformerAutoencoder(nn.Module):
    def __init__(
        self,
        file_type_vocab_size,
        file_extension_vocab_size,
        file_type_embed_dim=8,
        file_extension_embed_dim=8,
        presence_dim=16,
        magnitude_dim=48,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        max_len=200,
    ):
        super().__init__()
        d_model = presence_dim + magnitude_dim
        magnitude_input_dim = (
            len(MAGNITUDE_NUMERIC_INDICES) + file_type_embed_dim + file_extension_embed_dim
        )

        self.file_type_embedding = nn.Embedding(file_type_vocab_size, file_type_embed_dim)
        self.file_extension_embedding = nn.Embedding(
            file_extension_vocab_size, file_extension_embed_dim
        )

        self.presence_in_proj = nn.Linear(1, presence_dim)
        self.magnitude_in_proj = nn.Linear(magnitude_input_dim, magnitude_dim)

        self.positional_encoding = PositionalEncoding(d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Two separate reconstruction heads -- one per channel -- reading
        # off the SAME shared contextual encoding, so presence and
        # magnitude can inform each other during encoding but are scored
        # independently during reconstruction.
        self.presence_out_proj = nn.Linear(d_model, 1)
        self.magnitude_numeric_out_proj = nn.Linear(d_model, len(MAGNITUDE_NUMERIC_INDICES))
        self.file_type_out_proj = nn.Linear(d_model, file_type_vocab_size)
        self.file_extension_out_proj = nn.Linear(d_model, file_extension_vocab_size)

    def forward(self, batch):
        """
        batch: the dict produced by collate_sessions, containing
        "numeric" (B, L, 4), "file_type" (B, L), "file_extension" (B, L),
        "mask" (B, L) bool (True = real event, False = padding).

        Returns a dict of reconstructions, all shape (B, L, ...), to be
        compared against the batch's own inputs by reconstruction_loss.
        """
        numeric = batch["numeric"]
        mask = batch["mask"]

        presence_input = numeric[..., PRESENCE_COLUMN_INDEX : PRESENCE_COLUMN_INDEX + 1]
        magnitude_numeric = numeric[..., MAGNITUDE_NUMERIC_INDICES]

        file_type_embed = self.file_type_embedding(batch["file_type"])
        file_extension_embed = self.file_extension_embedding(batch["file_extension"])
        magnitude_input = torch.cat(
            [magnitude_numeric, file_type_embed, file_extension_embed], dim=-1
        )

        presence_embed = self.presence_in_proj(presence_input)
        magnitude_embed = self.magnitude_in_proj(magnitude_input)
        combined = torch.cat([presence_embed, magnitude_embed], dim=-1)
        combined = self.positional_encoding(combined)

        # nn.TransformerEncoder's src_key_padding_mask expects True where
        # a position should be IGNORED -- the opposite convention from
        # our own mask (True = real event) -- so invert it here.
        encoded = self.transformer_encoder(combined, src_key_padding_mask=~mask)

        return {
            "presence": self.presence_out_proj(encoded),
            "magnitude_numeric": self.magnitude_numeric_out_proj(encoded),
            "file_type_logits": self.file_type_out_proj(encoded),
            "file_extension_logits": self.file_extension_out_proj(encoded),
        }


def reconstruction_loss(outputs, batch):
    """
    Combines the presence and magnitude reconstruction errors into one
    scalar training loss, masked so padded timesteps (which carry no
    real information) never contribute.

    - presence: MSE against the true seconds_since_last_event.
    - magnitude_numeric: MSE against the true remaining numeric features.
    - file_type / file_extension: cross-entropy against the true vocab
      index (reconstructing "what kind of file was this" as a
      classification problem, not a regression one).

    All four terms are averaged only over real (non-padding) timesteps,
    then summed with equal weight -- keeping presence and magnitude as
    two independently-scored channels rather than one blended number, so
    training can't silently let one channel absorb all the error.
    """
    numeric = batch["numeric"]
    mask = batch["mask"]  # (B, L) bool
    num_real = mask.sum().clamp(min=1)  # avoid divide-by-zero on an empty batch

    presence_target = numeric[..., PRESENCE_COLUMN_INDEX : PRESENCE_COLUMN_INDEX + 1]
    presence_error = (outputs["presence"] - presence_target) ** 2
    presence_loss = (presence_error.squeeze(-1) * mask).sum() / num_real

    magnitude_target = numeric[..., MAGNITUDE_NUMERIC_INDICES]
    magnitude_error = (outputs["magnitude_numeric"] - magnitude_target) ** 2
    magnitude_loss = (magnitude_error.mean(dim=-1) * mask).sum() / num_real

    file_type_ce = nn.functional.cross_entropy(
        outputs["file_type_logits"].transpose(1, 2), batch["file_type"], reduction="none"
    )
    file_type_loss = (file_type_ce * mask).sum() / num_real

    file_extension_ce = nn.functional.cross_entropy(
        outputs["file_extension_logits"].transpose(1, 2),
        batch["file_extension"],
        reduction="none",
    )
    file_extension_loss = (file_extension_ce * mask).sum() / num_real

    total = presence_loss + magnitude_loss + file_type_loss + file_extension_loss
    return {
        "total": total,
        "presence": presence_loss,
        "magnitude_numeric": magnitude_loss,
        "file_type": file_type_loss,
        "file_extension": file_extension_loss,
    }


def session_anomaly_score(outputs, batch):
    """
    Per-session reconstruction error, for anomaly scoring at evaluation
    time (task 27). Same four components as reconstruction_loss, but
    reduced per-session (dim 1) rather than over the whole batch, so
    each session gets its own score rather than one batch-wide average.
    Returns a (B,) tensor.
    """
    numeric = batch["numeric"]
    mask = batch["mask"]
    lengths = mask.sum(dim=1).clamp(min=1)  # (B,)

    presence_target = numeric[..., PRESENCE_COLUMN_INDEX : PRESENCE_COLUMN_INDEX + 1]
    presence_error = ((outputs["presence"] - presence_target) ** 2).squeeze(-1)
    presence_score = (presence_error * mask).sum(dim=1) / lengths

    magnitude_target = numeric[..., MAGNITUDE_NUMERIC_INDICES]
    magnitude_error = ((outputs["magnitude_numeric"] - magnitude_target) ** 2).mean(dim=-1)
    magnitude_score = (magnitude_error * mask).sum(dim=1) / lengths

    file_type_ce = nn.functional.cross_entropy(
        outputs["file_type_logits"].transpose(1, 2), batch["file_type"], reduction="none"
    )
    file_type_score = (file_type_ce * mask).sum(dim=1) / lengths

    file_extension_ce = nn.functional.cross_entropy(
        outputs["file_extension_logits"].transpose(1, 2),
        batch["file_extension"],
        reduction="none",
    )
    file_extension_score = (file_extension_ce * mask).sum(dim=1) / lengths

    return presence_score + magnitude_score + file_type_score + file_extension_score