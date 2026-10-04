"""
Baseline 1: a standard, non-variational sequence autoencoder (Lorincz,
2023), used to isolate what the proposed model's Transformer encoder and
dual-channel presence/magnitude split (src/models/transformer_autoencoder.py)
buy over a plain, single-channel sequence autoencoder on the exact same
file-access feature space.

Unlike the proposed model, this treats a session as one undifferentiated
sequence -- no split between "when" (presence) and "what" (magnitude) --
and uses a standard LSTM encoder-decoder instead of self-attention. This
is deliberately the simplest architecture that's still a legitimate
sequence autoencoder, so any performance gap between this and the
proposed model can be attributed to the transformer + dual-channel
design choices rather than to some other confound.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from src.training.dataset import NUMERIC_FEATURE_COLUMNS


class LSTMAutoencoder(nn.Module):
    def __init__(
        self,
        file_type_vocab_size,
        file_extension_vocab_size,
        file_type_embed_dim=8,
        file_extension_embed_dim=8,
        hidden_dim=128,  # raised from 64 -- see scripts/tune_models.py results
        num_layers=1,
        dropout=0.0,
    ):
        super().__init__()
        input_dim = len(NUMERIC_FEATURE_COLUMNS) + file_type_embed_dim + file_extension_embed_dim

        self.file_type_embedding = nn.Embedding(file_type_vocab_size, file_type_embed_dim)
        self.file_extension_embedding = nn.Embedding(
            file_extension_vocab_size, file_extension_embed_dim
        )

        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Decoder is a second LSTM, fed the same repeated context vector
        # at every timestep -- a standard, simple way to drive a
        # fixed-length seq2seq decoder without an attention mechanism
        # (attention is what's reserved for the proposed model).
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.numeric_out_proj = nn.Linear(hidden_dim, len(NUMERIC_FEATURE_COLUMNS))
        self.file_type_out_proj = nn.Linear(hidden_dim, file_type_vocab_size)
        self.file_extension_out_proj = nn.Linear(hidden_dim, file_extension_vocab_size)

    def _build_input(self, batch):
        file_type_embed = self.file_type_embedding(batch["file_type"])
        file_extension_embed = self.file_extension_embedding(batch["file_extension"])
        return torch.cat([batch["numeric"], file_type_embed, file_extension_embed], dim=-1)

    def forward(self, batch):
        """
        batch: the same collate_sessions dict the proposed model consumes.
        Returns reconstructions for the whole sequence, shape (B, L, ...).
        Numeric features are reconstructed as ONE combined block (not
        split into presence/magnitude), since this baseline is
        deliberately single-channel.
        """
        x = self._build_input(batch)
        lengths = batch["mask"].sum(dim=1).clamp(min=1).cpu()

        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (hidden, _cell) = self.encoder_lstm(packed)

        # Final layer's hidden state = the session's fixed-size encoding
        # (the autoencoder's bottleneck), repeated at every timestep as
        # the decoder's input.
        max_len = x.shape[1]
        context = hidden[-1].unsqueeze(1).repeat(1, max_len, 1)  # (B, L, hidden_dim)
        decoded, _ = self.decoder_lstm(context)

        return {
            "numeric": self.numeric_out_proj(decoded),
            "file_type_logits": self.file_type_out_proj(decoded),
            "file_extension_logits": self.file_extension_out_proj(decoded),
        }


def reconstruction_loss(outputs, batch):
    """
    Single-channel reconstruction loss: MSE over all numeric features
    combined (no presence/magnitude split, unlike the proposed model),
    plus cross-entropy for the two categorical features. Masked to real
    (non-padding) timesteps only -- same masking convention as the
    proposed model's loss.
    """
    mask = batch["mask"]
    num_real = mask.sum().clamp(min=1)

    numeric_error = (outputs["numeric"] - batch["numeric"]) ** 2
    numeric_loss = (numeric_error.mean(dim=-1) * mask).sum() / num_real

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

    total = numeric_loss + file_type_loss + file_extension_loss
    return {
        "total": total,
        "numeric": numeric_loss,
        "file_type": file_type_loss,
        "file_extension": file_extension_loss,
    }


def session_anomaly_score(outputs, batch):
    """Per-session reconstruction error, for evaluation (task 27). Same
    three components as reconstruction_loss, reduced per-session rather
    than over the whole batch. Returns a (B,) tensor."""
    mask = batch["mask"]
    lengths = mask.sum(dim=1).clamp(min=1)

    numeric_error = ((outputs["numeric"] - batch["numeric"]) ** 2).mean(dim=-1)
    numeric_score = (numeric_error * mask).sum(dim=1) / lengths

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

    return numeric_score + file_type_score + file_extension_score