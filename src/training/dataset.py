import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.training.vocab import encode

SESSION_KEY_COLUMNS = ["user", "session_start", "session_end"]

# The columns that go into the model as plain numeric features. Notably
# absent: is_malicious / session_is_malicious (the label, not an input —
# including it here would leak the answer to the model), and file_type /
# file_extension (categorical — handled separately as integer indices,
# below, for an embedding layer rather than being crammed in here).
NUMERIC_FEATURE_COLUMNS = [
    "seconds_since_last_event",
    "seconds_since_device_connect",
    "copies_since_device_connect",
    "outside_working_hours",
]


def session_to_tensors(session_df, file_type_vocab, file_extension_vocab):
    """
    Converts one session's rows (all events sharing the same user +
    session_start + session_end) into the tensors a model will consume:
      - "numeric": float tensor, shape (num_events, len(NUMERIC_FEATURE_COLUMNS))
      - "file_type": long tensor of vocab indices, shape (num_events,)
      - "file_extension": long tensor of vocab indices, shape (num_events,)
      - "label": scalar float tensor, 1.0 if this session is malicious else 0.0

    Events are sorted by `date` first, so the sequence a model sees is in
    the order the events actually happened.

    file_type_vocab/file_extension_vocab must come from build_vocab() on
    the TRAINING split only, then be reused as-is here for every split
    (see src/training/vocab.py) — this function doesn't build vocabs
    itself, it only applies ones already built.
    """
    session_df = session_df.sort_values("date")

    numeric = torch.tensor(
        session_df[NUMERIC_FEATURE_COLUMNS].astype(float).values,
        dtype=torch.float32,
    )

    file_type = torch.tensor(
        encode(session_df["file_type"], file_type_vocab).values,
        dtype=torch.long,
    )
    file_extension = torch.tensor(
        encode(session_df["file_extension"], file_extension_vocab).values,
        dtype=torch.long,
    )

    # Every row in a session shares the same session_is_malicious value
    # (label_sessions() broadcasts it across the whole session), so the
    # first row's value is the session's label.
    label = torch.tensor(
        float(session_df["session_is_malicious"].iloc[0]),
        dtype=torch.float32,
    )

    return {
        "numeric": numeric,
        "file_type": file_type,
        "file_extension": file_extension,
        "label": label,
    }


class SessionDataset(Dataset):
    """
    One sample = one session's full sequence of file-copy events.

    `events_df` should already be filtered to a single split (train,
    validation, or test) before being passed in — this class doesn't
    filter by split itself, it just groups whatever rows it's given
    into sessions using SESSION_KEY_COLUMNS (user, session_start,
    session_end), the same grouping key label_sessions() used.

    file_type_vocab/file_extension_vocab must be built once from
    build_vocab() on the TRAINING split, then passed in unchanged for
    every split's SessionDataset — see src/training/vocab.py.
    """

    def __init__(self, events_df, file_type_vocab, file_extension_vocab):
        self.file_type_vocab = file_type_vocab
        self.file_extension_vocab = file_extension_vocab

        # Sorting the group keys gives a fixed, reproducible sample order
        # (dict/groupby order isn't guaranteed to be stable across runs
        # otherwise), which matters for things like debugging by index.
        self._sessions = [
            group for _, group in events_df.sort_values(SESSION_KEY_COLUMNS).groupby(
                SESSION_KEY_COLUMNS, sort=False
            )
        ]

    def __len__(self):
        return len(self._sessions)

    def __getitem__(self, index):
        session_df = self._sessions[index]
        return session_to_tensors(
            session_df, self.file_type_vocab, self.file_extension_vocab
        )


def collate_sessions(batch):
    """
    Combines a list of session samples (each a dict from session_to_tensors,
    with a different number of events) into one padded batch, for use as
    a DataLoader's collate_fn.

    Returns a dict:
      - "numeric": float tensor (batch_size, max_len, num_numeric_features)
      - "file_type": long tensor (batch_size, max_len)
      - "file_extension": long tensor (batch_size, max_len)
      - "mask": bool tensor (batch_size, max_len) -- True where a
        timestep is a real event, False where it's padding. Padded
        positions are filled with 0 in every tensor above, but 0 is
        also a real encoded value (e.g. the "<unknown>" vocab index),
        so the model must use this mask to ignore padding rather than
        relying on the padded value itself to be distinguishable.
      - "label": float tensor (batch_size,)
    """
    numeric = pad_sequence(
        [sample["numeric"] for sample in batch], batch_first=True, padding_value=0.0
    )
    file_type = pad_sequence(
        [sample["file_type"] for sample in batch], batch_first=True, padding_value=0
    )
    file_extension = pad_sequence(
        [sample["file_extension"] for sample in batch], batch_first=True, padding_value=0
    )

    lengths = torch.tensor([sample["numeric"].shape[0] for sample in batch])
    max_len = numeric.shape[1]
    # arange(max_len) < length, broadcast per-row -> True for real
    # timesteps, False for padding -- the same trick as the hand-typed
    # mask we built earlier, just vectorised across the whole batch.
    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)

    label = torch.stack([sample["label"] for sample in batch])

    return {
        "numeric": numeric,
        "file_type": file_type,
        "file_extension": file_extension,
        "mask": mask,
        "label": label,
    }
