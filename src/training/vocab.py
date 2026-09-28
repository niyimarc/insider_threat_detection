UNKNOWN_TOKEN = "<unknown>"


def build_vocab(series):
    """
    Builds a string -> integer mapping from the distinct values seen in
    `series` (typically a training-split column). Index 0 is always
    reserved for UNKNOWN_TOKEN, so that a value never seen during
    vocabulary-building (e.g. a rare file_extension that only shows up
    in validation/test) has somewhere safe to map to instead of crashing
    or being silently invented a new index.

    Only fit this on the training split. Reuse the same returned dict
    for encoding validation/test — rebuilding it per-split would let the
    same string map to a different integer in each split, which would
    make the encoded values meaningless to the model.
    """
    distinct_values = sorted(series.dropna().unique())
    vocab = {UNKNOWN_TOKEN: 0}
    for value in distinct_values:
        if value not in vocab:
            vocab[value] = len(vocab)
    return vocab


def encode(series, vocab):
    """
    Maps each value in `series` to its integer index in `vocab`, falling
    back to UNKNOWN_TOKEN's index (0) for any value not present in the
    vocabulary (e.g. an extension that only appears in validation/test).
    """
    unknown_index = vocab[UNKNOWN_TOKEN]
    return series.map(lambda value: vocab.get(value, unknown_index))
