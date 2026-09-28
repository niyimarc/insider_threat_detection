import pandas as pd
from src.training.vocab import build_vocab, encode, UNKNOWN_TOKEN


def test_build_vocab_assigns_unknown_to_index_zero():
    vocab = build_vocab(pd.Series(["pdf", "zip", "office_document"]))
    assert vocab[UNKNOWN_TOKEN] == 0


def test_build_vocab_gives_every_distinct_value_its_own_index():
    vocab = build_vocab(pd.Series(["pdf", "zip", "pdf", "office_document"]))
    # 3 distinct real values + the reserved unknown slot
    assert len(vocab) == 4
    assert len(set(vocab.values())) == 4  # every index is unique


def test_encode_maps_known_values_to_their_vocab_index():
    vocab = build_vocab(pd.Series(["pdf", "zip"]))
    result = encode(pd.Series(["pdf", "zip", "pdf"]), vocab)
    assert list(result) == [vocab["pdf"], vocab["zip"], vocab["pdf"]]


def test_encode_falls_back_to_unknown_for_unseen_values():
    # vocab built only from "train" data
    vocab = build_vocab(pd.Series(["pdf", "zip"]))
    # "exe" never appeared while building the vocab (e.g. only shows up
    # in validation/test) -- must fall back to the unknown index, not
    # crash or get its own new index.
    result = encode(pd.Series(["pdf", "exe"]), vocab)
    assert list(result) == [vocab["pdf"], vocab[UNKNOWN_TOKEN]]


def test_encode_is_stable_across_splits_built_from_same_vocab():
    train_vocab = build_vocab(pd.Series(["office_document", "pdf", "zip"]))
    train_encoded = encode(pd.Series(["pdf"]), train_vocab)
    val_encoded = encode(pd.Series(["pdf"]), train_vocab)
    # same vocab reused on both splits -> same string gives same index
    assert list(train_encoded) == list(val_encoded)
