"""
Diagnostic: do the already-trained models detect REAL malicious activity
when the ground truth is tightened using the answer key's event-level
records, instead of the whole insider time window?

Background: data/raw/answers/r4.2-<scenario>/r4.2-<scenario>-<USER>.csv
lists the exact log events the CERT authors injected for each insider
(http, device, ...). For scenarios 1 and 2 that includes the specific
removable-device Connect/Disconnect events used to steal data. File
copies made while one of THOSE device sessions was open are the real
exfiltration copies; every other file copy by the same user inside the
multi-week window is ordinary behaviour that the window-based label
mislabels as malicious (see diagnose_scenario2_dilution.py).

Tight label used here: a test session is "tight-malicious" if at least
one of its file events falls inside a Connect->Disconnect interval taken
from the answer key's own device records for that user.

Compared against two different negative sets, so the result can't be
explained by user-level differences alone:
  A. sessions of users that never appear in the answer key (clean benign)
  B. non-tight sessions of the insider users themselves (their own
     ordinary behaviour -- the hardest, fairest comparison)

Read-only: no retraining, no checkpoint or data changes. Training only
ever saw benign users, so changing which TEST sessions count as
malicious does not touch anything the models learned from.

IMPORTANT for reporting: this is a relabelling decided after seeing the
window-label results. Report BOTH labelings side by side and state why
the tight one was introduced (event-level answer key found), rather than
presenting only the better-looking number.

Run from the project root:
    python scripts/diagnose_tight_labels.py
"""

import csv
import glob
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
import joblib
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from src.training.vocab import load_vocab
from src.training.scaling import load_scaler
from src.training.feature_config import verify_feature_config
from src.training.dataset import SessionDataset, collate_sessions, SESSION_KEY_COLUMNS
from src.preprocessing.labeling import load_malicious_windows
from src.models.transformer_autoencoder import (
    DualChannelTransformerAutoencoder,
    session_anomaly_score as transformer_score_fn,
)
from src.models.lstm_autoencoder import LSTMAutoencoder, session_anomaly_score as lstm_score_fn
from src.models.static_vector_baseline import build_session_vectors
from src.evaluation.score import score_pytorch_model, score_static_model
from src.evaluation.partition import select_test_df

DATA_PATH = os.path.join(_PROJECT_ROOT, "data", "processed", "file_events_processed.parquet")
CHECKPOINT_DIR = os.path.join(_PROJECT_ROOT, "checkpoints")
ANSWERS_DIR = os.path.join(_PROJECT_ROOT, "data", "raw", "answers")
BATCH_SIZE = 64


def load_key_device_sessions():
    """Reads every r4.2-<scenario>/*.csv answer file and returns a
    DataFrame of (user, scenario, start, end): one row per
    Connect->Disconnect device session listed in the answer key."""
    rows = []
    for path in glob.glob(os.path.join(ANSWERS_DIR, "r4.2-*", "*.csv")):
        scenario = int(os.path.basename(os.path.dirname(path)).split("-")[1])
        events = []
        with open(path, newline="") as f:
            for row in csv.reader(f):
                if len(row) >= 6 and row[0] == "device":
                    events.append((pd.to_datetime(row[2], format="%m/%d/%Y %H:%M:%S"), row[3], row[4], row[5]))
        events.sort()
        open_connect = {}
        for date, user, pc, activity in events:
            key = (user, pc)
            if activity == "Connect":
                open_connect[key] = date
            elif activity == "Disconnect" and key in open_connect:
                rows.append({"user": user, "scenario": scenario, "start": open_connect.pop(key), "end": date})
    return pd.DataFrame(rows)


def label_tight(test_df, key_sessions):
    """Adds boolean column `tight_malicious` to each file event: True if
    it falls inside a key-listed device session for the same user."""
    test_df = test_df.copy()
    test_df["tight_malicious"] = False
    by_user = {u: g for u, g in key_sessions.groupby("user")}
    for user, windows in by_user.items():
        user_mask = test_df["user"] == user
        if not user_mask.any():
            continue
        dates = test_df.loc[user_mask, "date"]
        hit = pd.Series(False, index=dates.index)
        for _, w in windows.iterrows():
            hit |= (dates >= w["start"]) & (dates <= w["end"])
        test_df.loc[hit[hit].index, "tight_malicious"] = True
    return test_df


def session_labels(test_df, scenario_map):
    """Per-session tight label + user + scenario, in SessionDataset order."""
    rows = []
    for _, s in test_df.sort_values(SESSION_KEY_COLUMNS).groupby(SESSION_KEY_COLUMNS, sort=False):
        user = s["user"].iloc[0]
        rows.append({
            "user": user,
            "tight": bool(s["tight_malicious"].any()),
            "window": bool(s["session_is_malicious"].iloc[0]),
            "scenario": scenario_map.get(user, 0),
            "insider": user in scenario_map,
        })
    return pd.DataFrame(rows)


def auroc(labels, scores, mask):
    labels = np.asarray(labels)[mask]
    scores = np.asarray(scores)[mask]
    if labels.sum() == 0 or labels.sum() == len(labels):
        return None, None
    return roc_auc_score(labels, scores), average_precision_score(labels, scores)


def fmt(v):
    return "n/a" if v is None else f"{v:.4f}"


def report(name, scores, t):
    scores = np.asarray(scores)
    tight = t["tight"].values
    insider = t["insider"].values
    clean = ~insider

    print(f"\n=== {name} ===")
    a_roc, a_pr = auroc(tight, scores, tight | clean)
    print(f"  tight-malicious vs clean benign users (A)       AUROC={fmt(a_roc)}  AUPRC={fmt(a_pr)}")
    b_roc, b_pr = auroc(tight, scores, insider)
    print(f"  tight vs insiders' own other sessions (B)        AUROC={fmt(b_roc)}  AUPRC={fmt(b_pr)}")
    for sc in (1, 2, 3):
        sc_mask = (t["scenario"].values == sc) & tight
        keep = sc_mask | clean
        s_roc, _ = auroc(sc_mask, scores, keep)
        print(f"    scenario {sc} only vs clean (A)  n_tight={int(sc_mask.sum()):4d}  AUROC={fmt(s_roc)}")
    return {"model": name, "A": a_roc, "B": b_roc}


def main():
    print(f"Loading {DATA_PATH} ...")
    events_df = pd.read_parquet(DATA_PATH)
    test_df = select_test_df(events_df)

    print("Reading event-level answer key (device sessions) ...")
    key_sessions = load_key_device_sessions()
    print(f"  {len(key_sessions)} key-listed device sessions across "
          f"{key_sessions['user'].nunique()} insiders")

    scenario_map = load_malicious_windows().set_index("user")["scenario"].to_dict()
    test_df = label_tight(test_df, key_sessions)
    t = session_labels(test_df, scenario_map)

    n_tight = int(t["tight"].sum())
    n_window = int(t["window"].sum())
    print(f"\nTest sessions: {len(t)}")
    print(f"  window-malicious (old label): {n_window}")
    print(f"  tight-malicious  (new label): {n_tight}")
    print(f"  tight sessions by scenario: "
          f"{ {sc: int(((t['scenario'] == sc) & t['tight']).sum()) for sc in (1, 2, 3)} }")
    if n_tight == 0:
        print("No tight-malicious sessions found -- check that file events overlap key device sessions.")
        return

    file_type_vocab = load_vocab(os.path.join(CHECKPOINT_DIR, "file_type_vocab.json"))
    file_extension_vocab = load_vocab(os.path.join(CHECKPOINT_DIR, "file_extension_vocab.json"))
    verify_feature_config(CHECKPOINT_DIR)
    scaler = load_scaler(os.path.join(CHECKPOINT_DIR, "numeric_scaler.json"))

    test_dataset = SessionDataset(test_df, file_type_vocab, file_extension_vocab, scaler)
    assert test_dataset.session_users == t["user"].tolist(), "session order mismatch"
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_sessions)

    summary = []

    transformer_model = DualChannelTransformerAutoencoder(len(file_type_vocab), len(file_extension_vocab))
    transformer_model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "transformer_autoencoder.pt")))
    scores, _ = score_pytorch_model(transformer_model, test_loader, transformer_score_fn)
    summary.append(report("proposed_transformer_autoencoder", scores, t))

    lstm_model = LSTMAutoencoder(len(file_type_vocab), len(file_extension_vocab))
    lstm_model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "lstm_autoencoder.pt")))
    scores, _ = score_pytorch_model(lstm_model, test_loader, lstm_score_fn)
    summary.append(report("baseline_lstm_autoencoder", scores, t))

    static_detector = joblib.load(os.path.join(CHECKPOINT_DIR, "static_vector_detector.joblib"))
    vectors, _ = build_session_vectors(test_df, file_type_vocab, file_extension_vocab)
    scores = score_static_model(static_detector, vectors)
    summary.append(report("baseline_static_vector_isolation_forest", scores, t))

    print("\n=== Summary (AUROC) ===")
    print(f"{'model':42s} {'tight vs clean (A)':>20s} {'tight vs own (B)':>18s}")
    for r in summary:
        print(f"{r['model']:42s} {fmt(r['A']):>20s} {fmt(r['B']):>18s}")


if __name__ == "__main__":
    main()
