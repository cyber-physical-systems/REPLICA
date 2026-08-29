#!/usr/bin/env python3
"""
Prepare immutable HVAC Experiment 1 workload inputs from the 09A artifacts.

This script runs ONCE before hardware profiling.

Outputs:
- hvac_tabular_train_600k.npz
- hvac_tabular_eval_100k.npz
- hvac_lstm_train_600k.npz
- hvac_lstm_eval_100k.npz
- workload_input_manifest.json

The same frozen arrays are then used on A100, RTX 5090, and RTX 4080.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import gc
import hashlib
import json
import joblib
import numpy as np
import pandas as pd

RANDOM_SEED = 42
LSTM_WINDOW = 8

PROJECT_ROOT = Path("/workspace/sc26_rebuttal")
SOURCE_ROOT = PROJECT_ROOT / "outputs" / "09A_ai_normal_dynamics"
MODEL_ROOT = SOURCE_ROOT / "models"

TRAIN_PATH = SOURCE_ROOT / "09A_train_split.parquet"
TEST_PATH = SOURCE_ROOT / "09A_test_split.parquet"

X_SCALER_PATH = MODEL_ROOT / "09A_x_scaler.pkl"
Y_SCALER_PATH = MODEL_ROOT / "09A_y_scaler.pkl"
COLUMNS_PATH = MODEL_ROOT / "09A_columns.pkl"

DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "experiment1" / "workload_inputs"


def sha256(path: Path, block=1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(block)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_metadata():
    meta = joblib.load(COLUMNS_PATH)
    return meta, list(meta["x_cols"]), list(meta["y_cols"])


def fill_and_scale(df, cols, scaler):
    raw = df[cols].apply(pd.to_numeric, errors="coerce")
    fill = pd.Series(scaler.mean_, index=cols)
    raw = raw.fillna(fill)
    return scaler.transform(raw).astype(np.float32, copy=False)


def deterministic_sample(df, n, seed=RANDOM_SEED):
    if len(df) <= n:
        return df.reset_index(drop=True)
    idx = np.random.RandomState(seed).choice(len(df), n, replace=False)
    return df.iloc[idx].reset_index(drop=True)


def build_lstm_sequences(df, x_cols, y_cols, x_scaler, y_scaler, n_samples):
    # Build exact global sequence descriptors using run ordering.
    descriptors = []
    total = 0
    grouped = []

    for run_id, group in df.groupby("run_id", sort=False):
        group = group.sort_values("timestep").reset_index(drop=True)
        n_seq = max(len(group) - LSTM_WINDOW, 0)
        if n_seq:
            descriptors.append((len(grouped), total, total + n_seq))
            grouped.append(group)
            total += n_seq

    if total == 0:
        raise RuntimeError("No valid LSTM sequences found.")

    n_take = min(n_samples, total)
    selected = np.random.RandomState(RANDOM_SEED).choice(
        total, n_take, replace=False
    )
    # Preserve selected order to match the deterministic sample itself.
    X = np.empty((n_take, LSTM_WINDOW, len(x_cols)), dtype=np.float32)
    Y = np.empty((n_take, len(y_cols)), dtype=np.float32)

    x_fill = pd.Series(x_scaler.mean_, index=x_cols)
    y_fill = pd.Series(y_scaler.mean_, index=y_cols)

    for group_idx, start, stop in descriptors:
        positions = np.flatnonzero((selected >= start) & (selected < stop))
        if not len(positions):
            continue

        local = selected[positions] - start
        group = grouped[group_idx]

        x_raw = group[x_cols].apply(pd.to_numeric, errors="coerce").fillna(x_fill)
        y_raw = group[y_cols].apply(pd.to_numeric, errors="coerce").fillna(y_fill)

        xs = x_scaler.transform(x_raw).astype(np.float32, copy=False)
        ys = y_scaler.transform(y_raw).astype(np.float32, copy=False)

        for out_pos, local_idx in zip(positions, local):
            j = int(local_idx)
            X[out_pos] = xs[j:j + LSTM_WINDOW]
            Y[out_pos] = ys[j + LSTM_WINDOW]

        del x_raw, y_raw, xs, ys
        gc.collect()

    return X, Y, int(total)


def save_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--train-samples", type=int, default=600_000)
    ap.add_argument("--eval-samples", type=int, default=100_000)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta, x_cols, y_cols = load_metadata()
    x_scaler = joblib.load(X_SCALER_PATH)
    y_scaler = joblib.load(Y_SCALER_PATH)

    print("Loading tabular training columns...")
    train_tab = pd.read_parquet(
        TRAIN_PATH, columns=list(x_cols) + list(y_cols)
    )
    train_sample = deterministic_sample(train_tab, args.train_samples)
    X_train = fill_and_scale(train_sample, x_cols, x_scaler)
    Y_train = fill_and_scale(train_sample, y_cols, y_scaler)

    tab_train_path = out / f"hvac_tabular_train_{len(X_train)}.npz"
    save_npz(tab_train_path, X=X_train, Y=Y_train)
    print("Saved:", tab_train_path)

    del train_tab, train_sample, X_train, Y_train
    gc.collect()

    print("Loading tabular evaluation columns...")
    test_tab = pd.read_parquet(
        TEST_PATH, columns=list(x_cols) + list(y_cols)
    )
    eval_sample = deterministic_sample(test_tab, args.eval_samples)
    X_eval = fill_and_scale(eval_sample, x_cols, x_scaler)
    Y_eval = fill_and_scale(eval_sample, y_cols, y_scaler)

    tab_eval_path = out / f"hvac_tabular_eval_{len(X_eval)}.npz"
    save_npz(tab_eval_path, X=X_eval, Y=Y_eval)
    print("Saved:", tab_eval_path)

    del test_tab, eval_sample, X_eval, Y_eval
    gc.collect()

    # LSTM train sequences need run_id + timestep.
    print("Loading LSTM training source columns...")
    usecols = ["run_id", "timestep"] + list(x_cols) + list(y_cols)
    train_seq_df = pd.read_parquet(TRAIN_PATH, columns=usecols)
    X_lstm_train, Y_lstm_train, total_train_sequences = build_lstm_sequences(
        train_seq_df, x_cols, y_cols, x_scaler, y_scaler, args.train_samples
    )
    lstm_train_path = out / f"hvac_lstm_train_{len(X_lstm_train)}.npz"
    save_npz(lstm_train_path, X=X_lstm_train, Y=Y_lstm_train)
    print("Saved:", lstm_train_path)

    del train_seq_df, X_lstm_train, Y_lstm_train
    gc.collect()

    print("Loading LSTM evaluation source columns...")
    test_seq_df = pd.read_parquet(TEST_PATH, columns=usecols)
    X_lstm_eval, Y_lstm_eval, total_eval_sequences = build_lstm_sequences(
        test_seq_df, x_cols, y_cols, x_scaler, y_scaler, args.eval_samples
    )
    lstm_eval_path = out / f"hvac_lstm_eval_{len(X_lstm_eval)}.npz"
    save_npz(lstm_eval_path, X=X_lstm_eval, Y=Y_lstm_eval)
    print("Saved:", lstm_eval_path)

    files = [
        tab_train_path,
        tab_eval_path,
        lstm_train_path,
        lstm_eval_path,
    ]

    manifest = {
        "random_seed": RANDOM_SEED,
        "lstm_window": LSTM_WINDOW,
        "train_samples": args.train_samples,
        "eval_samples": args.eval_samples,
        "source_train_path": str(TRAIN_PATH),
        "source_test_path": str(TEST_PATH),
        "x_cols": x_cols,
        "y_cols": y_cols,
        "total_lstm_train_sequences_before_sampling": total_train_sequences,
        "total_lstm_eval_sequences_before_sampling": total_eval_sequences,
        "files": {
            p.name: {
                "path": str(p),
                "size_mb": p.stat().st_size / (1024**2),
                "sha256": sha256(p),
            }
            for p in files
        },
    }

    manifest_path = out / "workload_input_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print("\nFrozen workload inputs created.")
    print("Manifest:", manifest_path)
    for p in files:
        print(f" - {p.name}: {p.stat().st_size / (1024**2):.2f} MB")


if __name__ == "__main__":
    main()
