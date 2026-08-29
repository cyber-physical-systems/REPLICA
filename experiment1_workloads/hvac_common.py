#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json
import joblib
import numpy as np

PROJECT_ROOT = Path("/workspace/sc26_rebuttal")
SOURCE_ROOT = PROJECT_ROOT / "outputs" / "09A_ai_normal_dynamics"
MODEL_ROOT = SOURCE_ROOT / "models"
INPUT_ROOT = PROJECT_ROOT / "outputs" / "experiment1" / "workload_inputs"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "experiment1" / "candidates"

X_SCALER_PATH = MODEL_ROOT / "09A_x_scaler.pkl"
Y_SCALER_PATH = MODEL_ROOT / "09A_y_scaler.pkl"
COLUMNS_PATH = MODEL_ROOT / "09A_columns.pkl"

RANDOM_SEED = 42

def load_metadata():
    meta = joblib.load(COLUMNS_PATH)
    return meta, list(meta["x_cols"]), list(meta["y_cols"])

def load_scalers():
    return joblib.load(X_SCALER_PATH), joblib.load(Y_SCALER_PATH)

def load_npz(path):
    z = np.load(path)
    return z["X"], z["Y"]

def find_frozen(prefix):
    matches = sorted(INPUT_ROOT.glob(prefix))
    if not matches:
        raise FileNotFoundError(
            f"No frozen input matching {INPUT_ROOT / prefix}. "
            "Run prepare_hvac_workload_inputs.py first."
        )
    return matches[-1]

def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
