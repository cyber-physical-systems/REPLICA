#!/usr/bin/env python3
from pathlib import Path
import argparse
import joblib
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

from hvac_common import RANDOM_SEED, DEFAULT_OUT, find_frozen, load_npz, write_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None)
    ap.add_argument("--artifact", default=str(DEFAULT_OUT / "xgb_candidate.pkl"))
    ap.add_argument("--metric-file", required=True)
    args = ap.parse_args()

    inp = Path(args.input) if args.input else find_frozen("hvac_tabular_train_*.npz")
    X, Y = load_npz(inp)

    base = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        random_state=RANDOM_SEED,
        n_jobs=2,
        tree_method="hist",
        verbosity=1,
    )
    model = MultiOutputRegressor(base, n_jobs=1)
    model.fit(X, Y)

    artifact = Path(args.artifact)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact)

    write_json(args.metric_file, {
        "training_samples": int(len(X)),
        "input_file": str(inp),
        "n_features": int(X.shape[1]),
        "n_targets": int(Y.shape[1]),
        "model_family": "XGBoost",
        "service_operation": "retrain_frozen_input",
    })

if __name__ == "__main__":
    main()
