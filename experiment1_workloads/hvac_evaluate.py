#!/usr/bin/env python3
from pathlib import Path
import argparse
import joblib
import numpy as np

from hvac_common import find_frozen, load_npz, load_scalers, load_metadata, write_json

def regression_metrics(Y_scaled, pred_scaled):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    _, y_cols = load_metadata()[1:]
    _, y_scaler = load_scalers()
    Y = y_scaler.inverse_transform(Y_scaled)
    P = y_scaler.inverse_transform(pred_scaled)

    per_target = {}
    for i, col in enumerate(y_cols):
        per_target[col] = {
            "mae": float(mean_absolute_error(Y[:, i], P[:, i])),
            "rmse": float(mean_squared_error(Y[:, i], P[:, i]) ** 0.5),
            "r2": float(r2_score(Y[:, i], P[:, i])),
        }

    return {
        "mean_mae": float(np.mean([v["mae"] for v in per_target.values()])),
        "mean_rmse": float(np.mean([v["rmse"] for v in per_target.values()])),
        "mean_r2": float(np.mean([v["r2"] for v in per_target.values()])),
        "per_target": per_target,
        "n": int(len(Y)),
    }

def eval_tabular(model_path, model_type, input_path, metric_file):
    X, Y = load_npz(input_path)
    model = joblib.load(model_path)
    pred = model.predict(X)
    m = regression_metrics(Y, pred)
    m["model_family"] = model_type
    m["input_file"] = str(input_path)
    write_json(metric_file, m)

def eval_lstm(model_path, input_path, metric_file):
    import torch
    from hvac_lstm_update import LSTMNextState

    X, Y = load_npz(input_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(model_path, map_location=device)

    model = LSTMNextState(
        input_size=ckpt["input_size"],
        hidden_size=ckpt["hidden_size"],
        output_size=ckpt["output_size"],
        num_layers=ckpt["num_layers"],
        dropout=ckpt["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    preds = []
    batch_size = 4096
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[start:start + batch_size]).to(device)
            preds.append(model(xb).cpu().numpy())

    pred = np.concatenate(preds, axis=0)
    m = regression_metrics(Y, pred)
    m["model_family"] = "LSTM"
    m["input_file"] = str(input_path)
    write_json(metric_file, m)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-type", required=True, choices=["lstm", "rf", "xgb"])
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--input", default=None)
    ap.add_argument("--metric-file", required=True)
    args = ap.parse_args()

    if args.model_type == "lstm":
        inp = Path(args.input) if args.input else find_frozen("hvac_lstm_eval_*.npz")
        eval_lstm(args.model_path, inp, args.metric_file)
    else:
        inp = Path(args.input) if args.input else find_frozen("hvac_tabular_eval_*.npz")
        eval_tabular(
            args.model_path,
            "RandomForest" if args.model_type == "rf" else "XGBoost",
            inp,
            args.metric_file,
        )

if __name__ == "__main__":
    main()
