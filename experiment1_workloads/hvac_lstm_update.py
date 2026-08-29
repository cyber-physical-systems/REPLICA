#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from hvac_common import RANDOM_SEED, DEFAULT_OUT, find_frozen, load_npz, write_json

LSTM_EPOCHS = 25
LSTM_BATCH_SIZE = 512
LSTM_LR = 1e-3
LSTM_PATIENCE = 5
LSTM_WINDOW = 8

class LSTMNextState(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.10):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None)
    ap.add_argument("--epochs", type=int, default=LSTM_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=LSTM_BATCH_SIZE)
    ap.add_argument("--artifact", default=str(DEFAULT_OUT / "lstm_candidate.pt"))
    ap.add_argument("--metric-file", required=True)
    args = ap.parse_args()

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    inp = Path(args.input) if args.input else find_frozen("hvac_lstm_train_*.npz")
    X, Y = load_npz(inp)

    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    val_size = int(0.15 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=pin
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=pin
    )

    model = LSTMNextState(
        input_size=X.shape[2],
        hidden_size=64,
        output_size=Y.shape[1],
        num_layers=2,
        dropout=0.10,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_LR)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    best_state = None
    patience = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                val_losses.append(float(criterion(model(xb), yb).item()))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        print(f"Epoch {epoch:03d} | train={train_loss:.6f} | val={val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
            if patience >= LSTM_PATIENCE:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    artifact = Path(args.artifact)
    artifact.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "input_size": X.shape[2],
        "hidden_size": 64,
        "output_size": Y.shape[1],
        "num_layers": 2,
        "dropout": 0.10,
        "window": LSTM_WINDOW,
        "history": history,
        "best_val_loss": best_loss,
    }, artifact)

    write_json(args.metric_file, {
        "training_samples": int(len(X)),
        "input_file": str(inp),
        "epochs_completed": int(len(history)),
        "best_val_loss": float(best_loss),
        "batch_size": int(args.batch_size),
        "learning_rate": LSTM_LR,
        "model_family": "LSTM",
        "service_operation": "retrain_frozen_input",
    })

if __name__ == "__main__":
    main()
