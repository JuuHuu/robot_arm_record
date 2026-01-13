#!/usr/bin/env python3
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ========= CONFIG =========
MOVE_CSV = "/home/juu/Documents/robot_arm_record/exported/move_with_hammer/ml_segments_0.1.csv"
RANDOM_CSV = "/home/juu/Documents/robot_arm_record/exported/random_with_velocity_plan/ml_segments_0.1.csv"

MOVE_LABEL = 1
RANDOM_LABEL = 0

OUT_DIR = "/home/juu/Documents/robot_arm_record/exported/classify_move_vs_random"
MOVE_LABELED = os.path.join(OUT_DIR, "move_with_hammer_labeled.csv")
RANDOM_LABELED = os.path.join(OUT_DIR, "random_with_velocity_plan_labeled.csv")
COMBINED_LABELED = os.path.join(OUT_DIR, "combined_labeled.csv")
MODEL_OUT = os.path.join(OUT_DIR, "move_vs_random_classifier.pth")

TEST_SIZE = 0.25
BATCH_SIZE = 128
NUM_EPOCHS = 300
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_DIM = 256
DROPOUT_PROB = 0.1
PATIENCE = 30

LABEL_COL = "label"
# ==========================


def load_and_label(csv_path, label_value):
    df = pd.read_csv(csv_path)
    df[LABEL_COL] = int(label_value)
    return df


def pick_feature_columns(columns):
    suffixes = ("_position", "_velocity", "_effort", "_effort_lp")
    return [c for c in columns if c.endswith(suffixes)]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading: {MOVE_CSV}")
    move_df = load_and_label(MOVE_CSV, MOVE_LABEL)
    print(f"Loading: {RANDOM_CSV}")
    random_df = load_and_label(RANDOM_CSV, RANDOM_LABEL)

    # Keep only shared columns to avoid mismatches
    common_cols = sorted(set(move_df.columns).intersection(set(random_df.columns)))
    move_df = move_df[common_cols]
    random_df = random_df[common_cols]

    move_df.to_csv(MOVE_LABELED, index=False)
    random_df.to_csv(RANDOM_LABELED, index=False)
    print(f"Labeled data saved to: {MOVE_LABELED}")
    print(f"Labeled data saved to: {RANDOM_LABELED}")

    df = pd.concat([move_df, random_df], ignore_index=True)
    df.to_csv(COMBINED_LABELED, index=False)
    print(f"Combined labeled data saved to: {COMBINED_LABELED}")

    feature_cols = pick_feature_columns(df.columns)
    if not feature_cols:
        raise RuntimeError("No feature columns found with suffixes: _position, _velocity, _effort, _effort_lp")

    X = df[feature_cols].values.astype(np.float32)
    y = df[LABEL_COL].values.astype(np.float32).reshape(-1, 1)

    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    y = y[mask]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )

    # Normalize
    X_mean = X_train.mean(axis=0, keepdims=True)
    X_std = X_train.std(axis=0, keepdims=True)
    X_std[X_std < 1e-8] = 1.0

    X_train_n = (X_train - X_mean) / X_std
    X_val_n = (X_val - X_mean) / X_std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_ds = TensorDataset(torch.from_numpy(X_train_n), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val_n), torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    class Classifier(nn.Module):
        def __init__(self, in_dim, hidden_dim=128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=DROPOUT_PROB),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=DROPOUT_PROB),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, x):
            return self.net(x)

    model = Classifier(X.shape[1], HIDDEN_DIM).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    def eval_loader(loader):
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                total_correct += (preds == yb).sum().item()
                total += yb.numel()
                total_loss += loss.item() * xb.size(0)
        return total_loss / max(1, total), total_correct / max(1, total)

    best_val = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_loss, val_acc = eval_loader(val_loader)
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:04d} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

        if no_improve >= PATIENCE:
            print("Early stopping.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(
        {
            "model_state": model.state_dict(),
            "feature_cols": feature_cols,
            "x_mean": X_mean.astype(np.float32),
            "x_std": X_std.astype(np.float32),
            "label_mapping": {"random_with_velocity_plan": RANDOM_LABEL, "move_with_hammer": MOVE_LABEL},
        },
        MODEL_OUT,
    )
    print(f"Saved model to: {MODEL_OUT}")


if __name__ == "__main__":
    main()
