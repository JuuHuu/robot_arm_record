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
# If TEST_FILES is empty, a random train/test split is used from TRAIN_FILES.
TRAIN_FILES = [
    "/home/juu/Documents/robot_arm_record/exported/apply_force_A/force_segments_wide.csv",
    "/home/juu/Documents/robot_arm_record/exported/apply_force_B/force_segments_wide.csv",
    "/home/juu/Documents/robot_arm_record/exported/apply_force_C/force_segments_wide.csv",
    "/home/juu/Documents/robot_arm_record/exported/apply_force_D/force_segments_wide.csv",
    "/home/juu/Documents/robot_arm_record/exported/apply_force_E/force_segments_wide.csv",
    "/home/juu/Documents/robot_arm_record/exported/apply_force_F/force_segments_wide.csv",
    "/home/juu/Documents/robot_arm_record/exported/apply_force_G/force_segments_wide.csv",
    "/home/juu/Documents/robot_arm_record/exported/apply_force_H/force_segments_wide.csv",
]
TEST_FILES = [
    # "/home/juu/Documents/robot_arm_record/exported/apply_force_C/force_segments_wide.csv",
]

# Exclude data whose file path, parent folder, or source_folder starts with any prefix.
# Example: ["apply_force_B", "/home/juu/Documents/robot_arm_record/exported/apply_force_C"]
EXCLUDE_PREFIXES = []

# Use label column from CSV if present; otherwise set labels per file.
USE_LABEL_COLUMN = True
FILE_LABELS = {
    "/home/juu/Documents/robot_arm_record/exported/apply_force_A/force_segments_wide.csv": "A",
    "/home/juu/Documents/robot_arm_record/exported/apply_force_B/force_segments_wide.csv": "B",
    "/home/juu/Documents/robot_arm_record/exported/apply_force_C/force_segments_wide.csv": "C",
    "/home/juu/Documents/robot_arm_record/exported/apply_force_D/force_segments_wide.csv": "D",
    "/home/juu/Documents/robot_arm_record/exported/apply_force_E/force_segments_wide.csv": "E",
    "/home/juu/Documents/robot_arm_record/exported/apply_force_F/force_segments_wide.csv": "F",
    "/home/juu/Documents/robot_arm_record/exported/apply_force_G/force_segments_wide.csv": "G",
    "/home/juu/Documents/robot_arm_record/exported/apply_force_H/force_segments_wide.csv": "H",
}

LABEL_COL = "label"

# Feature selection
FEATURE_SUFFIXES = None # e.g. ("_position", "_velocity", "_effort", "_effort_lp", "_mean")
FEATURE_EXCLUDE = {"segment_id", "label", "source_folder", "t_start", "t_end", "t_center"}
# Exclude feature columns that start with any prefix in this tuple.
FEATURE_EXCLUDE_PREFIXES = ("ee_") # e.g. ("wrist_3_joint_", "temp_")

# Training params
TEST_SIZE = 0.25
BATCH_SIZE = 100
NUM_EPOCHS = 3000
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
HIDDEN_DIM = 64
DROPOUT_PROB = 0.02
PATIENCE = 500

OUT_DIR = "/home/juu/Documents/robot_arm_record/exported/contact_point_classifier"
MODEL_OUT = os.path.join(OUT_DIR, "contact_point_classifier.pth")
# ==========================


def load_csv_with_label(path):
    df = pd.read_csv(path)
    if USE_LABEL_COLUMN and LABEL_COL in df.columns:
        return df
    if path not in FILE_LABELS:
        raise RuntimeError(f"Missing label for {path} in FILE_LABELS.")
    df[LABEL_COL] = FILE_LABELS[path]
    return df


def pick_feature_columns(columns):
    numeric_cols = [c for c in columns if c not in FEATURE_EXCLUDE]
    if FEATURE_EXCLUDE_PREFIXES:
        numeric_cols = [
            c for c in numeric_cols if not any(c.startswith(p) for p in FEATURE_EXCLUDE_PREFIXES)
        ]
    if FEATURE_SUFFIXES is None:
        return numeric_cols
    return [c for c in numeric_cols if c.endswith(FEATURE_SUFFIXES)]


def is_excluded_path(path):
    if not EXCLUDE_PREFIXES:
        return False
    p = Path(path)
    p_str = p.as_posix()
    for prefix in EXCLUDE_PREFIXES:
        if p_str.startswith(prefix) or p.name.startswith(prefix) or p.parent.name.startswith(prefix):
            return True
    return False


def drop_excluded_sources(df):
    if not EXCLUDE_PREFIXES or "source_folder" not in df.columns:
        return df
    source = df["source_folder"].astype(str)
    return df[~source.str.startswith(tuple(EXCLUDE_PREFIXES))]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    train_files = [p for p in TRAIN_FILES if not is_excluded_path(p)]
    test_files = [p for p in TEST_FILES if not is_excluded_path(p)]
    if not train_files:
        raise RuntimeError("All TRAIN_FILES excluded. Check EXCLUDE_PREFIXES.")

    train_dfs = [load_csv_with_label(p) for p in train_files]
    train_df = pd.concat(train_dfs, ignore_index=True)
    train_df = drop_excluded_sources(train_df)

    if test_files:
        test_dfs = [load_csv_with_label(p) for p in test_files]
        test_df = pd.concat(test_dfs, ignore_index=True)
        test_df = drop_excluded_sources(test_df)
    else:
        test_df = None

    feature_cols = pick_feature_columns(train_df.columns)
    if not feature_cols:
        raise RuntimeError("No feature columns found. Check FEATURE_SUFFIXES/EXCLUDE.")

    X = train_df[feature_cols].values.astype(np.float32)
    y_raw = train_df[LABEL_COL].values

    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    y_raw = y_raw[mask]

    classes = sorted(pd.unique(y_raw).tolist())
    label_to_idx = {lbl: i for i, lbl in enumerate(classes)}
    y = np.array([label_to_idx[lbl] for lbl in y_raw], dtype=np.int64)

    if test_df is None:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=42, stratify=y
        )
    else:
        X_train = X
        y_train = y
        X_val = test_df[feature_cols].values.astype(np.float32)
        y_val_raw = test_df[LABEL_COL].values
        mask_val = ~np.isnan(X_val).any(axis=1)
        X_val = X_val[mask_val]
        y_val_raw = y_val_raw[mask_val]
        y_val = np.array([label_to_idx[lbl] for lbl in y_val_raw], dtype=np.int64)

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
        def __init__(self, in_dim, num_classes, hidden_dim=128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=DROPOUT_PROB),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=DROPOUT_PROB),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=DROPOUT_PROB),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=DROPOUT_PROB),
                nn.Linear(hidden_dim, num_classes),
            )

        def forward(self, x):
            return self.net(x)

    model = Classifier(X_train.shape[1], len(classes), HIDDEN_DIM).to(device)
    criterion = nn.CrossEntropyLoss()
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
                preds = torch.argmax(logits, dim=1)
                total_correct += (preds == yb).sum().item()
                total += yb.size(0)
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
            "label_mapping": label_to_idx,
        },
        MODEL_OUT,
    )
    print(f"Saved model to: {MODEL_OUT}")


if __name__ == "__main__":
    main()
