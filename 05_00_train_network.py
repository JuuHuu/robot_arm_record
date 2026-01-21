import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ========= CONFIG =========
CSV_PATH = "/home/juu/Documents/robot_arm_record/exported/move_with_hammer/ml_segments_0.1.csv"
MODEL_OUT = "/home/juu/Documents/robot_arm_record/exported/move_with_hammer/posvel_to_effort_model.pth" 

TEST_SIZE = 0.3
BATCH_SIZE = 128
NUM_EPOCHS = 5000
LEARNING_RATE = 1e-4
HIDDEN_DIM = 128
WEIGHT_DECAY = 1e-4     # <-- stronger regularization
PATIENCE = 1000             # <-- early stopping patience (epochs)
DROUPUT_PROB = 0.01

IGNORE_PREFIX = "wrist_3_joint_"

def main():
    # ===== 1. Load data =====
    print(f"Loading wide CSV from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    effort_cols = [c for c in df.columns if c.endswith("_effort")]
    pos_cols = [c for c in df.columns if c.endswith("_position")]
    velocity_cols = [c for c in df.columns if c.endswith("_velocity")]
    # accleration_cols = [c for c in df.columns if c.endswith("_joint_acceleration")]
    # wrench_cols = [c for c in df.columns if c.startswith("ee")]
    
    # velocity_cols = [c for c in df.columns if c.endswith("_velocity") and not c.startswith(IGNORE_PREFIX)]
    
    

    if not effort_cols:
        raise RuntimeError("No columns ending with '_effort_lp' found in CSV.")
    if not pos_cols:
        raise RuntimeError("No columns ending with '_position' found in CSV.")
    if not velocity_cols:
        raise RuntimeError("No columns ending with '_velocity' found in CSV.")
    # if not accleration_cols:
    #     raise RuntimeError("No columns ending with '_joint_acceleration' found in CSV.")

    # INPUT = position + velocity, OUTPUT = effort
    input_cols = sorted(pos_cols+ velocity_cols )
    output_cols = sorted(effort_cols)

    print("Input columns:")
    for c in input_cols:
        print("  ", c)
    print("Output columns:")
    for c in output_cols:
        print("  ", c)

    X = df[input_cols].values.astype(np.float32)
    y = df[output_cols].values.astype(np.float32)

    print("Dataset shape:")
    print("  X:", X.shape)
    print("  y:", y.shape)

    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y).any(axis=1)
    X = X[mask]
    y = y[mask]
    print(f"After removing NaNs: N = {X.shape[0]} samples")

    # ===== 2. Train/val split =====
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42
    )

    # ===== 3. Normalize =====
    X_mean = X_train.mean(axis=0, keepdims=True)
    X_std = X_train.std(axis=0, keepdims=True)
    X_std[X_std < 1e-8] = 1.0

    Y_mean = y_train.mean(axis=0, keepdims=True)
    Y_std = y_train.std(axis=0, keepdims=True)
    Y_std[Y_std < 1e-8] = 1.0

    X_train_n = (X_train - X_mean) / X_std
    X_val_n = (X_val - X_mean) / X_std

    y_train_n = (y_train - Y_mean) / Y_std
    y_val_n = (y_val - Y_mean) / Y_std

    # ===== 4. Datasets =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_ds = TensorDataset(
        torch.from_numpy(X_train_n),
        torch.from_numpy(y_train_n),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val_n),
        torch.from_numpy(y_val_n),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # ===== 5. Model =====
    input_dim = X.shape[1]
    output_dim = y.shape[1]

    class PosVelToEffortNet(nn.Module):
        def __init__(self, in_dim, out_dim, hidden_dim=128):
            super().__init__()
            ## for moving robots
            # self.net = nn.Sequential(
            #     nn.Linear(in_dim, hidden_dim),
            #     nn.ReLU(),
            #     nn.Dropout(p=DROUPUT_PROB),
            #     nn.Linear(hidden_dim, 128),
            #     nn.ReLU(),
            #     nn.Dropout(p=DROUPUT_PROB),
            #     nn.Linear(128, 64),
            #     nn.ReLU(),
            #     nn.Linear(64, out_dim),
            # )
            
            ## for static robots
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=DROUPUT_PROB),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=DROUPUT_PROB),
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, out_dim),
            )


        def forward(self, x):
            return self.net(x)

    model = PosVelToEffortNet(input_dim, output_dim, HIDDEN_DIM).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # ===== 6. Training loop with early stopping =====
    def run_epoch(loader, train_mode=True):
        if train_mode:
            model.train()
        else:
            model.eval()

        total_loss = 0.0
        n_samples = 0

        with torch.set_grad_enabled(train_mode):
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)

                pred = model(xb)
                loss = criterion(pred, yb)

                if train_mode:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                bs = xb.size(0)
                total_loss += loss.item() * bs
                n_samples += bs

        return total_loss / n_samples

    best_val = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = run_epoch(train_loader, train_mode=True)
        val_loss = run_epoch(val_loader, train_mode=False)

        # early stopping tracking
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | "
                f"train MSE: {train_loss:.6f} | "
                f"val MSE: {val_loss:.6f} | "
                f"best: {best_val:.6f}"
            )

        if no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

    # restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # ===== 7. Save model + normalization params =====
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_columns": input_cols,   # <-- FIXED
            "output_columns": output_cols, # <-- FIXED
            "X_mean": X_mean,
            "X_std": X_std,
            "Y_mean": Y_mean,
            "Y_std": Y_std,
        },
        MODEL_OUT,
    )
    print("Model saved to:", MODEL_OUT)

    # ===== 8. Quick sanity check =====
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(X_val_n[:5]).to(device)
        pred_n = model(xb).cpu().numpy()
        pred = pred_n * Y_std + Y_mean
        print("\nSample predictions (first 5):")
        print("True efforts:\n", y_val[:5])
        print("Pred efforts:\n", pred)

    # ===== 9. Per-joint RMSE in original units =====
    with torch.no_grad():
        xb = torch.from_numpy(X_val_n).to(device)
        pred_n = model(xb).cpu().numpy()
        pred = pred_n * Y_std + Y_mean

    mse_per_joint = ((pred - y_val) ** 2).mean(axis=0)
    rmse_per_joint = np.sqrt(mse_per_joint)

    print("\nPer-joint RMSE (effort units):")
    for name, rmse_d in zip(output_cols, rmse_per_joint):
        print(f"  {name}: RMSE = {rmse_d:.4f}")


if __name__ == "__main__":
    main()
