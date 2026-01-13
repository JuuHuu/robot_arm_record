import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path


# Paths to the trained model and the new dataset to evaluate
MODEL_PATH = Path("/home/juu/Documents/robot_arm_record/exported/new_move/posvel_to_effort_model.pth")
TEST_CSV = Path("/home/juu/Documents/robot_arm_record/exported/new_move/ml_segments_0.2.csv")


class PosVelToEffortNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def load_checkpoint():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    # Explicitly disable weights_only because the checkpoint stores numpy arrays
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

    required_keys = {"model_state_dict", "input_columns", "output_columns", "X_mean", "X_std", "Y_mean", "Y_std"}
    missing_keys = required_keys - set(ckpt.keys())
    if missing_keys:
        raise RuntimeError(f"Checkpoint missing keys: {missing_keys}")

    return ckpt


def build_model(state_dict, input_dim: int, output_dim: int):
    # Infer hidden dimension from the first linear layer weight matrix
    hidden_dim = state_dict["net.0.weight"].shape[0]
    model = PosVelToEffortNet(input_dim, output_dim, hidden_dim)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main():
    ckpt = load_checkpoint()

    input_cols = ckpt["input_columns"]
    output_cols = ckpt["output_columns"]
    X_mean = ckpt["X_mean"]
    X_std = ckpt["X_std"]
    Y_mean = ckpt["Y_mean"]
    Y_std = ckpt["Y_std"]

    print(f"Loaded checkpoint from: {MODEL_PATH}")
    print("Expecting inputs:", input_cols)
    print("Expecting outputs:", output_cols)

    df = pd.read_csv(TEST_CSV)
    print(f"\nLoaded test CSV: {TEST_CSV}  (rows: {len(df)})")

    missing_inputs = sorted(set(input_cols) - set(df.columns))
    missing_outputs = sorted(set(output_cols) - set(df.columns))
    if missing_inputs:
        raise RuntimeError(f"Missing input columns in CSV: {missing_inputs}")
    if missing_outputs:
        raise RuntimeError(f"Missing output columns in CSV: {missing_outputs}")

    X = df[input_cols].values.astype(np.float32)
    y = df[output_cols].values.astype(np.float32)

    # Drop any rows with NaNs
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y).any(axis=1)
    if not mask.all():
        print(f"Dropping {len(mask) - mask.sum()} rows containing NaNs")
        X = X[mask]
        y = y[mask]

    print(f"Evaluating on {len(X)} samples")

    # Normalize with training statistics
    X_std_safe = np.where(X_std < 1e-8, 1.0, X_std)
    Y_std_safe = np.where(Y_std < 1e-8, 1.0, Y_std)

    X_n = (X - X_mean) / X_std_safe

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(ckpt["model_state_dict"], input_dim=len(input_cols), output_dim=len(output_cols)).to(device)

    with torch.no_grad():
        xb = torch.from_numpy(X_n).to(device)
        pred_n = model(xb).cpu().numpy()

    # Convert back to original units
    preds = pred_n * Y_std_safe + Y_mean

    # Metrics
    mse_per_joint = ((preds - y) ** 2).mean(axis=0)
    rmse_per_joint = np.sqrt(mse_per_joint)
    overall_rmse = np.sqrt(((preds - y) ** 2).mean())

    print("\nSample predictions (first 5 rows):")
    for i in range(min(5, len(preds))):
        print(f"Row {i}:")
        print("  True :", y[i])
        print("  Pred :", preds[i])

    print("\nPer-joint RMSE (effort units):")
    for name, rmse_d in zip(output_cols, rmse_per_joint):
        print(f"  {name}: {rmse_d:.4f}")
    print(f"\nOverall RMSE: {overall_rmse:.4f}")


if __name__ == "__main__":
    main()
