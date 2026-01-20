import argparse
import glob
import math
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# input file
DEFAULT_ROOT = "/home/juu/Documents/robot_arm_record/auto_data"
DEFAULT_PATTERN = "autosave_[0-9][0-9]_[0-9][0-9][0-9]"
DEFAULT_JOINT_CSV = "fillted_joint_states.csv"
DEFAULT_WRENCH_CSV = "filtered_wrench.csv"
DEFAULT_SEGMENT_CSV = "selected_segments.csv"
DEFAULT_MODEL_OUT = "/home/juu/Documents/robot_arm_record/weights/03_joints.pth"

# input data
DEFAULT_JOINT_VALUE_COLS = "position,velocity,effort_lp" #"position,velocity,effort_lp"
DEFAULT_WRENCH_VALUE_COLS = "" #"fx_lp,fy_lp,fz_lp,tx_lp,ty_lp,tz_lp"
DEFAULT_AUGMENT_CROPS = 10
DEFAULT_NORMALIZE_INPUT = True
DEFAULT_TRAIN_WINDOW_SECONDS = 1.0
DEFAULT_SEQ_LEN = 50

# network
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 5000
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4 
DEFAULT_HIDDEN_DIM = 128
DEFAULT_NUM_LAYERS = 3
DEFAULT_DROPOUT = 0.2
DEFAULT_VAL_RATIO = 0.2
DEFAULT_PATIENCE = 500
DEFAULT_IMPROVE_DELTA = 1e-5
DEFAULT_LOG_EVERY = 10

# output results
DEFAULT_LABEL_A_WEIGHT = 1.0
DEFAULT_LABEL_B_WEIGHT = 1.0
DEFAULT_PREDICT_LABELS = "contact,label_a,label_b" # "contact,label_a,label_b"



SEED = 3407


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _flatten_columns(cols: pd.MultiIndex) -> List[str]:
    flat = []
    for value_name, joint_name in cols:
        flat.append(f"{joint_name}_{value_name}")
    return flat


def _parse_cols(value: str) -> List[str]:
    if value is None:
        return []
    value = value.strip()
    if not value:
        return []
    if value.lower() in {"none", "null"}:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def _parse_labels(value: str) -> List[str]:
    labels = _parse_cols(value)
    valid = {"contact", "label_a", "label_b"}
    for label in labels:
        if label not in valid:
            raise ValueError(f"Unknown label in --predict-labels: {label}")
    if not labels:
        raise ValueError("No labels selected. Use --predict-labels contact,label_a,label_b or subset.")
    return labels


def encode_angle_deg(angle_deg: np.ndarray) -> np.ndarray:
    angles = np.mod(angle_deg, 360.0)
    radians = np.deg2rad(angles)
    vec = np.stack([np.cos(radians), np.sin(radians)], axis=-1).astype(np.float32)
    invalid = angle_deg < 0
    if np.any(invalid):
        vec[invalid] = np.nan
    return vec


def circular_mae_deg(pred_vec: torch.Tensor, true_vec: torch.Tensor) -> torch.Tensor:
    pred_angle = torch.atan2(pred_vec[:, 1], pred_vec[:, 0])
    true_angle = torch.atan2(true_vec[:, 1], true_vec[:, 0])
    delta = torch.atan2(torch.sin(pred_angle - true_angle), torch.cos(pred_angle - true_angle))
    return torch.abs(delta) * (180.0 / math.pi)


def load_joint_wide(joint_csv: str, joint_cols: List[str]) -> pd.DataFrame:
    if not joint_cols:
        return pd.DataFrame(columns=["time"])
    df = pd.read_csv(joint_csv)
    missing = [c for c in joint_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing joint columns in {joint_csv}: {missing}")
    df = df[["time", "joint_name"] + joint_cols]
    wide = df.pivot_table(
        index="time",
        columns="joint_name",
        values=joint_cols,
        aggfunc="mean",
    )
    wide.columns = _flatten_columns(wide.columns)
    wide = wide.reset_index().sort_values("time")
    return wide


def load_wrench(wrench_csv: str, wrench_cols: List[str]) -> pd.DataFrame:
    if not wrench_cols:
        return pd.DataFrame(columns=["time"])
    df = pd.read_csv(wrench_csv)
    missing = [c for c in wrench_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing wrench columns in {wrench_csv}: {missing}")
    keep = ["time"] + wrench_cols
    return df[keep].sort_values("time")


def load_segments(segment_csv: str) -> pd.DataFrame:
    df = pd.read_csv(segment_csv)
    required = {"start_time", "end_time", "label_a", "label_b", "contact"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {segment_csv}: {sorted(missing)}")
    return df


def merge_streams(joint_df: pd.DataFrame, wrench_df: pd.DataFrame) -> pd.DataFrame:
    if joint_df.empty and wrench_df.empty:
        return pd.DataFrame()
    if joint_df.empty:
        return wrench_df.sort_values("time").dropna()
    if wrench_df.empty:
        return joint_df.sort_values("time").dropna()
    joint_df = joint_df.sort_values("time")
    wrench_df = wrench_df.sort_values("time")
    merged = pd.merge_asof(
        joint_df,
        wrench_df,
        on="time",
        direction="nearest",
        tolerance=0.02,
    )
    return merged.dropna()


def fixed_window_center(
    segment_df: pd.DataFrame,
    feature_cols: List[str],
    window_seconds: float,
    seq_len: int,
) -> np.ndarray:
    segment_df = segment_df.sort_values("time")
    times = segment_df["time"].to_numpy()
    if len(times) < 2:
        return None
    t_start = times[0]
    t_end = times[-1]
    if (t_end - t_start) < window_seconds:
        return None
    center = 0.5 * (t_start + t_end)
    half = 0.5 * window_seconds
    win_start = center - half
    win_end = center + half
    win_df = segment_df[(segment_df["time"] >= win_start) & (segment_df["time"] <= win_end)]
    if len(win_df) < 2:
        return None
    return resample_window(win_df, feature_cols, seq_len)


def random_subsegment(
    segment_df: pd.DataFrame,
    feature_cols: List[str],
    window_seconds: float,
    seq_len: int,
) -> np.ndarray:
    segment_df = segment_df.sort_values("time")
    times = segment_df["time"].to_numpy()
    if len(times) < 2:
        return None
    t_start = times[0]
    t_end = times[-1]
    duration = t_end - t_start
    if duration <= 0:
        return None
    if duration < window_seconds:
        return None
    crop_start = np.random.uniform(t_start, t_end - window_seconds)
    crop_end = crop_start + window_seconds
    sub_df = segment_df[(segment_df["time"] >= crop_start) & (segment_df["time"] <= crop_end)]
    if len(sub_df) < 2:
        return None
    return resample_window(sub_df, feature_cols, seq_len)


def resample_window(
    window_df: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int,
) -> np.ndarray:
    window_df = window_df.sort_values("time")
    values = window_df[feature_cols].to_numpy(dtype=np.float32)
    mask = ~np.isnan(values).any(axis=1)
    values = values[mask]
    times = window_df["time"].to_numpy()[mask]
    if len(times) < 2:
        return None
    t_new = np.linspace(times[0], times[-1], seq_len)
    out = np.empty((seq_len, len(feature_cols)), dtype=np.float32)
    for idx in range(len(feature_cols)):
        out[:, idx] = np.interp(t_new, times, values[:, idx])
    return out


def build_dataset(
    root: str,
    pattern: str,
    joint_csv_name: str,
    wrench_csv_name: str,
    segment_csv_name: str,
    seq_len: int,
    joint_cols: List[str],
    wrench_cols: List[str],
    augment_crops: int,
    window_seconds: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], Dict[str, str]]:
    sequences = []
    contacts = []
    label_a = []
    label_b = []
    feature_cols = None
    sources = []

    folders = sorted(glob.glob(os.path.join(root, pattern)))
    if not folders:
        raise RuntimeError(f"No folders found for {os.path.join(root, pattern)}")

    for folder in folders:
        joint_path = os.path.join(folder, joint_csv_name)
        wrench_path = os.path.join(folder, wrench_csv_name)
        segment_path = os.path.join(folder, segment_csv_name)
        if not (os.path.exists(joint_path) and os.path.exists(wrench_path) and os.path.exists(segment_path)):
            print(f"[SKIP] Missing files in {folder}")
            continue

        joint_df = load_joint_wide(joint_path, joint_cols)
        wrench_df = load_wrench(wrench_path, wrench_cols)
        merged = merge_streams(joint_df, wrench_df)
        if merged.empty:
            print(f"[SKIP] No merged data in {folder}")
            continue

        if feature_cols is None:
            feature_cols = [c for c in merged.columns if c != "time"]
        else:
            missing_cols = set(feature_cols).difference(merged.columns)
            if missing_cols:
                raise RuntimeError(f"Feature mismatch in {folder}: {sorted(missing_cols)}")

        segments = load_segments(segment_path)
        for _, row in segments.iterrows():
            start_t = float(row["start_time"])
            end_t = float(row["end_time"])
            seg_df = merged[(merged["time"] >= start_t) & (merged["time"] <= end_t)]
            if len(seg_df) < 2:
                continue
            base_seq = fixed_window_center(seg_df, feature_cols, window_seconds, seq_len)
            if base_seq is None:
                continue
            sequences.append(base_seq)
            contacts.append(int(row["contact"]))
            label_a.append(int(row["label_a"]))
            label_b.append(int(row["label_b"]))
            sources.append(folder)

            for _ in range(max(0, augment_crops)):
                aug_seq = random_subsegment(seg_df, feature_cols, window_seconds, seq_len)
                if aug_seq is None:
                    continue
                sequences.append(aug_seq)
                contacts.append(int(row["contact"]))
                label_a.append(int(row["label_a"]))
                label_b.append(int(row["label_b"]))
                sources.append(folder)

    if not sequences:
        raise RuntimeError("No sequences built. Check segment files and data.")

    X = np.stack(sequences).astype(np.float32)
    y_contact = np.array(contacts, dtype=np.int64)
    y_a = np.array(label_a, dtype=np.int64)
    y_b = np.array(label_b, dtype=np.int64)

    return X, y_contact, y_a, y_b, feature_cols, {"folders": ",".join(sorted(set(sources)))}


class ContactNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_a: int,
        num_b: int,
        predict_contact: bool,
        predict_label_a: bool,
        predict_label_b: bool,
        dropout: float,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.contact_head = nn.Linear(hidden_dim * 2, 1) if predict_contact else None
        self.label_a_head = nn.Linear(hidden_dim * 2, 1) if predict_label_a else None
        self.label_b_head = nn.Linear(hidden_dim * 2, 2) if predict_label_b else None

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.input_proj(x)
        out, _ = self.rnn(x)
        feat = out[:, -1, :]
        feat = self.dropout(feat)
        result = {}
        if self.contact_head is not None:
            result["contact"] = self.contact_head(feat).squeeze(-1)
        if self.label_a_head is not None:
            result["label_a"] = self.label_a_head(feat).squeeze(-1)
        if self.label_b_head is not None:
            result["label_b"] = self.label_b_head(feat)
        return result


def train(args: argparse.Namespace) -> None:
    set_seed(SEED)

    X, y_contact, y_a_raw, y_b_raw, feature_cols, source_info = build_dataset(
        args.root,
        args.pattern,
        args.joint_csv,
        args.wrench_csv,
        args.segment_csv,
        args.seq_len,
        _parse_cols(args.joint_cols),
        _parse_cols(args.wrench_cols),
        args.augment_crops,
        args.train_window_seconds,
    )

    y_a = y_a_raw.astype(np.float32)
    y_b = encode_angle_deg(y_b_raw.astype(np.float32))

    idx = np.arange(len(X))
    np.random.shuffle(idx)
    split = int(len(X) * (1.0 - args.val_ratio))
    train_idx, val_idx = idx[:split], idx[split:]

    X_train = X[train_idx]
    X_val = X[val_idx]
    y_contact_train = y_contact[train_idx]
    y_contact_val = y_contact[val_idx]
    y_a_train = y_a[train_idx]
    y_a_val = y_a[val_idx]
    y_b_train = y_b[train_idx]
    y_b_val = y_b[val_idx]

    feat_mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0, keepdims=True)
    feat_std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0, keepdims=True)
    feat_std[feat_std < 1e-6] = 1.0

    if args.normalize_input:
        X_train = (X_train - feat_mean) / feat_std
        X_val = (X_val - feat_mean) / feat_std
    else:
        feat_mean = np.zeros_like(feat_mean)
        feat_std = np.ones_like(feat_std)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_contact_train),
        torch.from_numpy(y_a_train),
        torch.from_numpy(y_b_train),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_contact_val),
        torch.from_numpy(y_a_val),
        torch.from_numpy(y_b_val),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    predict_labels = _parse_labels(args.predict_labels)
    predict_contact = "contact" in predict_labels
    predict_label_a = "label_a" in predict_labels
    predict_label_b = "label_b" in predict_labels
    if predict_label_a and not np.any(y_contact > 0):
        raise ValueError("label_a selected for prediction but no contact==1 samples found.")
    if predict_label_b and not np.any(y_contact > 0):
        raise ValueError("label_b selected for prediction but no contact==1 samples found.")

    model = ContactNet(
        in_dim=X_train.shape[-1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_a=0,
        num_b=0,
        predict_contact=predict_contact,
        predict_label_a=predict_label_a,
        predict_label_b=predict_label_b,
        dropout=args.dropout,
    ).to(device)

    contact_loss_fn = nn.BCEWithLogitsLoss() if predict_contact else None
    label_a_loss_fn = nn.MSELoss() if predict_label_a else None
    label_b_loss_fn = nn.MSELoss() if predict_label_b else None

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def run_epoch(loader: DataLoader, train_mode: bool) -> Tuple[float, Dict[str, float]]:
        model.train(train_mode)
        total_loss = 0.0
        total = 0
        metrics = {"contact_acc": 0.0, "label_a_mae": 0.0, "label_b_mae": 0.0}
        count_a = 0
        count_b = 0

        for xb, y_contact_b, y_a_b, y_b_b in loader:
            xb = xb.to(device)
            y_contact_b = y_contact_b.float().to(device)
            y_a_b = y_a_b.to(device)
            y_b_b = y_b_b.to(device)

            logits = model(xb)
            loss = 0.0
            if "contact" in logits:
                loss = loss + contact_loss_fn(logits["contact"], y_contact_b)
            if "label_a" in logits:
                mask_a = (y_contact_b > 0.5) & (y_a_b != -1)
                if mask_a.any():
                    loss_a = label_a_loss_fn(logits["label_a"][mask_a], y_a_b[mask_a])
                    loss = loss + args.label_a_weight * loss_a
            if "label_b" in logits:
                mask_b = (y_contact_b > 0.5) & (~torch.isnan(y_b_b).any(dim=1))
                if mask_b.any():
                    loss_b = label_b_loss_fn(logits["label_b"][mask_b], y_b_b[mask_b])
                    loss = loss + args.label_b_weight * loss_b

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            bs = xb.size(0)
            total_loss += loss.item() * bs
            total += bs

            with torch.no_grad():
                if "contact" in logits:
                    contact_pred = (torch.sigmoid(logits["contact"]) >= 0.5).long()
                    metrics["contact_acc"] += (contact_pred == y_contact_b.long()).sum().item()

                if "label_a" in logits:
                    mask_a = (y_contact_b > 0.5) & (y_a_b != -1)
                    if mask_a.any():
                        pred_a = logits["label_a"]
                        metrics["label_a_mae"] += torch.abs(pred_a[mask_a] - y_a_b[mask_a]).sum().item()
                        count_a += mask_a.sum().item()

                if "label_b" in logits:
                    mask_b = (y_contact_b > 0.5) & (~torch.isnan(y_b_b).any(dim=1))
                    if mask_b.any():
                        pred_b = logits["label_b"]
                        metrics["label_b_mae"] += circular_mae_deg(pred_b[mask_b], y_b_b[mask_b]).sum().item()
                        count_b += mask_b.sum().item()

        metrics["contact_acc"] = metrics["contact_acc"] / max(total, 1) if predict_contact else 0.0
        metrics["label_a_mae"] = metrics["label_a_mae"] / max(count_a, 1)
        metrics["label_b_mae"] = metrics["label_b_mae"] / max(count_b, 1)

        return total_loss / max(total, 1), metrics

    best_val = float("inf")
    best_state = None
    best_metrics = None
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(train_loader, True)
        val_loss, val_metrics = run_epoch(val_loader, False)

        if val_loss < best_val - args.improve_delta:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = val_metrics
            no_improve = 0
        else:
            no_improve += 1

        if epoch % args.log_every == 0 or epoch == 1:
            print(
                f"Epoch {epoch:04d} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
                f"train acc/mae {train_metrics['contact_acc']:.3f}/{train_metrics['label_a_mae']:.3f}/{train_metrics['label_b_mae']:.3f} | "
                f"val acc/mae {val_metrics['contact_acc']:.3f}/{val_metrics['label_a_mae']:.3f}/{val_metrics['label_b_mae']:.3f}"
            )

        if no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        if best_metrics is not None:
            print(
                f"Best val | loss {best_val:.4f} | "
                f"acc/mae {best_metrics['contact_acc']:.3f}/"
                f"{best_metrics['label_a_mae']:.3f}/"
                f"{best_metrics['label_b_mae']:.3f}"
            )

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_columns": feature_cols,
            "seq_len": args.seq_len,
            "train_window_seconds": args.train_window_seconds,
            "label_a_map": None,
            "label_b_map": None,
            "label_a_regression": True,
            "label_b_regression": True,
            "label_b_encoding": "sin_cos",
            "label_b_units": "deg",
            "label_b_range": 360.0,
            "predict_labels": predict_labels,
            "feat_mean": feat_mean,
            "feat_std": feat_std,
            "normalize_input": bool(args.normalize_input),
            "source_info": source_info,
        },
        args.model_out,
    )
    print(f"Model saved to: {args.model_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train contact + label network from selected segments.")
    parser.add_argument("--root", default=DEFAULT_ROOT, help="Root folder containing apply_force folders.")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN, help="Folder glob pattern under root.")
    parser.add_argument("--joint-csv", default=DEFAULT_JOINT_CSV, help="Joint CSV name.")
    parser.add_argument("--wrench-csv", default=DEFAULT_WRENCH_CSV, help="Wrench CSV name.")
    parser.add_argument("--segment-csv", default=DEFAULT_SEGMENT_CSV, help="Segment CSV name.")
    parser.add_argument(
        "--joint-cols",
        default=DEFAULT_JOINT_VALUE_COLS,
        help="Comma-separated joint columns to use (e.g. position,velocity,effort_lp).",
    )
    parser.add_argument(
        "--wrench-cols",
        default=DEFAULT_WRENCH_VALUE_COLS,
        help="Comma-separated wrench columns to use (e.g. fx_lp,fy_lp,fz_lp,tx_lp,ty_lp,tz_lp).",
    )
    parser.add_argument(
        "--predict-labels",
        default=DEFAULT_PREDICT_LABELS,
        help="Comma-separated labels to predict: contact,label_a,label_b",
    )
    parser.add_argument(
        "--normalize-input",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_NORMALIZE_INPUT,
        help="Normalize input features using training mean/std.",
    )
    parser.add_argument(
        "--augment-crops",
        type=int,
        default=DEFAULT_AUGMENT_CROPS,
        help="Number of random sub-sequence crops per segment.",
    )
    parser.add_argument(
        "--train-window-seconds",
        type=float,
        default=DEFAULT_TRAIN_WINDOW_SECONDS,
        help="Fixed time window (seconds) per sample; shorter segments are dropped.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=DEFAULT_SEQ_LEN,
        help="Sequence length (number of samples after resampling).",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="Weight decay.")
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM, help="GRU hidden size.")
    parser.add_argument("--num-layers", type=int, default=DEFAULT_NUM_LAYERS, help="GRU layers.")
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT, help="Dropout probability.")
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO, help="Validation split ratio.")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="Early stopping patience.")
    parser.add_argument(
        "--improve-delta",
        type=float,
        default=DEFAULT_IMPROVE_DELTA,
        help="Minimum validation loss improvement to reset early stopping.",
    )
    parser.add_argument("--log-every", type=int, default=DEFAULT_LOG_EVERY, help="Log every N epochs.")
    parser.add_argument("--label-a-weight", type=float, default=DEFAULT_LABEL_A_WEIGHT, help="Loss weight for label_a.")
    parser.add_argument("--label-b-weight", type=float, default=DEFAULT_LABEL_B_WEIGHT, help="Loss weight for label_b.")
    parser.add_argument("--model-out", default=DEFAULT_MODEL_OUT, help="Output model path.")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
