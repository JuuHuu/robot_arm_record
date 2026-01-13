#!/usr/bin/env python3
import argparse
import time
from collections import deque
from typing import Dict, List, Tuple

import numpy as np
import rclpy
import torch
import torch.nn as nn
from geometry_msgs.msg import WrenchStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState


# ========= CONFIG =========
DEFAULT_MODEL_PATH = "/home/juu/Documents/robot_arm_record/exported/test.pth"
DEFAULT_TOPIC_JOINT = "/joint_states_filtered"
DEFAULT_TOPIC_WRENCH = "/wrench_filtered"
DEFAULT_SYNC_TOL = 0.2  # seconds
DEFAULT_WINDOW_SECONDS = 0.5  # seconds of recent data used for sequence
DEFAULT_AVG_SECONDS = 0.5  # seconds for prediction averaging
DEFAULT_TIMER_HZ = 200.0
DEFAULT_STALE_SECONDS = 1.0
DEFAULT_WARN_EVERY = 2.0
DEFAULT_IGNORE_SYNC = False
DEFAULT_DEBUG_EVERY = 2.0
DEFAULT_USE_WALL_TIME = False
DEFAULT_USE_TRAIN_WINDOW = True
DEFAULT_RESAMPLE_TO_SEQ = True

# If training used *_lp names, map to ROS fields
JOINT_VALUE_MAP = {
    "position": "position",
    "velocity": "velocity",
    "effort": "effort",
    "effort_lp": "effort",
}
WRENCH_MAP = {
    "fx": "force.x",
    "fy": "force.y",
    "fz": "force.z",
    "tx": "torque.x",
    "ty": "torque.y",
    "tz": "torque.z",
}
# ==========================


def _ros_time_to_float(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _infer_num_layers(state_dict: Dict[str, torch.Tensor]) -> int:
    layers = set()
    for key in state_dict:
        if key.startswith("rnn.weight_ih_l"):
            parts = key.split("l")
            if len(parts) < 2:
                continue
            idx = parts[1].split("_")[0]
            if idx.isdigit():
                layers.add(int(idx))
    return max(layers) + 1 if layers else 1


def _infer_predict_labels(state_dict: Dict[str, torch.Tensor], ckpt: Dict) -> List[str]:
    if "predict_labels" in ckpt and ckpt["predict_labels"]:
        return list(ckpt["predict_labels"])
    labels = []
    if any(k.startswith("contact_head") for k in state_dict):
        labels.append("contact")
    if any(k.startswith("label_a_head") for k in state_dict):
        labels.append("label_a")
    if any(k.startswith("label_b_head") for k in state_dict):
        labels.append("label_b")
    return labels


class ContactNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        predict_contact: bool,
        predict_label_a: bool,
        predict_label_b: bool,
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
        self.dropout = nn.Dropout(p=0.0)
        self.contact_head = nn.Linear(hidden_dim * 2, 1) if predict_contact else None
        self.label_a_head = nn.Linear(hidden_dim * 2, 1) if predict_label_a else None
        self.label_b_head = nn.Linear(hidden_dim * 2, 1) if predict_label_b else None

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
            result["label_b"] = self.label_b_head(feat).squeeze(-1)
        return result


class OnlinePredictor(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("online_predictor")
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.feature_cols, self.seq_len, self.predict_labels, train_window = self._load_model(
            args.model_path
        )
        if args.use_train_window and train_window is not None:
            self.args.window_seconds = float(train_window)
            self.get_logger().info(f"Using train_window_seconds={self.args.window_seconds:.3f} from checkpoint")
        self.model.eval()

        self.latest_joint = None
        self.latest_wrench = None
        self.needs_joint, self.needs_wrench = self._infer_inputs(self.feature_cols)
        self.last_data_time = None
        self.last_warn_time = 0.0
        self.last_bad_time = 0.0
        self.last_debug_time = 0.0
        self.last_sample_time = None

        self.buffer = deque()
        self.pred_buffer = deque()

        if self.needs_joint:
            self.create_subscription(JointState, args.topic_joint, self._on_joint, 10)
        if self.needs_wrench:
            self.create_subscription(WrenchStamped, args.topic_wrench, self._on_wrench, 10)
        self.timer = self.create_timer(1.0 / args.timer_hz, self._on_timer)

        self.get_logger().info(
            f"Loaded model. seq_len={self.seq_len}, features={len(self.feature_cols)}, labels={self.predict_labels}"
        )
        if args.debug:
            self.get_logger().info(
                f"Inputs needed - joint: {self.needs_joint}, wrench: {self.needs_wrench} | "
                f"topics: {args.topic_joint}, {args.topic_wrench}"
            )

    def _infer_inputs(self, feature_cols: List[str]) -> Tuple[bool, bool]:
        needs_wrench = False
        needs_joint = False
        for col in feature_cols:
            if col in WRENCH_MAP or (col.endswith("_lp") and col[:-3] in WRENCH_MAP):
                needs_wrench = True
                continue
            if "_" in col:
                needs_joint = True
        return needs_joint, needs_wrench

    def _load_model(self, path: str) -> Tuple[nn.Module, List[str], int, List[str], float]:
        try:
            ckpt = torch.load(path, map_location="cpu")
        except Exception as exc:
            if "weights_only" in str(exc):
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
            else:
                raise
        state = ckpt["model_state_dict"]
        feature_cols = ckpt["feature_columns"]
        seq_len = int(ckpt["seq_len"])
        predict_labels = _infer_predict_labels(state, ckpt)

        in_dim = state["input_proj.weight"].shape[1]
        hidden_dim = state["input_proj.weight"].shape[0]
        num_layers = _infer_num_layers(state)
        model = ContactNet(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            predict_contact="contact" in predict_labels,
            predict_label_a="label_a" in predict_labels,
            predict_label_b="label_b" in predict_labels,
        )
        model.load_state_dict(state)
        model.to(self.device)

        self.feat_mean = ckpt["feat_mean"].astype(np.float32)
        self.feat_std = ckpt["feat_std"].astype(np.float32)
        train_window = ckpt.get("train_window_seconds")
        return model, feature_cols, seq_len, predict_labels, train_window

    def _on_joint(self, msg: JointState) -> None:
        self.latest_joint = msg

    def _on_wrench(self, msg: WrenchStamped) -> None:
        self.latest_wrench = msg

    def _build_feature_vector(self, joint_msg: JointState, wrench_msg: WrenchStamped) -> np.ndarray:
        joint_map = {}
        if joint_msg is not None:
            joint_map = {name: i for i, name in enumerate(joint_msg.name)}

        feat = np.zeros(len(self.feature_cols), dtype=np.float32)
        for i, col in enumerate(self.feature_cols):
            if col in {"time"}:
                feat[i] = 0.0
                continue
            if col in WRENCH_MAP or col.endswith("_lp") and col[:-3] in WRENCH_MAP:
                base = col[:-3] if col.endswith("_lp") else col
                field = WRENCH_MAP.get(base)
                if field is None:
                    raise KeyError(f"Unknown wrench field: {col}")
                if wrench_msg is None:
                    raise KeyError("Wrench message missing but wrench feature requested")
                if field == "force.x":
                    feat[i] = float(wrench_msg.wrench.force.x)
                elif field == "force.y":
                    feat[i] = float(wrench_msg.wrench.force.y)
                elif field == "force.z":
                    feat[i] = float(wrench_msg.wrench.force.z)
                elif field == "torque.x":
                    feat[i] = float(wrench_msg.wrench.torque.x)
                elif field == "torque.y":
                    feat[i] = float(wrench_msg.wrench.torque.y)
                elif field == "torque.z":
                    feat[i] = float(wrench_msg.wrench.torque.z)
                continue

            if "_" not in col:
                raise KeyError(f"Unknown feature column: {col}")
            parts = col.split("_")
            if len(parts) < 2:
                raise KeyError(f"Unknown feature column: {col}")
            if parts[-1] == "lp" and len(parts) >= 3:
                value_name = f"{parts[-2]}_{parts[-1]}"
                joint_name = "_".join(parts[:-2])
            else:
                value_name = parts[-1]
                joint_name = "_".join(parts[:-1])
            value_key = JOINT_VALUE_MAP.get(value_name)
            if value_key is None:
                raise KeyError(f"Unknown joint field: {value_name}")
            if joint_msg is None:
                raise KeyError("JointState message missing but joint feature requested")
            idx = joint_map.get(joint_name)
            if idx is None:
                raise KeyError(f"Joint {joint_name} not in JointState message")

            if value_key == "position":
                feat[i] = float(joint_msg.position[idx])
            elif value_key == "velocity":
                feat[i] = float(joint_msg.velocity[idx])
            elif value_key == "effort":
                feat[i] = float(joint_msg.effort[idx])
        return feat

    def _add_sample(self) -> None:
        if self.needs_joint and self.latest_joint is None:
            return
        if self.needs_wrench and self.latest_wrench is None:
            return
        if self.args.use_wall_time:
            t = time.time()
        else:
            if self.needs_joint and self.needs_wrench and not self.args.ignore_sync:
                tj = _ros_time_to_float(self.latest_joint.header.stamp)
                tw = _ros_time_to_float(self.latest_wrench.header.stamp)
                if abs(tj - tw) > self.args.sync_tol:
                    if self.args.debug:
                        self._debug_status(
                            "skip: timestamp delta too large",
                            joint_stamp=tj,
                            wrench_stamp=tw,
                        )
                    return
                t = max(tj, tw)
            elif self.needs_joint and self.latest_joint is not None:
                t = _ros_time_to_float(self.latest_joint.header.stamp)
            elif self.needs_wrench and self.latest_wrench is not None:
                t = _ros_time_to_float(self.latest_wrench.header.stamp)
            else:
                t = time.time()

        if self.last_sample_time is not None and t < self.last_sample_time:
            if self.args.debug:
                self._debug_status(
                    "non-monotonic timestamps; consider --use-wall-time",
                    joint_stamp=_ros_time_to_float(self.latest_joint.header.stamp)
                    if self.latest_joint is not None
                    else None,
                    wrench_stamp=_ros_time_to_float(self.latest_wrench.header.stamp)
                    if self.latest_wrench is not None
                    else None,
                )
            if self.args.use_wall_time:
                t = time.time()
            else:
                return
        self.last_sample_time = t
        try:
            feat = self._build_feature_vector(self.latest_joint, self.latest_wrench)
        except KeyError as exc:
            self.get_logger().warn(str(exc))
            return
        if not np.isfinite(feat).all():
            bad_cols = [c for c, v in zip(self.feature_cols, feat) if not np.isfinite(v)]
            now = time.time()
            if (now - self.last_bad_time) > self.args.warn_every:
                self.get_logger().warn(
                    f"Non-finite input values detected in {bad_cols}. "
                    "Skipping sample; check sensor data or filtering."
                )
                self.last_bad_time = now
            return

        self.buffer.append((t, feat))
        self.last_data_time = time.time()
        cutoff = t - self.args.window_seconds
        while self.buffer and self.buffer[0][0] < cutoff:
            self.buffer.popleft()
        if self.args.debug:
            self._debug_status("sample_added")

    def _make_sequence(self) -> np.ndarray:
        if len(self.buffer) < 2:
            return None
        times = np.array([t for t, _ in self.buffer], dtype=np.float64)
        values = np.stack([v for _, v in self.buffer]).astype(np.float32)
        if times[-1] <= times[0]:
            if self.args.debug:
                self._debug_status("non-increasing time window")
            return None
        if not self.args.resample_to_seq:
            if len(values) < self.seq_len:
                return None
            start_idx = max(0, len(values) - self.seq_len)
            return values[start_idx : start_idx + self.seq_len]
        t_new = np.linspace(times[0], times[-1], self.seq_len, dtype=np.float64)
        seq = np.empty((self.seq_len, values.shape[1]), dtype=np.float32)
        for i in range(values.shape[1]):
            seq[:, i] = np.interp(t_new, times, values[:, i])
        return seq

    def _predict(self) -> Dict[str, float]:
        seq = self._make_sequence()
        if seq is None:
            return None
        if not np.isfinite(seq).all():
            now = time.time()
            if (now - self.last_bad_time) > self.args.warn_every:
                self.get_logger().warn(
                    "Non-finite values detected in input sequence. "
                    "Skipping prediction; check sensor data."
                )
                self.last_bad_time = now
            return None

        feat_mean = self.feat_mean
        feat_std = self.feat_std
        if not np.isfinite(feat_mean).all() or not np.isfinite(feat_std).all() or (feat_std <= 0).any():
            now = time.time()
            if (now - self.last_bad_time) > self.args.warn_every:
                self.get_logger().warn(
                    "Non-finite or non-positive normalization stats detected in checkpoint. "
                    "Using safe defaults; consider retraining."
                )
                self.last_bad_time = now
            feat_mean = np.nan_to_num(feat_mean, nan=0.0, posinf=0.0, neginf=0.0)
            feat_std = np.nan_to_num(feat_std, nan=1.0, posinf=1.0, neginf=1.0)
            feat_std = np.where(feat_std <= 0, 1.0, feat_std)

        seq = (seq - feat_mean) / feat_std
        if not np.isfinite(seq).all():
            now = time.time()
            if (now - self.last_bad_time) > self.args.warn_every:
                self.get_logger().warn(
                    "Non-finite values after normalization. "
                    "Skipping prediction; check checkpoint stats and sensor data."
                )
                self.last_bad_time = now
            return None
        xb = torch.from_numpy(seq[None, :, :]).to(self.device)
        with torch.no_grad():
            logits = self.model(xb)
        result = {}
        if "contact" in logits:
            prob = torch.sigmoid(logits["contact"]).cpu().numpy().item()
            result["contact_prob"] = prob
        if "label_a" in logits:
            result["label_a"] = float(logits["label_a"].cpu().numpy().item())
        if "label_b" in logits:
            result["label_b"] = float(logits["label_b"].cpu().numpy().item())
        return result

    def _update_pred_buffer(self, pred: Dict[str, float]) -> Dict[str, float]:
        now = time.time()
        self.pred_buffer.append((now, pred))
        cutoff = now - self.args.avg_seconds
        while self.pred_buffer and self.pred_buffer[0][0] < cutoff:
            self.pred_buffer.popleft()
        if not self.pred_buffer:
            return pred

        avg = {}
        keys = pred.keys()
        for key in keys:
            vals = [p[key] for _, p in self.pred_buffer if key in p]
            if vals:
                avg[key] = float(np.mean(vals))
        return avg

    def _on_timer(self) -> None:
        self._add_sample()
        pred = self._predict()
        if pred is None:
            now = time.time()
            if self.last_data_time is None or (now - self.last_data_time) > self.args.stale_seconds:
                if (now - self.last_warn_time) > self.args.warn_every:
                    self.get_logger().warn(
                        "No predictions yet. Check topic publishing, sync tolerance, or window size."
                    )
                    self.last_warn_time = now
            if self.args.debug:
                self._debug_status("no_prediction")
            return
        avg = self._update_pred_buffer(pred)

        msg = " | ".join([f"{k}={v:.3f}" for k, v in avg.items()])
        self.get_logger().info(msg)

    def _debug_status(
        self,
        reason: str,
        joint_stamp: float = None,
        wrench_stamp: float = None,
    ) -> None:
        now = time.time()
        if (now - self.last_debug_time) < self.args.debug_every:
            return
        self.last_debug_time = now

        joint_age = None
        wrench_age = None
        if self.latest_joint is not None:
            jt = _ros_time_to_float(self.latest_joint.header.stamp)
            joint_age = now - jt
        if self.latest_wrench is not None:
            wt = _ros_time_to_float(self.latest_wrench.header.stamp)
            wrench_age = now - wt

        msg = (
            f"debug: {reason} | buffer={len(self.buffer)} | "
            f"joint_age={joint_age} wrench_age={wrench_age} | "
            f"window={self.args.window_seconds} seq_len={self.seq_len} "
            f"sync_tol={self.args.sync_tol} ignore_sync={self.args.ignore_sync}"
        )
        if joint_stamp is not None and wrench_stamp is not None:
            msg += f" | dt={abs(joint_stamp - wrench_stamp):.4f}"
        self.get_logger().info(msg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Online predictor for contact model.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to .pth checkpoint.")
    parser.add_argument("--topic-joint", default=DEFAULT_TOPIC_JOINT, help="JointState topic.")
    parser.add_argument("--topic-wrench", default=DEFAULT_TOPIC_WRENCH, help="WrenchStamped topic.")
    parser.add_argument("--sync-tol", type=float, default=DEFAULT_SYNC_TOL, help="Max time diff between topics.")
    parser.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS, help="Sequence window.")
    parser.add_argument("--avg-seconds", type=float, default=DEFAULT_AVG_SECONDS, help="Prediction averaging window.")
    parser.add_argument("--timer-hz", type=float, default=DEFAULT_TIMER_HZ, help="Timer frequency.")
    parser.add_argument(
        "--stale-seconds",
        type=float,
        default=DEFAULT_STALE_SECONDS,
        help="Warn if no data arrives for this long.",
    )
    parser.add_argument(
        "--warn-every",
        type=float,
        default=DEFAULT_WARN_EVERY,
        help="Seconds between warning logs when data is missing.",
    )
    parser.add_argument(
        "--ignore-sync",
        action="store_true",
        default=DEFAULT_IGNORE_SYNC,
        help="Ignore timestamp sync between joint and wrench topics.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable periodic debug logs.",
    )
    parser.add_argument(
        "--debug-every",
        type=float,
        default=DEFAULT_DEBUG_EVERY,
        help="Seconds between debug logs.",
    )
    parser.add_argument(
        "--use-wall-time",
        action="store_true",
        default=DEFAULT_USE_WALL_TIME,
        help="Use local wall time instead of message stamps for the sequence window.",
    )
    parser.add_argument(
        "--use-train-window",
        action="store_true",
        default=DEFAULT_USE_TRAIN_WINDOW,
        help="Use train_window_seconds from the checkpoint if available.",
    )
    parser.add_argument(
        "--resample-to-seq",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RESAMPLE_TO_SEQ,
        help="Resample the window to seq_len (matches training).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    try:
        node = OnlinePredictor(args)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
