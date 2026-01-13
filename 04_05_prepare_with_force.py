#!/usr/bin/env python3
import numpy as np
import pandas as pd


# ========= CONFIG =========
# Each folder must contain JOINT_FILENAME and WRENCH_FILENAME.
FOLDER_LABELS = [
    {
        "path": "/home/juu/Documents/robot_arm_record/exported/apply_force_H",
        "label": "H",
    },
    # {"path": "/home/juu/Documents/robot_arm_record/exported/apply_force_B", "label": "B"},
]

JOINT_FILENAME = "joint_states_filtered_all_joints.csv"
WRENCH_FILENAME = "wrench_filtered.csv"

# Use this wrench column to detect force segments
FY_COLUMN = "fy_lp"  # or "fy_lp" if you want filtered

# Baseline is the mean of the first N seconds of FY_COLUMN
BASELINE_SECONDS = 2.0

# Force threshold above baseline
FY_THRESHOLD = 0.25

# Minimal duration to accept a force segment
MIN_SEGMENT_DURATION = 0.1  # seconds

# Wrench columns to aggregate for training data (None = all except time)
WRENCH_COLUMNS = None  # e.g. ["fx", "fy", "fz", "tx", "ty", "tz", "fy_lp"]

# Outputs
OUTPUT_LONG = "/home/juu/Documents/robot_arm_record/exported/apply_force_H/force_segments_long.csv"
OUTPUT_WIDE = "/home/juu/Documents/robot_arm_record/exported/apply_force_H/force_segments_wide.csv"
# ==========================


def find_segments(time, condition_mask, min_duration):
    """
    condition_mask: boolean array where True = within force segment.
    Returns list of (t_start, t_end) where mask is True continuously.
    """
    segments = []
    in_seg = False
    start_idx = None

    for i, cond in enumerate(condition_mask):
        if cond and not in_seg:
            in_seg = True
            start_idx = i

        if in_seg and (not cond or i == len(condition_mask) - 1):
            end_idx = i if cond else i - 1
            t_start = time[start_idx]
            t_end = time[end_idx]
            if t_end - t_start >= min_duration:
                segments.append((t_start, t_end))
            in_seg = False

    return segments


def load_joint_df(folder_path):
    joint_path = f"{folder_path}/{JOINT_FILENAME}"
    print(f"Loading joint CSV from: {joint_path}")
    df = pd.read_csv(joint_path)
    if "time" not in df.columns or "joint_name" not in df.columns:
        raise RuntimeError(f"Joint CSV missing required columns: {joint_path}")
    return df.sort_values("time").reset_index(drop=True)


def load_wrench_df(folder_path):
    wrench_path = f"{folder_path}/{WRENCH_FILENAME}"
    print(f"Loading wrench CSV from: {wrench_path}")
    df = pd.read_csv(wrench_path)
    if "time" not in df.columns or FY_COLUMN not in df.columns:
        raise RuntimeError(f"Wrench CSV missing required columns: {wrench_path}")
    return df.sort_values("time").reset_index(drop=True)


def main():
    all_rows_long = []
    seg_wrench_rows = []
    segment_id = 0

    for item in FOLDER_LABELS:
        folder_path = item["path"]
        label = item["label"]

        joint_df = load_joint_df(folder_path)
        wrench_df = load_wrench_df(folder_path)

        w_time = wrench_df["time"].values
        fy = wrench_df[FY_COLUMN].values.astype(np.float32)

        t0 = w_time[0]
        baseline_mask = w_time <= (t0 + BASELINE_SECONDS)
        if not baseline_mask.any():
            raise RuntimeError(
                f"No samples found in first {BASELINE_SECONDS} seconds for {folder_path}"
            )
        baseline = fy[baseline_mask].mean()

        force_mask = fy > (baseline + FY_THRESHOLD)
        segments = find_segments(w_time, force_mask, MIN_SEGMENT_DURATION)
        print(
            f"{folder_path}: baseline={baseline:.4f}, segments={len(segments)}"
        )

        if not segments:
            continue

        numeric_cols = joint_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != "time"]

        if WRENCH_COLUMNS is None:
            wrench_cols = [c for c in wrench_df.columns if c != "time"]
        else:
            wrench_cols = [c for c in WRENCH_COLUMNS if c in wrench_df.columns]
        if not wrench_cols:
            raise RuntimeError("No wrench columns selected for aggregation.")

        for t_start, t_end in segments:
            seg_joint = joint_df[
                (joint_df["time"] >= t_start) & (joint_df["time"] <= t_end)
            ]
            if seg_joint.empty:
                continue

            seg_wrench = wrench_df[
                (wrench_df["time"] >= t_start) & (wrench_df["time"] <= t_end)
            ]

            avg = (
                seg_joint.groupby("joint_name")[numeric_cols]
                .mean()
                .reset_index()
            )
            avg["segment_id"] = segment_id
            avg["label"] = label
            avg["source_folder"] = folder_path
            avg["t_start"] = t_start
            avg["t_end"] = t_end
            avg["t_center"] = 0.5 * (t_start + t_end)
            all_rows_long.append(avg)

            if not seg_wrench.empty:
                w_mean = seg_wrench[wrench_cols].mean(axis=0).values.astype(np.float32)
            else:
                w_mean = np.full(len(wrench_cols), np.nan, dtype=np.float32)

            seg_row = {
                "segment_id": segment_id,
                "label": label,
                "source_folder": folder_path,
            }
            for col, val in zip(wrench_cols, w_mean):
                seg_row[f"ee_{col}_mean"] = val
            seg_wrench_rows.append(seg_row)

            segment_id += 1

    if not all_rows_long:
        print("No segments found across all folders.")
        return

    long_df = pd.concat(all_rows_long, ignore_index=True)
    long_df.to_csv(OUTPUT_LONG, index=False)
    print(f"Saved LONG format to: {OUTPUT_LONG}")

    numeric_cols = long_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [
        c for c in numeric_cols
        if c not in ("time", "segment_id", "t_start", "t_end", "t_center")
    ]
    wide = long_df.set_index(["segment_id", "joint_name"])[feature_cols].unstack("joint_name")
    wide.columns = [f"{joint}_{feat}" for feat, joint in wide.columns.to_flat_index()]

    meta = (
        long_df[["segment_id", "label", "source_folder", "t_start", "t_end", "t_center"]]
        .drop_duplicates(subset=["segment_id"])
        .set_index("segment_id")
    )
    wide = meta.join(wide).reset_index()

    seg_wrench_df = pd.DataFrame(seg_wrench_rows)
    wide = wide.merge(seg_wrench_df, on=["segment_id", "label", "source_folder"], how="left")

    wide.to_csv(OUTPUT_WIDE, index=False)
    print(f"Saved WIDE format to: {OUTPUT_WIDE}")


if __name__ == "__main__":
    main()
