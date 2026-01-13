#!/usr/bin/env python3
import numpy as np
import pandas as pd

# ========= CONFIG =========
CSV_PATH = "/home/juu/Documents/robot_arm_record/exported/move_with_hammer/joint_states_filtered_all_joints.csv"
WRENCH_CSV_PATH = "/home/juu/Documents/robot_arm_record/exported/move_with_hammer/wrench.csv"

# A joint is considered static when |velocity| < VEL_THRESH
VEL_THRESH = 0.005      # adjust if needed

# Minimal duration to accept a static segment
MIN_STATIC_DURATION = 0.5  # seconds

# Output
OUTPUT_LONG = "/home/juu/Documents/robot_arm_record/exported/move_with_hammer/static_segments_long.csv"
OUTPUT_WIDE = "/home/juu/Documents/robot_arm_record/exported/move_with_hammer/static_segments_wide.csv"
# ==========================


def find_static_segments(time, condition_mask, min_duration):
    """
    condition_mask: boolean array where True = static for that timestamp.
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


def main():
    print(f"Loading joint CSV from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    if "joint_name" not in df.columns or "velocity" not in df.columns:
        raise RuntimeError("CSV must contain 'joint_name' and 'velocity'.")

    # ---- Step 1: pivot table into time-major format ----
    # Make a table: one row per time stamp, one column per joint's velocity
    df_sorted = df.sort_values("time")
    pivot_vel = df_sorted.pivot(index="time", columns="joint_name", values="velocity")

    time = pivot_vel.index.values

    # Determine which timestamps are fully static across all joints
    static_mask = (pivot_vel.abs() < VEL_THRESH).all(axis=1).values

    print(f"Total timestamps: {len(time)}")
    print(f"Timestamps fully static: {static_mask.sum()}")

    # ---- Step 2: find static segments ----
    segments = find_static_segments(time, static_mask, MIN_STATIC_DURATION)

    if not segments:
        print("No static segments found.")
        return

    print("Static segments found:")
    for i, (ts, te) in enumerate(segments):
        print(f"  Segment {i}: {ts:.3f} -> {te:.3f}, duration {te-ts:.3f}s")

    # ---- Step 2.5: load wrench data and compute per-segment averages ----
    print(f"Loading wrench CSV from: {WRENCH_CSV_PATH}")
    wrench_df = pd.read_csv(WRENCH_CSV_PATH)

    # Expect: time, fx, fy, fz, tx, ty, tz
    required_wrench_cols = ["time", "fx", "fy", "fz", "tx", "ty", "tz"]
    missing = [c for c in required_wrench_cols if c not in wrench_df.columns]
    if missing:
        raise RuntimeError(f"wrench.csv missing columns: {missing}")

    wrench_df = wrench_df.sort_values("time").reset_index(drop=True)
    w_time = wrench_df["time"].values
    wrench_signals = ["fx", "fy", "fz", "tx", "ty", "tz"]
    w_values = wrench_df[wrench_signals].values.astype(np.float32)

    seg_wrench_rows = []
    print("Computing segment-wise wrench averages...")
    for seg_id, (t_start, t_end) in enumerate(segments):
        mask = (w_time >= t_start) & (w_time <= t_end)
        if mask.any():
            mean_vals = w_values[mask].mean(axis=0)
        else:
            # No wrench samples in this interval -> NaNs
            mean_vals = np.full(len(wrench_signals), np.nan, dtype=np.float32)

        seg_wrench_rows.append({
            "segment_id": seg_id,
            "ee_fx_mean": mean_vals[0],
            "ee_fy_mean": mean_vals[1],
            "ee_fz_mean": mean_vals[2],
            "ee_tx_mean": mean_vals[3],
            "ee_ty_mean": mean_vals[4],
            "ee_tz_mean": mean_vals[5],
        })

    seg_wrench_df = pd.DataFrame(seg_wrench_rows)
    print(f"Per-segment wrench table shape: {seg_wrench_df.shape}")

    # ---- Step 3: average joint info for each segment ----
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "time"]  # exclude time

    all_rows_long = []

    for seg_id, (t_start, t_end) in enumerate(segments):
        seg_df = df[(df["time"] >= t_start) & (df["time"] <= t_end)]
        if seg_df.empty:
            continue

        avg = (
            seg_df.groupby("joint_name")[numeric_cols]
            .mean()
            .reset_index()
        )
        avg["segment_id"] = seg_id
        avg["t_start"] = t_start
        avg["t_end"] = t_end
        avg["t_center"] = 0.5 * (t_start + t_end)

        all_rows_long.append(avg)

    long_df = pd.concat(all_rows_long, ignore_index=True)
    long_df.to_csv(OUTPUT_LONG, index=False)
    print(f"Saved LONG format to: {OUTPUT_LONG}")

    # ---- Step 4: create WIDE format ----
    wide = long_df.set_index(["segment_id", "joint_name"])[numeric_cols].unstack("joint_name")
    wide.columns = [f"{joint}_{feat}" for feat, joint in wide.columns.to_flat_index()]

    meta = (
        long_df[["segment_id", "t_start", "t_end", "t_center"]]
        .drop_duplicates(subset=["segment_id"])
        .set_index("segment_id")
    )
    wide = meta.join(wide).reset_index()

    # ---- Step 4.5: merge per-segment wrench means into WIDE ----
    wide = wide.merge(seg_wrench_df, on="segment_id", how="left")

    wide.to_csv(OUTPUT_WIDE, index=False)
    print(f"Saved WIDE format to: {OUTPUT_WIDE}")


if __name__ == "__main__":
    main()
