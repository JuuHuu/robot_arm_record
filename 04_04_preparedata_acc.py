#!/usr/bin/env python3
import numpy as np
import pandas as pd

# ========= CONFIG =========
JOINT_CSV_PATH = "/home/juu/Documents/robot_arm_record/exported/move_with_hammer/joint_states_filtered_all_joints.csv"

# Segment duration (seconds)
SEGMENT_DURATION = 0.2  # adjust as you like

# Output path
OUTPUT_CSV = "/home/juu/Documents/robot_arm_record/exported/move_with_hammer/ml_segments_acc_0.2.csv"

# Expected joints (output column order)
JOINTS = [
    "elbow_joint",
    "shoulder_lift_joint",
    "shoulder_pan_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
# ==========================


def main():
    # ---- 1. Load joint data ----
    df_j = pd.read_csv(JOINT_CSV_PATH)
    df_j = df_j.sort_values("time")

    # ---- 2. Pivot joints to wide format per timestamp ----
    wide = df_j.pivot_table(
        index="time",
        columns="joint_name",
        values=["position", "velocity", "effort", "effort_lp"],
        aggfunc="mean",
    )

    # Flatten MultiIndex column names
    wide.columns = [f"{lvl1}_{lvl2}" for lvl1, lvl2 in wide.columns]
    wide = wide.reset_index()  # 'time' becomes a column again

    # ---- 2.5. Compute per-joint accelerations from velocity ----
    # acceleration_j = d(velocity_j) / d(time)
    dt = wide["time"].diff()
    # Avoid division by zero
    dt = dt.replace(0, np.nan)

    for j in JOINTS:
        v_col = f"velocity_{j}"
        a_col = f"acceleration_{j}"
        if v_col in wide.columns:
            dv = wide[v_col].diff()
            wide[a_col] = (dv / dt).fillna(0.0)  # first sample / NaNs -> 0.0

    # ---- 3. Create segment ids using fixed time windows ----
    t0 = wide["time"].min()
    wide["segment_id"] = ((wide["time"] - t0) / SEGMENT_DURATION).astype(int)

    # ---- 4. Compute segment-level timing & means ----
    seg_time = wide.groupby("segment_id")["time"].agg(
        t_start="min",
        t_end="max",
        t_center="mean",
    )

    # This mean includes position_*, velocity_*, effort_*, effort_lp_*, acceleration_*
    seg_features = wide.groupby("segment_id").mean(numeric_only=True)

    # Merge timing and feature means
    seg = pd.concat([seg_time, seg_features], axis=1).reset_index()

    # ---- 5. Rename columns to desired naming format ----
    rename_map = {}
    for j in JOINTS:
        rename_map[f"position_{j}"] = f"{j}_position"
        rename_map[f"velocity_{j}"] = f"{j}_velocity"
        rename_map[f"effort_{j}"] = f"{j}_effort"
        rename_map[f"effort_lp_{j}"] = f"{j}_effort_lp"
        rename_map[f"acceleration_{j}"] = f"{j}_acceleration"

    seg = seg.rename(columns=rename_map)

    # ---- 6. Desired output column order ----
    desired_cols = [
        "segment_id",
        "t_start",
        "t_end",
        "t_center",
        # positions
        "elbow_joint_position",
        "shoulder_lift_joint_position",
        "shoulder_pan_joint_position",
        "wrist_1_joint_position",
        "wrist_2_joint_position",
        "wrist_3_joint_position",
        # velocities
        "elbow_joint_velocity",
        "shoulder_lift_joint_velocity",
        "shoulder_pan_joint_velocity",
        "wrist_1_joint_velocity",
        "wrist_2_joint_velocity",
        "wrist_3_joint_velocity",
        # accelerations
        "elbow_joint_acceleration",
        "shoulder_lift_joint_acceleration",
        "shoulder_pan_joint_acceleration",
        "wrist_1_joint_acceleration",
        "wrist_2_joint_acceleration",
        "wrist_3_joint_acceleration",
        # efforts
        "elbow_joint_effort",
        "shoulder_lift_joint_effort",
        "shoulder_pan_joint_effort",
        "wrist_1_joint_effort",
        "wrist_2_joint_effort",
        "wrist_3_joint_effort",
        # efforts low-pass
        "elbow_joint_effort_lp",
        "shoulder_lift_joint_effort_lp",
        "shoulder_pan_joint_effort_lp",
        "wrist_1_joint_effort_lp",
        "wrist_2_joint_effort_lp",
        "wrist_3_joint_effort_lp",
    ]

    # Keep only existing columns (in case some joints missing)
    final_cols = [c for c in desired_cols if c in seg.columns]
    seg = seg[final_cols]

    # ---- 7. Save ----
    seg.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(seg)} segments to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
