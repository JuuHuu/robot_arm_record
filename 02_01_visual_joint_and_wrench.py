import pandas as pd
import matplotlib.pyplot as plt


# ========= CONFIG =========
JOINT_CSV_PATH = "/home/juu/Documents/robot_arm_record/exported/apply_force_B/joint_states_filtered_all_joints.csv"
WRENCH_CSV_PATH = "/home/juu/Documents/robot_arm_record/exported/apply_force_B/wrench_filtered.csv"

# Plot only these joints (None = all joints)
PLOT_JOINTS = None

# Joint columns to plot (None = all available columns except time/joint_name)
JOINT_COLUMNS = ["effort_lp"] # e.g. ["position", "velocity", "effort", "effort_lp"]

# Wrench columns to plot (None = all available columns except time)
WRENCH_COLUMNS = ["fx", "fy", "fz", "tx", "ty", "tz"] # e.g. ["fx", "fy", "fz", "tx", "ty", "tz"]
# ==========================


def _select_joints(df, joints):
    if joints is None:
        return df
    return df[df["joint_name"].isin(joints)]


def plot_joint_style(df, title_prefix):
    df = df.sort_values("time")
    df = _select_joints(df, PLOT_JOINTS)
    value_cols = [c for c in df.columns if c not in ("time", "joint_name")]
    if JOINT_COLUMNS is not None:
        value_cols = [c for c in value_cols if c in JOINT_COLUMNS]
    if not value_cols:
        print(f"[WARN] No value columns to plot for {title_prefix}.")
        return

    for col in value_cols:
        for name in df["joint_name"].unique():
            df_j = df[df["joint_name"] == name]
            plt.plot(df_j["time"], df_j[col], label=f"{name}_{col}")


def plot_wrench_style(df, title_prefix):
    df = df.sort_values("time")
    value_cols = [c for c in df.columns if c != "time"]
    if WRENCH_COLUMNS is not None:
        value_cols = [c for c in value_cols if c in WRENCH_COLUMNS]
    if not value_cols:
        print(f"[WARN] No value columns to plot for {title_prefix}.")
        return

    for col in value_cols:
        plt.plot(df["time"], df[col], label=col)


def plot_csv(path, title_prefix):
    print(f"Loading CSV from: {path}")
    df = pd.read_csv(path)
    if "time" not in df.columns:
        raise RuntimeError(f"CSV '{path}' does not contain 'time' column.")

    if "joint_name" in df.columns:
        plot_joint_style(df, title_prefix)
    else:
        plot_wrench_style(df, title_prefix)


def main():
    plt.figure(figsize=(16, 8))
    plot_csv(JOINT_CSV_PATH, "Joint states")
    plot_csv(WRENCH_CSV_PATH, "Wrench data")
    plt.xlabel("Time (s)")
    plt.ylabel("Value")
    plt.title("Joint states + Wrench")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
