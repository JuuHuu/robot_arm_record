import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


# ========= CONFIG =========
CSV_PATH = "/home/juu/Documents/robot_arm_record/exported/apply_force_04_00/joint_states.csv"  # input CSV
OUTPUT_PATH = "/home/juu/Documents/robot_arm_record/exported/apply_force_04_00/fillted_joint_states.csv"


USE_MOVING_AVG = False
USE_EMA = False
USE_BUTTERWORTH = True

# Plot one joint to check result (set to None to disable)
PLOT_JOINT_NAME = "shoulder_lift_joint"

# Moving average parameters
MOVING_AVG_WINDOW = 40  # samples

# EMA parameters
EMA_ALPHA = 0.1  # 0 < alpha <= 1, smaller = smoother

# Butterworth low-pass parameters
BUTTER_ORDER = 4
CUTOFF_HZ = 3.0  # cutoff frequency in Hz (tune this!)
# ==========================


def butter_lowpass_filter(data, cutoff, fs, order=4):
    """
    Apply a zero-phase Butterworth low-pass filter using filtfilt.
    - data: 1D numpy array
    - cutoff: cutoff frequency in Hz
    - fs: sampling frequency in Hz
    - order: filter order
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    y = filtfilt(b, a, data)
    return y


def main():
    print(f"Loading CSV from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    # Basic checks
    if "joint_name" not in df.columns:
        raise RuntimeError("CSV does not contain 'joint_name' column.")
    if "effort" not in df.columns:
        raise RuntimeError("CSV does not contain 'effort' column.")
    if "time" not in df.columns:
        raise RuntimeError("CSV does not contain 'time' column.")

    # Prepare columns for filtered data (same length as df)
    if USE_MOVING_AVG:
        df["effort_ma"] = np.nan
    if USE_EMA:
        df["effort_ema"] = np.nan
    if USE_BUTTERWORTH:
        df["effort_lp"] = np.nan

    # Process each joint separately
    for joint_name, group in df.groupby("joint_name"):
        group = group.sort_values("time")
        idx = group.index

        # Drop NaNs in effort for this group
        group = group.dropna(subset=["effort"])
        if len(group) < 3:
            print(f"[WARN] Joint '{joint_name}' has too few samples ({len(group)}), skipping.")
            continue

        time = group["time"].values
        effort = group["effort"].values

        # Estimate sampling frequency
        dt = np.diff(time)
        dt_median = np.median(dt)
        if dt_median <= 0:
            print(f"[WARN] Non-positive dt for joint '{joint_name}', skipping.")
            continue

        fs = 1.0 / dt_median
        print(f"Joint '{joint_name}': fs ≈ {fs:.3f} Hz (median dt = {dt_median:.6f} s)")

        # Moving average
        if USE_MOVING_AVG:
            ma = (
                group["effort"]
                .rolling(window=MOVING_AVG_WINDOW, center=True)
                .mean()
                .values
            )
            df.loc[group.index, "effort_ma"] = ma

        # Exponential moving average
        if USE_EMA:
            ema = group["effort"].ewm(alpha=EMA_ALPHA).mean().values
            df.loc[group.index, "effort_ema"] = ema

        # Butterworth low-pass
        if USE_BUTTERWORTH:
            if fs <= 2 * CUTOFF_HZ:
                print(
                    f"[WARN] Joint '{joint_name}': fs={fs:.3f} Hz is too low for cutoff={CUTOFF_HZ} Hz, "
                    f"skipping Butterworth."
                )
            else:
                filtered = butter_lowpass_filter(
                    effort,
                    cutoff=CUTOFF_HZ,
                    fs=fs,
                    order=BUTTER_ORDER,
                )
                df.loc[group.index, "effort_lp"] = filtered

    # Save full filtered dataframe (all joints)
    df = df.sort_values(["time", "joint_name"]).reset_index(drop=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Filtered data for ALL joints saved to: {OUTPUT_PATH}")

    # Optional: plot one joint to visually check filtering
    if PLOT_JOINT_NAME is not None:
        df_plot = df[df["joint_name"] == PLOT_JOINT_NAME].copy()
        if df_plot.empty:
            print(f"[WARN] No data for PLOT_JOINT_NAME='{PLOT_JOINT_NAME}', skipping plot.")
            return

        plt.figure(figsize=(12, 6))
        plt.plot(df_plot["time"], df_plot["effort"], label="raw effort", alpha=0.4)

        if USE_MOVING_AVG and "effort_ma" in df_plot.columns:
            plt.plot(df_plot["time"], df_plot["effort_ma"],
                     label=f"moving avg (window={MOVING_AVG_WINDOW})")

        if USE_EMA and "effort_ema" in df_plot.columns:
            plt.plot(df_plot["time"], df_plot["effort_ema"],
                     label=f"EMA (alpha={EMA_ALPHA})")

        if USE_BUTTERWORTH and "effort_lp" in df_plot.columns:
            plt.plot(df_plot["time"], df_plot["effort_lp"],
                     label=f"Butterworth LP (cutoff={CUTOFF_HZ} Hz, order={BUTTER_ORDER})")

        plt.xlabel("Time [s]")
        plt.ylabel("Effort")
        plt.title(f"Effort filtering – {PLOT_JOINT_NAME}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
