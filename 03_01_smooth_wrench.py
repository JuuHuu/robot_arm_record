import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, detrend


# ========= CONFIG =========
CSV_PATH = "/home/juu/Documents/robot_arm_record/exported/apply_force_04_00/wrench.csv"
OUTPUT_PATH = "/home/juu/Documents/robot_arm_record/exported/apply_force_04_00/filtered_wrench.csv"

USE_BUTTERWORTH = True
USE_DETREND = False

# Plot one channel to check result (set to None to disable)
PLOT_CHANNEL = "fx"

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

    required_cols = ["time", "fx", "fy", "fz", "tx", "ty", "tz"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"CSV is missing required columns: {missing}")

    time = df["time"].values
    dt = np.diff(time)
    dt_median = np.median(dt)
    if dt_median <= 0:
        raise RuntimeError("Non-positive median dt; check time column ordering.")

    fs = 1.0 / dt_median
    print(f"Estimated fs ≈ {fs:.3f} Hz (median dt = {dt_median:.6f} s)")

    if USE_BUTTERWORTH:
        if fs <= 2 * CUTOFF_HZ:
            raise RuntimeError(
                f"Sampling rate too low for cutoff={CUTOFF_HZ} Hz (fs={fs:.3f} Hz)."
            )

        for col in ["fx", "fy", "fz", "tx", "ty", "tz"]:
            series = df[col].values
            filtered = butter_lowpass_filter(
                series,
                cutoff=CUTOFF_HZ,
                fs=fs,
                order=BUTTER_ORDER,
            )
            df[f"{col}_lp"] = filtered

    if USE_DETREND:
        source_suffix = "_lp" if USE_BUTTERWORTH else ""
        for col in ["fx", "fy", "fz", "tx", "ty", "tz"]:
            source_col = f"{col}{source_suffix}"
            if source_col not in df.columns:
                print(f"[WARN] Missing source column '{source_col}', skipping detrend.")
                continue
            df[f"{source_col}_dt"] = detrend(df[source_col].values, type="linear")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Filtered data saved to: {OUTPUT_PATH}")

    if PLOT_CHANNEL is not None:
        col_lp = f"{PLOT_CHANNEL}_lp"
        col_dt = f"{col_lp}_dt" if USE_DETREND else None
        if col_lp not in df.columns:
            print(f"[WARN] No filtered channel '{col_lp}', skipping plot.")
            return

        plt.figure(figsize=(12, 6))
        plt.plot(df["time"], df[PLOT_CHANNEL], label=f"raw {PLOT_CHANNEL}", alpha=0.4)
        plt.plot(
            df["time"],
            df[col_lp],
            label=f"Butterworth LP (cutoff={CUTOFF_HZ} Hz, order={BUTTER_ORDER})",
        )
        if USE_DETREND and col_dt in df.columns:
            plt.plot(
                df["time"],
                df[col_dt],
                label="detrended (linear)",
            )
        plt.xlabel("Time [s]")
        plt.ylabel(PLOT_CHANNEL)
        plt.title(f"Wrench filtering – {PLOT_CHANNEL}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
