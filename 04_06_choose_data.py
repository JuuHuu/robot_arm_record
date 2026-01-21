import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector, TextBox, Button


# ========= CONFIG =========
DEFAULT_FOLDER = "/home/juu/Documents/robot_arm_record/auto_data/autosave_04_180" 
JOINT_CSV_NAME = "fillted_joint_states.csv"
WRENCH_CSV_NAME = "filtered_wrench.csv"

# Plot only these joints (None = all joints)
PLOT_JOINTS = None

# Joint columns to plot (None = all available columns except time/joint_name)
JOINT_COLUMNS = ["effort_lp"]

# Wrench columns to plot (None = all available columns except time)
WRENCH_COLUMNS = ["fx_lp", "fy_lp", "fz_lp", "tx_lp", "ty_lp", "tz_lp"]

# Use to speed up large files (1 = no downsample)
DOWNSAMPLE_STEP = 1

# Output file name inside the folder
ANNOTATION_CSV_NAME = "selected_segments.csv"
# ==========================


def _select_joints(df, joints):
    if joints is None:
        return df
    return df[df["joint_name"].isin(joints)]


def _get_value_cols(df, exclude, desired):
    value_cols = [c for c in df.columns if c not in exclude]
    if desired is not None:
        value_cols = [c for c in value_cols if c in desired]
    return value_cols


def plot_joint_style(ax, df, title_prefix, time_col="time"):
    df = df.sort_values(time_col)
    df = _select_joints(df, PLOT_JOINTS)
    if DOWNSAMPLE_STEP > 1:
        df = df.iloc[::DOWNSAMPLE_STEP]
    value_cols = _get_value_cols(df, ("time", "time_rel", "joint_name"), JOINT_COLUMNS)
    if not value_cols:
        ax.set_title(f"{title_prefix} (no joint columns to plot)")
        return

    for col in value_cols:
        for name in df["joint_name"].unique():
            df_j = df[df["joint_name"] == name]
            ax.plot(df_j[time_col], df_j[col], label=f"{name}_{col}", alpha=0.8)

    ax.set_title(title_prefix)


def plot_wrench_style(ax, df, title_prefix, time_col="time"):
    df = df.sort_values(time_col)
    if DOWNSAMPLE_STEP > 1:
        df = df.iloc[::DOWNSAMPLE_STEP]
    value_cols = _get_value_cols(df, ("time", "time_rel"), WRENCH_COLUMNS)
    if not value_cols:
        ax.set_title(f"{title_prefix} (no wrench columns to plot)")
        return

    for col in value_cols:
        ax.plot(df[time_col], df[col], label=col, alpha=0.8)
    ax.set_title(title_prefix)


def parse_label(text, fallback):
    if text is None:
        return fallback
    text = text.strip()
    if not text:
        return fallback
    if text.lower() == "n":
        return (-1, -1)
    parts = text.split(",")
    if len(parts) != 2:
        return fallback
    try:
        return (int(parts[0].strip()), int(parts[1].strip()))
    except ValueError:
        return fallback


def format_label(label):
    if label == (-1, -1):
        return "-1,-1"
    return f"{label[0]},{label[1]}"


class AnnotationManager:
    def __init__(self, ax_list, time_base):
        self.ax_list = ax_list
        self.time_base = time_base
        self.annotations = []
        self.patches = []
        self.last_label = (-1, -1)

    def _to_abs(self, t):
        return t + self.time_base

    def _to_rel(self, t):
        return t - self.time_base

    def add_annotation(self, start, end, label, is_abs=False):
        if is_abs:
            start_rel, end_rel = sorted((self._to_rel(start), self._to_rel(end)))
            start_abs, end_abs = sorted((start, end))
        else:
            start_rel, end_rel = sorted((start, end))
            start_abs, end_abs = sorted((self._to_abs(start), self._to_abs(end)))
        color = "tab:green" if label != (-1, -1) else "tab:gray"
        patch_refs = []
        for ax in self.ax_list:
            patch = ax.axvspan(start_rel, end_rel, color=color, alpha=0.2)
            patch_refs.append(patch)
        self.annotations.append(
            {
                "start_time": start_abs,
                "end_time": end_abs,
                "label_a": label[0],
                "label_b": label[1],
                "contact": 0 if label == (-1, -1) else 1,
            }
        )
        self.patches.append(patch_refs)
        self.last_label = label
        print(f"[ADD] {start_abs:.3f} - {end_abs:.3f} label={label}")

    def undo(self):
        if not self.annotations:
            print("[UNDO] No annotations to remove.")
            return
        self.annotations.pop()
        patches = self.patches.pop()
        for p in patches:
            p.remove()
        print("[UNDO] Removed last annotation.")

    def load_existing(self, annotations):
        if not annotations:
            return
        max_time = max(item["start_time"] for item in annotations)
        assume_abs = max_time > 1e6
        for item in annotations:
            label = (int(item["label_a"]), int(item["label_b"]))
            self.add_annotation(
                float(item["start_time"]),
                float(item["end_time"]),
                label,
                is_abs=assume_abs,
            )

    def as_dataframe(self):
        if not self.annotations:
            return pd.DataFrame(
                columns=["start_time", "end_time", "label_a", "label_b", "contact"]
            )
        df = pd.DataFrame(self.annotations)
        return df.sort_values("start_time")


def load_annotations(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    required = {"start_time", "end_time", "label_a", "label_b"}
    if not required.issubset(df.columns):
        print(f"[WARN] Existing annotation file missing required columns: {path}")
        return None
    if "contact" not in df.columns:
        df["contact"] = (
            ~((df["label_a"] == -1) & (df["label_b"] == -1))
        ).astype(int)
    return df


def save_annotations(path, annotations):
    df = annotations.as_dataframe()
    df.to_csv(path, index=False)
    print(f"[SAVE] {len(df)} segments -> {path}")


def main():
    parser = argparse.ArgumentParser(description="Choose and annotate data segments.")
    parser.add_argument(
        "--folder",
        default=DEFAULT_FOLDER,
        help="Folder containing joint and wrench CSV files.",
    )
    args = parser.parse_args()

    folder = args.folder
    joint_path = os.path.join(folder, JOINT_CSV_NAME)
    wrench_path = os.path.join(folder, WRENCH_CSV_NAME)
    output_path = os.path.join(folder, ANNOTATION_CSV_NAME)

    if not os.path.exists(joint_path):
        raise FileNotFoundError(f"Joint CSV not found: {joint_path}")
    if not os.path.exists(wrench_path):
        raise FileNotFoundError(f"Wrench CSV not found: {wrench_path}")

    print(f"Loading joint CSV: {joint_path}")
    joint_df = pd.read_csv(joint_path)
    print(f"Loading wrench CSV: {wrench_path}")
    wrench_df = pd.read_csv(wrench_path)

    if "time" not in joint_df.columns or "time" not in wrench_df.columns:
        raise RuntimeError("Both CSV files must include a 'time' column.")

    time_base = min(joint_df["time"].min(), wrench_df["time"].min())
    joint_df["time_rel"] = joint_df["time"] - time_base
    wrench_df["time_rel"] = wrench_df["time"] - time_base

    fig, (ax_wrench, ax_joint) = plt.subplots(
        2, 1, figsize=(16, 8), sharex=True
    )
    plot_wrench_style(ax_wrench, wrench_df, "Wrench", time_col="time_rel")
    if "joint_name" in joint_df.columns:
        plot_joint_style(ax_joint, joint_df, "Joint States", time_col="time_rel")
    else:
        plot_wrench_style(
            ax_joint,
            joint_df,
            "Joint States (no joint_name column)",
            time_col="time_rel",
        )

    ax_joint.set_xlabel("Time (s, offset from start)")
    ax_wrench.set_ylabel("Wrench")
    ax_joint.set_ylabel("Joint")
    ax_wrench.legend(loc="upper right", fontsize=8)
    ax_joint.legend(loc="upper right", fontsize=8)
    for ax in (ax_wrench, ax_joint):
        ax.grid(True, alpha=0.3)

    manager = AnnotationManager([ax_wrench, ax_joint], time_base=time_base)

    fig.subplots_adjust(bottom=0.22)
    label_a_ax = fig.add_axes([0.1, 0.08, 0.1, 0.05])
    label_a_box = TextBox(label_a_ax, "Label A", initial="-1")

    label_b_ax = fig.add_axes([0.22, 0.08, 0.1, 0.05])
    label_b_box = TextBox(label_b_ax, "Label B", initial="-1")

    notouch_ax = fig.add_axes([0.37, 0.08, 0.12, 0.05])
    notouch_btn = Button(notouch_ax, "Set -1,-1")

    save_ax = fig.add_axes([0.51, 0.08, 0.08, 0.05])
    save_btn = Button(save_ax, "Save")

    undo_ax = fig.add_axes([0.61, 0.08, 0.08, 0.05])
    undo_btn = Button(undo_ax, "Undo")

    list_ax = fig.add_axes([0.72, 0.02, 0.26, 0.16])
    list_ax.axis("off")
    list_text = list_ax.text(0, 1, "", va="top", fontsize=9)
    existing = load_annotations(output_path)
    if existing is not None and not existing.empty:
        print(f"[INFO] Loading existing annotations from {output_path}")
        manager.load_existing(existing.to_dict(orient="records"))

    def update_segment_list():
        items = manager.annotations
        lines = [f"Segments: {len(items)}"]
        start_idx = max(len(items) - len(items[-6:]) + 1, 1)
        for idx, item in enumerate(items[-6:], start=start_idx):
            start_rel = item["start_time"] - time_base
            end_rel = item["end_time"] - time_base
            label = (int(item["label_a"]), int(item["label_b"]))
            lines.append(
                f"{idx:02d}  {start_rel:7.2f}-{end_rel:7.2f}  {format_label(label)}"
            )
        list_text.set_text("\n".join(lines))

    label_updating = {"active": False}

    def get_label_from_boxes():
        try:
            a = int(label_a_box.text.strip())
            b = int(label_b_box.text.strip())
            return (a, b)
        except ValueError:
            return manager.last_label

    def set_label_boxes(label):
        if label_updating["active"]:
            return
        label_updating["active"] = True
        label_a_box.set_val(str(label[0]))
        label_b_box.set_val(str(label[1]))
        label_updating["active"] = False

    def onselect(xmin, xmax):
        label = get_label_from_boxes()
        set_label_boxes(label)
        manager.add_annotation(xmin, xmax, label)
        update_segment_list()
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "z":
            manager.undo()
            update_segment_list()
            fig.canvas.draw_idle()
        elif event.key == "s":
            save_annotations(output_path, manager)
        elif event.key == "q":
            save_annotations(output_path, manager)
            plt.close(fig)

    def on_label_submit(text):
        label = get_label_from_boxes()
        set_label_boxes(label)

    def set_notouch(event):
        set_label_boxes((-1, -1))

    def on_save_click(event):
        save_annotations(output_path, manager)

    def on_undo_click(event):
        manager.undo()
        update_segment_list()
        fig.canvas.draw_idle()

    label_a_box.on_submit(on_label_submit)
    label_b_box.on_submit(on_label_submit)
    notouch_btn.on_clicked(set_notouch)
    save_btn.on_clicked(on_save_click)
    undo_btn.on_clicked(on_undo_click)

    span_style = dict(alpha=0.3, facecolor="tab:blue")
    span_kwargs = dict(
        useblit=True,
        button=1,
        minspan=0.01,
    )
    try:
        span_wrench = SpanSelector(
            ax_wrench,
            onselect,
            "horizontal",
            rectprops=span_style,
            **span_kwargs,
        )
        span_joint = SpanSelector(
            ax_joint,
            onselect,
            "horizontal",
            rectprops=span_style,
            **span_kwargs,
        )
    except TypeError:
        span_wrench = SpanSelector(
            ax_wrench,
            onselect,
            "horizontal",
            props=span_style,
            **span_kwargs,
        )
        span_joint = SpanSelector(
            ax_joint,
            onselect,
            "horizontal",
            props=span_style,
            **span_kwargs,
        )

    def sync_selector_state():
        toolbar = getattr(fig.canvas, "toolbar", None)
        mode = getattr(toolbar, "mode", "") if toolbar else ""
        active = not bool(mode)
        for selector in (span_wrench, span_joint):
            if hasattr(selector, "set_active"):
                selector.set_active(active)

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("button_press_event", lambda event: sync_selector_state())
    fig.suptitle(
        "Left-drag to select when pan/zoom is off. Keys: z=undo, s=save, q=save+quit",
        fontsize=11,
    )
    update_segment_list()
    plt.show()


if __name__ == "__main__":
    main()
