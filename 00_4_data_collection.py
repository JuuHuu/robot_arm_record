import itertools
import subprocess
import sys
import time
from pathlib import Path


# ========= CONFIG =========
# First variable: default 00..17 step 1
VAR1_START = 0
VAR1_STOP = 17
VAR1_STEP = 1
VAR1_SINGLE = None  # set to an int to use a single value only

# Second variable: default 00..360 step 30
VAR2_START = 0
VAR2_STOP = 360
VAR2_STEP = 30
VAR2_SINGLE = 000  # set to an int to use a single value only

PREP_SECONDS = 3.0
COLLECT_SECONDS = 3.0
COUNTDOWN_TICK = 0.1  # seconds

# Optional: show a separate GUI window (matplotlib)
USE_GUI = True
GUI_TICK = 0.1  # seconds, GUI refresh interval

# Optional: run a data preparation script before each collect phase
PREP_SCRIPT_PATH = "/home/juu/Documents/robot_arm_record/00_3_execute.py"  # e.g. "/home/juu/Documents/robot_arm_record/00_3_execute.py"

# Optional: limit how many cycles to run (None = all pairs)
MAX_CYCLES = None

# Repeat each (v1, v2) pair this many times
REPEAT_PER_PAIR = 1
# ==========================


def hit(times=1, gap=0.08):
    for _ in range(times):
        sys.stdout.write("\a")
        sys.stdout.flush()
        time.sleep(gap)


def color(text, code):
    return f"\033[{code}m{text}\033[0m"


def phase_label(name):
    if name == "PREPARE":
        return color(name, "33")  # yellow
    if name == "COLLECT":
        return color(name, "32")  # green
    return name


class PhaseGUI:
    def __init__(self):
        import matplotlib.pyplot as plt

        self.plt = plt
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.ax.set_axis_off()
        self.text = self.ax.text(
            0.5,
            0.5,
            "",
            ha="center",
            va="center",
            fontsize=36,
            fontweight="bold",
            family="monospace",
        )
        self.set_phase("READY", 0, 0, 0.0)
        plt.show(block=False)
        plt.pause(0.001)

    def set_phase(self, name, v1, v2, remaining, rep=None, rep_total=None):
        if name == "PREPARE":
            bg = "#f2c94c"
        elif name == "COLLECT":
            bg = "#27ae60"
        else:
            bg = "#95a5a6"
        self.fig.patch.set_facecolor(bg)
        self.ax.set_facecolor(bg)
        rep_text = ""
        if rep is not None and rep_total is not None:
            rep_text = f"\n\nrep {rep}/{rep_total}"
        self.text.set_text(
            f"{name}\n\nv1= {v1:02d}  v2= {v2:03d}{rep_text}\n\nremaining {remaining:4.1f}s"
        )
        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)


def inclusive_range(start, stop, step):
    if step == 0:
        raise ValueError("step must not be 0")
    if step > 0:
        return list(range(start, stop + 1, step))
    return list(range(start, stop - 1, step))


def build_values(single, start, stop, step, label):
    if single is not None:
        return [int(single)]
    if start is None or stop is None or step is None:
        raise ValueError(f"{label} range not configured")
    return inclusive_range(int(start), int(stop), int(step))


def run_prep_script():
    if PREP_SCRIPT_PATH is None:
        return
    path = Path(PREP_SCRIPT_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Prep script not found: {path}")
    subprocess.run([sys.executable, str(path)], check=True)


def countdown(label, seconds, v1=None, v2=None, gui=None, rep=None, rep_total=None):
    start = time.monotonic()
    label_colored = phase_label(label)
    while True:
        elapsed = time.monotonic() - start
        remaining = max(0.0, seconds - elapsed)
        rep_text = ""
        if rep is not None and rep_total is not None:
            rep_text = f" | rep {rep}/{rep_total}"
        if v1 is None or v2 is None:
            line = f"{label_colored} | remaining {remaining:4.1f}s"
        else:
            line = (
                f"{label_colored} | v1={v1:02d} v2={v2:03d}{rep_text} | "
                f"remaining {remaining:4.1f}s"
            )
        print("\r" + line + " " * 10, end="", flush=True)
        if gui is not None and v1 is not None and v2 is not None:
            gui.set_phase(label, v1, v2, remaining, rep=rep, rep_total=rep_total)
        if remaining <= 0:
            break
        tick = min(COUNTDOWN_TICK, remaining)
        if gui is not None:
            tick = min(GUI_TICK, remaining)
        time.sleep(tick)
    print()


def main():
    gui = PhaseGUI() if USE_GUI else None
    v1_values = build_values(VAR1_SINGLE, VAR1_START, VAR1_STOP, VAR1_STEP, "VAR1")
    v2_values = build_values(VAR2_SINGLE, VAR2_START, VAR2_STOP, VAR2_STEP, "VAR2")
    pairs = list(itertools.product(v1_values, v2_values))

    if MAX_CYCLES is not None:
        pairs = pairs[: int(MAX_CYCLES)]

    if not pairs:
        print("No pairs to process.")
        return

    total_seconds = len(pairs) * (PREP_SECONDS + COLLECT_SECONDS)
    print(f"Total cycles: {len(pairs)} | Estimated time: {total_seconds:.1f}s")
    print("Press Ctrl+C to stop.")

    cycle_count = 0
    rep_total = max(1, int(REPEAT_PER_PAIR))
    total_cycles = len(pairs) * rep_total
    for v1, v2 in pairs:
        for rep in range(rep_total):
            cycle_count += 1
            print(f"\nCycle {cycle_count}/{total_cycles} (rep {rep + 1}/{REPEAT_PER_PAIR})")
            hit(times=1)
            countdown(
                "PREPARE",
                PREP_SECONDS,
                v1=v1,
                v2=v2,
                gui=gui,
                rep=rep + 1,
                rep_total=rep_total,
            )

            run_prep_script()
            hit(times=2)
            countdown(
                "COLLECT",
                COLLECT_SECONDS,
                v1=v1,
                v2=v2,
                gui=gui,
                rep=rep + 1,
                rep_total=rep_total,
            )
            hit(times=1)

    print("\nDone.")


if __name__ == "__main__":
    main()
