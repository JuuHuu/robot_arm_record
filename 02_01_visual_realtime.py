import threading
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from geometry_msgs.msg import WrenchStamped
from matplotlib.animation import FuncAnimation
from rclpy.node import Node
from scipy.signal import butter, sosfilt, sosfilt_zi
from sensor_msgs.msg import JointState


# ========= CONFIG =========
TOPIC_JOINT_STATES = "/joint_states"
TOPIC_WRENCH = "/force_torque_sensor_broadcaster/wrench"

# Which joints/fields to show (None = all joints/fields in message)
PLOT_JOINTS = None  # e.g. ["shoulder_lift_joint", "wrist_1_joint"]
JOINT_FIELDS = ["effort"]#"position", "velocity", 

# Wrench channels to show (None = all)
WRENCH_FIELDS = ["force.x", "force.y", "force.z", "torque.x", "torque.y", "torque.z"]

# Publish filtered signals
PUBLISH_FILTERED = True
PUBLISH_JOINT_FIELDS = None  # None = all available fields in JointState
PUBLISH_WRENCH_FIELDS = None  # None = all available wrench fields
TOPIC_JOINT_STATES_FILTERED = "/joint_states_filtered"
TOPIC_WRENCH_FILTERED = "/wrench_filtered"

# Butterworth low-pass filter (streaming IIR)
FILTER_ENABLED = True
FILTER_ORDER = 4
FILTER_CUTOFF_HZ = 3.0
SAMPLE_RATE_HZ = 200.0

# Plot options
HISTORY_SECONDS = 50  # set to None to keep all history
SHOW_RAW = False
SHOW_FILTERED = True
# ==========================


class OnlineButterworth:
    def __init__(self, cutoff_hz, fs, order):
        nyq = 0.5 * fs
        normal_cutoff = cutoff_hz / nyq
        if normal_cutoff <= 0.0 or normal_cutoff >= 1.0:
            raise ValueError(
                f"Invalid cutoff {cutoff_hz} Hz for fs={fs} Hz (normalized={normal_cutoff:.3f})."
            )
        self.sos = butter(order, normal_cutoff, btype="low", output="sos")
        self.zi = None

    def filter(self, x):
        if self.zi is None:
            self.zi = sosfilt_zi(self.sos) * x
        y, self.zi = sosfilt(self.sos, [x], zi=self.zi)
        return float(y[0])


class ChannelData:
    def __init__(self, history_seconds, filter_obj, enable_filter):
        self.history_seconds = history_seconds
        self.t = deque()
        self.raw = deque()
        self.filtered = deque() if enable_filter else None
        self.filter = filter_obj

    def append(self, t, value):
        self.t.append(t)
        self.raw.append(value)
        filtered_value = value
        if self.filtered is not None:
            filtered_value = self.filter.filter(value)
            self.filtered.append(filtered_value)
        if self.history_seconds is None:
            return filtered_value
        cutoff = t - self.history_seconds
        while self.t and self.t[0] < cutoff:
            self.t.popleft()
            self.raw.popleft()
            if self.filtered is not None:
                self.filtered.popleft()
        return filtered_value


class DataStore:
    def __init__(self, history_seconds, enable_filter, filter_cfg):
        self.history_seconds = history_seconds
        self.enable_filter = enable_filter
        self.filter_cfg = filter_cfg
        self.channels = {}
        self.axes = {}
        self.lock = threading.Lock()
        self.start_time = None

    def _ensure_channel(self, name, axis_name):
        if name in self.channels:
            return
        filt = None
        if self.enable_filter:
            filt = OnlineButterworth(**self.filter_cfg)
        self.channels[name] = ChannelData(self.history_seconds, filt, self.enable_filter)
        self.axes[name] = axis_name

    def append(self, name, axis_name, stamp, value):
        if self.start_time is None:
            self.start_time = stamp
        t_rel = stamp - self.start_time
        with self.lock:
            self._ensure_channel(name, axis_name)
            return self.channels[name].append(t_rel, value)


class RealtimeVisualizer(Node):
    def __init__(self, store):
        super().__init__("realtime_joint_wrench_visualizer")
        self.store = store
        self.create_subscription(JointState, TOPIC_JOINT_STATES, self._on_joint, 10)
        self.create_subscription(WrenchStamped, TOPIC_WRENCH, self._on_wrench, 10)
        self.joint_pub = None
        self.wrench_pub = None
        if PUBLISH_FILTERED:
            self.joint_pub = self.create_publisher(JointState, TOPIC_JOINT_STATES_FILTERED, 10)
            self.wrench_pub = self.create_publisher(WrenchStamped, TOPIC_WRENCH_FILTERED, 10)

    @staticmethod
    def _stamp_to_float(stamp):
        if stamp.sec == 0 and stamp.nanosec == 0:
            return time.time()
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _on_joint(self, msg):
        stamp = self._stamp_to_float(msg.header.stamp)
        publish_fields = PUBLISH_JOINT_FIELDS
        if publish_fields is None:
            publish_fields = ["position", "velocity", "effort"]
        pub_msg = None
        if self.joint_pub is not None:
            pub_msg = JointState()
            pub_msg.header = msg.header
            pub_msg.name = list(msg.name)
        for idx, name in enumerate(msg.name):
            plot_position = JOINT_FIELDS is None or "position" in JOINT_FIELDS
            plot_velocity = JOINT_FIELDS is None or "velocity" in JOINT_FIELDS
            plot_effort = JOINT_FIELDS is None or "effort" in JOINT_FIELDS

            need_position = plot_position or "position" in publish_fields
            need_velocity = plot_velocity or "velocity" in publish_fields
            need_effort = plot_effort or "effort" in publish_fields

            if need_position:
                if idx < len(msg.position):
                    filt = self.store.append(f"{name}/position", "joint", stamp, msg.position[idx])
                    if pub_msg is not None and "position" in publish_fields:
                        pub_msg.position.append(filt if FILTER_ENABLED else msg.position[idx])
            if need_velocity:
                if idx < len(msg.velocity):
                    filt = self.store.append(f"{name}/velocity", "joint", stamp, msg.velocity[idx])
                    if pub_msg is not None and "velocity" in publish_fields:
                        pub_msg.velocity.append(filt if FILTER_ENABLED else msg.velocity[idx])
            if need_effort:
                if idx < len(msg.effort):
                    filt = self.store.append(f"{name}/effort", "joint", stamp, msg.effort[idx])
                    if pub_msg is not None and "effort" in publish_fields:
                        pub_msg.effort.append(filt if FILTER_ENABLED else msg.effort[idx])

        if pub_msg is not None:
            if "position" in publish_fields and len(pub_msg.position) != len(pub_msg.name):
                pub_msg.position = list(msg.position)
            if "velocity" in publish_fields and len(pub_msg.velocity) != len(pub_msg.name):
                pub_msg.velocity = list(msg.velocity)
            if "effort" in publish_fields and len(pub_msg.effort) != len(pub_msg.name):
                pub_msg.effort = list(msg.effort)
            self.joint_pub.publish(pub_msg)

    def _on_wrench(self, msg):
        stamp = self._stamp_to_float(msg.header.stamp)
        mapping = {
            "force.x": msg.wrench.force.x,
            "force.y": msg.wrench.force.y,
            "force.z": msg.wrench.force.z,
            "torque.x": msg.wrench.torque.x,
            "torque.y": msg.wrench.torque.y,
            "torque.z": msg.wrench.torque.z,
        }
        publish_fields = PUBLISH_WRENCH_FIELDS
        if publish_fields is None:
            publish_fields = list(mapping.keys())
        pub_msg = None
        if self.wrench_pub is not None:
            pub_msg = WrenchStamped()
            pub_msg.header = msg.header
        for key, value in mapping.items():
            filt = self.store.append(key, "wrench", stamp, value)
            if pub_msg is not None and key in publish_fields:
                if FILTER_ENABLED:
                    out = filt
                else:
                    out = value
                if key == "force.x":
                    pub_msg.wrench.force.x = out
                elif key == "force.y":
                    pub_msg.wrench.force.y = out
                elif key == "force.z":
                    pub_msg.wrench.force.z = out
                elif key == "torque.x":
                    pub_msg.wrench.torque.x = out
                elif key == "torque.y":
                    pub_msg.wrench.torque.y = out
                elif key == "torque.z":
                    pub_msg.wrench.torque.z = out
        if pub_msg is not None:
            self.wrench_pub.publish(pub_msg)


def _run_ros(node):
    rclpy.spin(node)


def main():
    filter_cfg = {
        "cutoff_hz": FILTER_CUTOFF_HZ,
        "fs": SAMPLE_RATE_HZ,
        "order": FILTER_ORDER,
    }
    store = DataStore(HISTORY_SECONDS, FILTER_ENABLED, filter_cfg)

    rclpy.init()
    node = RealtimeVisualizer(store)
    ros_thread = threading.Thread(target=_run_ros, args=(node,), daemon=True)
    ros_thread.start()

    fig, (ax_joint, ax_wrench) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    ax_joint.set_title("Joint States")
    ax_wrench.set_title("Wrench")
    ax_wrench.set_xlabel("Time (s)")
    ax_joint.set_ylabel("Value")
    ax_wrench.set_ylabel("Value")
    ax_joint.grid(True)
    ax_wrench.grid(True)

    lines = {}
    legend_dirty = {"joint": True, "wrench": True}

    def ensure_line(channel, axis_name, suffix, color=None):
        key = f"{channel}{suffix}"
        if key in lines:
            return lines[key]
        ax = ax_joint if axis_name == "joint" else ax_wrench
        line, = ax.plot([], [], label=key, linewidth=1.2)
        if color is not None:
            line.set_color(color)
        lines[key] = line
        legend_dirty[axis_name] = True
        return line

    def update(_):
        with store.lock:
            snapshot = {
                name: (
                    store.axes[name],
                    np.array(ch.t),
                    np.array(ch.raw),
                    np.array(ch.filtered) if ch.filtered is not None else None,
                )
                for name, ch in store.channels.items()
            }

        updated_lines = []
        for name, (axis_name, t, raw, filtered) in snapshot.items():
            if axis_name == "joint":
                joint_name, field = name.split("/", 1)
                if PLOT_JOINTS is not None and joint_name not in PLOT_JOINTS:
                    continue
                if JOINT_FIELDS is not None and field not in JOINT_FIELDS:
                    continue
            elif axis_name == "wrench":
                if WRENCH_FIELDS is not None and name not in WRENCH_FIELDS:
                    continue
            if SHOW_RAW:
                line = ensure_line(name, axis_name, " (raw)")
                line.set_data(t, raw)
                updated_lines.append(line)
            if SHOW_FILTERED and filtered is not None:
                line = ensure_line(name, axis_name, " (lp)")
                line.set_data(t, filtered)
                updated_lines.append(line)
            if not SHOW_FILTERED and not SHOW_RAW:
                line = ensure_line(name, axis_name, "")
                line.set_data(t, raw)
                updated_lines.append(line)

        if updated_lines:
            t_max = max(
                (line.get_xdata()[-1] for line in updated_lines if len(line.get_xdata()) > 0),
                default=0.0,
            )
            if HISTORY_SECONDS is None:
                t_min = 0.0
            else:
                t_min = max(0.0, t_max - HISTORY_SECONDS)
            ax_joint.set_xlim(t_min, t_max + 0.1)
            ax_wrench.set_xlim(t_min, t_max + 0.1)

        if legend_dirty["joint"]:
            ax_joint.legend(loc="upper left", ncol=2, fontsize=8)
            legend_dirty["joint"] = False
        if legend_dirty["wrench"]:
            ax_wrench.legend(loc="upper left", ncol=2, fontsize=8)
            legend_dirty["wrench"] = False

        for ax in (ax_joint, ax_wrench):
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)

        return updated_lines

    anim = FuncAnimation(fig, update, interval=50, blit=False)

    try:
        plt.tight_layout()
        plt.show()
    finally:
        anim.event_source.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
