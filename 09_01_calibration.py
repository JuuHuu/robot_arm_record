#!/usr/bin/env python3
import argparse
import math
import time
from typing import List, Tuple

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import DisplayTrajectory, RobotState, RobotTrajectory
from tf2_ros import Buffer, TransformListener


# ================== CONFIG DEFAULTS ==================
DEFAULT_SPEED = 0.1  # m/s
DEFAULT_ACCEL = 0.5  # m/s^2
DEFAULT_AMP_XY = 0.05  # m
DEFAULT_AMP_Z = 0.02  # m
DEFAULT_LOOPS = 1
DEFAULT_POINTS_PER_LOOP = 200
DEFAULT_MIN_SEGMENT_TIME = 0.03  # s
DEFAULT_MAX_JOINT_JUMP = 1.5  # rad
DEFAULT_WAIT_TIMEOUT = 10.0  # s
DEFAULT_BASE_FRAME = "base"
DEFAULT_EE_FRAME = "tool0"
DEFAULT_PLANNING_GROUP = "ur_manipulator"
DEFAULT_IK_SERVICE = "/compute_ik"
DEFAULT_TRAJ_TOPIC = "/scaled_joint_trajectory_controller/joint_trajectory"
DEFAULT_JOINT_STATES_TOPIC = "/joint_states"
DEFAULT_TF_TIMEOUT = 2.0  # s
DEFAULT_PREVIEW_ONLY = False

# UR7e 6 joints (order must match controller)
JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def _duration_to_msg(seconds: float):
    secs = int(seconds)
    nsecs = int((seconds - secs) * 1e9)
    msg = Duration(seconds=secs, nanoseconds=nsecs).to_msg()
    return msg


def _generate_eight_points(
    center: np.ndarray,
    amp_xy: float,
    amp_z: float,
    loops: int,
    points_per_loop: int,
) -> List[np.ndarray]:
    total = max(1, loops) * max(10, points_per_loop)
    points = []
    for i in range(total + 1):
        t = (2.0 * math.pi * loops) * (i / total)
        x = amp_xy * math.sin(t)
        y = amp_xy * math.sin(t) * math.cos(t)
        z = amp_z * math.sin(2.0 * t)
        points.append(center + np.array([x, y, z], dtype=float))
    return points


def _time_from_distance(s: float, total: float, v_max: float, accel: float) -> float:
    if total <= 0.0:
        return 0.0
    if accel <= 0.0:
        return s / max(v_max, 1e-6)

    d_acc = (v_max * v_max) / (2.0 * accel)
    if total < 2.0 * d_acc:
        v_peak = math.sqrt(total * accel)
        t_acc = v_peak / accel
        half = total / 2.0
        if s <= half:
            return math.sqrt(2.0 * s / accel)
        return 2.0 * t_acc - math.sqrt(2.0 * (total - s) / accel)

    t_acc = v_max / accel
    d_flat = total - 2.0 * d_acc
    if s <= d_acc:
        return math.sqrt(2.0 * s / accel)
    if s <= d_acc + d_flat:
        return t_acc + (s - d_acc) / v_max
    remaining = total - s
    return t_acc + d_flat / v_max + (t_acc - math.sqrt(2.0 * remaining / accel))


def _compute_time_profile(
    points: List[np.ndarray],
    speed: float,
    accel: float,
    min_segment_time: float,
) -> List[float]:
    if len(points) < 2:
        return [0.0]

    distances = []
    total = 0.0
    prev = points[0]
    for point in points[1:]:
        dist = float(np.linalg.norm(point - prev))
        distances.append(dist)
        total += dist
        prev = point

    times = [0.0]
    s = 0.0
    last_t = 0.0
    for dist in distances:
        s += dist
        t_target = _time_from_distance(s, total, speed, accel)
        t_target = max(t_target, last_t + min_segment_time)
        times.append(t_target)
        last_t = t_target
    return times


class EightShapeCalibration(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("eight_shape_calibration")
        self.args = args
        self.current_js = None
        self.current_q = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.traj_pub = self.create_publisher(JointTrajectory, args.traj_topic, 10)
        self.display_pub = self.create_publisher(DisplayTrajectory, "/display_planned_path", 10)
        self.ik_client = self.create_client(GetPositionIK, args.ik_service)
        self.get_logger().info(f"Waiting for IK service '{args.ik_service}'...")
        self.ik_client.wait_for_service()
        self.get_logger().info("IK service available.")

        self.js_sub = self.create_subscription(
            JointState, args.joint_states_topic, self.joint_state_cb, 10
        )

    def _publish_preview(self, traj: JointTrajectory, start_q: np.ndarray):
        msg = DisplayTrajectory()
        start_state = RobotState()
        start_state.joint_state.name = JOINT_NAMES
        start_state.joint_state.position = list(start_q)
        msg.trajectory_start = start_state

        robot_traj = RobotTrajectory()
        robot_traj.joint_trajectory = traj
        msg.trajectory.append(robot_traj)
        self.display_pub.publish(msg)

    def joint_state_cb(self, msg: JointState):
        name_to_pos = {n: p for n, p in zip(msg.name, msg.position)}
        q = []
        for name in JOINT_NAMES:
            if name not in name_to_pos:
                return
            q.append(float(name_to_pos[name]))
        self.current_js = msg
        self.current_q = np.array(q, dtype=float)

    def get_current_pose(self) -> Tuple[np.ndarray, List[float]]:
        timeout = Duration(seconds=self.args.tf_timeout)
        trans = self.tf_buffer.lookup_transform(
            self.args.base_frame,
            self.args.ee_frame,
            rclpy.time.Time(),
            timeout=timeout,
        )
        pos = trans.transform.translation
        ori = trans.transform.rotation
        center = np.array([pos.x, pos.y, pos.z], dtype=float)
        quat = [ori.x, ori.y, ori.z, ori.w]
        return center, quat

    def compute_ik(self, x: float, y: float, z: float, ori: List[float], seed_q=None):
        req = GetPositionIK.Request()
        req.ik_request.group_name = self.args.planning_group
        req.ik_request.avoid_collisions = True

        pose = PoseStamped()
        pose.header.frame_id = self.args.base_frame
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = float(z)
        pose.pose.orientation.x = float(ori[0])
        pose.pose.orientation.y = float(ori[1])
        pose.pose.orientation.z = float(ori[2])
        pose.pose.orientation.w = float(ori[3])
        req.ik_request.pose_stamped = pose

        if seed_q is not None:
            seed_js = JointState()
            seed_js.name = JOINT_NAMES
            seed_js.position = list(seed_q)
            req.ik_request.robot_state.joint_state = seed_js
        elif self.current_js is not None:
            req.ik_request.robot_state.joint_state = self.current_js

        future = self.ik_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        res = future.result()
        if not res:
            return None
        if res.error_code.val <= 0:
            return None

        js = res.solution.joint_state
        name_to_pos = {n: p for n, p in zip(js.name, js.position)}
        try:
            q = np.array([float(name_to_pos[n]) for n in JOINT_NAMES], dtype=float)
        except KeyError:
            return None
        return q

    def _build_trajectory(
        self,
        points: List[np.ndarray],
        ori: List[float],
        start_q: np.ndarray,
    ) -> JointTrajectory:
        traj = JointTrajectory()
        traj.joint_names = JOINT_NAMES

        prev_q = start_q
        t_from_start = 0.0
        q_list = []

        for point in points[1:]:
            q = self.compute_ik(point[0], point[1], point[2], ori, seed_q=prev_q)
            if q is None:
                self.get_logger().error("IK failed for path point; aborting without motion.")
                return None
            delta = np.abs(q - prev_q)
            if float(np.max(delta)) > self.args.max_joint_jump:
                self.get_logger().error("IK joint jump too large; aborting without motion.")
                return None
            q_list.append(q)
            prev_q = q

        times = _compute_time_profile(
            points,
            speed=self.args.speed,
            accel=self.args.accel,
            min_segment_time=self.args.min_segment_time,
        )
        q_points = [start_q] + q_list
        for q, t_target in zip(q_points, times):
            t_from_start = max(t_from_start, t_target)
            pt = JointTrajectoryPoint()
            pt.positions = list(q)
            pt.time_from_start = _duration_to_msg(t_from_start)
            traj.points.append(pt)

        vels = []
        accs = []
        for i in range(len(q_points)):
            if i == 0:
                v = np.zeros_like(q_points[0])
            else:
                dt = max(times[i] - times[i - 1], self.args.min_segment_time)
                v = (q_points[i] - q_points[i - 1]) / dt
            vels.append(v)

        for i in range(len(q_points)):
            if i == 0:
                a = np.zeros_like(q_points[0])
            else:
                dt = max(times[i] - times[i - 1], self.args.min_segment_time)
                a = (vels[i] - vels[i - 1]) / dt
            accs.append(a)

        for i, pt in enumerate(traj.points):
            pt.velocities = list(vels[i])
            pt.accelerations = list(accs[i])

        if traj.points:
            traj.points[0].velocities = [0.0] * len(JOINT_NAMES)
            traj.points[0].accelerations = [0.0] * len(JOINT_NAMES)
            traj.points[-1].velocities = [0.0] * len(JOINT_NAMES)
            traj.points[-1].accelerations = [0.0] * len(JOINT_NAMES)

        return traj

    def run(self):
        start_time = time.time()
        while rclpy.ok() and self.current_q is None:
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.time() - start_time > self.args.wait_timeout:
                raise RuntimeError("Timed out waiting for joint states.")

        start_time = time.time()
        while rclpy.ok():
            try:
                center, ori = self.get_current_pose()
                break
            except Exception:
                rclpy.spin_once(self, timeout_sec=0.1)
                if time.time() - start_time > self.args.wait_timeout:
                    raise RuntimeError("Timed out waiting for current pose transform.")

        start_q = np.array(self.current_q, dtype=float)
        points = _generate_eight_points(
            center=center,
            amp_xy=self.args.amp_xy,
            amp_z=self.args.amp_z,
            loops=self.args.loops,
            points_per_loop=self.args.points_per_loop,
        )

        if len(points) < 2:
            raise RuntimeError("Not enough points generated for trajectory.")

        traj = self._build_trajectory(points, ori, start_q)
        if traj is None:
            return

        if self.args.preview_only:
            self._publish_preview(traj, start_q)
            self.get_logger().info(
                f"Preview only: computed {len(traj.points)} points; "
                f"speed={self.args.speed:.3f} m/s, accel={self.args.accel:.3f} m/s^2, "
                f"amp_xy={self.args.amp_xy:.3f} m, amp_z={self.args.amp_z:.3f} m."
            )
            return

        self.get_logger().info(
            f"Sending {len(traj.points)} trajectory points "
            f"(speed={self.args.speed:.3f} m/s, accel={self.args.accel:.3f} m/s^2, "
            f"amp_xy={self.args.amp_xy:.3f} m, amp_z={self.args.amp_z:.3f} m)."
        )
        self.traj_pub.publish(traj)
        total_time = float(traj.points[-1].time_from_start.sec) + (
            float(traj.points[-1].time_from_start.nanosec) * 1e-9
        )
        time.sleep(total_time + 0.5)


def parse_args():
    parser = argparse.ArgumentParser(description="Run a 3D figure-eight calibration motion.")
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED, help="Cartesian speed (m/s).")
    parser.add_argument("--accel", type=float, default=DEFAULT_ACCEL, help="Cartesian accel (m/s^2).")
    parser.add_argument("--amp-xy", type=float, default=DEFAULT_AMP_XY, help="Figure-eight XY amplitude (m).")
    parser.add_argument("--amp-z", type=float, default=DEFAULT_AMP_Z, help="Z amplitude (m).")
    parser.add_argument("--loops", type=int, default=DEFAULT_LOOPS, help="Number of figure-eight loops.")
    parser.add_argument("--points-per-loop", type=int, default=DEFAULT_POINTS_PER_LOOP, help="Samples per loop.")
    parser.add_argument(
        "--min-segment-time",
        type=float,
        default=DEFAULT_MIN_SEGMENT_TIME,
        help="Minimum time per segment (s).",
    )
    parser.add_argument("--max-joint-jump", type=float, default=DEFAULT_MAX_JOINT_JUMP, help="Max joint delta allowed (rad).")
    parser.add_argument("--wait-timeout", type=float, default=DEFAULT_WAIT_TIMEOUT, help="Timeout for joint/pose (s).")
    parser.add_argument("--base-frame", default=DEFAULT_BASE_FRAME, help="Base frame for TF and IK.")
    parser.add_argument("--ee-frame", default=DEFAULT_EE_FRAME, help="End-effector frame for TF.")
    parser.add_argument("--planning-group", default=DEFAULT_PLANNING_GROUP, help="MoveIt planning group.")
    parser.add_argument("--ik-service", default=DEFAULT_IK_SERVICE, help="MoveIt IK service name.")
    parser.add_argument(
        "--traj-topic",
        default=DEFAULT_TRAJ_TOPIC,
        help="Joint trajectory topic.",
    )
    parser.add_argument(
        "--joint-states-topic",
        default=DEFAULT_JOINT_STATES_TOPIC,
        help="Joint states topic.",
    )
    parser.add_argument("--tf-timeout", type=float, default=DEFAULT_TF_TIMEOUT, help="TF lookup timeout (s).")
    parser.add_argument("--preview-only", action="store_true", default=DEFAULT_PREVIEW_ONLY, help="Compute only; do not publish.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.speed <= 0.0:
        raise ValueError("Speed must be > 0.")
    if args.accel <= 0.0:
        raise ValueError("Accel must be > 0.")

    rclpy.init()
    node = EightShapeCalibration(args)
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
