#!/usr/bin/env python3
import time
import yaml
import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, TransformStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.srv import GetPositionIK
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster

from moveit_msgs.msg import DisplayTrajectory, RobotState, RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory



# ============== CONFIG ==============
CORNERS_YAML = "/home/juu/Documents/robot_arm_record/corners.yaml"

# UR7e 6 joints (order must match controller)
JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

IK_SERVICE = "/compute_ik"
TRAJ_TOPIC = "/scaled_joint_trajectory_controller/joint_trajectory"
JOINT_STATES_TOPIC = "/joint_states"

PLANNING_GROUP = "ur_manipulator"   # your MoveIt group name
BASE_FRAME = "base"                 # matches frame_id in corners.yaml

# Box sampling (in local PCA coords)
NX, NY, NZ = 20, 20, 20   # grid resolution inside box
MOVE_TIME = 2             # seconds per move in trajectory
PAUSE_AT_POINT = 0.5        # extra pause at each point
TILT_DEG = 10.0           # maximum tilt angle away from -Z
MAX_JOINT_JUMP = 1.5  # max allowed delta per joint (rad) to avoid "crazy" IK
MAX_JOINT_VEL = 1.0   # rad/s cap per joint for speed planning
MAX_JOINT_ACC = 1.5   # rad/s^2 cap per joint for accel-based timing
MIN_MOVE_TIME = 0.75  # never plan faster than this even for tiny moves
PREVIEW_ONLY = True 
RANDOM_RUN = True
# ====================================


def load_corners(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    pos = []
    ori = []
    frames = []
    for c in data["corners"]:
        tcp = c["tcp_pose"]
        p = tcp["position"]
        o = tcp["orientation"]
        frames.append(tcp["frame_id"])
        pos.append([p["x"], p["y"], p["z"]])
        ori.append([o["x"], o["y"], o["z"], o["w"]])

    frame_id = frames[0]  # assume all same
    return np.array(pos, float), np.array(ori, float), frame_id


def sample_random_orientation(max_tilt_deg=10.0):
    """
    Generate a quaternion whose Z axis points roughly downward (-Z),
    but with a random tilt up to max_tilt_deg.
    """
    max_tilt = np.radians(max_tilt_deg)

    # random tilt direction (azimuth)
    phi = np.random.uniform(0, 2*np.pi)
    # random tilt magnitude
    theta = np.random.uniform(0, max_tilt)

    # ideal -Z direction
    z0 = np.array([0.0, 0.0, -1.0])

    # small rotation axis in XY plane
    axis = np.array([np.cos(phi), np.sin(phi), 0.0])
    axis /= np.linalg.norm(axis)

    # Rodrigues' formula
    k = axis
    ct = np.cos(theta)
    st = np.sin(theta)
    z_new = ct * z0 + st * np.cross(k, z0) + (1 - ct) * (np.dot(k, z0)) * k
    z_new /= np.linalg.norm(z_new)

    # pick an X perpendicular to z_new
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, z_new)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])

    x_new = np.cross(tmp, z_new)
    x_new /= np.linalg.norm(x_new)
    y_new = np.cross(z_new, x_new)

    R = np.vstack([x_new, y_new, z_new]).T  # rotation matrix

    # rotation matrix -> quaternion
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S

    return np.array([qx, qy, qz, qw], dtype=float)


def compute_ground_aligned_box(positions: np.ndarray):
    """
    Positions: (N,3) corner points in world frame.
    Z axis is forced to world Z; X axis follows longest direction in XY.
    """
    center = positions.mean(axis=0)

    # PCA in XY plane only
    XY = positions[:, :2] - center[:2]
    cov = np.cov(XY.T)
    eigval, eigvec = np.linalg.eig(cov)

    major_xy = eigvec[:, np.argmax(eigval)]
    major_xy = major_xy / np.linalg.norm(major_xy)

    z_axis = np.array([0.0, 0.0, 1.0])
    x_axis = np.array([major_xy[0], major_xy[1], 0.0])
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    R = np.vstack([x_axis, y_axis, z_axis]).T  # columns are axes

    X_local = (positions - center) @ R
    mins = X_local.min(axis=0)
    maxs = X_local.max(axis=0)

    return center, R, mins, maxs


def generate_grid_points_local(center, R, mins, maxs, nx, ny, nz):
    xs = np.linspace(mins[0], maxs[0], nx)
    ys = np.linspace(mins[1], maxs[1], ny)
    zs = np.linspace(mins[2], maxs[2], nz)

    pts_world = []
    for x in xs:
        for y in ys:
            for z in zs:
                p_local = np.array([x, y, z])
                p_world = center + R @ p_local
                pts_world.append((float(p_world[0]),
                                  float(p_world[1]),
                                  float(p_world[2])))
    return pts_world


class SimpleBoxScanner(Node):
    def __init__(self):
        super().__init__("simple_box_scanner")

        positions, orientations, frame_id = load_corners(CORNERS_YAML)
        self.frame_id = frame_id

        self.tf_broadcaster = TransformBroadcaster(self)

        center, R, mins, maxs = compute_ground_aligned_box(positions)
        self.center = center
        self.R = R
        self.mins = mins
        self.maxs = maxs

        self.get_logger().info(f"Frame: {self.frame_id}")
        self.get_logger().info(f"Center: {center}")
        self.get_logger().info(f"Local mins: {mins}")
        self.get_logger().info(f"Local maxs: {maxs}")
        self.get_logger().info(f"Axes (columns of R):\n{R}")

        self.points = generate_grid_points_local(center, R, mins, maxs,
                                                 NX, NY, NZ)
        self.get_logger().info(f"Generated {len(self.points)} points inside oriented box.")
        if RANDOM_RUN ==True:
            # Randomize visit order so the robot moves between points in a shuffled sequence each run.
            np.random.shuffle(self.points)

        self.traj_pub = self.create_publisher(JointTrajectory, TRAJ_TOPIC, 10)

        # IK client
        self.ik_client = self.create_client(GetPositionIK, IK_SERVICE)
        self.get_logger().info(f"Waiting for IK service '{IK_SERVICE}'...")
        self.ik_client.wait_for_service()
        self.get_logger().info("IK service available.")

        # joint state subscriber (for seeding IK and joint jump check)
        self.current_js = None
        self.current_q = None  # vector in JOINT_NAMES order
        self.js_sub = self.create_subscription(
            JointState,
            JOINT_STATES_TOPIC,
            self.joint_state_cb,
            10,
        )
        
        self.display_pub = self.create_publisher(DisplayTrajectory,
                                         "/display_planned_path", 10)
    def visualize_ik_solution(self, q):
        msg = DisplayTrajectory()

        # Start state (use current_q if you have it)
        start_state = RobotState()
        start_state.joint_state.name = JOINT_NAMES
        if self.current_q is not None:
            start_state.joint_state.position = list(self.current_q)
        else:
            start_state.joint_state.position = list(q)
        msg.trajectory_start = start_state

        # Build a one-point JointTrajectory
        jt = JointTrajectory()
        jt.joint_names = JOINT_NAMES

        pt = JointTrajectoryPoint()
        pt.positions = q
        pt.time_from_start.sec = 3  # just for RViz animation
        jt.points.append(pt)

        # Wrap in RobotTrajectory
        robot_traj = RobotTrajectory()
        robot_traj.joint_trajectory = jt

        # Append to DisplayTrajectory
        msg.trajectory.append(robot_traj)

        self.display_pub.publish(msg)

        
    def joint_state_cb(self, msg: JointState):
        """Store latest joint state in JOINT_NAMES order."""
        name_to_pos = {n: p for n, p in zip(msg.name, msg.position)}
        q = []
        for name in JOINT_NAMES:
            if name not in name_to_pos:
                return  # ignore incomplete states
            q.append(float(name_to_pos[name]))
        self.current_js = msg
        self.current_q = np.array(q, dtype=float)

    def compute_ik_with_ori(self, x, y, z, ori):
        req = GetPositionIK.Request()
        req.ik_request.group_name = PLANNING_GROUP
        req.ik_request.avoid_collisions = True

        pose = PoseStamped()
        pose.header.frame_id = self.frame_id
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.x = float(ori[0])
        pose.pose.orientation.y = float(ori[1])
        pose.pose.orientation.z = float(ori[2])
        pose.pose.orientation.w = float(ori[3])

        req.ik_request.pose_stamped = pose

        # Seed from current state if we have one
        if self.current_js is not None:
            req.ik_request.robot_state.joint_state = self.current_js

        future = self.ik_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        res = future.result()
        if not res:
            self.get_logger().warn("IK service call returned no result.")
            return None

        if res.error_code.val <= 0:
            self.get_logger().warn(f"IK failed, error_code = {res.error_code.val}")
            return None

        js = res.solution.joint_state
        name_to_pos = {n: p for n, p in zip(js.name, js.position)}
        try:
            q = np.array([float(name_to_pos[n]) for n in JOINT_NAMES], dtype=float)
        except KeyError as e:
            self.get_logger().warn(f"IK result missing joint: {e}")
            return None

        # ---- joint jump filter to avoid strange poses ----
        if self.current_q is not None:
            delta = np.abs(q - self.current_q)
            max_delta = np.max(delta)
            if max_delta > MAX_JOINT_JUMP:
                self.get_logger().warn(
                    f"IK solution has large joint jump (max {max_delta:.2f} rad), skipping."
                )
                return None

        return q.tolist()

    def publish_ik_pose(self, x, y, z, ori):
        t = TransformStamped()
        t.header.frame_id = self.frame_id
        t.header.stamp = self.get_clock().now().to_msg()
        t.child_frame_id = "ik_target"

        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = z

        t.transform.rotation.x = ori[0]
        t.transform.rotation.y = ori[1]
        t.transform.rotation.z = ori[2]
        t.transform.rotation.w = ori[3]

        self.tf_broadcaster.sendTransform(t)

    def plan_move_duration(self, q_target):
        """
        Compute a per-move duration that respects per-joint velocity/acc caps.
        Falls back to MOVE_TIME if current_q is unavailable (e.g. before the
        first joint state message arrives).
        """
        if self.current_q is None:
            return MOVE_TIME

        dq = np.abs(np.array(q_target, dtype=float) - self.current_q)
        max_dq = float(np.max(dq)) if dq.size else 0.0

        vel_time = float(np.max(dq / MAX_JOINT_VEL)) if dq.size else 0.0
        acc_time = 2.0 * np.sqrt(max_dq / MAX_JOINT_ACC) if max_dq > 0 else 0.0

        planned = max(vel_time, acc_time, MIN_MOVE_TIME)
        return planned

    def send_joint_trajectory(self, q, move_time):
        traj = JointTrajectory()
        traj.joint_names = JOINT_NAMES

        if self.current_q is not None:
            start_pt = JointTrajectoryPoint()
            start_pt.positions = list(self.current_q)
            start_pt.velocities = [0.0] * len(JOINT_NAMES)
            start_pt.time_from_start.sec = 0
            start_pt.time_from_start.nanosec = 0
            traj.points.append(start_pt)

        pt = JointTrajectoryPoint()
        pt.positions = q
        pt.velocities = [0.0] * len(JOINT_NAMES)  # end at rest
        pt.time_from_start.sec = int(move_time)
        pt.time_from_start.nanosec = int((move_time - int(move_time)) * 1e9)

        traj.points.append(pt)
        self.traj_pub.publish(traj)

    def run(self):
        for i, (x, y, z) in enumerate(self.points):
            self.get_logger().info(
                f"[{i+1}/{len(self.points)}] Target point: ({x:.3f}, {y:.3f}, {z:.3f})"
            )

            ori = sample_random_orientation(TILT_DEG)
            self.publish_ik_pose(x, y, z, ori)
            q = self.compute_ik_with_ori(x, y, z, ori)
            if q is None:
                self.get_logger().warn("Skipping point due to IK failure or joint jump.")
                continue
            
            self.visualize_ik_solution(q)
            if PREVIEW_ONLY:
                self.get_logger().info("Preview only mode, not sending trajectory.")
                time.sleep(0.1)
                continue
            move_time = self.plan_move_duration(q)
            self.get_logger().info(f"Sending trajectory (planned {move_time:.2f}s), q={q}")
            self.send_joint_trajectory(q, move_time)

            total_wait = move_time + PAUSE_AT_POINT
            self.get_logger().info(f"Waiting {total_wait:.1f} s at this point...")
            time.sleep(total_wait)

        self.get_logger().info("Finished visiting all points.")


def main():
    rclpy.init()
    node = SimpleBoxScanner()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
