#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import yaml
import numpy as np

from visualization_msgs.msg import Marker, MarkerArray


# ======= CONFIG =======
CORNERS_YAML = "/home/juu/Documents/robot_arm_record/corners.yaml"
MARKER_TOPIC = "oriented_box_pca_markers"

NX, NY, NZ = 30, 30, 20   # sample grid resolution inside box
# ======================


def load_corners(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    pos = []
    frames = []
    for c in data["corners"]:
        tcp = c["tcp_pose"]
        p = tcp["position"]
        frame = tcp["frame_id"]
        pos.append([p["x"], p["y"], p["z"]])
        frames.append(frame)

    positions = np.array(pos, dtype=float)
    frame_id = frames[0]  # assume all same
    return positions, frame_id


def compute_ground_aligned_box(positions):
    """
    positions: (N,3) array of corner points in world frame.
    Returns:
      center: mean of corners
      R: 3x3 rotation matrix (Z aligned to world Z)
      mins, maxs: bounds in this rotated frame
    """

    center = positions.mean(axis=0)

    # --- 1) Project to XY plane ---
    XY = positions[:, :2] - center[:2]

    # --- 2) PCA in 2D (XY only) ---
    # covariance matrix
    cov = np.cov(XY.T)
    eigval, eigvec = np.linalg.eig(cov)  # eigvec[:,i] = eigenvector

    # major axis in XY
    major_xy = eigvec[:, np.argmax(eigval)]
    major_xy = major_xy / np.linalg.norm(major_xy)

    # ensure right-handed orientation
    z_axis = np.array([0, 0, 1.0])
    x_axis = np.array([major_xy[0], major_xy[1], 0.0])
    x_axis /= np.linalg.norm(x_axis)

    # orthogonal y = z × x
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    # full rotation matrix
    R = np.vstack([x_axis, y_axis, z_axis]).T  # columns are the axes

    # --- 3) Transform corners into this aligned frame ---
    X_local = (positions - center) @ R

    mins = X_local.min(axis=0)
    maxs = X_local.max(axis=0)

    return center, R, mins, maxs



class PCABoxVisualizer(Node):
    def __init__(self):
        super().__init__("pca_box_visualizer")

        positions, frame_id = load_corners(CORNERS_YAML)
        self.positions = positions
        self.frame_id = frame_id

        center, R, mins, maxs = compute_ground_aligned_box(positions)
        self.center = center
        self.R = R
        self.mins = mins
        self.maxs = maxs

        self.get_logger().info(f"Frame: {self.frame_id}")
        self.get_logger().info(f"Center: {center}")
        self.get_logger().info(f"Principal axes (columns of R):\n{R}")
        self.get_logger().info(f"Local mins: {mins}")
        self.get_logger().info(f"Local maxs: {maxs}")

        self.samples = self._generate_samples()

        self.pub = self.create_publisher(MarkerArray, MARKER_TOPIC, 10)
        self.timer = self.create_timer(0.2, self._timer_cb)

    def _generate_samples(self):
        # sample grid in local PCA coords
        xs = np.linspace(self.mins[0], self.maxs[0], NX)
        ys = np.linspace(self.mins[1], self.maxs[1], NY)
        zs = np.linspace(self.mins[2], self.maxs[2], NZ)

        pts_world = []
        for x in xs:
            for y in ys:
                for z in zs:
                    p_local = np.array([x, y, z])
                    p_world = self.center + self.R @ p_local
                    pts_world.append(p_world)

        return np.array(pts_world)

    def _timer_cb(self):
        ma = MarkerArray()
        now = self.get_clock().now().to_msg()

        # 1) Original corners (orange spheres)
        for i, p in enumerate(self.positions):
            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp = now
            m.ns = "corners"
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.scale.x = m.scale.y = m.scale.z = 0.03
            m.color.r = 1.0
            m.color.g = 0.3
            m.color.b = 0.0
            m.color.a = 1.0
            m.pose.position.x = float(p[0])
            m.pose.position.y = float(p[1])
            m.pose.position.z = float(p[2])
            m.pose.orientation.w = 1.0
            ma.markers.append(m)

        # 2) Sampled points inside oriented box (blue)
        for i, p in enumerate(self.samples):
            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp = now
            m.ns = "samples"
            m.id = 1000 + i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.scale.x = m.scale.y = m.scale.z = 0.015
            m.color.r = 0.0
            m.color.g = 0.2
            m.color.b = 1.0
            m.color.a = 1.0
            m.pose.position.x = float(p[0])
            m.pose.position.y = float(p[1])
            m.pose.position.z = float(p[2])
            m.pose.orientation.w = 1.0
            ma.markers.append(m)

        self.pub.publish(ma)


def main():
    rclpy.init()
    node = PCABoxVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
