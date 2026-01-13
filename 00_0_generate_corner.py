import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
import yaml
import os
import numpy as np


class CornerRecorder(Node):
    def __init__(
        self,
        corner_count=8,
        vel_threshold=1e-5,
        stable_duration=0.5,
        output_path="corners.yaml",
    ):
        super().__init__("corner_recorder")

        self.corner_count = corner_count
        self.vel_threshold = vel_threshold
        self.stable_duration = stable_duration
        self.output_path = output_path

        # Subscriptions
        self.joint_state_sub = self.create_subscription(
            JointState, "/joint_states", self.joint_state_cb, 10
        )
        self.tcp_pose_sub = self.create_subscription(
            PoseStamped, "/tcp_pose_broadcaster/pose", self.tcp_pose_cb, 10
        )

        self.latest_joint_state = None
        self.latest_tcp_pose = None

        self.get_logger().info(
            f"CornerRecorder initialized: corners={corner_count}, "
            f"vel_threshold={vel_threshold}, stable_duration={stable_duration}s"
        )

    def joint_state_cb(self, msg: JointState):
        self.latest_joint_state = msg

    def tcp_pose_cb(self, msg: PoseStamped):
        self.latest_tcp_pose = msg

    def is_arm_stable(self):
        """
        Returns (stable: bool, max_vel: float)
        stable if all |vel| < vel_threshold.
        """
        if self.latest_joint_state is None:
            return False, None
        if not self.latest_joint_state.velocity:
            return False, None

        v = np.array(self.latest_joint_state.velocity, dtype=float)
        max_vel = float(np.max(np.abs(v)))
        stable = max_vel < self.vel_threshold
        return stable, max_vel

    def wait_until_stable(self):
        """
        Wait until the arm is stable for continuous stable_duration seconds.
        Uses joint_states velocity.
        """
        self.get_logger().info(
            f"Waiting for arm to be stable: |vel| < {self.vel_threshold} "
            f"for {self.stable_duration} s..."
        )

        stable_start_time = None

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)

            stable, max_vel = self.is_arm_stable()
            now = self.get_clock().now().nanoseconds / 1e9  # seconds

            if stable:
                if stable_start_time is None:
                    stable_start_time = now
                elapsed = now - stable_start_time
                if elapsed >= self.stable_duration:
                    self.get_logger().info(
                        f"Arm stable for {elapsed:.2f} s (max_vel={max_vel:.3e})."
                    )
                    return
            else:
                stable_start_time = None

    def record_corners(self):
        corners = []

        for i in range(self.corner_count):
            # Prompt user
            msg = (
                f"\n=== Corner {i+1}/{self.corner_count} ===\n"
                "Move the robot to this corner.\n"
                "When the arm is close to the desired pose, press ENTER..."
            )
            print(msg)
            input()  # wait for user

            # Make sure we have at least one TCP pose
            self.get_logger().info("Ensuring we have TCP pose and joint states...")
            while rclpy.ok() and (self.latest_tcp_pose is None or self.latest_joint_state is None):
                rclpy.spin_once(self, timeout_sec=0.1)

            # Wait for stability
            self.wait_until_stable()

            # Get final pose and joint state
            rclpy.spin_once(self, timeout_sec=0.1)
            pose = self.latest_tcp_pose
            js = self.latest_joint_state

            self.get_logger().info(
                f"Recording corner {i+1}: "
                f"position=({pose.pose.position.x:.3f}, "
                f"{pose.pose.position.y:.3f}, "
                f"{pose.pose.position.z:.3f})"
            )

            corner_data = {
                "index": i,
                "tcp_pose": {
                    "frame_id": pose.header.frame_id,
                    "position": {
                        "x": float(pose.pose.position.x),
                        "y": float(pose.pose.position.y),
                        "z": float(pose.pose.position.z),
                    },
                    "orientation": {
                        "x": float(pose.pose.orientation.x),
                        "y": float(pose.pose.orientation.y),
                        "z": float(pose.pose.orientation.z),
                        "w": float(pose.pose.orientation.w),
                    },
                },
                "joint_state": {
                    "name": list(js.name),
                    "position": [float(p) for p in js.position],
                    "velocity": [float(v) for v in js.velocity],
                    "effort": [float(e) for e in js.effort],
                },
            }

            corners.append(corner_data)

        # Save to YAML
        out_dir = os.path.dirname(self.output_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        data = {"corners": corners}
        with open(self.output_path, "w") as f:
            yaml.safe_dump(data, f)

        self.get_logger().info(f"\nSaved {len(corners)} corners to: {self.output_path}")


def main():
    rclpy.init()
    node = CornerRecorder(
        corner_count=8,
        vel_threshold=1e-3,        # |joint velocity| < 1e-3 rad/s
        stable_duration=0.5,       # stable for 0.5s
        output_path="corners.yaml" # change path if you like
    )

    try:
        node.record_corners()
    finally:
        node.get_logger().info("Shutting down CornerRecorder...")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
