import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, WrenchStamped
from visualization_msgs.msg import Marker


class ForceMarkerPublisher(Node):
    def __init__(self):
        super().__init__("force_marker_publisher")
        self.declare_parameter("wrench_topic", "/force_torque_sensor_broadcaster/wrench")
        self.declare_parameter("marker_topic", "/force_marker")
        self.declare_parameter("force_scale", 2.0)
        self.declare_parameter("arrow_shaft_diameter", 1.0)
        self.declare_parameter("arrow_head_diameter", 2.0)
        self.declare_parameter("arrow_head_length", 3.0)
        self.declare_parameter("ema_alpha", 0.2)

        wrench_topic = self.get_parameter("wrench_topic").get_parameter_value().string_value
        marker_topic = self.get_parameter("marker_topic").get_parameter_value().string_value

        self._force_scale = self.get_parameter("force_scale").get_parameter_value().double_value
        self._shaft_d = self.get_parameter("arrow_shaft_diameter").get_parameter_value().double_value
        self._head_d = self.get_parameter("arrow_head_diameter").get_parameter_value().double_value
        self._head_l = self.get_parameter("arrow_head_length").get_parameter_value().double_value
        self._ema_alpha = self.get_parameter("ema_alpha").get_parameter_value().double_value
        self._eps = 1e-6

        self._latest = None
        self._ema_wrench = None
        self._sub = self.create_subscription(WrenchStamped, wrench_topic, self._on_wrench, 10)
        self._pub = self.create_publisher(Marker, marker_topic, 10)
        self._timer = self.create_timer(0.05, self._publish_marker)

        self.get_logger().info(f"Listening to: {wrench_topic}")
        self.get_logger().info(f"Publishing marker: {marker_topic}")

    def _estimate_contact_z(self, fx, fy, tx, ty):
        z_candidates = []
        if abs(fy) > self._eps:
            z_candidates.append(tx / fy)
        if abs(fx) > self._eps:
            z_candidates.append(-ty / fx)
        if not z_candidates:
            return 0.0
        return sum(z_candidates) / len(z_candidates)

    def _on_wrench(self, msg: WrenchStamped):
        if self._ema_wrench is None:
            self._ema_wrench = {
                "fx": msg.wrench.force.x,
                "fy": msg.wrench.force.y,
                "fz": msg.wrench.force.z,
                "tx": msg.wrench.torque.x,
                "ty": msg.wrench.torque.y,
                "tz": msg.wrench.torque.z,
            }
        else:
            a = self._ema_alpha
            self._ema_wrench["fx"] = a * msg.wrench.force.x + (1.0 - a) * self._ema_wrench["fx"]
            self._ema_wrench["fy"] = a * msg.wrench.force.y + (1.0 - a) * self._ema_wrench["fy"]
            self._ema_wrench["fz"] = a * msg.wrench.force.z + (1.0 - a) * self._ema_wrench["fz"]
            self._ema_wrench["tx"] = a * msg.wrench.torque.x + (1.0 - a) * self._ema_wrench["tx"]
            self._ema_wrench["ty"] = a * msg.wrench.torque.y + (1.0 - a) * self._ema_wrench["ty"]
            self._ema_wrench["tz"] = a * msg.wrench.torque.z + (1.0 - a) * self._ema_wrench["tz"]
        self._latest = msg

    def _publish_marker(self):
        if self._latest is None:
            return

        msg = self._latest
        if self._ema_wrench is None:
            return
        fx = self._ema_wrench["fx"]
        fy = self._ema_wrench["fy"]
        fz = self._ema_wrench["fz"]
        tx = self._ema_wrench["tx"]
        ty = self._ema_wrench["ty"]

        mag = math.sqrt(fx * fx + fy * fy + fz * fz)
        length = mag * self._force_scale
        contact_z = self._estimate_contact_z(fx, fy, tx, ty)

        marker = Marker()
        marker.header = msg.header
        marker.ns = "force_vector"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        start = Point()
        start.x = 0.0
        start.y = 0.0
        start.z = contact_z
        end = Point()
        if mag > 1e-6:
            end.x = start.x + fx / mag * length
            end.y = start.y + fy / mag * length
            end.z = start.z + fz / mag * length
        else:
            end.x = start.x
            end.y = start.y
            end.z = start.z
        marker.points = [start, end]

        marker.scale.x = self._shaft_d
        marker.scale.y = self._head_d
        marker.scale.z = self._head_l

        marker.color.r = 1.0
        marker.color.g = 0.2
        marker.color.b = 0.0
        marker.color.a = 0.9

        self._pub.publish(marker)

        text = Marker()
        text.header = msg.header
        text.ns = "contact_point"
        text.id = 1
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = start.x
        text.pose.position.y = start.y
        text.pose.position.z = start.z + 0.03
        text.scale.z = 0.03
        text.color.r = 0.1
        text.color.g = 0.9
        text.color.b = 0.1
        text.color.a = 0.9
        text.text = f"z={contact_z:.4f}"
        self._pub.publish(text)


def main():
    rclpy.init()
    node = ForceMarkerPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
