from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class BridgeConfig:
    env_id: str = "TurtleBot3LineFollower-v0"
    gui: bool = False
    publish_tf: bool = True
    publish_odom: bool = True
    publish_camera: bool = True
    camera_topic: str = "/camera/down/image_raw"
    cmd_vel_topic: str = "/cmd_vel"
    odom_topic: str = "/odom"
    odom_frame: str = "odom"
    base_frame: str = "base_link"


def _require_ros2():
    try:
        import rclpy  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "ROS2 bridge requires ROS2 (rclpy) to be installed and sourced.\n"
            "Gym usage does not require ROS2.\n"
            "If you are on Windows, ROS2 Humble/Iron must be installed and your environment sourced."
        ) from e


def _yaw_to_quat(yaw: float):
    # Z-only yaw quaternion
    import math

    half = yaw * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))


class LineFollowerRos2Bridge:  # pragma: no cover (ROS2 runtime)
    """
    Runs the Gym environment as a ROS2 sim:
    - subscribes: geometry_msgs/Twist on /cmd_vel
    - publishes: sensor_msgs/Image (downward camera), nav_msgs/Odometry, optional TF

    ROS2 is separate from training. Training uses Gym directly.
    Deployment can reuse the same policy by writing a ROS2 node that consumes the same observation and publishes /cmd_vel.
    """

    def __init__(self, cfg: BridgeConfig):
        _require_ros2()
        import gymnasium as gym
        import rclpy
        from rclpy.node import Node
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import Image

        try:
            from tf2_ros import TransformBroadcaster
            from geometry_msgs.msg import TransformStamped
        except Exception:
            TransformBroadcaster = None
            TransformStamped = None

        import gym_line_follower  # noqa: F401 (register envs)

        self._rclpy = rclpy
        self._Twist = Twist
        self._Odometry = Odometry
        self._Image = Image
        self._TransformBroadcaster = TransformBroadcaster
        self._TransformStamped = TransformStamped

        class _BridgeNode(Node):
            pass

        self.node = _BridgeNode("gym_line_follower_bridge")

        self.cfg = cfg
        self.env = gym.make(cfg.env_id, gui=cfg.gui)
        self.env.reset()

        # Infer update rate from env configuration if present.
        sim_time_step = float(getattr(self.env.unwrapped, "sim_time_step", 1 / 250))
        sub_steps = int(getattr(self.env.unwrapped, "sub_steps", 10))
        self.dt = sim_time_step * sub_steps

        self.latest_cmd_vel = (0.0, 0.0)  # (vx, wz)

        self.sub_cmd = self.node.create_subscription(Twist, cfg.cmd_vel_topic, self._on_cmd_vel, 10)
        self.pub_img = self.node.create_publisher(Image, cfg.camera_topic, 10) if cfg.publish_camera else None
        self.pub_odom = self.node.create_publisher(Odometry, cfg.odom_topic, 10) if cfg.publish_odom else None
        self.tf_broadcaster = TransformBroadcaster(self.node) if (cfg.publish_tf and TransformBroadcaster) else None

        self.timer = self.node.create_timer(self.dt, self._tick)

        self.node.get_logger().info(
            f"Bridge started. env_id={cfg.env_id} dt={self.dt:.4f}s "
            f"sub:{cfg.cmd_vel_topic} pub:{cfg.camera_topic},{cfg.odom_topic}"
        )

    def _on_cmd_vel(self, msg):
        vx = float(msg.linear.x)
        wz = float(msg.angular.z)
        self.latest_cmd_vel = (vx, wz)

    def _publish_image(self, rgb: np.ndarray):
        if self.pub_img is None:
            return
        msg = self._Image()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = self.cfg.base_frame
        msg.height = int(rgb.shape[0])
        msg.width = int(rgb.shape[1])
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        msg.step = int(rgb.shape[1] * 3)
        msg.data = rgb.astype(np.uint8).tobytes()
        self.pub_img.publish(msg)

    def _publish_odom_and_tf(self, x: float, y: float, yaw: float):
        stamp = self.node.get_clock().now().to_msg()
        qx, qy, qz, qw = _yaw_to_quat(yaw)

        if self.pub_odom is not None:
            msg = self._Odometry()
            msg.header.stamp = stamp
            msg.header.frame_id = self.cfg.odom_frame
            msg.child_frame_id = self.cfg.base_frame
            msg.pose.pose.position.x = float(x)
            msg.pose.pose.position.y = float(y)
            msg.pose.pose.position.z = 0.0
            msg.pose.pose.orientation.x = qx
            msg.pose.pose.orientation.y = qy
            msg.pose.pose.orientation.z = qz
            msg.pose.pose.orientation.w = qw
            self.pub_odom.publish(msg)

        if self.tf_broadcaster is not None and self._TransformStamped is not None:
            tfm = self._TransformStamped()
            tfm.header.stamp = stamp
            tfm.header.frame_id = self.cfg.odom_frame
            tfm.child_frame_id = self.cfg.base_frame
            tfm.transform.translation.x = float(x)
            tfm.transform.translation.y = float(y)
            tfm.transform.translation.z = 0.0
            tfm.transform.rotation.x = qx
            tfm.transform.rotation.y = qy
            tfm.transform.rotation.z = qz
            tfm.transform.rotation.w = qw
            self.tf_broadcaster.sendTransform(tfm)

    def _tick(self):
        action = np.asarray(self.latest_cmd_vel, dtype=np.float32)
        obs, reward, terminated, truncated, info = self.env.step(action)

        if isinstance(obs, np.ndarray) and obs.ndim == 3 and obs.shape[2] == 3:
            self._publish_image(obs)

        x = float(info.get("x", 0.0))
        y = float(info.get("y", 0.0))
        yaw = float(info.get("yaw", 0.0))
        self._publish_odom_and_tf(x, y, yaw)

        if bool(terminated or truncated):
            self.env.reset()

    def spin(self):
        self._rclpy.spin(self.node)

    def shutdown(self):
        try:
            self.env.close()
        except Exception:
            pass
        try:
            self.node.destroy_node()
        except Exception:
            pass


def main(args: Optional[list[str]] = None) -> int:  # pragma: no cover
    _require_ros2()
    import rclpy

    rclpy.init(args=args)
    bridge = LineFollowerRos2Bridge(BridgeConfig())
    try:
        bridge.spin()
    except KeyboardInterrupt:
        pass
    finally:
        bridge.shutdown()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

