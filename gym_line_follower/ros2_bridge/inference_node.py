"""
ROS2 inference node: runs a trained SB3 PPO policy on a physical TurtleBot3.

Subscribes: sensor_msgs/Image  (downward camera)
Publishes:  geometry_msgs/Twist (/cmd_vel)

Preprocessing matches training exactly:
  BGR → grayscale → resize (84, 84) → frame stack of 4 → model.predict()
"""
from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def _require_ros2():
    try:
        import rclpy  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "inference_node requires ROS2 (rclpy). Source your ROS2 Humble workspace first."
        ) from e


class TurtleBot3InferenceNode:  # pragma: no cover (ROS2 runtime)
    """
    Loads a trained PPO model and runs it on the real robot at camera rate.

    Safety: if no image arrives within timeout_s seconds, publishes zero velocity
    and keeps doing so until images resume.
    """

    STACK_SIZE = 4
    IMG_H = 84
    IMG_W = 84

    def __init__(
        self,
        model_path: str,
        vecnorm_path: str,
        camera_topic: str = "/camera/down/image_raw",
        cmd_vel_topic: str = "/cmd_vel",
        timeout_s: float = 0.5,
    ):
        _require_ros2()
        import rclpy
        from rclpy.node import Node
        from geometry_msgs.msg import Twist
        from sensor_msgs.msg import Image

        self._rclpy = rclpy
        self._Twist = Twist

        class _Node(Node):
            pass

        self.node = _Node("turtlebot3_inference")
        self.timeout_s = timeout_s
        self._last_image_time: float = 0.0
        self._frame_stack: deque = deque(maxlen=self.STACK_SIZE)
        self._ready = False  # True once stack is full for the first time

        self._model, self._vecnorm = self._load_model(model_path, vecnorm_path)
        self.node.get_logger().info(f"Model loaded from {model_path}")

        self._pub_cmd = self.node.create_publisher(Twist, cmd_vel_topic, 10)
        self._sub_img = self.node.create_subscription(Image, camera_topic, self._on_image, 10)
        self._safety_timer = self.node.create_timer(0.1, self._safety_check)

        self.node.get_logger().info(
            f"Inference node ready. camera={camera_topic} cmd_vel={cmd_vel_topic}"
        )

    def _load_model(self, model_path: str, vecnorm_path: str):
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        import gymnasium as gym

        # Dummy env with matching obs/action spaces so VecNormalize can load.
        # We never actually step this env — it only provides space definitions.
        import gym_line_follower  # noqa: F401

        def _dummy():
            import numpy as _np
            env = gym.make("TurtleBot3LineFollower-v0", gui=False)
            env = gym.wrappers.GrayscaleObservation(env, keep_dim=False)
            env = gym.wrappers.ResizeObservation(env, shape=(self.IMG_H, self.IMG_W))
            env = gym.wrappers.FrameStackObservation(env, stack_size=self.STACK_SIZE)
            return env

        venv = DummyVecEnv([_dummy])
        vecnorm = VecNormalize.load(vecnorm_path, venv)
        vecnorm.training = False
        vecnorm.norm_reward = False

        model = PPO.load(model_path, env=vecnorm)
        return model, vecnorm

    def _preprocess(self, img_msg) -> Optional[np.ndarray]:
        h, w = img_msg.height, img_msg.width
        enc = img_msg.encoding.lower()

        raw = np.frombuffer(bytes(img_msg.data), dtype=np.uint8)

        if enc in ("rgb8", "rgb"):
            bgr = cv2.cvtColor(raw.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
        elif enc in ("bgr8", "bgr"):
            bgr = raw.reshape(h, w, 3)
        elif enc in ("mono8", "8uc1"):
            bgr = raw.reshape(h, w)
        else:
            self.node.get_logger().warn(f"Unsupported encoding: {enc}", throttle_duration_sec=5.0)
            return None

        if bgr.ndim == 3:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = bgr

        resized = cv2.resize(gray, (self.IMG_W, self.IMG_H), interpolation=cv2.INTER_AREA)
        return resized  # (84, 84) uint8

    def _on_image(self, msg):
        self._last_image_time = time.monotonic()

        frame = self._preprocess(msg)
        if frame is None:
            return

        self._frame_stack.append(frame)
        if len(self._frame_stack) < self.STACK_SIZE:
            return  # not enough frames yet

        if not self._ready:
            self._ready = True
            self.node.get_logger().info("Frame stack full — starting inference.")

        obs = np.stack(list(self._frame_stack), axis=0)[np.newaxis]  # (1, 4, 84, 84)
        obs = self._vecnorm.normalize_obs(obs)

        action, _ = self._model.predict(obs, deterministic=True)
        vx = float(action[0][0])
        wz = float(action[0][1])

        twist = self._Twist()
        twist.linear.x = vx
        twist.angular.z = wz
        self._pub_cmd.publish(twist)

    def _safety_check(self):
        if not self._ready:
            return
        age = time.monotonic() - self._last_image_time
        if age > self.timeout_s:
            self._pub_cmd.publish(self._Twist())  # zero velocity
            self.node.get_logger().warn(
                f"No camera image for {age:.2f}s — robot stopped.",
                throttle_duration_sec=1.0,
            )

    def spin(self):
        self._rclpy.spin(self.node)

    def shutdown(self):
        self._pub_cmd.publish(self._Twist())  # stop robot on exit
        try:
            self.node.destroy_node()
        except Exception:
            pass


def main(args=None) -> int:  # pragma: no cover
    _require_ros2()
    import rclpy

    parser = argparse.ArgumentParser(description="Run trained PPO policy on a real TurtleBot3.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to .zip model file.")
    parser.add_argument("--vecnorm-path", type=str, required=True, help="Path to VecNormalize .pkl file.")
    parser.add_argument("--camera-topic", type=str, default="/camera/down/image_raw")
    parser.add_argument("--cmd-vel-topic", type=str, default="/cmd_vel")
    parser.add_argument("--timeout", type=float, default=0.5, help="Seconds without image before stopping.")
    parsed, _ = parser.parse_known_args(args)

    rclpy.init(args=args)
    node = TurtleBot3InferenceNode(
        model_path=parsed.model_path,
        vecnorm_path=parsed.vecnorm_path,
        camera_topic=parsed.camera_topic,
        cmd_vel_topic=parsed.cmd_vel_topic,
        timeout_s=parsed.timeout,
    )
    try:
        node.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
