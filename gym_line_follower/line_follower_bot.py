from typing import Optional
import os

import numpy as np
import pybullet as p

from .reference_geometry import CameraWindow, ReferencePoint
from .line_interpolation import sort_points, interpolate_points
from .dc_motor import DCMotor
from .track import Track

DEFAULT_URDF = "follower_bot.urdf"
DEFAULT_WHEEL_JOINT_NAMES = {"left": "chassis_to_left_wheel", "right": "chassis_to_right_wheel"}
MOTOR_DIRECTIONS = {"left": -1, "right": -1}


class LineFollowerBot:
    """
    Class simulating a line following bot with differential steering.
    """
    SUPPORTED_OBSV_TYPE = ["points_visible", "points_latch", "points_latch_bool", "camera", "down_camera"]

    def __init__(
        self,
        pb_client,
        nb_cam_points,
        start_xy,
        start_yaw,
        config,
        obsv_type="visible",
        *,
        action_mode: str = "wheel_power",
    ):
        """
        Initialize bot.
        :param pb_client: pybullet client for simulation interfacing
        :param nb_cam_points: number of points describing line
        :param start_xy: starting x, y coordinates
        :param start_yaw: starting yaw
        :param config: configuration dictionary
        :param obsv_type: type of line observation generated:
                            "points_visible" - returns array shape (nb_cam_pts, 3)  - each line point has 3 parameters
                                [x, y, visibility] where visibility is 1.0 if point is visible in camera window
                                and 0.0 if not.
                            "points_latch" - returns array length nb_cam_points if at least 2 line points are visible in
                                camera window, returns empty array otherwise
                            "points_latch_bool" - same as "latch", se LineFollowerEnv implementation
                            "camera" - return (240, 320, 3) camera image RGB array
        """
        self.local_dir = os.path.dirname(__file__)
        self.config = config

        self.pb_client: p = pb_client
        self.bot = None
        self.wheel_joint_indices = {"left": None, "right": None}
        self.action_mode = action_mode.lower()
        if self.action_mode not in {"wheel_power", "cmd_vel"}:
            raise ValueError(f"Unsupported action_mode '{action_mode}'. Expected 'wheel_power' or 'cmd_vel'.")

        self.prev_pos = ((0., 0.), 0.)
        self.pos = ((0., 0.), 0.)

        self.prev_vel = ((0., 0.), 0.)
        self.vel = ((0., 0.), 0.)

        self.cam_window: CameraWindow = None
        self.nb_cam_pts = nb_cam_points

        self.track_ref_point: ReferencePoint = None
        self.cam_target_point: ReferencePoint = None  # POV Camera target point
        self.cam_pos_point: ReferencePoint = None  # POC Camera position

        self.cam_indicator = None  # PyBullet body ID for down-camera visual indicator

        self.volts = 0.

        self.left_motor: DCMotor = None
        self.right_motor: DCMotor = None

        self.obsv_type = obsv_type.lower()
        if self.obsv_type not in self.SUPPORTED_OBSV_TYPE:
            raise ValueError("Observation type '{}' not supported.".format(self.obsv_type))

        self.reset(start_xy, start_yaw)

    def _resolve_urdf_path(self):
        urdf = self.config.get("robot_urdf", DEFAULT_URDF)
        # allow absolute paths
        if os.path.isabs(urdf):
            return urdf
        return os.path.join(self.local_dir, urdf)

    def _find_joint_index(self, joint_name: str) -> Optional[int]:
        if not joint_name:
            return None
        try:
            n = self.pb_client.getNumJoints(self.bot)
        except Exception:
            return None
        for idx in range(n):
            info = self.pb_client.getJointInfo(self.bot, idx)
            name = info[1].decode("utf-8") if isinstance(info[1], (bytes, bytearray)) else str(info[1])
            if name == joint_name:
                return idx
        return None

    def _init_wheel_joints(self):
        # Preferred: resolve by joint names in config.
        left_name = self.config.get("wheel_joint_left_name") or DEFAULT_WHEEL_JOINT_NAMES["left"]
        right_name = self.config.get("wheel_joint_right_name") or DEFAULT_WHEEL_JOINT_NAMES["right"]

        li = self._find_joint_index(left_name)
        ri = self._find_joint_index(right_name)

        # Fallback: preserve legacy behavior if names don't match.
        if li is None or ri is None:
            li = 1
            ri = 2

        self.wheel_joint_indices["left"] = li
        self.wheel_joint_indices["right"] = ri

    def reset(self, xy, yaw):
        """
        Reload bot urdf, reposition bot, reinitialize camera window and other stuff.
        :param xy: starting xy coords
        :param yaw: starting yaw
        :return: None
        """
        self.bot = self.pb_client.loadURDF(
            self._resolve_urdf_path(),
            basePosition=[*xy, 0.0],
            baseOrientation=self.pb_client.getQuaternionFromEuler([0.0, 0.0, yaw]),
        )
        self._init_wheel_joints()
        self.pos = xy, yaw

        # Camera visual indicator (down_camera only) — remove old one first, then recreate.
        if self.cam_indicator is not None:
            try:
                self.pb_client.removeBody(self.cam_indicator)
            except Exception:
                pass
            self.cam_indicator = None

        cam_cfg = self.config.get("down_camera")
        if cam_cfg is not None:
            pos_in_base = np.asarray(cam_cfg.get("pos_in_base", [0.08, 0.0, 0.12]), dtype=np.float32)
            base_pos, base_orn = self.pb_client.getBasePositionAndOrientation(self.bot)
            rot = np.asarray(self.pb_client.getMatrixFromQuaternion(base_orn), dtype=np.float32).reshape(3, 3)
            cam_world_pos = np.asarray(base_pos, dtype=np.float32) + rot @ pos_in_base
            vis_shape = self.pb_client.createVisualShape(
                shapeType=self.pb_client.GEOM_SPHERE,
                radius=0.015,
                rgbaColor=[1.0, 0.5, 0.0, 1.0],  # orange
            )
            self.cam_indicator = self.pb_client.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=vis_shape,
                basePosition=cam_world_pos.tolist(),
            )

        # Reference geometry used by point-based observations and legacy POV camera.
        # Provide sane defaults so other render modes don't crash when using different configs.
        h = float(self.config.get("camera_window_height", 0.160))
        wt = float(self.config.get("camera_window_top_width", 0.270))
        wb = float(self.config.get("camera_window_bottom_width", 0.120))
        d = float(self.config.get("camera_window_distance", 0.112))
        win_points = [(d + h, wt / 2), (d + h, -wt / 2), (d, -wb / 2), (d, wb / 2)]

        self.cam_window = CameraWindow(win_points)
        self.cam_window.move(xy, yaw)

        tref_pt_x = float(self.config.get("track_ref_point_x", d))
        self.track_ref_point = ReferencePoint(xy_shift=(tref_pt_x, 0.0))
        self.track_ref_point.move(xy, yaw)

        cam_target_pt_x = float(self.config.get("camera_target_point_x", d + h))
        self.cam_target_point = ReferencePoint(xy_shift=(cam_target_pt_x, 0.0))
        self.cam_target_point.move(xy, yaw)

        cam_pos_pt_x = float(self.config.get("camera_position_point_x", 0.060))
        self.cam_pos_point = ReferencePoint(xy_shift=(cam_pos_pt_x, 0.0))
        self.cam_pos_point.move(xy, yaw)

        if self.action_mode == "wheel_power":
            nom_volt = float(self.config["motor_nominal_voltage"])
            no_load_speed = float(self.config["motor_no_load_speed"])
            stall_torque = float(self.config["motor_stall_torque"])
            self.left_motor = DCMotor(nom_volt, no_load_speed, stall_torque)
            self.right_motor = DCMotor(nom_volt, no_load_speed, stall_torque)
            self.volts = float(self.config["volts"])
        else:
            self.left_motor = None
            self.right_motor = None
            self.volts = 0.0

        # Disable default joint motors prior to explicit control
        self.pb_client.setJointMotorControl2(
            bodyIndex=self.bot,
            jointIndex=self.wheel_joint_indices["left"],
            controlMode=self.pb_client.VELOCITY_CONTROL,
            force=0,
        )
        self.pb_client.setJointMotorControl2(
            bodyIndex=self.bot,
            jointIndex=self.wheel_joint_indices["right"],
            controlMode=self.pb_client.VELOCITY_CONTROL,
            force=0,
        )

    def get_position(self):
        position, orientation = self.pb_client.getBasePositionAndOrientation(self.bot)
        x, y, z = position
        orientation = self.pb_client.getEulerFromQuaternion(orientation)
        pitch, roll, yaw = orientation
        return (x, y), yaw

    def _update_position_velocity(self):
        new_xy, new_yaw = self.get_position()
        self.cam_window.move(new_xy, new_yaw)
        self.track_ref_point.move(new_xy, new_yaw)
        self.cam_target_point.move(new_xy, new_yaw)
        self.cam_pos_point.move(new_xy, new_yaw)
        self.prev_pos = self.pos
        self.prev_vel = self.vel
        self.pos = new_xy, new_yaw
        self.vel = self.get_velocity()

        if self.cam_indicator is not None:
            cam_cfg = self.config.get("down_camera", {})
            pos_in_base = np.asarray(cam_cfg.get("pos_in_base", [0.08, 0.0, 0.12]), dtype=np.float32)
            base_pos, base_orn = self.pb_client.getBasePositionAndOrientation(self.bot)
            rot = np.asarray(self.pb_client.getMatrixFromQuaternion(base_orn), dtype=np.float32).reshape(3, 3)
            cam_world_pos = np.asarray(base_pos, dtype=np.float32) + rot @ pos_in_base
            self.pb_client.resetBasePositionAndOrientation(
                self.cam_indicator, cam_world_pos.tolist(), [0.0, 0.0, 0.0, 1.0]
            )

    def get_velocity(self):
        linear, angular = self.pb_client.getBaseVelocity(self.bot)
        vx, vy, vz = linear
        wx, wy, wz = angular
        return (vx, vy), wz

    def _get_wheel_velocity(self):
        l_pos, l_vel, l_react, l_torque = self.pb_client.getJointState(self.bot, self.wheel_joint_indices["left"])
        r_pos, r_vel, r_react, r_torque = self.pb_client.getJointState(self.bot, self.wheel_joint_indices["right"])
        return l_vel, r_vel

    def step(self, track: Track):
        """
        Should be called after simulation step.
        Update camera window and other reference geometry, generate observation.
        :param track: Track object
        :return: observation, according to self.obsv_type
        """
        self._update_position_velocity()
        visible_pts = self.cam_window.visible_points(track.mpt)

        if self.obsv_type == "points_visible":
            if len(visible_pts) > 0:
                pts = self.cam_window.convert_points_to_local(visible_pts)
                pts = sort_points(pts, origin=self.track_ref_point.get_xy())
                pts = interpolate_points(pts, segment_length=0.025)
            else:
                pts = np.zeros((0, 2))

            observation = np.zeros((self.nb_cam_pts, 3), dtype=np.float32)
            for i in range(self.nb_cam_pts):
                try:
                    x, y = pts[i]
                except IndexError:
                    x = np.random.uniform(0.0, 0.2)
                    y = np.random.uniform(-0.2, 0.2)
                    vis = 0.0
                else:
                    vis = 1.0
                observation[i] = [x, y, vis]
            observation = observation.flatten().tolist()
            return observation

        elif self.obsv_type in ["points_latch", "points_latch_bool"]:
            if len(visible_pts) > 0:
                visible_pts_local = self.cam_window.convert_points_to_local(visible_pts)
                visible_pts_local = sort_points(visible_pts_local)
                visible_pts_local = interpolate_points(visible_pts_local, self.nb_cam_pts)
                if len(visible_pts_local) > 1:
                    observation = visible_pts_local.flatten().tolist()
                    return observation
                else:
                    return []
            else:
                return []

        elif self.obsv_type == "camera":
            return self.get_pov_image()

        elif self.obsv_type == "down_camera":
            return self.get_down_camera_image()

    def _set_wheel_torque(self, l_torque, r_torque):
        """
        Apply torque to simulated wheels.
        :param l_torque: left wheel torque in Nm
        :param r_torque: right wheel torque in Nm
        """
        l_torque *= MOTOR_DIRECTIONS["left"]
        r_torque *= MOTOR_DIRECTIONS["right"]
        self.pb_client.setJointMotorControl2(self.bot,
                                             jointIndex=self.wheel_joint_indices["left"],
                                             controlMode=self.pb_client.TORQUE_CONTROL,
                                             force=l_torque)
        self.pb_client.setJointMotorControl2(self.bot,
                                             jointIndex=self.wheel_joint_indices["right"],
                                             controlMode=self.pb_client.TORQUE_CONTROL,
                                             force=r_torque)

    def _power_to_volts(self, l_pow, r_pow):
        """
        Convert power to volts
        :param l_pow:
        :param r_pow:
        :return:
        """
        l_pow = np.clip(l_pow, -1., 1.)
        r_pow = np.clip(r_pow, -1., 1.)
        return l_pow * self.volts, r_pow * self.volts

    def apply_action(self, action):
        """
        Apply action to the simulated base.

        - wheel_power: action = (left_power, right_power) in [-1, 1] (legacy)
        - cmd_vel: action = (vx, wz) in [m/s, rad/s]

        motor_noise_std (from config) adds Gaussian noise scaled to the command magnitude,
        simulating actuator uncertainty for sim-to-real transfer.
        """
        motor_noise_std = float(self.config.get("motor_noise_std", 0.0))

        if self.action_mode == "wheel_power":
            l_volts, r_volts = self._power_to_volts(*action)
            if motor_noise_std > 0.0:
                l_volts += np.random.normal(0.0, motor_noise_std * self.volts)
                r_volts += np.random.normal(0.0, motor_noise_std * self.volts)
            l_vel, r_vel = self._get_wheel_velocity()
            l_vel *= MOTOR_DIRECTIONS["left"]
            r_vel *= MOTOR_DIRECTIONS["right"]
            l_torque = self.left_motor.get_torque(l_volts, l_vel)
            r_torque = self.right_motor.get_torque(r_volts, r_vel)
            self._set_wheel_torque(l_torque, r_torque)
            return

        if self.action_mode == "cmd_vel":
            vx, wz = float(action[0]), float(action[1])
            if motor_noise_std > 0.0:
                vx_lim = float(self.config.get("cmd_vel_vx_limit", 0.22))
                wz_lim = float(self.config.get("cmd_vel_wz_limit", 2.84))
                vx += np.random.normal(0.0, motor_noise_std * vx_lim)
                wz += np.random.normal(0.0, motor_noise_std * wz_lim)
            r = float(self.config.get("wheel_radius", 0.033))
            L = float(self.config.get("wheel_separation", 0.16))
            max_force = float(self.config.get("max_wheel_force", 2.0))

            wl = (vx - (wz * L / 2.0)) / r
            wr = (vx + (wz * L / 2.0)) / r

            # Respect legacy motor direction flags.
            wl *= MOTOR_DIRECTIONS["left"]
            wr *= MOTOR_DIRECTIONS["right"]

            self.pb_client.setJointMotorControl2(
                bodyIndex=self.bot,
                jointIndex=self.wheel_joint_indices["left"],
                controlMode=self.pb_client.VELOCITY_CONTROL,
                targetVelocity=wl,
                force=max_force,
            )
            self.pb_client.setJointMotorControl2(
                bodyIndex=self.bot,
                jointIndex=self.wheel_joint_indices["right"],
                controlMode=self.pb_client.VELOCITY_CONTROL,
                targetVelocity=wr,
                force=max_force,
            )
            return

        raise RuntimeError(f"Unhandled action_mode {self.action_mode!r}")

    def get_pov_image(self):
        """
        Render virtual camera image.
        :return: RGB Array shape (240, 320, 3)
        """
        cam_x, cam_y = self.cam_pos_point.get_xy()
        cam_z = 0.095
        target_x, target_y = self.cam_target_point.get_xy()
        vm = self.pb_client.computeViewMatrix(cameraEyePosition=[cam_x, cam_y, cam_z],
                                              cameraTargetPosition=[target_x, target_y, 0.0],
                                              cameraUpVector=[0.0, 0.0, 1.0])
        pm = self.pb_client.computeProjectionMatrixFOV(fov=49,
                                                       aspect=320 / 240,
                                                       nearVal=0.0001,
                                                       farVal=1)
        w, h, rgb, deth, seg = self.pb_client.getCameraImage(width=320,
                                                             height=240,
                                                             viewMatrix=vm,
                                                             projectionMatrix=pm,
                                                             renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgb = np.asarray(rgb, dtype=np.uint8)
        if rgb.ndim == 1:
            rgb = rgb.reshape((h, w, 4))
        rgb = rgb[:, :, :3]
        return rgb

    def get_down_camera_image(self):
        """
        Render a downward-facing camera image (intended to approximate a Raspberry Pi camera).
        Camera intrinsics are approximated via a pinhole projection with FOV.
        Camera extrinsics are defined as a transform in the robot base frame (config key 'down_camera').
        """
        cam_cfg = self.config.get("down_camera", {})
        width = int(cam_cfg.get("width", 160))
        height = int(cam_cfg.get("height", 120))
        fov = float(cam_cfg.get("fov", 90))
        near = float(cam_cfg.get("near", 0.01))
        far = float(cam_cfg.get("far", 1.5))

        pos_in_base = np.asarray(cam_cfg.get("pos_in_base", [0.08, 0.0, 0.12]), dtype=np.float32)
        rpy_in_base = np.asarray(cam_cfg.get("rpy_in_base", [np.pi, 0.0, 0.0]), dtype=np.float32)

        pitch_noise = float(self.config.get("camera_pitch_noise", 0.0))
        roll_noise = float(self.config.get("camera_roll_noise", 0.0))
        height_noise = float(self.config.get("camera_height_noise", 0.0))
        rpy_in_base = rpy_in_base + np.array([roll_noise, pitch_noise, 0.0], dtype=np.float32)
        pos_in_base = pos_in_base + np.array([0.0, 0.0, height_noise], dtype=np.float32)

        base_pos, base_orn = self.pb_client.getBasePositionAndOrientation(self.bot)
        base_pos = np.asarray(base_pos, dtype=np.float32)

        # Rotate offset and orientation by base orientation.
        rot = np.asarray(self.pb_client.getMatrixFromQuaternion(base_orn), dtype=np.float32).reshape(3, 3)
        cam_pos = base_pos + rot @ pos_in_base

        cam_orn_local = self.pb_client.getQuaternionFromEuler(rpy_in_base.tolist())
        cam_orn_world = self.pb_client.multiplyTransforms([0, 0, 0], base_orn, [0, 0, 0], cam_orn_local)[1]
        cam_rot_world = np.asarray(self.pb_client.getMatrixFromQuaternion(cam_orn_world), dtype=np.float32).reshape(3, 3)

        # In camera frame, look along +X with up +Z (matches typical Bullet conventions for view matrix usage below).
        fwd = cam_rot_world @ np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        up = cam_rot_world @ np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        target = cam_pos + fwd * 0.2

        vm = self.pb_client.computeViewMatrix(
            cameraEyePosition=cam_pos.tolist(),
            cameraTargetPosition=target.tolist(),
            cameraUpVector=up.tolist(),
        )
        pm = self.pb_client.computeProjectionMatrixFOV(
            fov=fov,
            aspect=float(width) / float(height),
            nearVal=near,
            farVal=far,
        )
        w, h, rgb, deth, seg = self.pb_client.getCameraImage(
            width=width,
            height=height,
            viewMatrix=vm,
            projectionMatrix=pm,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )
        rgb = np.asarray(rgb, dtype=np.uint8)
        if rgb.ndim == 1:
            rgb = rgb.reshape((h, w, 4))
        rgb = rgb[:, :, :3]
        return rgb