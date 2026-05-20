"""ROS 2 node that turns /human/keypoints into G1 upper_body_controller commands.

The flow per timer tick:
  1. Read the latest PoseArray on /human/keypoints.
  2. Convert to (N,3) keypoints + visibility arrays in the camera optical frame.
  3. Optionally swap left/right (mirror mode).
  4. Run geometric_ik.compute_joint_angles.
  5. For each joint we have a fresh angle for: apply sign flip, clamp to URDF
     limits, exponential-smooth with previous command.
  6. Publish the full 15-vector as std_msgs/Float64MultiArray on
     /upper_body_controller/commands in the controller's joint order.

If a frame produces no IK solution (operator out of view), the previous
command is republished unchanged.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from humanoid_pose_estimator.keypoints import KEYPOINT_COUNT
from .geometric_ik import compute_joint_angles, swap_left_right_joints

CONTROLLER_JOINT_ORDER: list[str] = [
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


class RetargeterNode(Node):
    def __init__(self) -> None:
        super().__init__("humanoid_retargeter")

        self.declare_parameter("mirror", True)
        self.declare_parameter("smoothing_alpha", 0.4)
        self.declare_parameter("command_rate_hz", 30.0)
        self.declare_parameter("keypoint_min_visibility", 0.5)
        self.declare_parameter("keypoints_topic", "/human/keypoints")
        self.declare_parameter("command_topic", "/upper_body_controller/commands")

        # Per-joint sign flips and limits — declared once with defaults, then
        # overridden by the YAML parameter file.
        self._signs: dict[str, float] = {}
        self._limits: dict[str, tuple[float, float]] = {}
        for j in CONTROLLER_JOINT_ORDER:
            self.declare_parameter(f"joint_signs.{j}", 1.0)
            self.declare_parameter(f"joint_limits.{j}.min", -3.14)
            self.declare_parameter(f"joint_limits.{j}.max", 3.14)
            self._signs[j] = float(self.get_parameter(f"joint_signs.{j}").value)
            self._limits[j] = (
                float(self.get_parameter(f"joint_limits.{j}.min").value),
                float(self.get_parameter(f"joint_limits.{j}.max").value),
            )

        gp = self.get_parameter
        self.mirror: bool = bool(gp("mirror").value)
        self.alpha: float = float(gp("smoothing_alpha").value)
        self.min_vis: float = float(gp("keypoint_min_visibility").value)
        self.rate_hz: float = float(gp("command_rate_hz").value)
        kps_topic: str = gp("keypoints_topic").value
        cmd_topic: str = gp("command_topic").value

        self._last_msg: Optional[PoseArray] = None
        # Cached commanded position per joint; None = never commanded.
        self._cmd: dict[str, Optional[float]] = {j: None for j in CONTROLLER_JOINT_ORDER}

        self.sub = self.create_subscription(
            PoseArray, kps_topic, self._on_kps, 10
        )
        self.pub = self.create_publisher(Float64MultiArray, cmd_topic, 10)
        self.timer = self.create_timer(1.0 / self.rate_hz, self._tick)

        self.get_logger().info(
            f"retargeter up; mirror={self.mirror} alpha={self.alpha} "
            f"rate={self.rate_hz} Hz, sub={kps_topic} pub={cmd_topic}"
        )

    # ------------------------------------------------------------------ helpers

    def _on_kps(self, msg: PoseArray) -> None:
        self._last_msg = msg

    def _decode_msg(self, msg: PoseArray) -> tuple[np.ndarray, np.ndarray]:
        n = KEYPOINT_COUNT
        if len(msg.poses) < n:
            # Pad if upstream published less for some reason.
            poses = list(msg.poses) + [None] * (n - len(msg.poses))
        else:
            poses = msg.poses
        kps = np.zeros((n, 3), dtype=np.float64)
        vis = np.zeros((n,), dtype=np.float64)
        for i, p in enumerate(poses):
            if p is None:
                continue
            kps[i, 0] = p.position.x
            kps[i, 1] = p.position.y
            kps[i, 2] = p.position.z
            # Pose estimator encodes visibility in orientation.w (see node).
            vis[i] = float(p.orientation.w)
        return kps, vis

    def _apply_one(self, joint: str, raw_angle: float) -> float:
        a = self._signs.get(joint, 1.0) * raw_angle
        lo, hi = self._limits.get(joint, (-3.14, 3.14))
        a = float(np.clip(a, lo, hi))
        prev = self._cmd[joint]
        if prev is None:
            blended = a
        else:
            # Exponential smoothing: alpha=0 -> instant (no smoothing).
            blended = (1.0 - self.alpha) * a + self.alpha * prev
        self._cmd[joint] = blended
        return blended

    # ------------------------------------------------------------------ tick

    def _tick(self) -> None:
        if self._last_msg is None:
            return
        kps, vis = self._decode_msg(self._last_msg)

        result = compute_joint_angles(kps, vis, min_vis=self.min_vis)

        angles = result.angles
        if self.mirror:
            angles = swap_left_right_joints(angles)

        # Apply each newly computed joint; hold last for the rest.
        for joint, val in angles.items():
            if joint not in self._cmd:
                continue
            self._apply_one(joint, val)

        # Wrists are not driven; force them to zero (smoothed).
        for joint in CONTROLLER_JOINT_ORDER:
            if "wrist" in joint and joint not in result.angles:
                self._apply_one(joint, 0.0)

        # If we still have None values (never seen a valid frame), hold at 0.
        msg = Float64MultiArray()
        msg.data = [
            self._cmd[j] if self._cmd[j] is not None else 0.0
            for j in CONTROLLER_JOINT_ORDER
        ]
        self.pub.publish(msg)


def main(argv: list[str] | None = None) -> None:
    rclpy.init(args=argv)
    node = RetargeterNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
