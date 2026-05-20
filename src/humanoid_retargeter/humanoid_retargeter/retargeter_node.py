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

import math
import time
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseArray
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Float64MultiArray

from humanoid_pose_estimator.keypoints import KEYPOINT_COUNT
from .geometric_ik import (
    compute_head_yaw,
    compute_joint_angles,
    swap_left_right_joints,
)

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
        # Fraction of the operator's head yaw to add into waist_yaw.  The
        # G1's URDF has no neck joint, so the only way for the robot's
        # head to follow the operator's head is to spin the whole torso.
        # 0.0 = ignore head yaw entirely (torso-twist-only, original
        # behaviour).  1.0 = waist follows head 1:1 (the robot's body
        # turns whenever you turn your head).  Default 0.7 keeps both
        # signals: head dominates, but torso-twist still adds in.
        self.declare_parameter("head_to_waist_gain", 0.7)
        # Sign flip for the head_yaw signal (camera mirror, operator
        # facing).  Positive head_yaw from compute_head_yaw means the
        # operator's left ear moved forward relative to their shoulders,
        # i.e. they turned their face *to their right*.  With mirror=true
        # we want the robot to turn *to its left* (operator's right side
        # of view), and waist_yaw on the G1 increases CCW about +z, which
        # is the robot's-left direction -- so +1.0 is the natural default.
        self.declare_parameter("head_yaw_sign", 1.0)
        # If >0, dump the commanded joint angles (post-sign, post-clamp,
        # post-smoothing) at this period in seconds.  Off by default;
        # turn on for IK debugging.  Example:
        #   ros2 param set /humanoid_retargeter debug_log_period_s 1.0
        self.declare_parameter("debug_log_period_s", 0.0)

        # Per-joint sign flips, offsets, and limits — declared once with
        # defaults, then overridden by the YAML parameter file.  The
        # commanded angle is ``sign * raw_ik + offset``, clamped to
        # ``[min, max]``, then exponentially smoothed.
        #
        # ``joint_offsets`` (rad) was added because the G1's kinematic
        # zero pose does not coincide with the IK's geometric zero pose:
        # the G1's URDF has the forearm extending forward (along +x in
        # elbow_link) at elbow_joint=0, but the IK is derived assuming a
        # "fully straight arm" rest where forearm continues the upper
        # arm.  These poses differ by ~pi/2 of elbow rotation, so the
        # elbow needs both a sign flip (-1) and an offset (+pi/2) for
        # the operator's anatomical rest to map to the G1's anatomical
        # rest.  Generalising the mechanism to all joints lets us absorb
        # any future URDF-zero quirks the same way without touching the
        # IK math.
        self._signs: dict[str, float] = {}
        self._offsets: dict[str, float] = {}
        self._limits: dict[str, tuple[float, float]] = {}
        for j in CONTROLLER_JOINT_ORDER:
            self.declare_parameter(f"joint_signs.{j}", 1.0)
            self.declare_parameter(f"joint_offsets.{j}", 0.0)
            self.declare_parameter(f"joint_limits.{j}.min", -3.14)
            self.declare_parameter(f"joint_limits.{j}.max", 3.14)
            self._signs[j] = float(self.get_parameter(f"joint_signs.{j}").value)
            self._offsets[j] = float(self.get_parameter(f"joint_offsets.{j}").value)
            self._limits[j] = (
                float(self.get_parameter(f"joint_limits.{j}.min").value),
                float(self.get_parameter(f"joint_limits.{j}.max").value),
            )

        gp = self.get_parameter
        self.mirror: bool = bool(gp("mirror").value)
        self.alpha: float = float(gp("smoothing_alpha").value)
        self.min_vis: float = float(gp("keypoint_min_visibility").value)
        self.rate_hz: float = float(gp("command_rate_hz").value)
        self.head_to_waist_gain: float = float(gp("head_to_waist_gain").value)
        self.head_yaw_sign: float = float(gp("head_yaw_sign").value)
        self.debug_log_period_s: float = float(gp("debug_log_period_s").value)
        self._last_debug_log_t: float = 0.0
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

        # Register a param callback so signs / limits / mirror / smoothing can
        # be live-tuned via ``ros2 param set``.  Calibrating shoulder/wrist
        # sign flips is iterative -- having to restart the demo every time you
        # want to test ``left_shoulder_roll_joint: -1.0`` vs ``+1.0`` is
        # painful (gazebo+camera bring-up alone is ~30s), so we expose every
        # knob the node owns instead.
        self.add_on_set_parameters_callback(self._on_param_set)

        self.get_logger().info(
            f"retargeter up; mirror={self.mirror} alpha={self.alpha} "
            f"rate={self.rate_hz} Hz, sub={kps_topic} pub={cmd_topic}"
        )

    def _on_param_set(self, params: list[Parameter]) -> SetParametersResult:
        """Re-pull self._signs/self._limits/self.mirror/etc from ROS params.

        We don't validate aggressively -- if a user types
        ``joint_signs.left_shoulder_roll_joint:=2.5`` they presumably know
        what they're doing.  We do refuse non-finite values (NaN/Inf) so a
        typo can't poison the smoother.

        Each accepted change is echoed at INFO so the user can verify in
        the launch console that their ``ros2 param set`` actually reached
        this node (vs being silently dropped by a DDS-implementation
        mismatch between the launch and their tuning terminal -- a common
        failure mode in this workspace, see README).
        """
        applied: list[str] = []
        for p in params:
            name = p.name
            try:
                value = p.value
            except Exception:
                return SetParametersResult(successful=False, reason=f"bad value for {name}")
            if isinstance(value, float) and not np.isfinite(value):
                return SetParametersResult(successful=False, reason=f"non-finite {name}")

            if name.startswith("joint_signs."):
                joint = name[len("joint_signs."):]
                self._signs[joint] = float(value)
                applied.append(f"{name}={value:+.3f}")
            elif name.startswith("joint_offsets."):
                joint = name[len("joint_offsets."):]
                self._offsets[joint] = float(value)
                applied.append(f"{name}={value:+.3f}")
            elif name.startswith("joint_limits.") and name.endswith(".min"):
                joint = name[len("joint_limits."):-len(".min")]
                lo, hi = self._limits.get(joint, (-3.14, 3.14))
                self._limits[joint] = (float(value), hi)
                applied.append(f"{name}={value:+.3f}")
            elif name.startswith("joint_limits.") and name.endswith(".max"):
                joint = name[len("joint_limits."):-len(".max")]
                lo, hi = self._limits.get(joint, (-3.14, 3.14))
                self._limits[joint] = (lo, float(value))
                applied.append(f"{name}={value:+.3f}")
            elif name == "mirror":
                self.mirror = bool(value)
                applied.append(f"mirror={self.mirror}")
            elif name == "smoothing_alpha":
                self.alpha = float(value)
                applied.append(f"smoothing_alpha={self.alpha:.3f}")
            elif name == "keypoint_min_visibility":
                self.min_vis = float(value)
                applied.append(f"keypoint_min_visibility={self.min_vis:.3f}")
            elif name == "head_to_waist_gain":
                self.head_to_waist_gain = float(value)
                applied.append(f"head_to_waist_gain={self.head_to_waist_gain:.3f}")
            elif name == "head_yaw_sign":
                self.head_yaw_sign = float(value)
                applied.append(f"head_yaw_sign={self.head_yaw_sign:+.1f}")
            elif name == "debug_log_period_s":
                self.debug_log_period_s = float(value)
                applied.append(f"debug_log_period_s={self.debug_log_period_s:.2f}")
        if applied:
            self.get_logger().info("param set: " + ", ".join(applied))
        return SetParametersResult(successful=True)

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
        a = self._signs.get(joint, 1.0) * raw_angle + self._offsets.get(joint, 0.0)
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

        # The G1 URDF has no neck joint, so we mimic head yaw by mixing it
        # into waist_yaw.  Done *after* swap_left_right_joints because the
        # waist is not chirality-flipped under mirror mode (it's a
        # midline joint).  Skipped silently if either ear or either
        # shoulder is below min_vis, which keeps the prior waist_yaw
        # value intact rather than snapping to zero.
        if self.head_to_waist_gain > 0.0:
            head_yaw = compute_head_yaw(kps, vis, min_vis=self.min_vis)
            if head_yaw is not None and "waist_yaw_joint" in angles:
                g = self.head_to_waist_gain
                angles["waist_yaw_joint"] = (
                    (1.0 - g) * angles["waist_yaw_joint"]
                    + g * self.head_yaw_sign * head_yaw
                )

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

        if self.debug_log_period_s > 0.0:
            now = time.monotonic()
            if now - self._last_debug_log_t >= self.debug_log_period_s:
                self._last_debug_log_t = now
                # Compact one-line dump (degrees, 1 decimal).  Useful for
                # diagnosing "which joint is wrong" without setting up a
                # plotter: pose statically, read the line, compare to
                # what the joint physically *should* be.
                deg = {j: math.degrees(self._cmd[j] or 0.0) for j in CONTROLLER_JOINT_ORDER}
                self.get_logger().info(
                    "cmd(deg): "
                    f"wy={deg['waist_yaw_joint']:+.1f} | "
                    f"LSp={deg['left_shoulder_pitch_joint']:+.1f} "
                    f"LSr={deg['left_shoulder_roll_joint']:+.1f} "
                    f"LSy={deg['left_shoulder_yaw_joint']:+.1f} "
                    f"LE={deg['left_elbow_joint']:+.1f} | "
                    f"RSp={deg['right_shoulder_pitch_joint']:+.1f} "
                    f"RSr={deg['right_shoulder_roll_joint']:+.1f} "
                    f"RSy={deg['right_shoulder_yaw_joint']:+.1f} "
                    f"RE={deg['right_elbow_joint']:+.1f}"
                )


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
