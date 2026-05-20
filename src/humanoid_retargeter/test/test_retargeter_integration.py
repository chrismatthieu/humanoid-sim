"""Integration test: spin the retargeter node, publish a fake PoseArray,
expect a Float64MultiArray on the controller topic.

Run with: `python3 -m pytest src/humanoid_retargeter/test/test_retargeter_integration.py`
The test launches an rclpy node inline (no DDS network discovery needed
beyond localhost) so it works in CI as well.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
import rclpy
from geometry_msgs.msg import Pose, PoseArray
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from humanoid_pose_estimator.keypoints import KEYPOINT_COUNT, Kp
from humanoid_retargeter.retargeter_node import (
    CONTROLLER_JOINT_ORDER,
    RetargeterNode,
)


def _make_pose_array() -> PoseArray:
    """A canonical operator pose: square shoulders, left arm out to the side."""
    msg = PoseArray()
    msg.header.frame_id = "camera_color_optical_frame"
    poses = [Pose() for _ in range(KEYPOINT_COUNT)]

    def set(kp: Kp, xyz, vis=1.0):
        p = poses[int(kp)]
        p.position.x, p.position.y, p.position.z = xyz
        p.orientation.w = vis

    set(Kp.NOSE,           (0.0, -0.20, 2.0))
    set(Kp.LEFT_SHOULDER,  (+0.20, 0.0, 2.0))
    set(Kp.RIGHT_SHOULDER, (-0.20, 0.0, 2.0))
    set(Kp.LEFT_ELBOW,     (+0.50, 0.0, 2.0))   # out to the side
    set(Kp.RIGHT_ELBOW,    (-0.20, +0.30, 2.0)) # hanging
    set(Kp.LEFT_WRIST,     (+0.80, 0.0, 2.0))
    set(Kp.RIGHT_WRIST,    (-0.20, +0.60, 2.0))
    set(Kp.LEFT_HIP,       (+0.10, 0.50, 2.0))
    set(Kp.RIGHT_HIP,      (-0.10, 0.50, 2.0))
    msg.poses = poses
    return msg


class _CommandSink(Node):
    def __init__(self) -> None:
        super().__init__("test_command_sink")
        self.received: list[list[float]] = []
        self.create_subscription(
            Float64MultiArray, "/upper_body_controller/commands",
            self._on_cmd, 10,
        )

    def _on_cmd(self, msg: Float64MultiArray) -> None:
        self.received.append(list(msg.data))


@pytest.fixture(scope="module")
def ros_context():
    rclpy.init()
    yield
    if rclpy.ok():
        rclpy.shutdown()


def test_retargeter_publishes_commands_on_keypoint_input(ros_context):
    retargeter = RetargeterNode()
    sink = _CommandSink()

    publisher_node = rclpy.create_node("test_publisher")
    pub = publisher_node.create_publisher(PoseArray, "/human/keypoints", 10)

    exe = SingleThreadedExecutor()
    exe.add_node(retargeter)
    exe.add_node(sink)
    exe.add_node(publisher_node)

    def spin_for(seconds: float) -> None:
        deadline = time.time() + seconds
        while time.time() < deadline:
            exe.spin_once(timeout_sec=0.05)

    try:
        # Allow publishers/subscribers to discover each other.
        spin_for(0.5)
        msg = _make_pose_array()
        # Send a few keypoint frames so the timer's _tick callback fires.
        for _ in range(20):
            pub.publish(msg)
            spin_for(0.1)

        assert len(sink.received) >= 1, "retargeter never published commands"
        first = sink.received[-1]
        assert len(first) == len(CONTROLLER_JOINT_ORDER), (
            f"expected {len(CONTROLLER_JOINT_ORDER)} values, got {len(first)}"
        )
        # Sanity: at least one joint moved away from 0 (we set the left arm out).
        arr = np.asarray(first)
        assert np.any(np.abs(arr) > 0.05), (
            f"expected non-zero command after sending an arm-out pose; got {arr}"
        )
    finally:
        exe.shutdown()
        retargeter.destroy_node()
        sink.destroy_node()
        publisher_node.destroy_node()
