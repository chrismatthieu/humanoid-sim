"""Pytest unit tests for humanoid_retargeter.geometric_ik.

These exercise the IK math against hand-picked synthetic keypoint
configurations in the camera optical frame (x=right, y=down, z=forward), with
the operator standing 2 m in front of the camera, shoulders square.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from humanoid_pose_estimator.keypoints import KEYPOINT_COUNT, Kp
from humanoid_retargeter.geometric_ik import (
    IKResult,
    compute_joint_angles,
    swap_left_right_joints,
)


def _build_pose(
    *,
    l_elbow_offset=(0, 0.30, 0),
    l_wrist_offset=(0, 0.30, 0),
    r_elbow_offset=(0, 0.30, 0),
    r_wrist_offset=(0, 0.30, 0),
    torso_yaw_deg: float = 0.0,
):
    """Compose a canonical operator pose in the camera optical frame.

    All offsets are in *camera* coordinates relative to the corresponding
    shoulder.  Camera y is down, so "arms hanging" is (0, +0.30, 0).
    `torso_yaw_deg` rotates the shoulders about the world-up (-y) axis.
    """
    L_hip = np.array([+0.10, 0.5, 2.0])
    R_hip = np.array([-0.10, 0.5, 2.0])
    L_sh = np.array([+0.20, 0.0, 2.0])
    R_sh = np.array([-0.20, 0.0, 2.0])

    if torso_yaw_deg:
        a = math.radians(torso_yaw_deg)
        c, s = math.cos(a), math.sin(a)
        # Rotation about world up = -y_cam, applied to shoulders only.
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        sh_mid = 0.5 * (L_sh + R_sh)
        L_sh = sh_mid + R @ (L_sh - sh_mid)
        R_sh = sh_mid + R @ (R_sh - sh_mid)

    L_el = L_sh + np.array(l_elbow_offset)
    L_wr = L_el + np.array(l_wrist_offset)
    R_el = R_sh + np.array(r_elbow_offset)
    R_wr = R_el + np.array(r_wrist_offset)

    kps = np.zeros((KEYPOINT_COUNT, 3))
    vis = np.ones(KEYPOINT_COUNT)
    kps[int(Kp.LEFT_HIP)] = L_hip
    kps[int(Kp.RIGHT_HIP)] = R_hip
    kps[int(Kp.LEFT_SHOULDER)] = L_sh
    kps[int(Kp.RIGHT_SHOULDER)] = R_sh
    kps[int(Kp.LEFT_ELBOW)] = L_el
    kps[int(Kp.RIGHT_ELBOW)] = R_el
    kps[int(Kp.LEFT_WRIST)] = L_wr
    kps[int(Kp.RIGHT_WRIST)] = R_wr
    kps[int(Kp.NOSE)] = 0.5 * (L_sh + R_sh) + np.array([0, -0.2, 0])
    return kps, vis


def _approx(actual: float, expected_deg: float, tol_deg: float = 1.0) -> None:
    assert abs(math.degrees(actual) - expected_deg) <= tol_deg, (
        f"got {math.degrees(actual):.2f} deg, expected {expected_deg:.2f} ± {tol_deg}"
    )


# ---------- body-frame construction ----------

def test_rejects_when_hips_missing():
    kps, vis = _build_pose()
    vis[int(Kp.LEFT_HIP)] = 0.0
    res = compute_joint_angles(kps, vis)
    assert res.body_frame_ok is False
    assert res.angles == {}


def test_rejects_when_shoulders_missing():
    kps, vis = _build_pose()
    vis[int(Kp.LEFT_SHOULDER)] = 0.0
    res = compute_joint_angles(kps, vis)
    assert res.body_frame_ok is False


# ---------- waist yaw ----------

def test_arms_hanging_produces_zero_angles():
    kps, vis = _build_pose()
    res = compute_joint_angles(kps, vis)
    assert res.body_frame_ok
    for joint, val in res.angles.items():
        _approx(val, 0.0, tol_deg=2.0)


@pytest.mark.parametrize("yaw_deg", [-45.0, -30.0, -10.0, 10.0, 30.0, 45.0])
def test_waist_yaw_follows_torso_rotation(yaw_deg: float):
    kps, vis = _build_pose(torso_yaw_deg=yaw_deg)
    res = compute_joint_angles(kps, vis)
    assert res.body_frame_ok
    # The sign convention is "atan2(-sh_in_hip.x, sh_in_hip.y)" which produces
    # the opposite sign of the input torso_yaw_deg parameter (rotation about
    # world up is anti-clockwise viewed from above).  Match magnitudes.
    waist = res.angles["waist_yaw_joint"]
    assert abs(math.degrees(waist) - (-yaw_deg)) <= 3.0, (
        f"waist_yaw={math.degrees(waist):.1f} deg for torso_yaw={yaw_deg} deg"
    )


# ---------- per-arm angles ----------

def test_left_arm_forward():
    """Operator's left arm extended forward (toward camera) means -z_cam."""
    kps, vis = _build_pose(
        l_elbow_offset=(0, 0, -0.30),
        l_wrist_offset=(0, 0, -0.30),
    )
    res = compute_joint_angles(kps, vis)
    a = res.angles
    _approx(a["left_shoulder_pitch_joint"], -90.0, tol_deg=2.0)
    _approx(a["left_shoulder_roll_joint"], 0.0, tol_deg=2.0)
    _approx(a["left_elbow_joint"], 0.0, tol_deg=2.0)
    # Right arm untouched -> all zeros
    _approx(a["right_shoulder_pitch_joint"], 0.0, tol_deg=2.0)
    _approx(a["right_elbow_joint"], 0.0, tol_deg=2.0)


def test_left_arm_to_side():
    """Operator's left arm extended to the side (operator's left = +x_cam)."""
    kps, vis = _build_pose(
        l_elbow_offset=(+0.30, 0, 0),
        l_wrist_offset=(+0.30, 0, 0),
    )
    res = compute_joint_angles(kps, vis)
    _approx(res.angles["left_shoulder_roll_joint"], 90.0, tol_deg=2.0)


def test_left_elbow_90deg_forward():
    """Upper arm hangs down, forearm forward => elbow=90."""
    kps, vis = _build_pose(
        l_elbow_offset=(0, +0.30, 0),       # hang down
        l_wrist_offset=(0, 0, -0.30),       # forearm forward
    )
    res = compute_joint_angles(kps, vis)
    _approx(res.angles["left_elbow_joint"], 90.0, tol_deg=3.0)


def test_left_elbow_fully_bent():
    """Forearm bent back parallel to upper arm => elbow ~180 deg.

    The retargeter doesn't clamp here; the caller's joint_limits do that.
    """
    kps, vis = _build_pose(
        l_elbow_offset=(0, +0.30, 0),
        l_wrist_offset=(0, -0.30, 0),  # back up
    )
    res = compute_joint_angles(kps, vis)
    _approx(res.angles["left_elbow_joint"], 180.0, tol_deg=3.0)


def test_skips_arm_when_elbow_missing():
    kps, vis = _build_pose()
    vis[int(Kp.LEFT_ELBOW)] = 0.0
    res = compute_joint_angles(kps, vis, min_vis=0.5)
    assert res.body_frame_ok
    # Left arm has no angles, right arm still does.
    assert "left_shoulder_pitch_joint" not in res.angles
    assert "right_shoulder_pitch_joint" in res.angles
    # Waist yaw still computed because hips+shoulders are valid.
    assert "waist_yaw_joint" in res.angles


def test_skips_yaw_when_wrist_missing_but_keeps_pitch_roll():
    kps, vis = _build_pose(
        l_elbow_offset=(+0.30, 0, 0),
        l_wrist_offset=(+0.30, 0, 0),
    )
    vis[int(Kp.LEFT_WRIST)] = 0.0
    res = compute_joint_angles(kps, vis, min_vis=0.5)
    assert "left_shoulder_pitch_joint" in res.angles
    assert "left_shoulder_roll_joint" in res.angles
    assert "left_elbow_joint" not in res.angles
    assert "left_shoulder_yaw_joint" not in res.angles


# ---------- mirror (swap_left_right_joints) ----------

def test_swap_left_right_joints_basic():
    angles = {
        "waist_yaw_joint": 0.2,
        "left_shoulder_pitch_joint": -1.5,
        "right_shoulder_pitch_joint": 0.1,
        "left_elbow_joint": 0.5,
    }
    swapped = swap_left_right_joints(angles)
    assert swapped["waist_yaw_joint"] == 0.2
    assert swapped["right_shoulder_pitch_joint"] == -1.5
    assert swapped["left_shoulder_pitch_joint"] == 0.1
    assert swapped["right_elbow_joint"] == 0.5
    # No stray keys
    assert set(swapped) == {
        "waist_yaw_joint",
        "right_shoulder_pitch_joint",
        "left_shoulder_pitch_joint",
        "right_elbow_joint",
    }


def test_mirror_routes_operator_left_arm_to_robot_right():
    """Operator's left arm forward, run IK, swap L<->R: G1's right arm should
    take the same pitch/roll/elbow that the operator's left did.
    """
    kps, vis = _build_pose(
        l_elbow_offset=(0, 0, -0.30),
        l_wrist_offset=(0, 0, -0.30),
    )
    res = compute_joint_angles(kps, vis)
    swapped = swap_left_right_joints(res.angles)
    _approx(swapped["right_shoulder_pitch_joint"],
            math.degrees(res.angles["left_shoulder_pitch_joint"]),
            tol_deg=0.01)
    _approx(swapped["right_elbow_joint"],
            math.degrees(res.angles["left_elbow_joint"]),
            tol_deg=0.01)
    # The other side mirrors the operator's resting right arm (~0).
    _approx(swapped["left_shoulder_pitch_joint"], 0.0, tol_deg=2.0)
