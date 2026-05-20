"""Closed-form geometric retargeting from human 3D keypoints to G1 joints.

We solve the upper body in two parts.

1. Body frame and waist yaw
   - Build a "hip frame" anchored on the operator's hips with z = world up
     (which is -y in camera optical frame), y = operator left (from R_hip to
     L_hip, projected horizontal), and x = forward (operator's chest direction).
   - waist_yaw = signed angle by which the shoulder line is rotated about z
     relative to the hip line.

2. Per-arm shoulder + elbow
   - Compute the upper-arm and forearm direction unit vectors expressed in a
     "shoulder frame" (= hip frame rotated by waist_yaw).  At the G1's zero
     pose, the upper arm hangs straight down, i.e. along (0, 0, -1) in this
     frame, and the forearm continues straight from it.
   - shoulder_roll  = asin(u.y)              (lift to the side)
   - shoulder_pitch = atan2(-u.x, -u.z)      (lift forward/back)
   - elbow          = angle between u and f (0 = straight, +PI = touching shoulder)
   - shoulder_yaw   = twist about the upper arm such that the forearm direction
                      matches f after R_y(pitch) * R_x(roll) * R_z(yaw) *
                      R_y(elbow) * (0,0,-1).

Wrists are not driven (we don't have reliable hand orientation from MediaPipe
Pose alone).  They stay at 0.0.

The output dict contains only joints whose source keypoints were valid this
frame, so the caller can decide to hold the last value for missing entries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from humanoid_pose_estimator.keypoints import Kp


@dataclass
class IKResult:
    angles: dict[str, float]  # joint_name -> rad
    body_frame_ok: bool       # False if hips/shoulders were unusable


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return np.zeros_like(v)
    return v / n


def _rot_y(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rot_x(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _arm_angles(
    shoulder: np.ndarray,
    elbow: np.ndarray,
    wrist: np.ndarray,
    R_shoulder_in_cam: np.ndarray,
    elbow_valid: bool,
    wrist_valid: bool,
) -> dict[str, float]:
    out: dict[str, float] = {}
    if not elbow_valid:
        return out

    u_cam = elbow - shoulder
    u_b = _normalize(R_shoulder_in_cam.T @ u_cam)
    if np.linalg.norm(u_b) < 1e-6:
        return out

    # Shoulder frame: x=forward, y=left, z=up.  Rest arm = (0, 0, -1).
    roll = math.asin(float(np.clip(u_b[1], -1.0, 1.0)))
    pitch = math.atan2(-u_b[0], -u_b[2])
    out["shoulder_pitch"] = pitch
    out["shoulder_roll"] = roll

    if not wrist_valid:
        return out

    f_cam = wrist - elbow
    f_b = _normalize(R_shoulder_in_cam.T @ f_cam)
    if np.linalg.norm(f_b) < 1e-6:
        return out

    cos_el = float(np.clip(np.dot(u_b, f_b), -1.0, 1.0))
    elbow_angle = math.acos(cos_el)
    out["elbow"] = elbow_angle

    # Yaw: undo pitch/roll, then read the forearm's azimuth.
    #
    # We chose ``+x`` of the (post-pitch-roll, post-yaw) frame as the
    # "yaw = 0" reference direction so this convention agrees with the
    # G1's URDF: at ``shoulder_yaw_joint = 0, elbow_joint = 0``, the G1's
    # forearm extends along ``+x_elbow_link`` (the ``wrist_roll_joint``
    # origin in elbow_link is ``(0.100, 0.002, -0.010)``, dominantly
    # ``+x``).  Before this we used ``atan2(-y, -x)`` which corresponded
    # to a "yaw=0 means forearm in -x" convention; that's the +/- pi
    # mirror of the URDF and the IK was saturating ``shoulder_yaw`` at
    # the URDF limit of +/- 2.618 rad in every common pose, producing
    # the "hands turned the opposite direction" symptom (the forearm was
    # being rotated ~180 deg about the upper-arm axis relative to
    # what the operator was doing).
    R_pr = _rot_y(pitch) @ _rot_x(roll)
    f_in_yaw_frame = R_pr.T @ f_b
    sin_el = math.sin(elbow_angle)
    if sin_el > 0.08:  # ~5deg; below this, yaw is ill-defined
        yaw = math.atan2(f_in_yaw_frame[1], f_in_yaw_frame[0])
    else:
        yaw = 0.0
    out["shoulder_yaw"] = yaw

    return out


def compute_joint_angles(
    keypoints: np.ndarray,  # shape (N, 3) in camera optical frame, meters
    visibility: np.ndarray,  # shape (N,), in [0, 1]
    *,
    min_vis: float = 0.5,
) -> IKResult:
    """Compute G1 upper-body joint angles from 3D human keypoints.

    `keypoints` and `visibility` are indexed by `humanoid_pose_estimator.keypoints.Kp`.
    """
    angles: dict[str, float] = {}

    needed = [Kp.LEFT_HIP, Kp.RIGHT_HIP, Kp.LEFT_SHOULDER, Kp.RIGHT_SHOULDER]
    if any(visibility[int(k)] < min_vis for k in needed):
        return IKResult(angles=angles, body_frame_ok=False)

    L_hip = keypoints[int(Kp.LEFT_HIP)]
    R_hip = keypoints[int(Kp.RIGHT_HIP)]
    L_sh = keypoints[int(Kp.LEFT_SHOULDER)]
    R_sh = keypoints[int(Kp.RIGHT_SHOULDER)]

    # Camera optical frame: x_right, y_down, z_forward => world up ≈ -y.
    world_up_cam = np.array([0.0, -1.0, 0.0])

    hip_left = L_hip - R_hip
    # Project onto horizontal plane (perpendicular to world up) for a stable z.
    hip_left_proj = hip_left - np.dot(hip_left, world_up_cam) * world_up_cam
    if np.linalg.norm(hip_left_proj) < 1e-4:
        return IKResult(angles=angles, body_frame_ok=False)

    hip_y = _normalize(hip_left_proj)
    hip_z = world_up_cam
    hip_x = _normalize(np.cross(hip_y, hip_z))
    # Re-orthonormalize y for numerical safety.
    hip_y = _normalize(np.cross(hip_z, hip_x))
    R_hip_in_cam = np.column_stack([hip_x, hip_y, hip_z])

    # Waist yaw: rotate the shoulder line to express it in hip frame, then
    # measure its angle relative to hip_y about hip_z.
    sh_left = L_sh - R_sh
    sh_in_hip = R_hip_in_cam.T @ sh_left
    waist_yaw = math.atan2(-sh_in_hip[0], sh_in_hip[1])
    angles["waist_yaw_joint"] = waist_yaw

    cz, sz = math.cos(waist_yaw), math.sin(waist_yaw)
    R_yaw = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    R_shoulder_in_cam = R_hip_in_cam @ R_yaw

    for side, sh_kp, el_kp, wr_kp in (
        ("left",  Kp.LEFT_SHOULDER,  Kp.LEFT_ELBOW,  Kp.LEFT_WRIST),
        ("right", Kp.RIGHT_SHOULDER, Kp.RIGHT_ELBOW, Kp.RIGHT_WRIST),
    ):
        elbow_valid = visibility[int(el_kp)] >= min_vis
        wrist_valid = visibility[int(wr_kp)] >= min_vis
        arm = _arm_angles(
            shoulder=keypoints[int(sh_kp)],
            elbow=keypoints[int(el_kp)],
            wrist=keypoints[int(wr_kp)],
            R_shoulder_in_cam=R_shoulder_in_cam,
            elbow_valid=elbow_valid,
            wrist_valid=wrist_valid,
        )
        for short, val in arm.items():
            angles[f"{side}_{short}_joint"] = val

    return IKResult(angles=angles, body_frame_ok=True)


def compute_head_yaw(
    keypoints: np.ndarray,
    visibility: np.ndarray,
    *,
    min_vis: float = 0.5,
) -> float | None:
    """Estimate the operator's head yaw in radians.

    The G1's URDF has ``head_joint type="fixed"`` -- the head is rigidly
    bolted to ``torso_link`` and there is no neck joint to drive.  The only
    way to make the robot's head visually follow the operator's head turn
    is to route the operator's head yaw into ``waist_yaw_joint`` (which
    rotates the whole torso including the head).  The retargeter blends
    this signal with the torso-twist yaw using the ``head_to_waist_gain``
    parameter.

    Definition of "head yaw" here: the signed angle (about world up,
    which is -y in the camera optical frame) from the shoulder-line
    (``LEFT_SHOULDER - RIGHT_SHOULDER``) to the ear-line
    (``LEFT_EAR - RIGHT_EAR``), both projected onto the horizontal plane.
    This is *relative to the torso*, so it captures "turn just your head"
    rather than "turn your whole body".

    Sign convention: a *positive* return value means the operator turned
    their face to their own **left** (the ear-line rotated CCW about
    world up when viewed from above).  This matches the G1's
    ``waist_yaw_joint`` axis (``<axis xyz="0 0 1"/>``, CCW positive about
    world up), so under mirror mode you can feed this value directly into
    waist_yaw without an extra sign flip -- and the robot's head rotates
    in the *same direction* as the operator's, exactly like a mirror.

    Returns ``None`` if either ear or either shoulder is below ``min_vis``,
    or if the projected vectors are too short to be reliable (operator
    looking ~90 deg sideways so both ears collapse).
    """
    needed = [Kp.LEFT_EAR, Kp.RIGHT_EAR, Kp.LEFT_SHOULDER, Kp.RIGHT_SHOULDER]
    if any(visibility[int(k)] < min_vis for k in needed):
        return None

    world_up_cam = np.array([0.0, -1.0, 0.0])

    ear_left = keypoints[int(Kp.LEFT_EAR)] - keypoints[int(Kp.RIGHT_EAR)]
    sh_left = keypoints[int(Kp.LEFT_SHOULDER)] - keypoints[int(Kp.RIGHT_SHOULDER)]
    ear_proj = ear_left - np.dot(ear_left, world_up_cam) * world_up_cam
    sh_proj = sh_left - np.dot(sh_left, world_up_cam) * world_up_cam
    if np.linalg.norm(ear_proj) < 0.03 or np.linalg.norm(sh_proj) < 0.03:
        # ~3 cm projected width: operator is in profile (face perpendicular
        # to camera) and one ear is occluded behind the head.  Bailing
        # rather than returning a meaningless atan2 result.
        return None

    e = _normalize(ear_proj)
    s = _normalize(sh_proj)
    # Signed angle from shoulder-line to ear-line about world up (=-y_cam).
    # cross . up = sin(theta), dot = cos(theta).
    sin_t = float(np.dot(np.cross(s, e), world_up_cam))
    cos_t = float(np.dot(s, e))
    return math.atan2(sin_t, cos_t)


def swap_left_right_joints(angles: dict[str, float]) -> dict[str, float]:
    """Swap `left_*` and `right_*` joint names in a joint-angle dict.

    This is the correct way to implement "mirror mode": run the geometric IK
    on the un-modified keypoints (so the body frame is built from the real
    hip layout), then re-route the operator's left-arm result to the G1's
    right arm and vice versa.  Joints without an L/R prefix (e.g.
    `waist_yaw_joint`) are passed through unchanged.

    Sign tuning per joint is the responsibility of `joint_signs` in
    `retargeter.yaml`; with mirror=true you typically end up with the
    opposite signs to mirror=false on the chirality-flipping joints
    (roll, yaw, wrist_roll, wrist_yaw).
    """
    out: dict[str, float] = {}
    for joint, val in angles.items():
        if joint.startswith("left_"):
            out["right_" + joint[len("left_"):]] = val
        elif joint.startswith("right_"):
            out["left_" + joint[len("right_"):]] = val
        else:
            out[joint] = val
    return out
