"""Shared keypoint index convention for /human/keypoints.

The pose estimator publishes a geometry_msgs/PoseArray whose entries map to
these landmark names by index.  The retargeter reads positions by the same
indices.  Both sides import from here so they stay in lock-step.
"""

from __future__ import annotations

from enum import IntEnum


class Kp(IntEnum):
    NOSE = 0
    LEFT_EAR = 1
    RIGHT_EAR = 2
    LEFT_SHOULDER = 3
    RIGHT_SHOULDER = 4
    LEFT_ELBOW = 5
    RIGHT_ELBOW = 6
    LEFT_WRIST = 7
    RIGHT_WRIST = 8
    LEFT_HIP = 9
    RIGHT_HIP = 10


KEYPOINT_COUNT: int = len(Kp)

# Mapping from our compact index to MediaPipe's pose landmark index.
# See https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
MEDIAPIPE_INDEX: dict[Kp, int] = {
    Kp.NOSE: 0,
    Kp.LEFT_EAR: 7,
    Kp.RIGHT_EAR: 8,
    Kp.LEFT_SHOULDER: 11,
    Kp.RIGHT_SHOULDER: 12,
    Kp.LEFT_ELBOW: 13,
    Kp.RIGHT_ELBOW: 14,
    Kp.LEFT_WRIST: 15,
    Kp.RIGHT_WRIST: 16,
    Kp.LEFT_HIP: 23,
    Kp.RIGHT_HIP: 24,
}


SKELETON_EDGES: list[tuple[Kp, Kp]] = [
    (Kp.LEFT_SHOULDER, Kp.RIGHT_SHOULDER),
    (Kp.LEFT_HIP, Kp.RIGHT_HIP),
    (Kp.LEFT_SHOULDER, Kp.LEFT_HIP),
    (Kp.RIGHT_SHOULDER, Kp.RIGHT_HIP),
    (Kp.LEFT_SHOULDER, Kp.LEFT_ELBOW),
    (Kp.LEFT_ELBOW, Kp.LEFT_WRIST),
    (Kp.RIGHT_SHOULDER, Kp.RIGHT_ELBOW),
    (Kp.RIGHT_ELBOW, Kp.RIGHT_WRIST),
    (Kp.NOSE, Kp.LEFT_EAR),
    (Kp.NOSE, Kp.RIGHT_EAR),
]
