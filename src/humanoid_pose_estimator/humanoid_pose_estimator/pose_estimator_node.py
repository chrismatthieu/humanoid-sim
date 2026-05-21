"""ROS 2 node: MediaPipe PoseLandmarker (Tasks API) + RealSense depth
=> 3D human keypoints.

Subscribes (color triggers inference, depth + info are latest-cached):
  - color image  (sensor_msgs/Image, encoding rgb8 or bgr8)
  - depth image  (sensor_msgs/Image, 16UC1 in mm or 32FC1 in m).  Either
    the realsense-side aligned-depth-to-color stream (default on the
    Intel N100 / D415 path) or the raw depth-frame stream (default on
    the Jetson AGX Orin / D457 path -- see ``align_mode`` below).
  - color or depth camera_info  (sensor_msgs/CameraInfo) -- which one is
    semantically the "color intrinsics" depends on ``align_mode``; either
    way it's the K used for back-projecting MediaPipe landmark pixels.
  - (manual mode only) realsense2_camera_msgs/Extrinsics
    /camera/.../extrinsics/depth_to_color  and a separate
    /camera/.../depth/camera_info for the depth-sensor intrinsics.

Publishes:
  - /human/keypoints (geometry_msgs/PoseArray) in the color optical frame.
    Each Pose.position is the 3D point of a landmark; Pose.orientation.w
    carries the MediaPipe `visibility` in 0..1.  Indices follow `keypoints.Kp`.
  - /human/markers (visualization_msgs/MarkerArray) for RViz.
  - /human/debug_image (sensor_msgs/Image) overlay of the pose, for debugging.

Model files (`pose_landmarker_{lite,full,heavy}.task`) are auto-downloaded
to ~/.cache/humanoid_pose_estimator/models on first use.

``align_mode`` (default ``realsense``):

  * ``realsense`` -- subscribe to ``aligned_depth_to_color/image_raw`` and
    look up depth at the color pixel directly.  Works on any RealSense
    that exposes UVC frame metadata (the D415 over USB does, and so does
    the D435/D455 over USB).
  * ``manual`` -- subscribe to the raw ``depth/image_rect_raw`` plus the
    ``extrinsics/depth_to_color`` topic and the depth-sensor intrinsics,
    and warp each landmark from the color pixel into the depth pixel
    ourselves.  Required on the Jetson AGX Orin with a D457 connected
    via GMSL2: that hardware path does not expose UVC frame metadata,
    so librealsense's sync filter cannot pair depth+color frames and the
    realsense-side aligned-depth pipeline silently produces no messages
    even though the topic is advertised.  See
    ``/opt/ros/humble/share/realsense2_camera/examples/align_depth/`` and
    the README "Jetson AGX Orin" section for the diagnostic.
"""

from __future__ import annotations

import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseArray
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from .keypoints import KEYPOINT_COUNT, MEDIAPIPE_INDEX, SKELETON_EDGES, Kp


_MODEL_URLS: dict[str, str] = {
    "lite":  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full":  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}

# 0=lite, 1=full, 2=heavy.  Matches the historical `model_complexity` knob.
_MODEL_BY_COMPLEXITY: dict[int, str] = {0: "lite", 1: "full", 2: "heavy"}


def _sensor_qos(depth: int = 5) -> QoSProfile:
    """BEST_EFFORT sensor QoS matching ``realsense2_camera`` image topics.

    A small but non-trivial queue (``depth=5``) is important: MediaPipe
    inference on a CPU can run slower than the 30 Hz camera, and a depth-1
    middleware buffer would silently drop the only fresh message every time
    the detector is busy, starving the callbacks.
    """
    return QoSProfile(
        depth=depth,
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        durability=QoSDurabilityPolicy.VOLATILE,
        history=QoSHistoryPolicy.KEEP_LAST,
    )


def _ensure_model(variant: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    fname = f"pose_landmarker_{variant}.task"
    out = cache_dir / fname
    if out.exists() and out.stat().st_size > 0:
        return out
    url = _MODEL_URLS[variant]
    tmp = out.with_suffix(".task.partial")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as fh:
            fh.write(resp.read())
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Failed to download {url}: {exc}.  Place the file manually at {out}."
        ) from exc
    tmp.rename(out)
    return out


class PoseEstimatorNode(Node):
    def __init__(self) -> None:
        super().__init__("humanoid_pose_estimator")

        self.declare_parameter("color_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic",
                               "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("info_topic",
                               "/camera/camera/aligned_depth_to_color/camera_info")
        # ``align_mode``: ``realsense`` uses librealsense's depth->color
        # align filter (depth is sampled directly at the color pixel);
        # ``manual`` runs the alignment in this node using the extrinsics
        # topic.  See the module docstring.  Auto-flipping the defaults
        # of ``depth_topic`` and ``info_topic`` is deliberate so the user
        # only has to set ``align_mode:=manual`` to switch the whole
        # pipeline; explicit topic overrides still win.
        self.declare_parameter("align_mode", "realsense")
        self.declare_parameter(
            "depth_info_topic", "/camera/camera/depth/camera_info"
        )
        self.declare_parameter(
            "extrinsics_topic", "/camera/camera/extrinsics/depth_to_color"
        )
        self.declare_parameter("output_frame", "camera_color_optical_frame")
        self.declare_parameter("min_confidence", 0.5)
        self.declare_parameter("depth_patch", 5)
        self.declare_parameter("model_complexity", 0)
        self.declare_parameter("model_path", "")  # if non-empty, overrides the download
        self.declare_parameter("model_cache_dir", "")  # default ~/.cache/humanoid_pose_estimator/models
        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter("publish_markers", True)
        # Wrist-depth sanity check.  When the operator's hand is near the
        # body (e.g. hands resting on the hips, arms folded across chest),
        # the median over the depth patch around the wrist pixel can bleed
        # into the torso behind the hand and snap the wrist's 3D position
        # backward by 10-30 cm.  That makes the IK think the forearm is
        # almost colinear with the upper arm (under-bent elbow).  When
        # enabled, this re-projects the wrist pixel at a depth that makes
        # the elbow-to-wrist distance match MediaPipe's metric
        # ``pose_world_landmarks`` forearm length.
        self.declare_parameter("wrist_correction", True)
        self.declare_parameter("wrist_forearm_tol_m", 0.05)
        # Elbow-depth sanity check.  Same mechanism as the wrist correction,
        # one link upstream: re-back-project the elbow if its stereo
        # shoulder-to-elbow length disagrees with MediaPipe's world-landmark
        # upper-arm length by more than ``elbow_upper_arm_tol_m``.  This
        # matters when the operator's elbow rests against the body (hands
        # on hips, folded arms) -- depth lookups around the elbow then
        # bleed into the torso, anchoring everything downstream (including
        # the wrist correction) to a wrong elbow position.  Runs *before*
        # the wrist correction so the wrist sees the corrected elbow.
        self.declare_parameter("elbow_correction", True)
        self.declare_parameter("elbow_upper_arm_tol_m", 0.05)

        gp = self.get_parameter
        self.color_topic: str = gp("color_topic").value
        self.depth_topic: str = gp("depth_topic").value
        self.info_topic: str = gp("info_topic").value
        self.align_mode: str = str(gp("align_mode").value).strip().lower()
        if self.align_mode not in ("realsense", "manual"):
            raise ValueError(
                f"align_mode must be 'realsense' or 'manual', got {self.align_mode!r}"
            )
        # In manual mode swap the defaults to raw streams unless the
        # operator has overridden them.  ``info_topic`` in manual mode is
        # the *color* camera_info because that's the K we use for back-
        # projection downstream.  The depth-sensor K comes from a
        # separate ``depth_info_topic`` and is only used for the
        # color-pixel -> depth-pixel warp.
        if self.align_mode == "manual":
            if self.depth_topic == "/camera/camera/aligned_depth_to_color/image_raw":
                self.depth_topic = "/camera/camera/depth/image_rect_raw"
            if self.info_topic == "/camera/camera/aligned_depth_to_color/camera_info":
                self.info_topic = "/camera/camera/color/camera_info"
        self.depth_info_topic: str = gp("depth_info_topic").value
        self.extrinsics_topic: str = gp("extrinsics_topic").value
        self.output_frame: str = gp("output_frame").value
        self.min_confidence: float = float(gp("min_confidence").value)
        self.depth_patch: int = int(gp("depth_patch").value)
        self.model_complexity: int = int(gp("model_complexity").value)
        self.publish_debug_image: bool = bool(gp("publish_debug_image").value)
        self.publish_markers: bool = bool(gp("publish_markers").value)
        self.wrist_correction: bool = bool(gp("wrist_correction").value)
        self.wrist_forearm_tol_m: float = float(gp("wrist_forearm_tol_m").value)
        self.elbow_correction: bool = bool(gp("elbow_correction").value)
        self.elbow_upper_arm_tol_m: float = float(gp("elbow_upper_arm_tol_m").value)
        # Diagnostic counters for the heartbeat.  Reset each tick.
        self._elbow_fix_count = 0
        self._wrist_fix_count = 0

        # Resolve the model file.
        model_path_param = str(gp("model_path").value)
        if model_path_param:
            model_path = Path(model_path_param).expanduser()
            if not model_path.exists():
                raise FileNotFoundError(f"model_path does not exist: {model_path}")
        else:
            variant = _MODEL_BY_COMPLEXITY.get(self.model_complexity, "lite")
            cache = gp("model_cache_dir").value
            cache_dir = (
                Path(cache).expanduser() if cache
                else Path.home() / ".cache" / "humanoid_pose_estimator" / "models"
            )
            self.get_logger().info(
                f"Resolving MediaPipe {variant} model in {cache_dir} "
                "(downloading on first use)"
            )
            model_path = _ensure_model(variant, cache_dir)
        self.get_logger().info(f"Using PoseLandmarker model: {model_path}")

        # Build the detector (deferred import keeps launch introspection fast).
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except Exception as exc:  # pragma: no cover - env specific
            self.get_logger().fatal(
                f"MediaPipe import failed: {exc}.  "
                "Install with `pip3 install --user --break-system-packages "
                "'mediapipe>=0.10.14'` and `'numpy<2'`."
            )
            raise
        self._mp = mp
        self._mp_vision = mp_vision

        # IMAGE mode is stateless: each call is independent, no monotonic
        # timestamp requirement and (more importantly) no internal tracker
        # state that can wedge on Mesa-Intel GL drivers after a few frames,
        # which we observed with VIDEO mode.  We rely on the downstream
        # retargeter's EMA filter for temporal smoothing, so we lose nothing
        # by going stateless here.
        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=self.min_confidence,
            min_pose_presence_confidence=self.min_confidence,
            min_tracking_confidence=self.min_confidence,
            output_segmentation_masks=False,
        )
        self._detector = mp_vision.PoseLandmarker.create_from_options(options)
        self._lock = threading.Lock()  # detector is not thread-safe

        self._bridge = CvBridge()
        self._cam_info: Optional[CameraInfo] = None
        self._latest_depth: Optional[Image] = None
        self._depth_lock = threading.Lock()
        self._frames_in = 0
        self._frames_out = 0
        self._color_seen = False
        # Last per-frame inference wall-time, in ms (for the heartbeat).
        self._last_infer_ms = 0.0
        # Wall-time of when we last *entered* _on_color.  If this stops
        # advancing while the heartbeat keeps ticking, we know detect() hung.
        self._last_color_enter_t = time.monotonic()
        # Subscription-resync bookkeeping (see _watchdog).
        self._node_start_t = time.monotonic()
        self._last_resubscribe_t = 0.0
        self._resubscribe_count = 0
        self._info_sub: Optional[Any] = None
        self._depth_sub: Optional[Any] = None
        self._color_sub: Optional[Any] = None
        # Manual-alignment state (populated only when align_mode=="manual").
        self._depth_K: Optional[np.ndarray] = None   # 3x3 depth intrinsics
        self._R_dc: Optional[np.ndarray] = None      # depth->color rotation (3x3)
        self._t_dc: Optional[np.ndarray] = None      # depth->color translation (3,)
        self._depth_info_sub: Optional[Any] = None
        self._extrinsics_sub: Optional[Any] = None
        # Cache the last good operator distance to seed the
        # color-pixel->depth-pixel iteration; 2.0 m is a reasonable
        # initial guess for the demo's calibration setup.
        self._last_z_guess: float = 2.0
        # Lazy-load realsense2_camera_msgs only when actually needed -- it
        # is an optional dependency for the realsense (aligned) path.
        self._extrinsics_msg_type = None
        if self.align_mode == "manual":
            try:
                from realsense2_camera_msgs.msg import Extrinsics
                self._extrinsics_msg_type = Extrinsics
            except ImportError as exc:
                raise RuntimeError(
                    "align_mode='manual' requires realsense2_camera_msgs. "
                    "Install with apt: ros-${ROS_DISTRO}-realsense2-camera-msgs"
                ) from exc

        self.pub_kps = self.create_publisher(PoseArray, "/human/keypoints", 10)
        self.pub_markers = self.create_publisher(MarkerArray, "/human/markers", 10)
        self.pub_dbg = self.create_publisher(Image, "/human/debug_image", 10)

        # Single-threaded executor (see ``main``) drives all three callbacks
        # from one thread.  MediaPipe's PoseLandmarker initializes a GL/EGL
        # context lazily on the first ``detect_for_video`` call and binds
        # that context to the calling thread, so we must keep all inference
        # on the same thread.  The deeper QoS queue absorbs inference jitter
        # (depth=10 lets ~330 ms of buffered frames survive before drops).
        self._create_subscriptions()

        # Heartbeat on a real timer (independent of color callback firing).
        # If this keeps ticking while _on_color stops, we know detect() hung.
        self._heartbeat_timer = self.create_timer(1.0, self._heartbeat)
        # Subscription-resync watchdog.  When the underlying DDS reader
        # silently unmatches from the realsense writer (a well-known
        # Cyclone-DDS-on-Jazzy symptom triggered by bursty participant
        # churn -- gazebo + ros2_control bring up ~6 new participants in a
        # ~2 s window, which we observed reliably ``in=0``-freezes the
        # subscription even though ``ros2 topic hz`` shows the topic is
        # still being published at full rate), tear the subs down and
        # rebuild them.  Fires every 2 s and only acts if we *had* been
        # seeing frames and have now been silent for >= 5 s; this is
        # cheap when healthy and self-healing when not.
        self._watchdog_timer = self.create_timer(2.0, self._watchdog)

        extra = ""
        if self.align_mode == "manual":
            extra = (
                f" align=manual depth_info={self.depth_info_topic} "
                f"extrinsics={self.extrinsics_topic}"
            )
        self.get_logger().info(
            f"pose_estimator up; color={self.color_topic} depth={self.depth_topic} "
            f"info={self.info_topic} frame={self.output_frame}{extra}"
        )

    # ------------------------------------------------------------------ helpers

    def _create_subscriptions(self) -> None:
        """(Re)create the camera_info / depth / color subscriptions.

        Pulled into a helper so the watchdog can rebuild the readers when
        Cyclone DDS drops the subscriber<->writer match during gazebo
        bring-up.  In manual-align mode we additionally (re)create the
        depth-sensor camera_info and extrinsics subscriptions, which are
        latched-style (the publisher sends them once at startup) so
        rebuilding the reader is important if the original message was
        sent before the subscriber matched.  Destroying the old
        subscription before recreating is important: ``rclpy``'s
        subscription objects hold a reference to
        the underlying DDS reader, and that reader is the thing we
        actually need to rebuild.
        """
        for attr in (
            "_info_sub", "_depth_sub", "_color_sub",
            "_depth_info_sub", "_extrinsics_sub",
        ):
            sub = getattr(self, attr, None)
            if sub is not None:
                self.destroy_subscription(sub)
        self._info_sub = self.create_subscription(
            CameraInfo, self.info_topic, self._on_info, _sensor_qos(depth=5)
        )
        self._depth_sub = self.create_subscription(
            Image, self.depth_topic, self._on_depth, _sensor_qos(depth=10)
        )
        self._color_sub = self.create_subscription(
            Image, self.color_topic, self._on_color, _sensor_qos(depth=10)
        )
        if self.align_mode == "manual":
            self._depth_info_sub = self.create_subscription(
                CameraInfo,
                self.depth_info_topic,
                self._on_depth_info,
                _sensor_qos(depth=5),
            )
            # The extrinsics topic is published once at startup with
            # TRANSIENT_LOCAL durability on the realsense2_camera side,
            # so the subscriber must request a compatible (or weaker)
            # durability to receive the late-joiner replay.
            self._extrinsics_sub = self.create_subscription(
                self._extrinsics_msg_type,
                self.extrinsics_topic,
                self._on_extrinsics,
                QoSProfile(
                    depth=1,
                    reliability=QoSReliabilityPolicy.RELIABLE,
                    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                    history=QoSHistoryPolicy.KEEP_LAST,
                ),
            )

    def _watchdog(self) -> None:
        """Detect a silent subscription death and rebuild the readers.

        Two cases are handled:

        1. We *had* been receiving color frames (``_color_seen``) and
           have now been silent for >= 5 s.  Classic mid-run
           subscription death from a DDS participant churn.

        2. We have *never* received color, but have been receiving depth
           or camera_info for >= 8 s.  That's a signal that the
           subscription infrastructure works (other topics on the same
           writer process are matching), but specifically the color
           subscription is dead -- or, more often, the realsense
           publisher's RGB sensor is itself wedged.  Either way the
           reader-level rebuild is worth a try; if it doesn't help, the
           escalation message at attempt 4 will tell the user to
           power-cycle the camera (unplug+replug the D415).

        Has a 5 s cooldown so we don't churn subscriptions when the
        publisher genuinely went away.

        After ~3 failed rebuilds we know the reader-level fix won't work
        (the DDS *participant* is poisoned, not just the reader, or the
        publisher's color stream is itself dead), so we escalate the log
        level and stop spamming.
        """
        now = time.monotonic()
        node_age = now - self._node_start_t
        other_topic_alive = (
            self._cam_info is not None or self._latest_depth is not None
        )
        if self._color_seen:
            if now - self._last_color_enter_t < 5.0:
                return
        elif other_topic_alive and node_age >= 8.0:
            pass  # case 2: never-saw-color but depth/info arrived
        else:
            return
        if now - self._last_resubscribe_t < 5.0:
            return
        attempt = self._resubscribe_count + 1
        case = "color-died-mid-run" if self._color_seen else "color-never-arrived"
        if attempt <= 3:
            if case == "color-died-mid-run":
                self.get_logger().warn(
                    f"No color frames for "
                    f"{now - self._last_color_enter_t:.1f}s but topic "
                    f"should be active -- the DDS reader appears to "
                    f"have unmatched.  Rebuilding subscriptions "
                    f"(attempt #{attempt})."
                )
            else:
                self.get_logger().warn(
                    f"Color never arrived after {node_age:.1f}s, even "
                    f"though depth/info matched fine -- "
                    f"realsense's RGB stream is silently wedged OR the "
                    f"color subscription specifically failed to match.  "
                    f"Rebuilding subscriptions (attempt #{attempt})."
                )
        elif attempt == 4:
            # First "we give up on the reader-level fix" message -- log
            # loud once so the user can find it after the fact, then go
            # quiet to avoid drowning the rest of the log.
            if case == "color-died-mid-run":
                self.get_logger().error(
                    f"Reader-level rebuild has failed "
                    f"{self._resubscribe_count} times in a row.  Your "
                    f"DDS *participant* is poisoned, not just the "
                    f"subscription, and no amount of resubscribing "
                    f"will help.  Kill this launch and re-run with "
                    f"`RMW_IMPLEMENTATION=rmw_fastrtps_cpp` prefixed "
                    f"(or just use ``demo.launch.py``, which now "
                    f"defaults to it).  This watchdog will keep trying "
                    f"silently in case your stack eventually recovers, "
                    f"but don't hold your breath."
                )
            else:
                self.get_logger().error(
                    f"Color subscription has been silent since "
                    f"startup ({self._resubscribe_count} rebuilds, "
                    f"node age {node_age:.1f}s) while depth and "
                    f"camera_info match cleanly.  This is almost "
                    f"always a wedged realsense RGB sensor (often left "
                    f"over from a previous run that died in "
                    f"`xioctl(VIDIOC_S_FMT)`).  Kill this launch, then "
                    f"physically unplug+replug the D415 (or run "
                    f"`pkill -9 -f realsense2_camera_node` to make "
                    f"sure no stale process is holding the camera), "
                    f"then relaunch.  This watchdog will keep trying "
                    f"silently in case the camera comes back on its "
                    f"own, but it usually won't."
                )
        # attempts >= 5 stay silent so we don't drown the log; the
        # rebuild itself runs every cycle regardless, in case the stack
        # ever does recover (e.g. a flaky network bouncing back).
        self._create_subscriptions()
        self._last_resubscribe_t = now
        self._resubscribe_count += 1
        # Reset _last_color_enter_t so the watchdog gives the new readers
        # the full 5 s cooldown to start firing before tripping again.
        self._last_color_enter_t = now

    def _on_info(self, msg: CameraInfo) -> None:
        if self._cam_info is None:
            self.get_logger().info(
                f"camera_info: first message received (K=fx={msg.k[0]:.1f},fy={msg.k[4]:.1f},"
                f"cx={msg.k[2]:.1f},cy={msg.k[5]:.1f}, size={msg.width}x{msg.height})"
            )
        self._cam_info = msg

    def _on_depth(self, msg: Image) -> None:
        if self._latest_depth is None:
            self.get_logger().info(
                f"depth: first message received ({msg.width}x{msg.height}, enc={msg.encoding})"
            )
        with self._depth_lock:
            self._latest_depth = msg

    def _on_depth_info(self, msg: CameraInfo) -> None:
        """Cache the *depth-sensor* intrinsics for manual alignment.

        Distinct from :py:meth:`_on_info` (which receives the color/aligned
        intrinsics used for landmark back-projection).  We only consume
        the K matrix here -- distortion is ignored because the realsense
        depth stream is already rectified.
        """
        if self._depth_K is None:
            self.get_logger().info(
                f"depth_info: first message received (K=fx={msg.k[0]:.1f},"
                f"fy={msg.k[4]:.1f},cx={msg.k[2]:.1f},cy={msg.k[5]:.1f}, "
                f"size={msg.width}x{msg.height})"
            )
        self._depth_K = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)

    def _on_extrinsics(self, msg) -> None:
        """Cache the depth->color rigid transform for manual alignment.

        ``realsense2_camera_msgs/Extrinsics`` carries a column-major 3x3
        rotation and a 3-vector translation in metres.  Convention from
        ``rs2_transform_point_to_point``::

            P_color = R * P_depth + t

        i.e. ``rotation`` and ``translation`` are *from* the depth optical
        frame *to* the color optical frame.  We store ``R`` row-major
        (numpy default) by transposing the column-major payload.
        """
        # The wire format is a length-9 column-major array.
        R_colmajor = np.asarray(msg.rotation, dtype=np.float64).reshape(3, 3)
        R = R_colmajor.T  # column-major -> row-major (i.e. numpy standard)
        t = np.asarray(msg.translation, dtype=np.float64)
        if self._R_dc is None:
            self.get_logger().info(
                f"extrinsics: depth->color t=({t[0]:+.3f},{t[1]:+.3f},"
                f"{t[2]:+.3f}) m, R~I+{np.linalg.norm(R - np.eye(3)):.2e}"
            )
        self._R_dc = R
        self._t_dc = t

    @staticmethod
    def _depth_to_meters(depth_img: np.ndarray) -> np.ndarray:
        """Convert RealSense 16UC1 (mm) or 32FC1 (m) to meters as float32."""
        if depth_img.dtype == np.uint16:
            return depth_img.astype(np.float32) * 1e-3
        if depth_img.dtype == np.float32:
            return depth_img
        return depth_img.astype(np.float32)

    def _median_depth(self, depth_m: np.ndarray, u: int, v: int) -> float:
        h, w = depth_m.shape
        if not (0 <= u < w and 0 <= v < h):
            return 0.0
        k = self.depth_patch
        u0, u1 = max(0, u - k), min(w, u + k + 1)
        v0, v1 = max(0, v - k), min(h, v + k + 1)
        patch = depth_m[v0:v1, u0:u1]
        valid = patch[(patch > 0.05) & (patch < 8.0)]
        if valid.size == 0:
            return 0.0
        return float(np.median(valid))

    def _z_for_color_pixel(
        self,
        u_color: float,
        v_color: float,
        depth_m: np.ndarray,
        K_color: np.ndarray,
    ) -> float:
        """Return the depth (Z, in *color* frame, meters) at a color pixel.

        In ``align_mode='realsense'`` this is a direct median lookup on
        the already-aligned depth image, identical to the pre-existing
        behavior of the node (depth-pixel and color-pixel are the same).

        In ``align_mode='manual'`` we don't have an aligned image, so we
        warp the color pixel onto the depth image using the extrinsics:

          1. Start with a guess ``Z0`` for how far the 3D point is from
             the camera (seeded from the last successful sample so the
             iteration converges in 1 step at steady state).
          2. Back-project (u_color, v_color, Z0) to a 3D point in the
             color optical frame.
          3. Apply the *inverse* depth->color extrinsics to express that
             point in the depth optical frame.
          4. Project into the depth image with the depth intrinsics.
          5. Sample depth at that pixel.
          6. Re-build the true 3D point from the sampled depth + depth
             intrinsics, and convert back to color frame.  Its Z is the
             answer; feed it back as ``Z0`` for one refinement pass to
             handle large-disparity edge cases (operator very close, or
             landmark near the edge of the color FOV).

        Returns 0.0 if any step fails so callers can treat it identically
        to ``_median_depth`` returning 0.0 (i.e. ignore the landmark).
        """
        if self.align_mode == "realsense":
            return self._median_depth(
                depth_m, int(round(u_color)), int(round(v_color))
            )

        if self._depth_K is None or self._R_dc is None or self._t_dc is None:
            # Manual mode but extrinsics / depth_info haven't arrived yet.
            return 0.0

        fxc, fyc, cxc, cyc = K_color[0, 0], K_color[1, 1], K_color[0, 2], K_color[1, 2]
        fxd, fyd, cxd, cyd = (
            self._depth_K[0, 0], self._depth_K[1, 1],
            self._depth_K[0, 2], self._depth_K[1, 2],
        )
        R = self._R_dc      # depth -> color
        t = self._t_dc

        # Ray direction in color frame (unit-Z so Z component scales it).
        ax = (u_color - cxc) / fxc
        ay = (v_color - cyc) / fyc

        Z = self._last_z_guess if self._last_z_guess > 0.1 else 2.0
        final_z = 0.0
        for _ in range(2):
            # 3D point at depth Z in color frame.
            Pc = np.array([ax * Z, ay * Z, Z], dtype=np.float64)
            # Express in depth frame: P_color = R*P_depth + t  =>
            # P_depth = R^T (P_color - t).
            Pd_guess = R.T @ (Pc - t)
            if Pd_guess[2] < 0.05:
                return 0.0
            u_d = fxd * Pd_guess[0] / Pd_guess[2] + cxd
            v_d = fyd * Pd_guess[1] / Pd_guess[2] + cyd
            ud_i = int(round(u_d))
            vd_i = int(round(v_d))
            Z_d = self._median_depth(depth_m, ud_i, vd_i)
            if Z_d <= 0.05:
                # Hole in the depth image at the warped pixel.  Fall back
                # to looking up at the *color* pixel position directly,
                # which is at worst a few-cm error on the D457 baseline
                # but better than dropping the landmark entirely.
                Z_d = self._median_depth(
                    depth_m, int(round(u_color)), int(round(v_color))
                )
                if Z_d <= 0.05:
                    return 0.0
            # True 3D point in depth frame, then convert to color frame.
            Pd_true = np.array([
                (ud_i - cxd) / fxd * Z_d,
                (vd_i - cyd) / fyd * Z_d,
                Z_d,
            ], dtype=np.float64)
            Pc_true = R @ Pd_true + t
            final_z = float(Pc_true[2])
            Z = final_z  # refine on next iteration
        if 0.1 < final_z < 8.0:
            self._last_z_guess = final_z
        return final_z

    @staticmethod
    def _solve_wrist_depth_for_forearm(
        u_w: float,
        v_w: float,
        cx: float,
        cy: float,
        fx: float,
        fy: float,
        elbow_xyz: tuple[float, float, float],
        forearm_len_m: float,
        prefer_nearer: bool = True,
    ) -> Optional[float]:
        """Solve for the wrist's camera-Z so that the back-projected wrist
        3D point is exactly ``forearm_len_m`` away from ``elbow_xyz``.

        Back-projection from a pinhole camera at depth ``Z`` gives
        ``P_w = (a*Z, b*Z, Z)`` where ``a = (u - cx)/fx`` and
        ``b = (v - cy)/fy``.  Imposing ``||P_w - P_e|| == L`` yields a
        quadratic in ``Z`` with up to two positive roots: the two points
        where the wrist back-projection ray intersects the sphere of
        radius ``L`` around the elbow.  ``prefer_nearer=True`` returns the
        smaller (closer to camera) root, which is the right answer in
        the depth-bleed scenario this helper exists for -- when the
        depth patch mistakenly samples the body behind the hand, the
        true wrist always sits in *front* of the bled-into surface, so
        the closer root is correct.  Returns None if no positive real
        root exists.
        """
        X_e, Y_e, Z_e = elbow_xyz
        a = (u_w - cx) / fx
        b = (v_w - cy) / fy
        A = a * a + b * b + 1.0
        B = -2.0 * (a * X_e + b * Y_e + Z_e)
        C = X_e * X_e + Y_e * Y_e + Z_e * Z_e - forearm_len_m * forearm_len_m
        disc = B * B - 4.0 * A * C
        if disc < 0.0:
            return None
        sq = float(np.sqrt(disc))
        z1 = (-B + sq) / (2.0 * A)
        z2 = (-B - sq) / (2.0 * A)
        candidates = [z for z in (z1, z2) if z > 0.05]
        if not candidates:
            return None
        return min(candidates) if prefer_nearer else max(candidates)

    def _heartbeat(self) -> None:
        """Independent 1 Hz heartbeat.

        Fires from a Timer callback, NOT from inside ``_on_color``, so if
        the inference call hangs the heartbeat will still print and tell
        us exactly when the color callback last fired.
        """
        idle_s = time.monotonic() - self._last_color_enter_t
        resub_str = (
            f" resub={self._resubscribe_count}"
            if self._resubscribe_count
            else ""
        )
        fixes = ""
        if self._elbow_fix_count or self._wrist_fix_count:
            fixes = (
                f" fix(el={self._elbow_fix_count},"
                f"wr={self._wrist_fix_count})"
            )
        extra_status = ""
        if self.align_mode == "manual":
            extra_status = (
                f", dK={'ok' if self._depth_K is not None else 'none'}"
                f", ext={'ok' if self._R_dc is not None else 'none'}"
            )
        self.get_logger().info(
            f"hb: in={self._frames_in} out={self._frames_out} "
            f"infer={self._last_infer_ms:.0f}ms "
            f"idle={idle_s:.1f}s "
            f"(depth={'ok' if self._latest_depth else 'none'}, "
            f"info={'ok' if self._cam_info else 'none'}{extra_status})"
            f"{fixes}"
            f"{resub_str}"
        )
        self._frames_in = 0
        self._frames_out = 0
        self._elbow_fix_count = 0
        self._wrist_fix_count = 0

    # ------------------------------------------------------------------ callback

    def _on_color(self, color_msg: Image) -> None:
        self._last_color_enter_t = time.monotonic()
        if not self._color_seen:
            self.get_logger().info(
                f"color: first message received ({color_msg.width}x{color_msg.height}, "
                f"enc={color_msg.encoding})"
            )
            self._color_seen = True
        self._frames_in += 1

        if self._cam_info is None:
            return
        with self._depth_lock:
            depth_msg = self._latest_depth
        if depth_msg is None:
            return

        try:
            self._process_frame(color_msg, depth_msg)
        except Exception:
            # Surface any silent failure that would otherwise just freeze
            # the pipeline (e.g. cv_bridge, MediaPipe internals, OpenCV).
            self.get_logger().error(
                "Pose estimation crashed:\n" + traceback.format_exc()
            )

    def _process_frame(self, color_msg: Image, depth_msg: Image) -> None:
        color = self._bridge.imgmsg_to_cv2(color_msg, desired_encoding="rgb8")
        depth_raw = self._bridge.imgmsg_to_cv2(depth_msg)

        depth_m = self._depth_to_meters(depth_raw)

        K = np.asarray(self._cam_info.k, dtype=np.float64).reshape(3, 3)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        h_color, w_color = color.shape[:2]
        # Color-pixel -> depth-pixel mapping happens inside
        # ``_z_for_color_pixel`` (a direct lookup for the aligned mode,
        # an extrinsics-based warp for the manual mode), so we don't
        # need any resolution-scaling factors here.

        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=color)
        t0 = time.monotonic()
        with self._lock:
            result = self._detector.detect(mp_image)
        self._last_infer_ms = (time.monotonic() - t0) * 1000.0

        out = PoseArray()
        out.header.stamp = color_msg.header.stamp
        out.header.frame_id = self.output_frame
        out.poses = [Pose() for _ in range(KEYPOINT_COUNT)]
        for p in out.poses:
            p.orientation.w = 0.0

        debug_img = None
        if self.publish_debug_image:
            debug_img = cv2.cvtColor(color, cv2.COLOR_RGB2BGR).copy()

        landmarks = result.pose_landmarks[0] if result.pose_landmarks else None

        if landmarks is not None:
            for kp_enum, mp_idx in MEDIAPIPE_INDEX.items():
                m = landmarks[mp_idx]
                visibility = float(getattr(m, "visibility", 1.0))

                u_color = m.x * w_color
                v_color = m.y * h_color
                z = self._z_for_color_pixel(u_color, v_color, depth_m, K)

                pose = out.poses[int(kp_enum)]
                if z > 0.0 and visibility >= self.min_confidence:
                    pose.position.x = (u_color - cx) * z / fx
                    pose.position.y = (v_color - cy) * z / fy
                    pose.position.z = z
                    pose.orientation.w = visibility
                else:
                    pose.orientation.w = 0.0

                if debug_img is not None:
                    color_dot = (0, 255, 0) if pose.orientation.w > 0 else (0, 0, 255)
                    cv2.circle(debug_img, (int(u_color), int(v_color)), 4, color_dot, -1)

            world_landmarks = (
                result.pose_world_landmarks[0]
                if getattr(result, "pose_world_landmarks", None)
                else None
            )
            if self.wrist_correction or self.elbow_correction:
                self._correct_arm_depths(
                    out,
                    landmarks=landmarks,
                    world_landmarks=world_landmarks,
                    w_color=w_color,
                    h_color=h_color,
                    cx=cx,
                    cy=cy,
                    fx=fx,
                    fy=fy,
                    debug_img=debug_img,
                )

            if debug_img is not None:
                for a, b in SKELETON_EDGES:
                    pa = out.poses[int(a)]
                    pb = out.poses[int(b)]
                    if pa.orientation.w > 0 and pb.orientation.w > 0:
                        ma = landmarks[MEDIAPIPE_INDEX[a]]
                        mb = landmarks[MEDIAPIPE_INDEX[b]]
                        cv2.line(debug_img,
                                 (int(ma.x * w_color), int(ma.y * h_color)),
                                 (int(mb.x * w_color), int(mb.y * h_color)),
                                 (255, 255, 0), 2)

        self.pub_kps.publish(out)

        if self.publish_markers:
            self._publish_markers(out)

        if debug_img is not None:
            dbg_msg = self._bridge.cv2_to_imgmsg(debug_img, encoding="bgr8")
            dbg_msg.header = color_msg.header
            self.pub_dbg.publish(dbg_msg)

        self._frames_out += 1

    # Adult anthropometric defaults & clamps used by the depth-bleed fix.
    # Upper-arm and forearm both live in the 18-40 cm range for adults;
    # clamping protects us from world-landmark blow-ups (single bad frame
    # producing a 1.2 m "forearm" can otherwise poison the back-projection
    # for several smoothed frames downstream).
    _DEFAULT_UPPER_ARM_M = 0.30
    _DEFAULT_FOREARM_M = 0.27
    _LIMB_MIN_M = 0.18
    _LIMB_MAX_M = 0.40

    def _correct_distal_depth(
        self,
        out: PoseArray,
        *,
        anchor_kp: Kp,
        distal_kp: Kp,
        default_len_m: float,
        tol_m: float,
        landmarks: list,
        world_landmarks: Optional[list],
        w_color: int,
        h_color: int,
        cx: float,
        cy: float,
        fx: float,
        fy: float,
        debug_img: Optional[np.ndarray],
        debug_color: tuple[int, int, int],
    ) -> int:
        """Re-back-project ``distal_kp`` so the 3D ``anchor_kp -> distal_kp``
        distance matches MediaPipe's world-landmark limb length.

        Same trick as the wrist correction (and uses the same closed-form
        quadratic), generalised so the elbow can be corrected too: the
        anchor's 3D position is trusted (it's higher up the kinematic chain
        and less likely to bleed into the body in this pose), and we
        re-solve for the distal joint's Z along its pixel back-projection
        ray to match the world-landmark limb length.

        Returns 1 if a correction was applied, 0 if not.
        """
        anchor_pose = out.poses[int(anchor_kp)]
        distal_pose = out.poses[int(distal_kp)]
        if anchor_pose.orientation.w <= 0.0:
            return 0

        md = landmarks[MEDIAPIPE_INDEX[distal_kp]]
        if float(getattr(md, "visibility", 0.0)) < self.min_confidence:
            return 0

        P_a = (
            anchor_pose.position.x,
            anchor_pose.position.y,
            anchor_pose.position.z,
        )
        if distal_pose.orientation.w > 0.0:
            dx = distal_pose.position.x - P_a[0]
            dy = distal_pose.position.y - P_a[1]
            dz = distal_pose.position.z - P_a[2]
            d_observed = float(np.sqrt(dx * dx + dy * dy + dz * dz))
        else:
            d_observed = float("inf")

        if world_landmarks is not None:
            wa = world_landmarks[MEDIAPIPE_INDEX[anchor_kp]]
            wd = world_landmarks[MEDIAPIPE_INDEX[distal_kp]]
            limb_len_m = float(np.sqrt(
                (wd.x - wa.x) ** 2
                + (wd.y - wa.y) ** 2
                + (wd.z - wa.z) ** 2
            ))
        else:
            limb_len_m = default_len_m

        limb_len_m = float(np.clip(limb_len_m, self._LIMB_MIN_M, self._LIMB_MAX_M))

        needs_fix = (
            d_observed == float("inf")
            or abs(d_observed - limb_len_m) > tol_m
        )
        if not needs_fix:
            return 0

        u_d = md.x * w_color
        v_d = md.y * h_color
        z_new = self._solve_wrist_depth_for_forearm(
            u_w=u_d, v_w=v_d,
            cx=cx, cy=cy, fx=fx, fy=fy,
            elbow_xyz=P_a,
            forearm_len_m=limb_len_m,
        )
        if z_new is None:
            return 0

        distal_pose.position.x = (u_d - cx) * z_new / fx
        distal_pose.position.y = (v_d - cy) * z_new / fy
        distal_pose.position.z = z_new
        distal_pose.orientation.w = max(
            float(getattr(md, "visibility", 0.0)),
            self.min_confidence,
        )
        if debug_img is not None:
            cv2.circle(
                debug_img, (int(u_d), int(v_d)), 7, debug_color, 2
            )
        return 1

    def _correct_arm_depths(
        self,
        out: PoseArray,
        *,
        landmarks: list,
        world_landmarks: Optional[list],
        w_color: int,
        h_color: int,
        cx: float,
        cy: float,
        fx: float,
        fy: float,
        debug_img: Optional[np.ndarray],
    ) -> None:
        """Run depth-bleed correction on both arms.

        Two passes per side: shoulder -> elbow first (so the wrist
        correction below sees the corrected elbow), then elbow -> wrist.
        Corrected joints are drawn in the debug image: magenta rings for
        the wrist, cyan rings for the elbow.
        """
        for shoulder, elbow, wrist in (
            (Kp.LEFT_SHOULDER,  Kp.LEFT_ELBOW,  Kp.LEFT_WRIST),
            (Kp.RIGHT_SHOULDER, Kp.RIGHT_ELBOW, Kp.RIGHT_WRIST),
        ):
            if self.elbow_correction:
                self._elbow_fix_count += self._correct_distal_depth(
                    out,
                    anchor_kp=shoulder,
                    distal_kp=elbow,
                    default_len_m=self._DEFAULT_UPPER_ARM_M,
                    tol_m=self.elbow_upper_arm_tol_m,
                    landmarks=landmarks,
                    world_landmarks=world_landmarks,
                    w_color=w_color,
                    h_color=h_color,
                    cx=cx, cy=cy, fx=fx, fy=fy,
                    debug_img=debug_img,
                    debug_color=(255, 255, 0),  # cyan
                )
            if self.wrist_correction:
                self._wrist_fix_count += self._correct_distal_depth(
                    out,
                    anchor_kp=elbow,
                    distal_kp=wrist,
                    default_len_m=self._DEFAULT_FOREARM_M,
                    tol_m=self.wrist_forearm_tol_m,
                    landmarks=landmarks,
                    world_landmarks=world_landmarks,
                    w_color=w_color,
                    h_color=h_color,
                    cx=cx, cy=cy, fx=fx, fy=fy,
                    debug_img=debug_img,
                    debug_color=(255, 0, 255),  # magenta
                )

    def _publish_markers(self, kps: PoseArray) -> None:
        ma = MarkerArray()

        points = Marker()
        points.header = kps.header
        points.ns = "keypoints"
        points.id = 0
        points.type = Marker.SPHERE_LIST
        points.action = Marker.ADD
        points.scale.x = points.scale.y = points.scale.z = 0.04
        points.color = ColorRGBA(r=0.1, g=1.0, b=0.1, a=1.0)
        points.pose.orientation.w = 1.0
        for p in kps.poses:
            if p.orientation.w > 0:
                points.points.append(p.position)
        ma.markers.append(points)

        edges = Marker()
        edges.header = kps.header
        edges.ns = "skeleton"
        edges.id = 1
        edges.type = Marker.LINE_LIST
        edges.action = Marker.ADD
        edges.scale.x = 0.015
        edges.color = ColorRGBA(r=1.0, g=0.9, b=0.2, a=1.0)
        edges.pose.orientation.w = 1.0
        for a, b in SKELETON_EDGES:
            pa = kps.poses[int(a)]
            pb = kps.poses[int(b)]
            if pa.orientation.w > 0 and pb.orientation.w > 0:
                edges.points.extend([pa.position, pb.position])
        ma.markers.append(edges)

        self.pub_markers.publish(ma)


def main(argv: list[str] | None = None) -> None:
    rclpy.init(args=argv)
    node = PoseEstimatorNode()
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
