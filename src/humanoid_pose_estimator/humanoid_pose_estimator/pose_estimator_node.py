"""ROS 2 node: MediaPipe PoseLandmarker (Tasks API) + RealSense aligned depth
=> 3D human keypoints.

Subscribes (color triggers inference, depth + info are latest-cached):
  - color image  (sensor_msgs/Image, encoding rgb8 or bgr8)
  - aligned depth image  (sensor_msgs/Image, 16UC1 in mm or 32FC1 in m)
  - depth camera_info  (sensor_msgs/CameraInfo, captured once)

Publishes:
  - /human/keypoints (geometry_msgs/PoseArray) in the color optical frame.
    Each Pose.position is the 3D point of a landmark; Pose.orientation.w
    carries the MediaPipe `visibility` in 0..1.  Indices follow `keypoints.Kp`.
  - /human/markers (visualization_msgs/MarkerArray) for RViz.
  - /human/debug_image (sensor_msgs/Image) overlay of the pose, for debugging.

Model files (`pose_landmarker_{lite,full,heavy}.task`) are auto-downloaded
to ~/.cache/humanoid_pose_estimator/models on first use.
"""

from __future__ import annotations

import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

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
        self.declare_parameter("output_frame", "camera_color_optical_frame")
        self.declare_parameter("min_confidence", 0.5)
        self.declare_parameter("depth_patch", 5)
        self.declare_parameter("model_complexity", 0)
        self.declare_parameter("model_path", "")  # if non-empty, overrides the download
        self.declare_parameter("model_cache_dir", "")  # default ~/.cache/humanoid_pose_estimator/models
        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter("publish_markers", True)

        gp = self.get_parameter
        self.color_topic: str = gp("color_topic").value
        self.depth_topic: str = gp("depth_topic").value
        self.info_topic: str = gp("info_topic").value
        self.output_frame: str = gp("output_frame").value
        self.min_confidence: float = float(gp("min_confidence").value)
        self.depth_patch: int = int(gp("depth_patch").value)
        self.model_complexity: int = int(gp("model_complexity").value)
        self.publish_debug_image: bool = bool(gp("publish_debug_image").value)
        self.publish_markers: bool = bool(gp("publish_markers").value)

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

        self.pub_kps = self.create_publisher(PoseArray, "/human/keypoints", 10)
        self.pub_markers = self.create_publisher(MarkerArray, "/human/markers", 10)
        self.pub_dbg = self.create_publisher(Image, "/human/debug_image", 10)

        # Single-threaded executor (see ``main``) drives all three callbacks
        # from one thread.  MediaPipe's PoseLandmarker initializes a GL/EGL
        # context lazily on the first ``detect_for_video`` call and binds
        # that context to the calling thread, so we must keep all inference
        # on the same thread.  The deeper QoS queue absorbs inference jitter
        # (depth=10 lets ~330 ms of buffered frames survive before drops).
        self._info_sub = self.create_subscription(
            CameraInfo, self.info_topic, self._on_info, _sensor_qos(depth=5)
        )
        self._depth_sub = self.create_subscription(
            Image, self.depth_topic, self._on_depth, _sensor_qos(depth=10)
        )
        self._color_sub = self.create_subscription(
            Image, self.color_topic, self._on_color, _sensor_qos(depth=10)
        )

        # Heartbeat on a real timer (independent of color callback firing).
        # If this keeps ticking while _on_color stops, we know detect() hung.
        self._heartbeat_timer = self.create_timer(1.0, self._heartbeat)

        self.get_logger().info(
            f"pose_estimator up; color={self.color_topic} depth={self.depth_topic} "
            f"info={self.info_topic} frame={self.output_frame}"
        )

    # ------------------------------------------------------------------ helpers

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

    def _heartbeat(self) -> None:
        """Independent 1 Hz heartbeat.

        Fires from a Timer callback, NOT from inside ``_on_color``, so if
        the inference call hangs the heartbeat will still print and tell
        us exactly when the color callback last fired.
        """
        idle_s = time.monotonic() - self._last_color_enter_t
        self.get_logger().info(
            f"hb: in={self._frames_in} out={self._frames_out} "
            f"infer={self._last_infer_ms:.0f}ms "
            f"idle={idle_s:.1f}s "
            f"(depth={'ok' if self._latest_depth else 'none'}, "
            f"info={'ok' if self._cam_info else 'none'})"
        )
        self._frames_in = 0
        self._frames_out = 0

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
        h_depth, w_depth = depth_m.shape
        scale_u = w_depth / w_color
        scale_v = h_depth / h_color

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
                ud = int(round(u_color * scale_u))
                vd = int(round(v_color * scale_v))
                z = self._median_depth(depth_m, ud, vd)

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
