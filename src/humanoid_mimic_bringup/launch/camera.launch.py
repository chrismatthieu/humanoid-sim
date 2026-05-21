"""Bring up the RealSense camera and the MediaPipe-based pose estimator only.

Useful for tuning the pose pipeline without spinning up the simulator.

We launch the realsense2_camera node *directly* rather than going through the
upstream rs_launch.py.  rs_launch.py munges dotted parameter names (e.g.
`align_depth.enable`) when they are passed via IncludeLaunchDescription's
`launch_arguments=`, which silently leaves the aligned depth disabled.
Calling the node directly with `parameters=[...]` always works.

Camera + pose CPU affinity is controlled by the machine profile (see
``_machine_profile.py``).  We pin the realsense USB thread and MediaPipe
inference to cores 0-1 on the 4-core N100 because the Gazebo bring-up
storm starves the USB transfer thread otherwise; on the Jetson AGX Orin
and other >=8-core machines we leave the OS scheduler alone.
"""

from __future__ import annotations

import importlib.util
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def _load_machine_profile():
    """See sim.launch.py for the rationale on sys.modules registration."""
    import sys
    name = "_humanoid_machine_profile"
    if name in sys.modules:
        return sys.modules[name]
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(here, "_machine_profile.py"),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load _machine_profile.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def generate_launch_description() -> LaunchDescription:
    profile_mod = _load_machine_profile()
    profile = profile_mod.detect()

    retargeter_yaml = PathJoinSubstitution([
        FindPackageShare("humanoid_mimic_bringup"), "config", "retargeter.yaml",
    ])
    # Override of pose_estimator.model_complexity.  0=lite, 1=full, 2=heavy.
    # On a 4-core CPU (N100), 0 keeps the inference loop comfortably under
    # the 30 Hz camera frame budget; 1 buys substantially better landmark
    # accuracy (helps with hands-on-hips & folded-arms poses) at ~2x CPU.
    # The Jetson AGX Orin's 8x Cortex-A78AE handle 1 (and arguably 2)
    # comfortably with cycles to spare.
    model_complexity = LaunchConfiguration("model_complexity")

    # NB: launch_ros's ``prefix`` is joined into a *single* string with no
    # separator -- pass one string, NOT a list (``["taskset","-c","0,1"]``
    # would try to exec the literal ``taskset-c0,1``).  The profile gives
    # us an empty string when affinity is disabled, which makes Node()
    # behave as if no prefix was passed.
    camera_prefix = profile.taskset(profile.camera_cores)

    realsense = Node(
        package="realsense2_camera",
        executable="realsense2_camera_node",
        name="camera",
        namespace="camera",
        output="screen",
        prefix=camera_prefix,
        parameters=[{
            # NB: ``initial_reset:=true`` is intentionally *off*.  It was
            # tempting because it papers over the "Device or resource
            # busy" wedge from a previous unclean Ctrl-C, but on the D415
            # the reset is asynchronous -- librealsense issues
            # ``rs2::device::hardware_reset()`` and returns immediately,
            # the kernel re-enumerates the device 5-6 s later with a new
            # ``/dev/videoN`` node, and *that* re-enumeration arrives
            # in the middle of streaming and kills the sensor with
            # ``xioctl(VIDIOC_S_FMT) failed, errno=5 Input/output error``.
            # We saw exactly that pattern in the logs.  The same caveat
            # applies to the D457 we use on the Jetson.  Leave the reset
            # off; if you do hit "Device or resource busy" (rare, only
            # after a SIGKILL of the previous launch), unplug+replug the
            # camera once or run
            # ``sudo bash -c 'echo -1 > /sys/module/usbcore/parameters/autosuspend'``
            # to let the kernel recycle the port.
            "initial_reset": False,
            # Streams
            "enable_color": True,
            "enable_depth": True,
            "enable_infra1": False,
            "enable_infra2": False,
            "enable_gyro": False,
            "enable_accel": False,
            # Alignment + topic generation.
            #
            # In the ``realsense`` align mode we ask the SDK to publish
            # the aligned-depth-to-color stream and pair frames with the
            # sync filter (``align_depth.enable=true`` alone is *not*
            # enough on realsense2_camera 4.57.x: the depth-to-color
            # align processor only runs frames through when
            # ``enable_sync=true``, see
            # ``/opt/ros/humble/share/realsense2_camera/examples/align_depth/``
            # which sets both together).
            #
            # In the ``manual`` align mode (Jetson + D457 over GMSL2)
            # the SDK align filter is silent regardless of these flags
            # because UVC frame metadata isn't exposed on that hardware
            # path, so we don't waste cycles asking for it.  We disable
            # the sync filter too to remove its added latency, and we
            # let the pose-estimator node do the warp from raw depth +
            # extrinsics.
            "enable_sync": profile.align_mode == "realsense",
            "align_depth.enable": profile.align_mode == "realsense",
            "pointcloud.enable": False,
            # Resolutions kept modest to stay real-time on the N100; the
            # Jetson has the headroom to go higher but 640x480 keeps the
            # pose-estimator latency identical across hosts.
            "rgb_camera.color_profile": "640,480,30",
            "depth_module.depth_profile": "640,480,30",
            "publish_tf": True,
        }],
    )

    pose = Node(
        package="humanoid_pose_estimator",
        executable="pose_estimator_node",
        name="humanoid_pose_estimator",
        output="screen",
        prefix=camera_prefix,
        parameters=[
            retargeter_yaml,
            {"model_complexity": ParameterValue(model_complexity, value_type=int)},
            # ``align_mode`` is fed in from the machine profile so the
            # estimator picks the right depth topology automatically:
            # "realsense" on x86 / D415, "manual" on Jetson / D457.  In
            # manual mode the node defaults to subscribing to
            # /camera/camera/depth/image_rect_raw and warps each
            # landmark using the depth->color extrinsics topic.  See
            # the pose_estimator_node module docstring.
            {"align_mode": profile.align_mode},
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument(
            "model_complexity",
            default_value="0",
            choices=["0", "1", "2"],
            description=(
                "MediaPipe Pose model: 0=lite (fast, default), 1=full "
                "(better wrist/elbow localization, ~2x CPU), 2=heavy."
            ),
        ),
        LogInfo(msg="[humanoid-sim camera] " + " | ".join(profile.notes)),
        realsense,
        pose,
    ])
