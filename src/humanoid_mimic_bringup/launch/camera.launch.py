"""Bring up the RealSense camera and the MediaPipe-based pose estimator only.

Useful for tuning the pose pipeline without spinning up the simulator.

We launch the realsense2_camera node *directly* rather than going through the
upstream rs_launch.py.  rs_launch.py munges dotted parameter names (e.g.
`align_depth.enable`) when they are passed via IncludeLaunchDescription's
`launch_arguments=`, which silently leaves the aligned depth disabled.
Calling the node directly with `parameters=[...]` always works.
"""

from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    retargeter_yaml = PathJoinSubstitution([
        FindPackageShare("humanoid_mimic_bringup"), "config", "retargeter.yaml",
    ])
    # Override of pose_estimator.model_complexity.  0=lite, 1=full, 2=heavy.
    # Default 0 keeps the N100 inference loop comfortably under the 30 Hz
    # camera frame budget; 1 buys substantially better landmark accuracy
    # (helps with hands-on-hips & folded-arms poses) at ~2x CPU cost.
    model_complexity = LaunchConfiguration("model_complexity")

    # CPU affinity: pin the camera's USB transfer thread and MediaPipe
    # inference to cores 0-1, leaving cores 2-3 for Gazebo + RViz (pinned
    # there in ``sim.launch.py``).  Without this, the gazebo bring-up
    # spike preempts the realsense thread long enough to overrun the
    # D415's UVC buffer; after that the camera firmware silently stops
    # delivering frames for the rest of the run.  ``taskset`` ships with
    # ``util-linux`` so no extra install.
    # NB: launch_ros's ``prefix`` is joined into a *single* string with no
    # separator -- pass one string, NOT a list (``["taskset","-c","0,1"]``
    # would try to exec the literal ``taskset-c0,1``).
    camera_prefix = "taskset -c 0,1"

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
            # We saw exactly that pattern in the logs.  Leave the reset
            # off; if you do hit "Device or resource busy" (rare, only
            # after a SIGKILL of the previous launch), unplug+replug the
            # D415 once or run
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
            # Alignment + topic generation
            "align_depth.enable": True,
            "pointcloud.enable": False,
            # Resolutions kept modest to stay real-time on the N100.
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
        realsense,
        pose,
    ])
