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
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    retargeter_yaml = PathJoinSubstitution([
        FindPackageShare("humanoid_mimic_bringup"), "config", "retargeter.yaml",
    ])

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
            # Hardware-reset the camera at startup.  Without this, killing
            # a previous demo with Ctrl-C leaves the D415 in a half-open
            # state on the USB bus; the next launch opens streams, emits
            # one or two frames, then silently dies with
            # ``Hardware Notification: Depth stream start failure`` and
            # ``get_xu(ctrl=1) failed! Last Error: Device or resource busy``.
            # ``initial_reset:=true`` issues a hard reset that clears that
            # stale state.  Adds ~2 s to bring-up.
            "initial_reset": True,
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
        parameters=[retargeter_yaml],
    )

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        realsense,
        pose,
    ])
