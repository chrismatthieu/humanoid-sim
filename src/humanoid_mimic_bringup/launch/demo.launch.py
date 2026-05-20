"""Top-level launch: camera + pose estimator + retargeter + Gazebo + G1.

This is the demo entry point.
"""

from __future__ import annotations

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    bringup = FindPackageShare("humanoid_mimic_bringup")
    retargeter_yaml = PathJoinSubstitution([bringup, "config", "retargeter.yaml"])

    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([bringup, "launch", "sim.launch.py"])
        ),
        launch_arguments={"use_rviz": LaunchConfiguration("use_rviz")}.items(),
    )

    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([bringup, "launch", "camera.launch.py"])
        ),
    )

    retargeter = Node(
        package="humanoid_retargeter",
        executable="retargeter_node",
        name="humanoid_retargeter",
        output="screen",
        parameters=[retargeter_yaml],
    )

    # Anchor the camera 4 m behind the G1, 1.2 m off the floor, facing the
    # robot along world +x.  We attach to `camera_link` (the ROS-convention
    # frame published by realsense2_camera: x=forward, y=left, z=up) so that
    # the realsense node's own internal TF chain
    #   camera_link -> camera_color_frame -> camera_color_optical_frame
    # remains the unique parent of the optical frame.  Publishing
    # `world -> camera_color_optical_frame` directly would give the optical
    # frame two parents and break the TF tree (which is what RViz was
    # complaining about).
    camera_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="world_to_camera_link_tf",
        arguments=[
            "--x", "-4.0", "--y", "0.0", "--z", "1.2",
            "--roll", "0.0", "--pitch", "0.0", "--yaw", "0.0",
            "--frame-id", "world",
            "--child-frame-id", "camera_link",
        ],
        output="screen",
    )

    # On a CPU-only N100, bringing up Gazebo, the controllers, the realsense
    # node, MediaPipe and the retargeter simultaneously pegs all four cores
    # for ~10 s.  Empirically that starves the realsense USB transfer
    # thread, and even once the streams start, the resulting buffer overrun
    # silently freezes color delivery for the rest of the run (heartbeats
    # show ``in=0 out=0`` forever, and we never recover).
    #
    # Mitigations layered together:
    #   1. CPU affinity (see ``sim.launch.py`` / ``camera.launch.py``):
    #      pin realsense + pose_estimator to cores 0-1 and gazebo + rviz
    #      to cores 2-3 via ``taskset``.  This is the load-bearing fix --
    #      the USB transfer thread now has guaranteed CPU and cannot be
    #      preempted by gazebo's startup spike.
    #   2. Bring the camera up first and wait 12 s for its USB streams to
    #      stabilize before launching gazebo.  Belt to the affinity
    #      suspenders.
    #   3. Default ``use_rviz=false``.  RViz adds another big CPU spike
    #      and a second consumer of the iGPU; even pinned to cores 2-3 it
    #      slows bring-up noticeably.  For the pose-debug image, prefer
    #      ``ros2 run rqt_image_view rqt_image_view`` (and select
    #      ``/human/debug_image``), or just open RViz manually once the
    #      heartbeats are steady:
    #         ``taskset -c 2,3 ros2 run rviz2 rviz2 \
    #             -d src/humanoid_mimic_bringup/rviz/demo.rviz``.
    #      Pass ``use_rviz:=true`` to include it in the demo (slower).
    return LaunchDescription([
        DeclareLaunchArgument("use_rviz", default_value="false"),
        camera_launch,
        camera_tf,
        TimerAction(period=12.0, actions=[sim_launch, retargeter]),
    ])
