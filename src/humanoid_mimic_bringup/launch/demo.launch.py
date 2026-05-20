"""Top-level launch: camera + pose estimator + retargeter + Gazebo + G1.

This is the demo entry point.
"""

from __future__ import annotations

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _sim_launch(bringup):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([bringup, "launch", "sim.launch.py"])
        ),
        launch_arguments={
            "use_rviz": LaunchConfiguration("use_rviz"),
            "headless": LaunchConfiguration("headless"),
        }.items(),
    )


def _retargeter(retargeter_yaml):
    return Node(
        package="humanoid_retargeter",
        executable="retargeter_node",
        name="humanoid_retargeter",
        output="screen",
        parameters=[retargeter_yaml],
    )


def generate_launch_description() -> LaunchDescription:
    bringup = FindPackageShare("humanoid_mimic_bringup")
    retargeter_yaml = PathJoinSubstitution([bringup, "config", "retargeter.yaml"])

    # Default the whole launch to Fast-DDS.  Jazzy ships with Cyclone DDS
    # as the system default, but Cyclone has a participant-poisoning bug
    # on this workspace's bring-up profile: when gazebo + ros2_control
    # bring ~6 new DDS participants up in a ~2 s window, any pre-existing
    # subscription on the bus silently un-matches from its writer and
    # cannot be recovered (we tried -- the watchdog in
    # ``pose_estimator_node`` will retry 3 times then give up).  The
    # symptom is heartbeats freezing at ``in=0`` for the rest of the run
    # even though ``ros2 topic hz`` shows the writer publishing happily.
    # Fast-DDS' discovery is robust to the same churn.
    #
    # We force this unconditionally rather than "honour an already-set
    # env var" because Jazzy's system setup *already* sets
    # ``RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`` for you, which means
    # there's no way to distinguish "user wants Cyclone" from "user
    # never touched the env var".  If you genuinely need Cyclone (for
    # debugging interop with another machine that uses Cyclone, say),
    # pass ``dds:=cyclonedds`` on the command line.
    dds = LaunchConfiguration("dds")
    rmw_default = SetEnvironmentVariable(
        "RMW_IMPLEMENTATION",
        PythonExpression([
            "'rmw_fastrtps_cpp' if '", dds,
            "' == 'fastrtps' else 'rmw_cyclonedds_cpp'",
        ]),
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
    #      to cores 2-3 via ``taskset``.
    #   2. ``headless:=true`` (server-only gazebo, no Ogre) avoids any
    #      iGPU contention with MediaPipe's EGL context.
    #   3. ``bringup_order:=gazebo_first`` brings the camera up *after*
    #      gazebo's startup storm has fully settled, so realsense's USB
    #      stream and MediaPipe's first inference happen in a quiet
    #      environment.  This sidesteps the cross-affinity disturbance
    #      (page-fault/mmap floods, DDS discovery, memory-bandwidth spike)
    #      that affinity alone cannot isolate.  Trade-off: bring-up time
    #      goes from ~15 s (camera_first) to ~35 s.  Default is
    #      ``camera_first`` for fast iteration; switch to ``gazebo_first``
    #      if the camera dies on gazebo start despite headless + affinity.
    #   4. Default ``use_rviz=false``.  Use rqt_image_view for the pose
    #      debug image, or launch RViz manually once heartbeats are steady
    #      (and pin it to cores 2-3):
    #         ``taskset -c 2,3 ros2 run rviz2 rviz2 \
    #             -d src/humanoid_mimic_bringup/rviz/demo.rviz``.
    bringup_order = LaunchConfiguration("bringup_order")
    is_camera_first = IfCondition(
        PythonExpression(["'", bringup_order, "' == 'camera_first'"])
    )
    is_gazebo_first = IfCondition(
        PythonExpression(["'", bringup_order, "' == 'gazebo_first'"])
    )

    # camera_first: camera at t=0, gazebo + retargeter at t=12.
    cf_camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([bringup, "launch", "camera.launch.py"])
        ),
        condition=is_camera_first,
    )
    cf_delayed = TimerAction(
        period=12.0,
        actions=[_sim_launch(bringup), _retargeter(retargeter_yaml)],
        condition=is_camera_first,
    )

    # gazebo_first: gazebo + retargeter at t=0, camera at t=30.
    gf_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([bringup, "launch", "sim.launch.py"])
        ),
        launch_arguments={
            "use_rviz": LaunchConfiguration("use_rviz"),
            "headless": LaunchConfiguration("headless"),
        }.items(),
        condition=is_gazebo_first,
    )
    gf_retargeter = Node(
        package="humanoid_retargeter",
        executable="retargeter_node",
        name="humanoid_retargeter",
        output="screen",
        parameters=[retargeter_yaml],
        condition=is_gazebo_first,
    )
    gf_camera_delayed = TimerAction(
        period=30.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([
                        bringup, "launch", "camera.launch.py",
                    ])
                ),
            ),
        ],
        condition=is_gazebo_first,
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "dds",
            default_value="fastrtps",
            choices=["fastrtps", "cyclonedds"],
            description=(
                "Which DDS implementation to use.  Defaults to Fast-DDS "
                "to sidestep Cyclone DDS' participant-poisoning bug on "
                "this workspace's bring-up profile (see the long comment "
                "in this launch file)."
            ),
        ),
        rmw_default,
        DeclareLaunchArgument("use_rviz", default_value="false"),
        DeclareLaunchArgument(
            "headless",
            default_value="false",
            description=(
                "Run gazebo server-only (no GUI).  Removes Ogre2 from the "
                "iGPU; required if your machine shares Mesa between gazebo "
                "and MediaPipe."
            ),
        ),
        DeclareLaunchArgument(
            "bringup_order",
            default_value="camera_first",
            choices=["camera_first", "gazebo_first"],
            description=(
                "camera_first: camera at t=0, gazebo at t=12 (default, "
                "fast bring-up).  gazebo_first: gazebo at t=0, camera at "
                "t=30 -- lets gazebo's startup storm finish before "
                "realsense's USB stream comes up, which sidesteps the "
                "cross-affinity disturbance that kills the camera on "
                "shared-memory-channel CPUs like the N100."
            ),
        ),
        camera_tf,
        cf_camera,
        cf_delayed,
        gf_sim,
        gf_retargeter,
        gf_camera_delayed,
    ])
