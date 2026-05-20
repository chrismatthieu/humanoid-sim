"""Launch Gazebo Sim with the G1 spawned and ros2_control controllers active.

Useful on its own for sanity-checking the model and joint commanding loop
without bringing up the camera or pose pipeline.  Try:

    ros2 topic pub /upper_body_controller/commands std_msgs/Float64MultiArray \\
      "data: [0, 0,0,0, 0.5, 0,0,0, 0,0,0, 0.5, 0,0,0]" --once

(the 5th and 12th entries are left/right elbow_joint).
"""

from __future__ import annotations

import os
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _build_robot_description(context, *args, **kwargs):
    """Run process_g1_urdf.py to produce the Gazebo-ready URDF string."""
    import importlib.util

    g1_share = get_package_share_directory("g1_description")
    bringup_share = get_package_share_directory("humanoid_mimic_bringup")

    source_urdf = os.path.join(g1_share, "urdf", "g1_29dof_lock_waist.urdf")
    controllers_yaml = os.path.join(bringup_share, "config", "controllers.yaml")
    processor_py = os.path.join(g1_share, "scripts", "process_g1_urdf.py")

    spec = importlib.util.spec_from_file_location("process_g1_urdf", processor_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {processor_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    urdf_xml: str = mod.generate(
        source_urdf=source_urdf,
        controllers_yaml=controllers_yaml,
    )

    # Persist to /tmp so we (and `ros2 run robot_state_publisher`) can also
    # inspect it easily if needed.
    out_path = os.path.join(tempfile.gettempdir(), "g1_gz.urdf")
    with open(out_path, "w") as fh:
        fh.write(urdf_xml)

    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{
            "robot_description": urdf_xml,
            "use_sim_time": True,
        }],
    )

    spawn = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-name", "g1",
            "-topic", "robot_description",
            "-x", "0", "-y", "0", "-z", "0",
        ],
        output="screen",
    )

    return [rsp, spawn]


def generate_launch_description() -> LaunchDescription:
    bringup_share = FindPackageShare("humanoid_mimic_bringup")
    rviz_cfg = PathJoinSubstitution([bringup_share, "rviz", "demo.rviz"])

    # Tell Gazebo where to find the meshes we vendored.
    bringup_share_dir = get_package_share_directory("humanoid_mimic_bringup")
    g1_share = get_package_share_directory("g1_description")
    extra_resource = os.path.dirname(g1_share)  # parent so package:// resolves
    existing = os.environ.get("GZ_SIM_RESOURCE_PATH", "")
    os.environ["GZ_SIM_RESOURCE_PATH"] = (
        f"{extra_resource}:{existing}" if existing else extra_resource
    )
    world_path_str = os.path.join(bringup_share_dir, "worlds", "mimic.sdf")

    # CPU affinity (see camera.launch.py comment about cores 0-1).  On the
    # N100 (4 cores) we pin Gazebo to cores 2-3 so it physically cannot
    # preempt the realsense USB transfer thread and MediaPipe inference,
    # which are pinned to cores 0-1.  Without this, the gazebo bring-up
    # spike silently starves the realsense node enough to overrun the
    # D415's UVC buffer and freeze color delivery for the rest of the run.
    #
    # We launch ``gz sim`` ourselves rather than going through
    # ``ros_gz_sim``'s ``gz_sim.launch.py`` because ``IncludeLaunchDescription``
    # has no ``prefix=`` knob -- the cleanest way to get ``taskset`` in
    # front of the gazebo binary is to do the ExecuteProcess directly.
    gz_sim = ExecuteProcess(
        cmd=[
            "taskset", "-c", "2,3",
            "gz", "sim", world_path_str,
            "-r", "--render-engine", "ogre2",
        ],
        output="screen",
        # If gazebo dies, tear down the whole launch.
        on_exit=__import__("launch.actions", fromlist=["Shutdown"]).Shutdown(),
    )

    # ros2_control spawners.  ``taskset -c 2,3`` keeps the spawners on
    # gazebo's cores too, so they can never steal cycles from realsense.
    # NB: launch_ros's ``prefix`` is joined to a *single* string with no
    # separators -- pass it as one string, not a list, or you get errors
    # like ``FileNotFoundError: 'taskset-c2,3'``.
    spawner_prefix = "taskset -c 2,3"
    jsb_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
        output="screen",
        prefix=spawner_prefix,
    )
    ub_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["upper_body_controller", "--controller-manager", "/controller_manager"],
        output="screen",
        prefix=spawner_prefix,
    )

    # /clock bridge so ROS nodes that opt into use_sim_time get sim time.
    clock_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
        output="screen",
    )

    use_rviz = LaunchConfiguration("use_rviz")
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_cfg],
        condition=__import__("launch.conditions", fromlist=["IfCondition"]).IfCondition(use_rviz),
        output="screen",
        # Keep RViz off the camera's cores (see gazebo comment above).
        prefix="taskset -c 2,3",
    )

    return LaunchDescription([
        DeclareLaunchArgument("use_rviz", default_value="false"),
        gz_sim,
        clock_bridge,
        OpaqueFunction(function=_build_robot_description),
        # Sequence: spawn -> jsb -> upper_body
        RegisterEventHandler(
            OnProcessExit(target_action=jsb_spawner, on_exit=[ub_spawner])
        ),
        jsb_spawner,
        rviz,
    ])
