"""Launch Gazebo Sim with the G1 spawned and ros2_control controllers active.

Useful on its own for sanity-checking the model and joint commanding loop
without bringing up the camera or pose pipeline.  Try:

    ros2 topic pub /upper_body_controller/commands std_msgs/Float64MultiArray \\
      "data: [0, 0,0,0, 0.5, 0,0,0, 0,0,0, 0.5, 0,0,0]" --once

(the 5th and 12th entries are left/right elbow_joint).

This launch auto-detects which Gazebo backend to use based on the current
ROS distro:

  * Jazzy (Ubuntu 24.04, ROS 2 Jazzy)     -> ``gz sim`` (Gazebo Harmonic).
  * Humble (Ubuntu 22.04, ROS 2 Humble) -> ``ign gazebo`` (Gazebo
    Fortress).  Required on the Jetson AGX Orin because the apt-installed
    ``ros-humble-ros-gz-*`` and ``ros-humble-gz-ros2-control`` packages
    are linked against ignition-gazebo6 / ignition-transport11; they
    *cannot* talk to a Harmonic ``gz sim`` server even if it is on the
    same machine, so the spawned ``create`` and ``parameter_bridge``
    processes hang forever in "Requesting list of world names".

See ``_machine_profile.py`` for the full decision table; override the
backend with ``HUMANOID_SIM_GZ_BACKEND={harmonic,fortress}`` and the
CPU pinning with ``HUMANOID_SIM_USE_AFFINITY={0,1}``.
"""

from __future__ import annotations

import importlib.util
import os
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
    Shutdown,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _load_machine_profile():
    """Import ``_machine_profile`` from the installed launch directory.

    We use importlib so this launch file works whether it's run from the
    install space (``share/humanoid_mimic_bringup/launch``) or the source
    tree, without needing the launch dir to be a proper Python package.
    The module is registered in ``sys.modules`` BEFORE ``exec_module`` --
    Python 3.10's ``@dataclass`` decorator (used inside the module) looks
    up ``sys.modules[cls.__module__]`` at class-creation time and raises
    ``AttributeError: 'NoneType' object has no attribute '__dict__'``
    if the module isn't registered yet.  This bites only on Humble/Jetson
    where Python is 3.10; 3.12 (Jazzy) is more permissive but the
    registration is harmless there.
    """
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


def _build_robot_description(context, *args, **kwargs):
    """Run process_g1_urdf.py to produce the Gazebo-ready URDF string."""

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
    profile_mod = _load_machine_profile()
    profile = profile_mod.detect()

    bringup_share = FindPackageShare("humanoid_mimic_bringup")
    rviz_cfg = PathJoinSubstitution([bringup_share, "rviz", "demo.rviz"])

    # Tell Gazebo where to find the meshes we vendored.  Resource path env
    # var is ``GZ_SIM_RESOURCE_PATH`` on Harmonic and ``IGN_GAZEBO_RESOURCE_PATH``
    # on Fortress; we set both for robustness against tooling that reads
    # the "wrong" one.
    bringup_share_dir = get_package_share_directory("humanoid_mimic_bringup")
    g1_share = get_package_share_directory("g1_description")
    extra_resource = os.path.dirname(g1_share)  # parent so package:// resolves
    for var in ("GZ_SIM_RESOURCE_PATH", "IGN_GAZEBO_RESOURCE_PATH"):
        existing = os.environ.get(var, "")
        os.environ[var] = (
            f"{extra_resource}:{existing}" if existing else extra_resource
        )

    # Render the world template to /tmp with the right plugin filenames /
    # class namespaces for the chosen backend.  Loading the raw template
    # with un-substituted ``{plugin_prefix}`` placeholders would fail SDF
    # validation, so the launch ALWAYS goes through this rendering step
    # even on the original Jazzy/Harmonic path.
    world_template = os.path.join(bringup_share_dir, "worlds", "mimic.sdf")
    world_sdf = profile_mod.render_world_sdf(world_template, profile)
    world_path_str = os.path.join(tempfile.gettempdir(), "humanoid_sim_world.sdf")
    with open(world_path_str, "w", encoding="utf-8") as fh:
        fh.write(world_sdf)

    # ``ros_gz_sim``'s gz_sim.launch.py normally folds ``LD_LIBRARY_PATH``
    # into ``GZ_SIM_SYSTEM_PLUGIN_PATH`` (or its Fortress sibling
    # ``IGN_GAZEBO_SYSTEM_PLUGIN_PATH``) so the simulator can find ROS
    # plugins like ``libgz_ros2_control-system.so`` (which lives in
    # ``/opt/ros/<distro>/lib``).  We bypass that include below (see the
    # comment on ``gz_sim`` ExecuteProcess), so do the same injection
    # ourselves -- otherwise the simulator logs ``Failed to load system
    # plugin [gz_ros2_control-system] : Could not find shared library.``
    # and the spawners then sit forever in "waiting for service
    # /controller_manager/list_controllers".
    plugin_var = f"{profile.sim_env_prefix}_SYSTEM_PLUGIN_PATH"
    ld_lib = os.environ.get("LD_LIBRARY_PATH", "")
    for var in (plugin_var, "GZ_SIM_SYSTEM_PLUGIN_PATH", "IGN_GAZEBO_SYSTEM_PLUGIN_PATH"):
        existing_plug = os.environ.get(var, "")
        os.environ[var] = os.pathsep.join(
            p for p in (existing_plug, ld_lib) if p
        )

    # CPU affinity.  On a 4-core N100 we *must* pin Gazebo to cores 2-3 so
    # it physically cannot preempt the realsense USB transfer thread and
    # MediaPipe inference (pinned to cores 0-1).  Without this, the
    # gazebo bring-up spike silently starves the realsense node enough
    # to overrun the D415's UVC buffer and freeze color delivery for the
    # rest of the run.  Larger CPUs (Jetson AGX Orin's 8x Cortex-A78AE,
    # 12-thread x86 desktops) have enough headroom that pinning is
    # unnecessary and we leave the OS scheduler to its job.
    #
    # We launch ``gz sim`` (or ``ign gazebo``) ourselves rather than
    # through ros_gz_sim's ``gz_sim.launch.py`` because IncludeLaunchDescription
    # has no ``prefix=`` knob -- the cleanest way to get ``taskset`` in
    # front of the gazebo binary is to do the ExecuteProcess directly.
    # ``headless:=true`` runs gazebo server-only (no GUI), which removes
    # the second iGPU consumer (after MediaPipe).
    headless = LaunchConfiguration("headless")
    aff_prefix_cmd = profile.taskset_cmd(profile.gz_cores)
    render_args = ["--render-engine", profile.render_engine] if profile.render_engine else []

    gz_sim_full = ExecuteProcess(
        cmd=[
            *aff_prefix_cmd,
            *profile.gz_cmd,
            world_path_str,
            "-r",
            *render_args,
        ],
        output="screen",
        condition=UnlessCondition(headless),
        # If gazebo dies, tear down the whole launch.
        on_exit=Shutdown(),
    )
    gz_sim_headless = ExecuteProcess(
        cmd=[
            *aff_prefix_cmd,
            *profile.gz_cmd,
            world_path_str,
            "-r", "-s",
            *render_args,
        ],
        output="screen",
        condition=IfCondition(headless),
        on_exit=Shutdown(),
    )

    # ros2_control spawners.  Pin them to gazebo's cores too (where applicable)
    # so they can never steal cycles from the realsense pipeline.
    # NB: launch_ros's ``prefix`` is joined to a *single* string with no
    # separators -- pass it as one string, not a list, or you get errors
    # like ``FileNotFoundError: 'taskset-c2,3'``.
    spawner_prefix = profile.taskset(profile.gz_cores)
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
    # The Gazebo message namespace is backend-specific (gz.msgs vs
    # ignition.msgs), so we ask the profile for the full mapping.
    clock_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[profile.clock_bridge_arg()],
        output="screen",
    )

    use_rviz = LaunchConfiguration("use_rviz")
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_cfg],
        condition=IfCondition(use_rviz),
        output="screen",
        # Keep RViz off the camera's cores (see gazebo comment above) on
        # CPUs where we actually use affinity.
        prefix=spawner_prefix,
    )

    return LaunchDescription([
        DeclareLaunchArgument("use_rviz", default_value="false"),
        DeclareLaunchArgument(
            "headless",
            default_value="false",
            description=(
                "If true, run gazebo with -s (server only, no GUI).  Useful "
                "if the iGPU contention between Ogre2 and MediaPipe is "
                "starving the camera even with CPU affinity."
            ),
        ),
        LogInfo(msg="[humanoid-sim] machine profile: " + " | ".join(profile.notes)),
        gz_sim_full,
        gz_sim_headless,
        clock_bridge,
        OpaqueFunction(function=_build_robot_description),
        # Sequence: spawn -> jsb -> upper_body
        RegisterEventHandler(
            OnProcessExit(target_action=jsb_spawner, on_exit=[ub_spawner])
        ),
        jsb_spawner,
        rviz,
    ])
