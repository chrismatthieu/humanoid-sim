"""Per-machine launch profile.

The demo was originally written for an x86 Ubuntu 24.04 / ROS 2 Jazzy /
Gazebo Harmonic / Intel iGPU box.  We now also want to run on a Jetson AGX
Orin (Ubuntu 22.04 / ROS 2 Humble / Gazebo Fortress *or* Harmonic / NVIDIA
Tegra GPU).  The combinations differ in:

  * Which Gazebo binary is on PATH (``gz sim`` for Harmonic, ``ign gazebo``
    for Fortress) and which env-var prefix it consumes
    (``GZ_SIM_*`` vs ``IGN_GAZEBO_*``).
  * Which messages namespace the ros_gz bridge expects (``gz.msgs.*`` for
    Harmonic-built ros_gz, ``ignition.msgs.*`` for Fortress-built ros_gz).
  * Which world-plugin filenames are valid (``gz-sim-*-system`` vs
    ``ignition-gazebo-*-system``).
  * Whether per-process CPU pinning helps (it does on a 4-core N100,
    matters less on a 12-core Orin) and which cores to use.
  * Whether a GPU is available for the Gazebo renderer.

Rather than littering the launch files with ``os.environ.get("ROS_DISTRO")``
branches, every per-machine decision is centralised in :func:`detect` so
the launch files can stay readable.  All decisions can be overridden via
environment variables for debugging (``HUMANOID_SIM_GZ_BACKEND=harmonic``,
``HUMANOID_SIM_USE_AFFINITY=0`` etc.).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional


# Map ROS distro -> the Gazebo backend whose ros_gz/gz_ros2_control debs are
# shipped for that distro on the OSRF + ROS apt repos.  This is the default
# unless an explicit override is provided.  See README "Hardware tested" /
# "Jetson / Humble" sections.
_DISTRO_DEFAULT_BACKEND = {
    "jazzy": "harmonic",
    "humble": "fortress",
    "iron": "fortress",
    "rolling": "harmonic",
}


@dataclass(frozen=True)
class MachineProfile:
    """All per-machine knobs for the demo's launch files."""

    # --- ROS / Gazebo identity ---
    ros_distro: str
    gz_backend: str                     # "harmonic" or "fortress"
    gz_cmd: List[str]                   # e.g. ["gz", "sim"]   or ["ign", "gazebo"]
    gz_gui_cmd: List[str]               # e.g. ["gz", "sim", "-g"] or ["ign", "gazebo", "-g"]
    msg_namespace: str                  # e.g. "gz.msgs" or "ignition.msgs"
    sim_env_prefix: str                 # "GZ_SIM" or "IGN_GAZEBO"
    world_plugin_prefix: str            # "gz-sim" or "ignition-gazebo"
    world_plugin_class_ns: str          # "gz::sim" or "ignition::gazebo"
    render_engine: str                  # "ogre2" / "ogre" / "" (default)

    # --- CPU / GPU ---
    is_jetson: bool
    has_gpu: bool
    cpu_count: int
    use_affinity: bool                  # if False, no taskset prefix at all
    camera_cores: str                   # comma-separated for taskset -c
    gz_cores: str                       # comma-separated for taskset -c

    # --- Camera / depth alignment ---
    # ``realsense``: trust the SDK's depth-to-color align filter
    #   (publishes /camera/.../aligned_depth_to_color/image_raw).
    # ``manual``: subscribe to raw depth + extrinsics and align in the
    #   pose-estimator node.  Required on the Jetson AGX Orin with a
    #   D457 via GMSL2, where the SDK align filter never emits frames
    #   because UVC metadata isn't exposed.  See README "Jetson AGX
    #   Orin" + pose_estimator_node module docstring.
    align_mode: str = "realsense"

    # --- Cosmetic / tuning ---
    notes: List[str] = field(default_factory=list)

    # --- Convenience builders ---
    def taskset(self, cores: str) -> str:
        """Return an empty string (no affinity) or a ``taskset -c CORES`` prefix.

        The result is suitable for ``launch_ros.actions.Node(prefix=...)``,
        which concatenates the prefix into the executable command line.
        """
        if not self.use_affinity or not cores:
            return ""
        return f"taskset -c {cores}"

    def taskset_cmd(self, cores: str) -> List[str]:
        """Return ``["taskset", "-c", CORES]`` or ``[]`` for ``ExecuteProcess``."""
        if not self.use_affinity or not cores:
            return []
        return ["taskset", "-c", cores]

    def clock_bridge_arg(self) -> str:
        """``ros_gz_bridge parameter_bridge`` mapping for the /clock topic."""
        return f"/clock@rosgraph_msgs/msg/Clock[{self.msg_namespace}.Clock"


def _detect_ros_distro() -> str:
    distro = os.environ.get("ROS_DISTRO", "").strip()
    if distro:
        return distro
    # Best-effort fallback: look for /opt/ros/<distro>/setup.bash.
    for candidate in ("jazzy", "humble", "iron", "rolling"):
        if os.path.exists(f"/opt/ros/{candidate}/setup.bash"):
            return candidate
    return "unknown"


def _detect_jetson() -> bool:
    return os.path.exists("/etc/nv_tegra_release")


def _detect_gpu(is_jetson: bool) -> bool:
    if is_jetson:
        # Every Jetson has an integrated NVIDIA GPU.  We don't try to call
        # nvidia-smi here because it doesn't work on Tegra (use tegrastats).
        return True
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return False
    try:
        res = subprocess.run(
            [nvidia_smi, "-L"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=2.0,
        )
        return res.returncode == 0 and b"GPU" in res.stdout
    except (OSError, subprocess.SubprocessError):
        return False


def _select_gz_backend(ros_distro: str) -> str:
    explicit = os.environ.get("HUMANOID_SIM_GZ_BACKEND", "").strip().lower()
    if explicit in ("harmonic", "fortress"):
        return explicit
    default = _DISTRO_DEFAULT_BACKEND.get(ros_distro, "harmonic")
    # If the default backend's binary isn't installed but the other one is,
    # fall back to the installed one so we at least try to come up.
    has_gz = shutil.which("gz") is not None
    has_ign = shutil.which("ign") is not None
    if default == "harmonic" and not has_gz and has_ign:
        return "fortress"
    if default == "fortress" and not has_ign and has_gz:
        return "harmonic"
    return default


def _backend_settings(backend: str) -> dict:
    if backend == "harmonic":
        return dict(
            gz_cmd=["gz", "sim"],
            gz_gui_cmd=["gz", "sim", "-g"],
            msg_namespace="gz.msgs",
            sim_env_prefix="GZ_SIM",
            world_plugin_prefix="gz-sim",
            world_plugin_class_ns="gz::sim",
        )
    if backend == "fortress":
        return dict(
            gz_cmd=["ign", "gazebo"],
            gz_gui_cmd=["ign", "gazebo", "-g"],
            msg_namespace="ignition.msgs",
            sim_env_prefix="IGN_GAZEBO",
            world_plugin_prefix="ignition-gazebo",
            world_plugin_class_ns="ignition::gazebo",
        )
    raise ValueError(f"unknown gz backend: {backend!r}")


def _select_affinity(cpu_count: int, is_jetson: bool) -> dict:
    """Pick taskset cores based on the CPU layout.

    On a 4-core N100 we *must* pin the camera+pose pipeline to cores 0-1 and
    everything heavy (gazebo, rviz) to cores 2-3, otherwise the realsense
    USB transfer thread gets preempted by the gazebo bring-up storm and
    color delivery silently freezes for the rest of the run.

    On larger machines (Jetson AGX Orin has 8-12 Cortex-A78AE cores; most
    x86 desktops have 8+ threads) the pinning is unnecessary -- the OS
    scheduler has enough room.  Pinning is even mildly counter-productive
    because it caps total parallelism.  We disable it by default for >=8
    cores.  Override with ``HUMANOID_SIM_USE_AFFINITY=1`` to force-enable.
    """

    override = os.environ.get("HUMANOID_SIM_USE_AFFINITY", "").strip()
    if override == "1":
        use_affinity = True
    elif override == "0":
        use_affinity = False
    else:
        use_affinity = cpu_count <= 6 and not is_jetson

    if not use_affinity:
        return dict(use_affinity=False, camera_cores="", gz_cores="")

    if cpu_count <= 4:
        return dict(use_affinity=True, camera_cores="0,1", gz_cores="2,3")
    if cpu_count <= 6:
        return dict(use_affinity=True, camera_cores="0,1", gz_cores="2,3,4,5")
    # Defensive: shouldn't reach here because of the >=8 check above.
    return dict(use_affinity=False, camera_cores="", gz_cores="")


def _select_render_engine(is_jetson: bool, has_gpu: bool) -> str:
    explicit = os.environ.get("HUMANOID_SIM_RENDER_ENGINE", "").strip()
    if explicit:
        return explicit
    if is_jetson:
        # Tegra's GL stack mostly works with Ogre2 once nvidia-l4t userspace
        # is installed, but headless EGL is brittle.  Let the user fall
        # back via the env var if needed.
        return "ogre2"
    # CPU-only N100 path: ogre2 is fine because gazebo is server-only.
    return "ogre2"


def _select_align_mode(is_jetson: bool) -> str:
    """Default the depth-to-color alignment strategy per host.

    On an x86 desktop with a D415 over USB, the realsense SDK exposes
    full UVC metadata and its built-in align-to-color filter works out
    of the box; the aligned-depth topic streams at 30 Hz and the pose
    estimator can do a direct color-pixel depth lookup.

    On an NVIDIA Jetson AGX Orin with a D457 over GMSL2, the kernel
    backend path (``/dev/video-rs-depth-0``) does not expose UVC frame
    metadata, so the SDK's align filter is silent forever: the topic
    is advertised but no messages flow.  Reproduced even by upstream's
    own example, ``rs_align_depth_launch.py``.  We work around it by
    doing the warp ourselves in :class:`PoseEstimatorNode`, using the
    extrinsics topic that the realsense node *does* publish reliably.

    Override either way with ``HUMANOID_SIM_ALIGN_MODE=realsense`` or
    ``HUMANOID_SIM_ALIGN_MODE=manual``.
    """
    explicit = os.environ.get("HUMANOID_SIM_ALIGN_MODE", "").strip().lower()
    if explicit in ("realsense", "manual"):
        return explicit
    return "manual" if is_jetson else "realsense"


def detect() -> MachineProfile:
    """Build the :class:`MachineProfile` for the current machine."""

    ros_distro = _detect_ros_distro()
    is_jetson = _detect_jetson()
    has_gpu = _detect_gpu(is_jetson)
    cpu_count = os.cpu_count() or 4

    backend = _select_gz_backend(ros_distro)
    bs = _backend_settings(backend)
    aff = _select_affinity(cpu_count, is_jetson)
    render = _select_render_engine(is_jetson, has_gpu)
    align_mode = _select_align_mode(is_jetson)

    notes: List[str] = []
    notes.append(f"ros_distro={ros_distro}")
    notes.append(f"gz_backend={backend} ({' '.join(bs['gz_cmd'])})")
    notes.append(f"is_jetson={is_jetson} has_gpu={has_gpu} cpu_count={cpu_count}")
    notes.append(
        "affinity=" + (
            f"camera={aff['camera_cores']} gz={aff['gz_cores']}"
            if aff["use_affinity"] else "off"
        )
    )
    notes.append(f"render_engine={render}")
    notes.append(f"align_mode={align_mode}")

    return MachineProfile(
        ros_distro=ros_distro,
        gz_backend=backend,
        gz_cmd=bs["gz_cmd"],
        gz_gui_cmd=bs["gz_gui_cmd"],
        msg_namespace=bs["msg_namespace"],
        sim_env_prefix=bs["sim_env_prefix"],
        world_plugin_prefix=bs["world_plugin_prefix"],
        world_plugin_class_ns=bs["world_plugin_class_ns"],
        render_engine=render,
        is_jetson=is_jetson,
        has_gpu=has_gpu,
        cpu_count=cpu_count,
        use_affinity=aff["use_affinity"],
        camera_cores=aff["camera_cores"],
        gz_cores=aff["gz_cores"],
        align_mode=align_mode,
        notes=notes,
    )


def render_world_sdf(template_path: str, profile: Optional[MachineProfile] = None) -> str:
    """Read ``template_path`` and substitute the Gazebo plugin names.

    The world SDF stores plugin filenames + class namespaces as template
    placeholders ``{plugin_prefix}`` and ``{plugin_ns}`` so we can emit
    either a Harmonic or Fortress build of the same world from a single
    source file.  Anything that isn't a placeholder is preserved verbatim.
    """

    if profile is None:
        profile = detect()
    with open(template_path, "r", encoding="utf-8") as fh:
        text = fh.read()
    return text.format(
        plugin_prefix=profile.world_plugin_prefix,
        plugin_ns=profile.world_plugin_class_ns,
    )
