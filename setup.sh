#!/usr/bin/env bash
# One-shot dependency install for the G1 RealSense mimic demo.
# Idempotent: safe to re-run.
#
# Supports two host profiles:
#   * Ubuntu 24.04 (noble) + ROS 2 Jazzy + Gazebo Harmonic     -- original
#   * Ubuntu 22.04 (jammy) + ROS 2 Humble + Gazebo Fortress    -- Jetson AGX Orin
#
# The ROS distro is auto-detected from $ROS_DISTRO if set, else from
# /opt/ros/<distro>/setup.bash, else from /etc/os-release.  Override with
# ``ROS_DISTRO=humble ./setup.sh`` to force a particular distro.
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  SUDO="sudo"
else
  SUDO=""
fi

# --- Detect ROS distro -------------------------------------------------------
detect_ros_distro() {
  if [[ -n "${ROS_DISTRO:-}" ]]; then
    echo "${ROS_DISTRO}"
    return
  fi
  for d in jazzy humble iron rolling; do
    if [[ -f "/opt/ros/${d}/setup.bash" ]]; then
      echo "${d}"
      return
    fi
  done
  # Fallback: guess from Ubuntu codename.
  local codename
  codename="$(. /etc/os-release && echo "${VERSION_CODENAME:-}")"
  case "${codename}" in
    noble)  echo "jazzy" ;;
    jammy)  echo "humble" ;;
    *)      echo "" ;;
  esac
}

ROS_DISTRO="$(detect_ros_distro)"
if [[ -z "${ROS_DISTRO}" ]]; then
  echo "[setup] ERROR: could not detect ROS distro." >&2
  echo "[setup]   Set ROS_DISTRO=jazzy (Ubuntu 24.04) or ROS_DISTRO=humble (Ubuntu 22.04, Jetson)." >&2
  exit 1
fi
echo "[setup] Detected ROS distro: ${ROS_DISTRO}"

# --- Detect Jetson -----------------------------------------------------------
IS_JETSON="0"
if [[ -f /etc/nv_tegra_release ]]; then
  IS_JETSON="1"
  echo "[setup] Detected Jetson (L4T): $(head -n1 /etc/nv_tegra_release)"
fi

# --- Pick Gazebo backend per distro ------------------------------------------
# On Jazzy the apt-installed ``ros-jazzy-ros-gz-*`` packages are built against
# Gazebo Harmonic; on Humble they're built against Ignition Fortress.  We
# install whichever Gazebo userspace matches the ROS bindings -- mixing them
# (e.g. Harmonic + Humble) leaves ``ros_gz_sim create``, ``parameter_bridge``
# and ``gz_ros2_control`` unable to talk to the simulator because their
# transport namespaces differ.
case "${ROS_DISTRO}" in
  jazzy)
    GZ_PACKAGES=(gz-harmonic)
    ;;
  humble|iron)
    GZ_PACKAGES=(ignition-fortress)
    ;;
  *)
    GZ_PACKAGES=()
    ;;
esac

# --- Install OSRF apt key + repo (needed for gz / ignition packages) --------
ensure_osrf_repo() {
  if [[ -f /etc/apt/sources.list.d/gazebo-stable.list ]]; then
    return
  fi
  echo "[setup] Adding OSRF Gazebo apt repo..."
  ${SUDO} apt-get install -y --no-install-recommends curl ca-certificates gnupg lsb-release
  ${SUDO} install -d -m 0755 /etc/apt/keyrings
  curl -fsSL https://packages.osrfoundation.org/gazebo.gpg \
    | ${SUDO} tee /etc/apt/keyrings/osrf.gpg.asc >/dev/null
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/osrf.gpg.asc] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
    | ${SUDO} tee /etc/apt/sources.list.d/gazebo-stable.list >/dev/null
}

# --- ROS package install ------------------------------------------------------
ROS_PACKAGES=(
  ros-${ROS_DISTRO}-gz-ros2-control
  ros-${ROS_DISTRO}-ros2-controllers
  ros-${ROS_DISTRO}-controller-manager
  ros-${ROS_DISTRO}-joint-state-broadcaster
  ros-${ROS_DISTRO}-forward-command-controller
  ros-${ROS_DISTRO}-joint-trajectory-controller
  ros-${ROS_DISTRO}-xacro
  ros-${ROS_DISTRO}-robot-state-publisher
  ros-${ROS_DISTRO}-ros-gz
  ros-${ROS_DISTRO}-ros-gz-sim
  ros-${ROS_DISTRO}-ros-gz-bridge
  ros-${ROS_DISTRO}-ros-gz-image
  ros-${ROS_DISTRO}-realsense2-camera
  ros-${ROS_DISTRO}-realsense2-description
  ros-${ROS_DISTRO}-rviz2
  ros-${ROS_DISTRO}-cv-bridge
  ros-${ROS_DISTRO}-image-transport-plugins
  ros-${ROS_DISTRO}-image-view
  ros-${ROS_DISTRO}-rqt-image-view
)

echo "[setup] Installing apt packages..."
${SUDO} apt-get update
if [[ "${#GZ_PACKAGES[@]}" -gt 0 ]]; then
  ensure_osrf_repo
  ${SUDO} apt-get update
fi
${SUDO} apt-get install -y --no-install-recommends \
  "${ROS_PACKAGES[@]}" \
  "${GZ_PACKAGES[@]}" \
  python3-pip \
  python3-colcon-common-extensions \
  util-linux

# --- pip --break-system-packages is Noble-only ------------------------------
# Ubuntu 24.04 enforces PEP 668 (the system Python is "externally managed")
# and refuses ``pip install`` without ``--break-system-packages``.  Ubuntu
# 22.04 / Jammy is still permissive, and the Jetson's pip predates PEP 668
# entirely -- passing the flag there is harmless but unnecessary.
UBUNTU_CODENAME="$(. /etc/os-release && echo "${VERSION_CODENAME:-}")"
PIP_FLAGS=(--user --upgrade)
if [[ "${UBUNTU_CODENAME}" == "noble" ]]; then
  PIP_FLAGS+=(--break-system-packages)
fi

echo "[setup] Installing pip packages (${PIP_FLAGS[*]}) ..."
# We deliberately install user-site packages alongside ROS's python so
# MediaPipe is importable from rclpy nodes without a venv.
pip3 install "${PIP_FLAGS[@]}" -r requirements.txt

# --- Jetson-specific reminders ----------------------------------------------
if [[ "${IS_JETSON}" == "1" ]]; then
  echo "[setup] Jetson note: librealsense on JetPack 6 ships pre-built with"
  echo "[setup]   the apt ros-${ROS_DISTRO}-realsense2-camera package above."
  echo "[setup]   If lsusb shows the D45x but the realsense node doesn't see"
  echo "[setup]   it, check the user is in the 'video' and 'plugdev' groups."
fi

echo "[setup] Done.  Now run:"
echo "[setup]   source /opt/ros/${ROS_DISTRO}/setup.bash && colcon build --symlink-install"
echo "[setup]   source install/setup.bash && ros2 launch humanoid_mimic_bringup demo.launch.py"
