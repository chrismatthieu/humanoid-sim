#!/usr/bin/env bash
# One-shot dependency install for the G1 RealSense mimic demo.
# Idempotent: safe to re-run.
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  SUDO="sudo"
else
  SUDO=""
fi

echo "[setup] Installing apt packages..."
${SUDO} apt-get update
${SUDO} apt-get install -y --no-install-recommends \
  ros-jazzy-gz-ros2-control \
  ros-jazzy-ros2-controllers \
  ros-jazzy-controller-manager \
  ros-jazzy-joint-state-broadcaster \
  ros-jazzy-forward-command-controller \
  ros-jazzy-joint-trajectory-controller \
  ros-jazzy-xacro \
  ros-jazzy-robot-state-publisher \
  ros-jazzy-ros-gz \
  ros-jazzy-ros-gz-sim \
  ros-jazzy-ros-gz-bridge \
  ros-jazzy-ros-gz-image \
  ros-jazzy-realsense2-camera \
  ros-jazzy-realsense2-description \
  ros-jazzy-rviz2 \
  ros-jazzy-cv-bridge \
  ros-jazzy-image-transport-plugins \
  ros-jazzy-image-view \
  ros-jazzy-rqt-image-view \
  python3-pip \
  python3-colcon-common-extensions

echo "[setup] Installing pip packages..."
# Ubuntu 24.04 (Noble) enforces PEP 668 system-managed python; we deliberately
# install user-site packages alongside ROS's python so MediaPipe is importable
# from rclpy nodes without a venv.
pip3 install --user --upgrade --break-system-packages -r requirements.txt

echo "[setup] Done. Now: source /opt/ros/jazzy/setup.bash && colcon build --symlink-install"
