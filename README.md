# humanoid-sim — G1 RealSense Mimic Demo

A ROS 2 Jazzy + Gazebo Harmonic demo that makes a Unitree G1 humanoid (pelvis welded
to the world) mimic the operator's upper body in real time, using a single Intel
RealSense D415 and a CPU-only pose-estimation pipeline (MediaPipe Pose + aligned
depth).

```
RealSense D415 ──► realsense2_camera ──► pose_estimator (MediaPipe + depth)
                                          │   /human/keypoints (PoseArray, 3D)
                                          ▼
                                       retargeter (geometric IK + filter)
                                          │   /upper_body_controller/commands
                                          ▼
                                gz_ros2_control ──► Gazebo Sim (G1, pelvis fixed)
```

## What this is and isn't

- **Is**: a desktop teleop demo of upper-body motion (waist yaw + both arms,
  including wrists), running in Gazebo Sim 8 on a CPU-only Intel N100.
- **Isn't**: a balancing/standing controller, a learning policy, or anything that
  drives the real robot. The G1's pelvis is welded to the world and the legs are
  rigidified, so we only worry about arms and waist yaw.

## Hardware tested

- Ubuntu 24.04, ROS 2 Jazzy, Gazebo Sim 8 (Harmonic)
- Intel N100 (no GPU)
- Intel RealSense D415

## Quick start

```bash
# 1. One-time system + python deps (uses apt + pip; needs sudo)
./setup.sh

# 2. Source ROS and build
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install

# 3. Run everything in one shell
source install/setup.bash
ros2 launch humanoid_mimic_bringup demo.launch.py
```

The first launch downloads the MediaPipe `pose_landmarker_lite.task` model
to `~/.cache/humanoid_pose_estimator/models/` (≈5.6 MB).  Set parameter
`model_complexity: 1` for the `full` variant or `2` for `heavy` if you have
the CPU headroom.

## Packages

| Package | Role |
| --- | --- |
| `g1_description` | URDF + meshes (vendored from `unitreerobotics/unitree_ros`) and a `g1_gz.urdf` generated for Gazebo + ros2_control. |
| `humanoid_mimic_bringup` | Launch files, controller configs, world SDF, RViz config. |
| `humanoid_pose_estimator` | MediaPipe Pose + aligned-depth deprojection node. |
| `humanoid_retargeter` | Geometric IK that maps human 3D keypoints to G1 joint commands. |

## Parameters worth knowing

- `humanoid_retargeter` ROS params:
  - `mirror` (bool, default `true`) — when `true`, user's right hand controls
    G1's left arm so the simulation looks like a mirror to the operator.
  - `smoothing_alpha` (double, default `0.4`) — exponential filter on joint commands.
  - `command_rate_hz` (double, default `30.0`).
- `humanoid_pose_estimator` ROS params:
  - `min_confidence` (double, default `0.5`) — minimum MediaPipe landmark visibility.
  - `depth_patch` (int, default `5`) — pixel half-window for median depth lookup.

See [`src/humanoid_mimic_bringup/config`](src/humanoid_mimic_bringup/config) for the
controller and retargeter configuration.

## Calibration / operator setup

1. Stand 1.5–2.5 m in front of the RealSense, fully visible from waist up.
2. Square your shoulders to the camera — the retargeter builds a torso frame from
   shoulders + hip midpoint, so a clean facing pose at startup produces the best
   results.
3. Move slowly the first few seconds; the smoothing filter takes a moment to
   settle.

## Testing

Unit + integration tests for the geometric IK and the retargeter node:

```bash
source install/setup.bash
python3 -m pytest src/humanoid_retargeter/test -v
```

There are 18 tests covering the body-frame construction, per-arm angles,
the mirror swap, and a live integration test that spins the retargeter node
and verifies it publishes `Float64MultiArray` commands when fed a synthetic
`/human/keypoints`.

## Known caveats

- **NumPy must stay on 1.x** — Ubuntu 24.04's system matplotlib and parts of
  ROS Python are compiled against the 1.x ABI.  `requirements.txt` already
  pins `numpy>=1.24,<2.0` and `opencv-python<4.10` for the same reason; do
  not relax these unless you know your stack tolerates 2.x.
- **MediaPipe uses the Tasks API** — the legacy `mediapipe.solutions.pose`
  module was dropped in the wheels for Python 3.12.  The pose estimator uses
  `mediapipe.tasks.python.vision.PoseLandmarker` and downloads the `.task`
  model on first use.
- **Wrist joints are commanded to 0**.  MediaPipe Pose alone doesn't give
  reliable hand orientation; commanding the wrists would amount to noise.
- **Shoulder-roll ±90° is a singularity** in the closed-form IK; the
  smoothing alpha hides most of it, and visually the arm still reaches the
  right place.

## Troubleshooting

- **No RealSense topics**: check `lsusb | grep Intel`, then `ros2 launch
  realsense2_camera rs_launch.py align_depth.enable:=true`.
- **Robot doesn't move**: confirm controllers loaded via `ros2 control list_controllers`.
  Expect `joint_state_broadcaster` and `upper_body_controller` both `active`.
- **Joint signs look wrong**: tune
  `src/humanoid_mimic_bringup/config/retargeter.yaml` `joint_signs` map.
- **`No module named 'mediapipe.solutions'`**: your installed MediaPipe is
  newer than 0.10.14 and only ships the Tasks API.  No fix needed — the
  pose estimator already targets that API.
- **`A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`**:
  ran `pip install` without the version pin.  Re-run
  `pip3 install --user --break-system-packages 'numpy<2' 'opencv-python<4.10'`.
- **Pose debug image freezes after 1–2 frames, with `Hardware Notification: Depth
  stream start failure` / `get_xu(ctrl=1) failed! Last Error: Device or resource
  busy` in the log**: the D415 was left in a half-open state by a previous run
  (e.g. Ctrl-C didn't reach the realsense node fast enough).  `camera.launch.py`
  already passes `initial_reset:=true` which hardware-resets the camera on
  startup, so this should self-heal on the *next* launch.  If you still see it,
  physically unplug+replug the D415 and relaunch; on persistent systems also
  add `sudo bash -c 'echo -1 > /sys/module/usbcore/parameters/autosuspend'`.
- **Heartbeats from `humanoid_pose_estimator` go to `in=0 out=0` the moment
  Gazebo finishes loading**: realsense's USB transfer thread is being starved
  by the bring-up CPU spike, and once the D415's UVC buffer overruns the
  camera silently stops delivering frames for the rest of the run.
  `demo.launch.py` already mitigates this by:
    (a) **CPU affinity** -- the realsense node and `pose_estimator_node` are
        pinned to cores 0-1 via `taskset`, and Gazebo + RViz are pinned to
        cores 2-3.  This is the load-bearing fix.
    (b) bringing the camera up first and waiting 12 s before Gazebo, and
    (c) defaulting `use_rviz:=false`.
  If you still see it, verify cores are isolated with
  `ps -o pid,psr,comm -C realsense2_camera_,gz,rviz2` -- realsense should sit
  on CPU 0 or 1, gz on 2 or 3.  For the pose-debug image, run
  `ros2 run rqt_image_view rqt_image_view` in another terminal and select
  `/human/debug_image`, or open RViz manually *after* the demo's heartbeats
  are steady (still keeping it off the camera's cores):
  `taskset -c 2,3 ros2 run rviz2 rviz2 -d src/humanoid_mimic_bringup/rviz/demo.rviz`.

## Credits / licenses

- Unitree G1 description (URDF + meshes) © Unitree Robotics, vendored from
  [`unitreerobotics/unitree_ros`](https://github.com/unitreerobotics/unitree_ros).
  Subject to the upstream BSD-3 license — see `src/g1_description/LICENSE.unitree`.
- The rest of this workspace is MIT.
