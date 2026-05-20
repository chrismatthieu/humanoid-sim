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
- **Pose debug image freezes after 1–2 frames, with
  `xioctl(VIDIOC_S_FMT) failed, errno=5 Input/output error` in the
  realsense log and a `Device with physical ID /sys/.../video4linux/videoN`
  whose `N` *changed* mid-run**: this is the D415 re-enumerating itself
  on the USB bus, which the realsense node interprets as the device
  disappearing mid-stream and then trips ``VIDIOC_S_FMT`` when it tries
  to re-configure the new ``/dev/videoN``.  We used to set
  ``initial_reset:=true`` on the realsense node to clear "Device or
  resource busy" from prior unclean shutdowns, but on the D415 that
  reset is asynchronous -- librealsense returns immediately, the kernel
  re-enumerates the device 5-6 s later with a new ``/dev/videoN``, and
  that arrives mid-stream and kills the sensor.  ``camera.launch.py``
  now sets ``initial_reset:=false`` for this reason.
- **`get_xu(ctrl=1) failed! Last Error: Device or resource busy` on the
  *very first* launch after killing a previous one with Ctrl-C / SIGKILL**:
  the D415 was left in a half-open state by the previous run because
  the realsense node didn't get to drain its USB endpoints.  Recovery:
  physically unplug+replug the D415 once, or on persistent systems run
  `sudo bash -c 'echo -1 > /sys/module/usbcore/parameters/autosuspend'`
  to let the kernel power-cycle the port automatically.  We deliberately
  do *not* enable ``initial_reset:=true`` to paper over this -- the
  reset itself causes a worse failure mode (see entry above).
- **Heartbeats from `humanoid_pose_estimator` go to `in=0 out=0` shortly
  after Gazebo starts, but `ros2 topic hz /camera/camera/color/image_raw`
  in another terminal still shows the topic publishing at ~10-30 Hz**:
  this is a *Cyclone DDS subscription death*.  When gazebo + ros2_control
  bring ~6 new DDS participants onto the bus in a ~2 s window, Cyclone's
  participant gets poisoned and any pre-existing subscriber<->writer
  match silently un-pairs.  After that the writer keeps publishing
  happily but the reader is permanently disconnected, and re-creating
  the subscription from the *same* participant doesn't help (we tried
  -- the watchdog in ``pose_estimator_node`` retries 3 times then logs
  an ``ERROR`` recommending you swap DDS stacks).

  **This is already fixed by default**: ``demo.launch.py`` runs a
  ``SetEnvironmentVariable("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")``
  at the top of the launch description, which overrides Jazzy's default
  Cyclone with Fast-DDS for the entire demo.  Fast-DDS' discovery is
  robust to the same bring-up storm.  If you want Cyclone back for some
  reason, ``export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`` *before*
  ``ros2 launch`` -- the launch file honours an explicitly-set env var.

  Secondary mitigations that the demo still applies in case the
  subscription dies anyway:

    - **Subscription watchdog**: ``pose_estimator_node`` runs a 2 s
      timer that, after seeing color frames go silent for >= 5 s,
      tears down and re-creates all three image subscriptions.  You'll
      see a `WARN` line ``No color frames for X.Xs ... Rebuilding
      subscriptions (attempt #N)`` in the log, and the next heartbeat
      will start showing `in=...` with a `resub=N` suffix.  If
      ``resub=N`` keeps climbing without ever seeing ``in>0``, the
      DDS *participant* is poisoned (not just the reader) and you need
      to swap DDS stacks -- the watchdog will escalate to a single
      ``ERROR`` line at attempt #4 with that exact recommendation.
    - **CPU affinity**: realsense + pose_estimator pinned to cores 0-1
      via ``taskset``, gazebo + rviz to cores 2-3.  Verify with
      ``ps -o pid,psr,comm -C realsense2_camera_,gz,rviz2``.
    - ``headless:=true``: gazebo server-only (no Ogre2), avoids any
      fight with MediaPipe's EGL context on a shared Intel iGPU.
      Recommended on the Intel N100.
    - ``bringup_order:=gazebo_first``: gazebo at t=0, camera at t=30
      so the camera comes up in a quiet steady-state environment.
      Bring-up time ~35 s vs ~15 s for the default ``camera_first``.

  For the pose-debug image, in another terminal run
  `ros2 run rqt_image_view rqt_image_view` and select `/human/debug_image`,
  or open RViz manually *after* the demo's heartbeats are steady (still
  keeping it off the camera's cores):
  `taskset -c 2,3 ros2 run rviz2 rviz2 -d src/humanoid_mimic_bringup/rviz/demo.rviz`.
- **`Failed to load system plugin [gz_ros2_control-system] : Could not find
  shared library` in the gazebo log, followed by spawners that hang on
  `waiting for service /controller_manager/list_controllers`, and a
  G1 whose arms swing freely under gravity**: gazebo can't find
  `libgz_ros2_control-system.so` (in `/opt/ros/jazzy/lib`) because the
  custom `gz sim` ExecuteProcess we use (so we can `taskset` it) doesn't
  go through `ros_gz_sim`'s launch file, which is what normally folds
  `LD_LIBRARY_PATH` into `GZ_SIM_SYSTEM_PLUGIN_PATH`.  `sim.launch.py`
  now does that injection itself; if you still see the error, double-check
  that `/opt/ros/jazzy/setup.bash` was sourced *before* `ros2 launch`
  (otherwise `LD_LIBRARY_PATH` has no ROS entries to inherit).

## Credits / licenses

- Unitree G1 description (URDF + meshes) © Unitree Robotics, vendored from
  [`unitreerobotics/unitree_ros`](https://github.com/unitreerobotics/unitree_ros).
  Subject to the upstream BSD-3 license — see `src/g1_description/LICENSE.unitree`.
- The rest of this workspace is MIT.
