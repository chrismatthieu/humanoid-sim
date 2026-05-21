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

That single command brings up the camera, MediaPipe pose estimator,
retargeter, Gazebo (server-only, headless by default), the G1 with its
controllers, RViz, and the Gazebo GUI client.  Bring-up takes ~25 s.  The
first launch also downloads the MediaPipe ``pose_landmarker_lite.task``
model to ``~/.cache/humanoid_pose_estimator/models/`` (~5.6 MB).

### Useful launch arguments

All flags below are passed as ``arg:=value`` after the launch file:

| arg | default | choices | what it does |
| --- | --- | --- | --- |
| ``model_complexity`` | ``0`` | ``0``, ``1``, ``2`` | MediaPipe Pose model: 0=lite (~80 ms infer on N100), 1=full (~150 ms, much better wrist/elbow localization when the hand is near the body), 2=heavy (~400 ms, slower than the capture rate). |
| ``gui`` | ``true`` | ``true``, ``false`` | Auto-attach the Gazebo GUI + RViz a few seconds after the pipeline is up.  Set ``false`` for fully headless runs (SSH, remote box). |
| ``headless`` | ``true`` | ``true``, ``false`` | Run Gazebo server-only (no Ogre2 inside the gazebo process).  Keep ``true`` on a shared-iGPU machine so the gazebo renderer doesn't fight MediaPipe's EGL context. |
| ``use_rviz`` | ``false`` | ``true``, ``false`` | In-process RViz inside ``sim.launch.py``.  Off by default because we launch RViz separately via ``gui`` after a delay. |
| ``bringup_order`` | ``camera_first`` | ``camera_first``, ``gazebo_first`` | ``gazebo_first`` lets gazebo's startup storm finish before realsense's USB stream comes up; safer on CPUs with shared memory channels (N100), at the cost of ~20 s extra bring-up. |
| ``dds`` | ``fastrtps`` | ``fastrtps``, ``cyclonedds`` | DDS implementation for the demo's processes.  Defaults to Fast-DDS to sidestep Cyclone DDS' participant-poisoning bug on this workspace; see *Troubleshooting* if you need Cyclone. |

Example: hands-on-hips and arms-crossed poses look much better with the
``full`` MediaPipe model, if you have the CPU:

```bash
ros2 launch humanoid_mimic_bringup demo.launch.py model_complexity:=1
```

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
  - `head_to_waist_gain` (double, default `0.7`) — mixes operator head yaw into
    `waist_yaw_joint` since the G1 has no neck joint.  See *Head tracking* below.
  - `debug_log_period_s` (double, default `2.0`) — periodic dump of commanded
    joint angles to the launch console.  Set to `0` to silence.
- `humanoid_pose_estimator` ROS params:
  - `min_confidence` (double, default `0.5`) — minimum MediaPipe landmark visibility.
  - `depth_patch` (int, default `5`) — pixel half-window for median depth lookup.
  - `model_complexity` (int, default `0`) — see the launch-arg of the same name above.
  - `wrist_correction` / `elbow_correction` (bool, default `true`) — world-landmark-anchored
    depth-bleed fix for arms-near-body poses.  See *"Hand near body" poses straighten one
    arm* below.
  - `wrist_forearm_tol_m` / `elbow_upper_arm_tol_m` (double, default `0.05`) —
    stereo-vs.-world-landmark limb-length disagreement that triggers a re-projection.

See [`src/humanoid_mimic_bringup/config/retargeter.yaml`](src/humanoid_mimic_bringup/config/retargeter.yaml)
for the canonical defaults of every parameter above plus the per-joint
sign / offset / limit tables.

## Calibration / operator setup

1. Stand 1.5–2.5 m in front of the RealSense, fully visible from waist up.
2. Square your shoulders to the camera — the retargeter builds a torso frame from
   shoulders + hip midpoint, so a clean facing pose at startup produces the best
   results.
3. Move slowly the first few seconds; the smoothing filter takes a moment to
   settle.

### How a commanded angle is computed

For each joint, every tick the retargeter does:

```
cmd  =  sign * raw_ik  +  offset      # absorb URDF zero-pose mismatch
cmd  =  clamp(cmd, joint_limits.min, joint_limits.max)
cmd  =  (1 - alpha) * cmd + alpha * previous_cmd   # exponential smoothing
```

The `joint_offsets` map was added because the G1's URDF zero pose is *not*
"straight arm down": at `elbow_joint=0` the forearm extends forward (+x in
elbow_link), not parallel to the upper arm.  Geometrically the IK assumes a
clean "arm fully extended" zero, so the elbow needs `sign=-1` and
`offset=+pi/2` to map the operator's straight-arm rest onto the G1's
forearm-down pose.  Every other joint defaults to `offset=0`; tune via the
YAML or `ros2 param set` if you discover similar zero-pose mismatches.

### Tuning joint signs / offsets / limits live

The retargeter has a `set_parameter` callback registered, so every entry
under ``joint_signs.*``, ``joint_offsets.*``, ``joint_limits.*``, plus
``mirror`` / ``smoothing_alpha`` / ``keypoint_min_visibility`` /
``head_to_waist_gain`` / ``head_yaw_sign`` / ``debug_log_period_s``, can be
changed at runtime without restarting the demo.

> **IMPORTANT - DDS implementation must match the launch.**  ``demo.launch.py``
> exports ``RMW_IMPLEMENTATION=rmw_fastrtps_cpp`` *for the processes it
> spawns*, to work around Cyclone DDS' subscription-poisoning bug on this
> workspace's bring-up profile.  That env var does NOT propagate to other
> terminals.  If you open a second terminal to ``ros2 param set`` or
> ``ros2 topic echo``, you must also set this in that terminal:
>
> ```bash
> export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
> ros2 param set /humanoid_retargeter joint_offsets.left_elbow_joint 1.5708
> ```
>
> Without the export, the second terminal will use Jazzy's system default
> (Cyclone DDS), and any ``ros2`` CLI call will silently hang / time out:
> the values you "set" are never delivered to the running node, and the
> debug log never turns on.  When the param set DOES reach the node, the
> launch console prints a ``param set: name=value`` confirmation line.

```bash
# Example: try the *opposite* sign on left_shoulder_roll while the robot is
# running, e.g. because one arm is crossing the body.
ros2 param set /humanoid_retargeter joint_signs.left_shoulder_roll_joint -1.0

# Or temporarily disable mirror mode to see same-side mimicry:
ros2 param set /humanoid_retargeter mirror false

# Or turn on the joint-command debug log (degrees, every 1 s).  Pose
# statically and inspect each joint's commanded angle:
ros2 param set /humanoid_retargeter debug_log_period_s 1.0
```

The debug log line looks like:

```
cmd(deg): wy=+12.3 | LSp=+5.1 LSr=-8.4 LSy=-1.2 LE=+45.0 | RSp=+5.0 RSr=+8.6 RSy=+1.1 RE=+44.8
```

Recommended calibration workflow once the demo is up:

1. Stand still in the neutral pose (arms hanging at sides).  The robot should
   also be in its neutral pose.  If a joint is far off in neutral, its sign
   or limit centre is wrong, not the IK.
2. Raise each arm straight out to the side (T-pose).  The robot's matching
   arm should also T-pose outward.  If it crosses the body instead, flip the
   sign on that arm's ``shoulder_roll_joint``.
3. Reach each arm straight forward.  The robot's arm should reach forward
   too.  If it reaches backward, flip the sign on that arm's
   ``shoulder_pitch_joint``.
4. Touch each shoulder with the same-side hand (full elbow flex).  Robot
   elbow should bend the same amount.  Wrong sign → elbow extends backward
   (which the URDF limit clamps to 0).
5. With arms straight out, twist the upper arm so the palm faces up vs down.
   That's ``shoulder_yaw``; if rotation looks reversed, flip that sign.

Each chirality-flipping joint (anything with ``roll`` or ``yaw`` in the name,
on either side) must have **opposite** signs on L vs R under ``mirror=true``.
Pitch / elbow are symmetric and should be the same sign on both sides.

### "The elbows look wrong" / "the hands are turned the opposite way"

This was the original symptom that motivated `joint_offsets` -- see *How a
commanded angle is computed* above.  The fix is in `retargeter.yaml`
(`left_elbow_joint`/`right_elbow_joint` have `sign: -1.0` and `offset:
1.5708`).  If you still see issues:

1. Turn on ``debug_log_period_s 1.0``.
2. Pose deliberately and read the line:
   ```
   cmd(deg): wy=+12.3 | LSp=... LE=+45.0 | RSp=... RE=+44.8
   ```
3. Holding your arm straight forward with elbow bent 90° (forearm
   horizontal, palm facing the floor) should show `LE` (or `RE`) near 0°,
   because the G1's elbow=0 *is* forearm-forward.  Holding your arm
   straight down should show `LE`/`RE` near +90°, because the G1 needs
   to rotate the forearm 90° from rest-forward to point down.
4. If the relationship is inverted, flip the relevant elbow sign:
   ```bash
   ros2 param set /humanoid_retargeter joint_signs.right_elbow_joint +1.0
   # AND adjust the offset accordingly:
   ros2 param set /humanoid_retargeter joint_offsets.right_elbow_joint 0.0
   ```

For *forearm twist* / hand direction (not bend), the culprit is
``shoulder_yaw`` instead.  Try toggling that sign per side:

```bash
ros2 param set /humanoid_retargeter joint_signs.right_shoulder_yaw_joint +1.0
```

### "Hand near body" poses straighten one arm

When you put a hand on your hip or fold your arms across your chest, the
wrist (and sometimes the elbow) landmark sits in pixel-space *immediately
next to your torso*.  The ``humanoid_pose_estimator`` reads each joint's
depth as the median over an 11x11 patch (``depth_patch=5``).  When part of
that patch lands on the body behind the limb, the median snaps backward
by ~10-30 cm.  The 3D wrist (or elbow) then collapses onto the torso
plane, the elbow-to-wrist vector loses its forward/inward component, and
the IK reports ``elbow_angle`` near zero -- the arm visually stays
half-extended.  Critically this misfires *asymmetrically*: which patch
pixels fall on limb vs. body depends on subpixel offsets, so one side
can blow up while the other looks fine.

The pose estimator runs a two-pass world-landmark-anchored fix per arm,
``shoulder -> elbow`` then ``elbow -> wrist``.  Each pass compares the
3D limb length from stereo against MediaPipe's metric
``pose_world_landmarks`` limb length; if they disagree by more than the
matching ``*_tol_m`` parameter (default 5 cm), it back-projects the
distal joint's pixel at the depth that *does* match the world-landmark
length (closed form, single quadratic, pick the nearer of the two
positive roots since the depth bleed is always *behind* the true joint).
The distal-joint pixel is always reliable -- only the depth is the
failure point.

Corrected joints are drawn in ``/human/debug_image``:

  - **magenta** ring on a wrist that was re-projected
  - **cyan** ring on an elbow that was re-projected

The pose estimator's 1 Hz heartbeat also counts how many corrections
fired in the last second:

```
hb: in=9 out=9 infer=82ms idle=0.1s (depth=ok, info=ok) fix(el=3,wr=7)
```

If ``el=0`` and ``wr=0`` for a pose you *expect* to trigger (e.g.
hands-on-hips), MediaPipe's 2D landmarks must already be giving stereo
the right depth -- no correction needed.  If you're still seeing visual
asymmetry in that case, the issue is 2D landmark accuracy, not depth.
Bump ``model_complexity`` to ``1`` (full) or ``2`` (heavy) -- those
have substantially better wrist/elbow localization when limbs occlude
the torso.

Tunable parameters (defaults in ``retargeter.yaml`` under the
``humanoid_pose_estimator`` block):

| parameter                | default | meaning |
| ------------------------ | ------- | ------- |
| ``wrist_correction``        | ``true``  | Set ``false`` to disable the elbow->wrist pass.  Reach-back poses (the rare case where the wrist is *behind* the elbow in depth) are the case to disable for. |
| ``wrist_forearm_tol_m``     | ``0.05``  | Stereo vs. world-landmark forearm-length disagreement that triggers a wrist re-projection, in meters. |
| ``elbow_correction``        | ``true``  | Set ``false`` to disable the shoulder->elbow pass. |
| ``elbow_upper_arm_tol_m``   | ``0.05``  | Stereo vs. world-landmark upper-arm-length disagreement that triggers an elbow re-projection, in meters. |

### Head tracking

The G1 URDF in this workspace (``g1_29dof_lock_waist.urdf``) has
``head_joint type="fixed"`` -- the head is rigidly bolted to ``torso_link``,
so there is no neck joint to drive directly.  To make the robot's head
follow yours, we route the operator's head yaw (computed from the ear-line
relative to the shoulder-line, both projected horizontal) into
``waist_yaw_joint``.  The blend is controlled by:

| parameter            | default | meaning                                                                   |
| -------------------- | ------- | ------------------------------------------------------------------------- |
| ``head_to_waist_gain`` | ``0.7``   | 0 = waist follows torso-twist only; 1 = waist follows head 1:1.           |
| ``head_yaw_sign``      | ``+1.0``  | flip to ``-1.0`` if the robot turns its body to the *opposite* side from where you look. |

If the robot ignores your head turn entirely, check that both ears are above
``keypoint_min_visibility`` -- in profile (one ear behind the head),
``compute_head_yaw`` deliberately returns ``None`` and the waist holds its
last value.

## Testing

Unit + integration tests for the geometric IK and the retargeter node:

```bash
source install/setup.bash
python3 -m pytest src/humanoid_retargeter/test -v
```

There are 19 tests covering the body-frame construction, per-arm angles,
the mirror swap, head-yaw estimation, and a live integration test that
spins the retargeter node and verifies it publishes `Float64MultiArray`
commands when fed a synthetic `/human/keypoints`.

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
- **No neck joint in the G1 URDF**.  ``head_joint`` is ``type="fixed"``, so
  head mimicking is faked by routing the operator's head yaw into
  ``waist_yaw_joint`` (see *Head tracking* above).  Side effects: the
  robot's torso twists when you just turn your head, and head pitch/roll
  can't be mimicked at all.
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
