#!/usr/bin/env python3
"""Generate a Gazebo + ros2_control-ready URDF from the vendored G1 description.

The upstream Unitree G1 URDF is pure mechanical description: no Gazebo plugin,
no <ros2_control> tag, all mesh paths relative, and the pelvis is the URDF root
(no `world` link).  For our upper-body mimic demo we want:

  * Pelvis welded to the world (no legs walking, no balancing).
  * Leg joints rigidified so the robot stands like a mannequin.
  * Upper body (waist_yaw + both arms incl. wrists = 15 joints) commandable via
    ros2_control with a position interface.
  * gz_ros2_control plugin loaded with our controllers.yaml.
  * Mesh paths resolvable through ROS package:// URIs.

This module is importable (call `generate(...)`) and runnable as a CLI.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

UPPER_BODY_JOINTS: list[str] = [
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# Leg joints get changed to type="fixed" so they don't flop under gravity.
LEG_JOINTS_TO_FREEZE: list[str] = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]


def _rewrite_mesh_paths(root: ET.Element, package: str) -> None:
    for mesh in root.iter("mesh"):
        fn = mesh.get("filename", "")
        if fn.startswith("meshes/"):
            mesh.set("filename", f"package://{package}/{fn}")


def _freeze_joints(root: ET.Element, names: Iterable[str]) -> None:
    name_set = set(names)
    for joint in root.iter("joint"):
        if joint.get("name") in name_set:
            joint.set("type", "fixed")
            # Strip elements that have no meaning on a fixed joint.
            for tag in ("axis", "limit", "dynamics", "safety_controller", "mimic"):
                for child in list(joint.findall(tag)):
                    joint.remove(child)


def _add_world_weld(root: ET.Element, root_link: str = "pelvis") -> None:
    world_link = ET.Element("link", {"name": "world"})
    weld = ET.Element("joint", {"name": "world_to_pelvis", "type": "fixed"})
    ET.SubElement(weld, "parent", {"link": "world"})
    ET.SubElement(weld, "child", {"link": root_link})
    ET.SubElement(weld, "origin", {"xyz": "0 0 0.793", "rpy": "0 0 0"})
    # Place "world" and the weld joint before everything else so the URDF
    # parser sees the world link first.
    root.insert(0, weld)
    root.insert(0, world_link)


def _add_ros2_control(root: ET.Element, joint_names: Iterable[str]) -> None:
    rc = ET.SubElement(root, "ros2_control", {"name": "GazeboSystem", "type": "system"})
    hw = ET.SubElement(rc, "hardware")
    plugin = ET.SubElement(hw, "plugin")
    plugin.text = "gz_ros2_control/GazeboSimSystem"
    for j in joint_names:
        je = ET.SubElement(rc, "joint", {"name": j})
        ci = ET.SubElement(je, "command_interface", {"name": "position"})
        # Provide reasonable default min/max to satisfy controllers that read
        # them; the actual limits in <joint><limit/> still govern Gazebo.
        ET.SubElement(ci, "param", {"name": "min"}).text = "-3.14"
        ET.SubElement(ci, "param", {"name": "max"}).text = "3.14"
        si_pos = ET.SubElement(je, "state_interface", {"name": "position"})
        ET.SubElement(si_pos, "param", {"name": "initial_value"}).text = "0.0"
        ET.SubElement(je, "state_interface", {"name": "velocity"})
        ET.SubElement(je, "state_interface", {"name": "effort"})


def _add_gz_plugin(root: ET.Element, controllers_yaml: str) -> None:
    gz = ET.SubElement(root, "gazebo")
    plugin = ET.SubElement(gz, "plugin", {
        "filename": "gz_ros2_control-system",
        "name": "gz_ros2_control::GazeboSimROS2ControlPlugin",
    })
    ET.SubElement(plugin, "parameters").text = controllers_yaml
    ET.SubElement(plugin, "ros").append(ET.Element("namespace"))  # default ns
    # Use the simulated controller manager update period derived from the
    # controller_manager config, so we don't need <update_rate> here.


def generate(
    source_urdf: Path | str,
    controllers_yaml: str,
    package: str = "g1_description",
    freeze_legs: bool = True,
    upper_body_joints: Iterable[str] | None = None,
) -> str:
    """Return the transformed URDF as an XML string."""
    upper = list(upper_body_joints) if upper_body_joints else UPPER_BODY_JOINTS
    tree = ET.parse(str(source_urdf))
    root = tree.getroot()
    if root.tag != "robot":
        raise ValueError(f"Expected <robot> root, got <{root.tag}>")

    _rewrite_mesh_paths(root, package)
    if freeze_legs:
        _freeze_joints(root, LEG_JOINTS_TO_FREEZE)
    _add_world_weld(root, root_link="pelvis")
    _add_ros2_control(root, upper)
    _add_gz_plugin(root, controllers_yaml)

    # Pretty-print is not strictly required; ROS parsers don't care about whitespace.
    return ET.tostring(root, encoding="unicode")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", required=True, type=Path,
        help="Path to the source G1 URDF (e.g. g1_29dof_lock_waist_rev_1_0.urdf).",
    )
    parser.add_argument(
        "--controllers-yaml", required=True,
        help="Absolute path passed to the gz_ros2_control plugin's <parameters>.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("-"),
        help="Output URDF path. '-' (default) writes to stdout.",
    )
    parser.add_argument(
        "--no-freeze-legs", action="store_true",
        help="Keep leg joints revolute (they will be passive and dangle under gravity).",
    )
    args = parser.parse_args(argv)

    xml = generate(
        source_urdf=args.source,
        controllers_yaml=args.controllers_yaml,
        freeze_legs=not args.no_freeze_legs,
    )
    if str(args.output) == "-":
        sys.stdout.write(xml)
    else:
        args.output.write_text(xml)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
