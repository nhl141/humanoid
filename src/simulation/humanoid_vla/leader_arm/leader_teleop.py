"""Drive the Pioneer left arm in Isaac Sim from the 7-servo leader arm.

The leader is zeroed wherever it is when this program starts. Its relative
angles are added to the simulated arm's default pose:

    A (servo ID 2) -> joint1L (shoulder flexion)
    B (servo ID 3) -> joint2l (shoulder abduction)
    C (servo ID 1) -> joint3l (shoulder rotation)
    D (servo ID 5) -> joint4l (elbow flexion)
    E (servo ID 4) -> joint5l (forearm rotation)
    F (servo ID 7) -> joint6l (wrist)
    G (servo ID 6) -> gripper (starts open at 41.5 deg, closes toward 0)

The remaining robot joints are held at the canonical default pose. Leader
torque is always disabled; the physical arm is an input device only.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

from arm_limits import gripper_fraction, joint_limits_deg, limited_target
from isaaclab.app import AppLauncher

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_SDK_ROOT = _REPO_ROOT / "servo_test" / "STServo_Python" / "stservo-env"

parser = argparse.ArgumentParser(
    description="7-servo leader control of the Pioneer arm in Isaac Sim."
)
parser.add_argument("--port", default="/dev/ttyACM0", help="Leader serial port")
parser.add_argument(
    "--baud", type=int, default=1_000_000, help="Leader serial baud rate"
)
parser.add_argument(
    "--sdk-root",
    type=Path,
    default=_DEFAULT_SDK_ROOT if _DEFAULT_SDK_ROOT.is_dir() else None,
    help="Directory containing a vendored scservo_sdk package. Omitted when that "
    "directory is absent, in which case the installed SDK is used.",
)
parser.add_argument("--scene", default="bare", help="Registered humanoid_scenes scene")
parser.add_argument(
    "--arm",
    choices=("left", "right"),
    default="left",
    help="Sim arm to command (left is the L-suffixed physical-left chain)",
)
parser.add_argument(
    "--signs",
    default="1,-1,1,-1,1,1,1",
    help="Per-axis direction for A,B,C,D,E,F,G (default: 1,-1,1,-1,1,1,1)",
)
parser.add_argument(
    "--scale", type=float, default=1.0, help="Leader-to-sim angular scale"
)
parser.add_argument(
    "--verify-travel",
    action="store_true",
    help="Test sim-only travel past old limits and return to zero before connecting leader",
)
parser.add_argument(
    "--filter-alpha",
    type=float,
    default=0.35,
    help="Target low-pass coefficient in (0,1]; 1 disables filtering",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import omni.ui as ui
import isaaclab.sim as sim_utils
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
from isaaclab.scene import InteractiveScene

sys.path.insert(0, str(_REPO_ROOT / "src" / "pioneer_humanoid"))
from humanoid_scenes import list_scenes, make_scene_cfg, scene_camera
from pioneer_humanoid.bimanual_arm import (
    BIMANUAL_ARM_CFG,
    LEFT_ARM_JOINTS,
    LEFT_GRIPPER_CLOSED,
    LEFT_GRIPPER_JOINTS,
    LEFT_GRIPPER_OPEN,
    RIGHT_ARM_JOINTS,
    RIGHT_GRIPPER_CLOSED,
    RIGHT_GRIPPER_JOINTS,
    RIGHT_GRIPPER_OPEN,
    resolve_joint_name,
)

# Leader reading lives in servo_leader.py so this script and
# vla_imitation_record.py share one implementation.
from servo_leader import ARM_SERVOS, GRIPPER_SERVO, SERVO_IDS, ServoLeader, parse_signs


def _joint_ids(robot, names: list[str]) -> list[int]:
    name_to_id = {name: idx for idx, name in enumerate(robot.data.joint_names)}
    resolved = [resolve_joint_name(robot, name) for name in names]
    return [name_to_id[name] for name in resolved]


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene) -> None:
    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()
    scene.update(sim_dt)

    arm_joints = LEFT_ARM_JOINTS if args_cli.arm == "left" else RIGHT_ARM_JOINTS
    controlled_names = arm_joints[: len(ARM_SERVOS)]
    controlled_ids = _joint_ids(robot, controlled_names)
    if args_cli.arm == "left":
        gripper_names, g_open, g_closed = LEFT_GRIPPER_JOINTS, LEFT_GRIPPER_OPEN, LEFT_GRIPPER_CLOSED
    else:
        gripper_names, g_open, g_closed = RIGHT_GRIPPER_JOINTS, RIGHT_GRIPPER_OPEN, RIGHT_GRIPPER_CLOSED
    gripper_ids = _joint_ids(robot, gripper_names)
    gripper_open = torch.tensor([[g_open[j] for j in gripper_names]], device=sim.device)
    gripper_closed = torch.tensor([[g_closed[j] for j in gripper_names]], device=sim.device)
    gripper_axis = tuple(SERVO_IDS).index(GRIPPER_SERVO)
    gripper_frac = 0.0
    actual_limits = robot.root_physx_view.get_dof_limits()[:, controlled_ids, :]
    bounds = joint_limits_deg(args_cli.arm)
    print(
        f"[INFO] Human-style limits (degrees): {dict(zip(ARM_SERVOS, bounds))}; "
        f"physics bounds={actual_limits.tolist()}",
        flush=True,
    )

    signs = parse_signs(args_cli.signs)
    print(f"[INFO] Active directions: {dict(zip(SERVO_IDS, signs))}", flush=True)
    if not 0.0 < args_cli.filter_alpha <= 1.0:
        raise ValueError("--filter-alpha must be in (0, 1]")

    default_pos = robot.data.default_joint_pos.clone()
    default_vel = robot.data.default_joint_vel.clone()
    # The canonical articulation pose bends joint4l by -75 degrees to avoid an
    # IK singularity. This direct joint-space leader instead uses the physical
    # arm's fully straight resting pose as zero, so A-E all start at 0 degrees.
    neutral_pos = default_pos.clone()
    neutral_pos[:, controlled_ids] = 0.0
    robot.write_joint_state_to_sim(neutral_pos, default_vel)
    filtered_target = neutral_pos[:, controlled_ids].clone()

    if args_cli.verify_travel:
        for label, degrees in (
            ("upper overshoot", [hi + 5 for lo, hi in bounds]),
            ("return to zero", [0] * len(bounds)),
            ("lower overshoot", [lo - 5 for lo, hi in bounds]),
            ("return to zero", [0] * len(bounds)),
        ):
            target = neutral_pos.clone()
            target[:, controlled_ids] = torch.tensor(
                [
                    limited_target(math.radians(v), 1, 1, b)
                    for v, b in zip(degrees, bounds)
                ],
                device=sim.device,
            )
            for _ in range(500):
                robot.set_joint_position_target(target)
                robot.set_joint_velocity_target(torch.zeros_like(default_vel))
                scene.write_data_to_sim()
                sim.step()
                scene.update(sim_dt)
            measured = robot.data.joint_pos[:, controlled_ids]
            error = (
                torch.rad2deg((measured - target[:, controlled_ids]).abs()).max().item()
            )
            print(
                f"[VERIFY] {label}: measured_deg={torch.rad2deg(measured).tolist()} "
                f"max_error_deg={error:.2f}",
                flush=True,
            )
            if not math.isfinite(error) or error > (
                1 if label == "return to zero" else 5
            ):
                raise RuntimeError(f"Sim travel verification failed: {label}")
        robot.write_joint_state_to_sim(neutral_pos, default_vel)

    leader = ServoLeader(args_cli.port, args_cli.baud, args_cli.sdk_root)
    should_rezero = False

    keyboard = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.0, rot_sensitivity=0.0))

    def request_rezero() -> None:
        nonlocal should_rezero
        should_rezero = True

    keyboard.add_callback("R", request_rezero)

    controls_window = ui.Window("Leader Arm Controls", width=420, height=240)
    with controls_window.frame:
        with ui.VStack(spacing=8, height=0):
            ui.Label("Hold the leader in the desired straight zero pose")
            ui.Button("ZERO LEADER", height=44, clicked_fn=request_rezero)
            for label, (lo, hi) in zip(ARM_SERVOS, bounds):
                ui.Label(f"{label}: {lo} to {hi} degrees")
            ui.Label(f"{GRIPPER_SERVO}: gripper, zero it open")

    print("[INFO] Six-joint + gripper leader teleop ready.", flush=True)
    print(
        "[INFO] Mapping: "
        + " | ".join(
            f"{label}(ID{SERVO_IDS[label]}) -> {joint}"
            for label, joint in zip(ARM_SERVOS, controlled_names)
        )
        + f" | {GRIPPER_SERVO}(ID{SERVO_IDS[GRIPPER_SERVO]}) -> gripper",
        flush=True,
    )
    print(
        "[INFO] Move the leader by hand. Use ZERO LEADER or press R to re-zero.",
        flush=True,
    )

    last_angles = (0.0,) * len(SERVO_IDS)
    last_report = time.monotonic()
    last_warning = 0.0

    try:
        while simulation_app.is_running():
            loop_started = time.monotonic()
            keyboard.advance()  # pumps callbacks; motion output is intentionally unused

            if should_rezero:
                leader.rezero()
                robot.write_joint_state_to_sim(neutral_pos, default_vel)
                robot.reset()
                filtered_target.copy_(neutral_pos[:, controlled_ids])
                gripper_frac = 0.0
                last_angles = (0.0,) * len(SERVO_IDS)
                should_rezero = False
                print("[INFO] Leader and simulated arm re-zeroed.", flush=True)

            try:
                last_angles = leader.read_radians()
            except RuntimeError as exc:
                # A transient serial miss should hold the last safe target, not stop physics.
                now = time.monotonic()
                if now - last_warning >= 1.0:
                    print(f"[WARN] {exc}; holding last target", flush=True)
                    last_warning = now

            desired = neutral_pos[:, controlled_ids].clone()
            for axis in range(len(controlled_names)):
                desired[0, axis] = limited_target(
                    last_angles[axis], signs[axis], args_cli.scale, bounds[axis]
                )
            filtered_target.lerp_(desired, args_cli.filter_alpha)
            gripper_frac += args_cli.filter_alpha * (
                gripper_fraction(last_angles[gripper_axis], signs[gripper_axis])
                - gripper_frac
            )

            # Hold every non-driven joint at its canonical pose and command the
            # six arm joints and the gripper from the leader.
            robot.set_joint_position_target(neutral_pos)
            robot.set_joint_velocity_target(torch.zeros_like(default_vel))
            robot.set_joint_position_target(filtered_target, joint_ids=controlled_ids)
            robot.set_joint_position_target(
                torch.lerp(gripper_open, gripper_closed, gripper_frac),
                joint_ids=gripper_ids,
            )
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

            now = time.monotonic()
            if now - last_report >= 0.5:
                degrees = [math.degrees(value) for value in last_angles]
                targets = [math.degrees(float(value)) for value in filtered_target[0]]
                print(
                    "\r[LEADER] "
                    + " ".join(
                        f"{label}={value:+6.1f}deg"
                        for label, value in zip(SERVO_IDS, degrees)
                    )
                    + f" grip={gripper_frac:.2f}"
                    + " -> "
                    + " ".join(
                        f"{name}={value:+6.1f}deg"
                        for name, value in zip(controlled_names, targets)
                    ),
                    end="",
                    flush=True,
                )
                last_report = now

            # Cap polling at 50 Hz, matching the standalone reader's cadence.
            # Only one process may own this serial port at a time.
            elapsed = time.monotonic() - loop_started
            if elapsed < 0.02:
                time.sleep(0.02 - elapsed)
    finally:
        leader.close()
        print("\n[INFO] Stopped. Leader torque is OFF.", flush=True)


def main() -> None:
    if args_cli.scene not in list_scenes():
        raise SystemExit(
            f"unknown --scene {args_cli.scene!r}; available: {list_scenes()}"
        )
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    )
    sim.set_camera_view(
        *(scene_camera(args_cli.scene) or ([2.5, 2.5, 2.0], [0.0, 0.0, 0.8]))
    )
    scene = InteractiveScene(
        make_scene_cfg(args_cli.scene, BIMANUAL_ARM_CFG, num_envs=1, env_spacing=2.0)
    )

    # Set the same bounds in USD before PhysX creates the articulation.
    from pxr import Usd, UsdPhysics
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    names = (LEFT_ARM_JOINTS if args_cli.arm == "left" else RIGHT_ARM_JOINTS)[
        : len(ARM_SERVOS)
    ]
    limits_by_name = dict(zip(names, joint_limits_deg(args_cli.arm)))
    changed = set()
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/envs/env_0/Robot")):
        if prim.GetName() in names and prim.IsA(UsdPhysics.RevoluteJoint):
            joint = UsdPhysics.RevoluteJoint(prim)
            lo, hi = limits_by_name[prim.GetName()]
            joint.GetLowerLimitAttr().Set(float(lo))
            joint.GetUpperLimitAttr().Set(float(hi))
            changed.add(prim.GetName())
    if changed != set(names):
        raise RuntimeError(
            f"Missing joints while setting limits: {set(names) - changed}"
        )

    sim.reset()
    run_simulator(sim, scene)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
