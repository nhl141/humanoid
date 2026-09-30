"""Keyboard teleop + LeRobot recording for the Pioneer left arm.

W/A/S/D/Q/E jog an end-effector pose target through a differential IK solver;
C and V jog the two wrist joints directly, K works the gripper, R resets the
scene, and B/N/M drive episode recording.

This is the keyboard-only variant. For the same scene and recorder driven by
the physical 5-servo leader arm instead, use vla_imitation_record.py -- the two
produce interchangeable datasets (identical action/state layout and schema).
"""
import argparse
import random
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

_IL_PKG = Path(__file__).resolve().parents[2] / "il"
_DEFAULT_SIM_SCHEMA = _IL_PKG / "config" / "dataset_schema_pioneer_vla.yaml"

# Robot USD. Overrides BIMANUAL_ARM_CFG's own usd_path (assets/.../pioneer_bimanual_arm.usd) for
# THIS script only -- that cfg is shared with the RL tasks, quest teleop and task_space_ik's
# real-arm publisher, so it is not edited in place. new_pioneer_all_cams adds the wrist cameras
# and the base/wrist RealSense rigs; same 12 revolute + 4 prismatic joints.
_DEFAULT_ROBOT_USD = str(
    Path(__file__).resolve().parents[3]
    / "assets" / "pioneer_bimanual_arm" / "usd" / "new_pioneer_all_cams.usd"
)

# pioneer_humanoid package (canonical arm config). Editable-installed in the image; this fallback
# keeps a bare bind-mounted checkout working.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pioneer_humanoid"))

# Camera prims available as live Kit viewport windows, keyed by --cam-viewport name.
# Each value is (path suffix under the robot prim, window title). Suffixes are the same
# prims MySceneCfg's CameraCfg sensors bind to -- see the note on open_camera_viewports
# for why both a sensor and a viewport exist for the same camera.
_CAM_VIEWPORT_PRIMS = {
    "base": (
        "/base_link/Realsense/RSD455/Camera_OmniVision_OV9782_Color",
        "RealSense RGB (base)",
    ),
    "wrist_left": ("/link6l/left_wrist_camera_link/left_wrist_camera", "Wrist RGB (left)"),
    "wrist_right": ("/link6/right_wrist_camera_link/right_wrist_camera", "Wrist RGB (right)"),
}

parser = argparse.ArgumentParser(
    description="Keyboard teleoperation + VLA recording for the Pioneer bimanual arm (left only)."
)
# Recording is always available from the keyboard (B / N / M, see run_simulator); these only
# configure it. Needs humanoid-il: pip install -e src/il[sim].
parser.add_argument(
    "--schema",
    type=str,
    default=str(_DEFAULT_SIM_SCHEMA),
    help="dataset_schema YAML (default: src/il/config/dataset_schema_pioneer_vla.yaml)",
)
parser.add_argument(
    "--dataset_root",
    type=str,
    default=None,
    help="Override record.root from schema (e.g. datasets/record_sim)",
)
parser.add_argument(
    "--max_episode_s",
    type=float,
    default=90.0,
    help="longest recordable episode; sizes the GPU + pinned-CPU frame buffers "
    "(~0.6 GB each per 30 s with three 256x256 cameras at 25 fps)",
)
parser.add_argument("--num_envs", type=int, default=1, help="parallel envs to clone")
# pi0.5 reads this string as its language prompt, so phrase it as the instruction.
parser.add_argument("--task_description", type=str, default="pick up the cube and place it in the tray")
parser.add_argument(
    "--scene",
    type=str,
    default="bare",
    help="scene name: 'bare' (arm only), 'push', or any scene registered in "
    "humanoid_scenes (validated after launch — pass an unknown name to list them)",
)
parser.add_argument(
    "--robot-usd",
    type=str,
    default=_DEFAULT_ROBOT_USD,
    help="robot USD to spawn (default: assets/pioneer_bimanual_arm/usd/new_pioneer_all_cams.usd)",
)
parser.add_argument(
    "--cam-viewport",
    type=str,
    default="base,wrist_left",
    help="comma-separated live camera viewport windows: "
    + ", ".join(_CAM_VIEWPORT_PRIMS)
    + ", all, or none (default: base,wrist_left -- the base RealSense RGB plus the "
    "wrist camera on the actuated (L-suffixed) arm)",
)
parser.add_argument(
    "--cam-viewport-size",
    type=str,
    default="480x360",
    help="WxH of each camera viewport window (default: 480x360). Window size only -- the "
    "CameraCfg sensor resolutions used for recording are separate.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Validate the viewport args HERE, before AppLauncher: a typo should fail in a
# millisecond, not after Isaac Sim has spent a minute booting.
_cam_viewport_names = [s.strip() for s in args_cli.cam_viewport.split(",") if s.strip()]
if "none" in _cam_viewport_names:
    _cam_viewport_names = []
elif "all" in _cam_viewport_names:
    _cam_viewport_names = list(_CAM_VIEWPORT_PRIMS)
for _name in _cam_viewport_names:
    if _name not in _CAM_VIEWPORT_PRIMS:
        parser.error(
            f"--cam-viewport: unknown camera '{_name}'. "
            f"Choose from: {', '.join(_CAM_VIEWPORT_PRIMS)}, all, none"
        )
try:
    _w, _h = (int(v) for v in args_cli.cam_viewport_size.lower().split("x", 1))
except ValueError:
    parser.error(f"--cam-viewport-size: expected WxH (e.g. 480x360), got '{args_cli.cam_viewport_size}'")
_CAM_VIEWPORT_SIZE = (_w, _h)

# MySceneCfg's CameraCfg sensors, the viewport windows and recording all need the render
# pipeline, and a Camera sensor refuses to build without it -- so it is always on here
# rather than a flag that every invocation has to remember.
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
import omni.appwindow
import torch

import isaaclab.sim as sim_utils
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.utils.math import subtract_frame_transforms
import math
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

# This script actuates the L-suffixed chain (physical LEFT arm) and holds the unsuffixed
# one. Canonical names it LEFT_*; the aliases below keep this file's RIGHT_*/GRIPPER_* local
# names (RIGHT_* = the actuated arm) so the body is unchanged.
from pioneer_humanoid.bimanual_arm import (
    BIMANUAL_ARM_CFG,
    LEFT_GRIPPER_CLOSED as GRIPPER_CLOSED,
    LEFT_GRIPPER_OPEN as GRIPPER_OPEN,
    LEFT_ARM_JOINTS as RIGHT_ARM_JOINTS,
    LEFT_EE_BODY as RIGHT_EE_BODY,
    LEFT_GRIPPER_JOINTS as RIGHT_GRIPPER_JOINTS,
    LEFT_FINGER_TIP_BODIES as RIGHT_FINGER_TIP_BODIES,
    RIGHT_ARM_JOINTS as LEFT_ARM_JOINTS,
)

from pioneer_humanoid.bimanual_arm import (
    apply_joint_limits,
    compute_gripper_tip_pose_b,
    compute_tip_ik_jacobian,
    patch_joint_pos_limits_on_prim,
    resolve_body_ids,
    resolve_joint_name,
)
from humanoid_scenes.pick_and_place_vla.scene import ROBOT_PRIM_IN_SCENE


# Wrist damping, LOCAL to this script -- same reasoning as _DEFAULT_ROBOT_USD at the top:
# BIMANUAL_ARM_CFG is shared with the RL tasks, quest teleop and task_space_ik's real-arm
# publisher, so it is not edited in place.
#
# Stock is 18.0 N m s/rad against joint6l's 0.73 N m effort cap. An implicit actuator is
# tau = k*err + d*qd clipped at the cap, so damping alone fixes the joint's top speed at
# effort/damping = 0.041 rad/s -- about 2 deg/s, and ~19% less once gravity takes its cut
# with the arm loaded. That is tens of times slower than any teleop jog, which is why the
# wrist reads as frozen. 18.0 is also roughly 15x critical damping for joint6l's ~1e-3
# kg m^2 reflected inertia, so it is overdamped by a wide margin.
#
# Lowering damping buys speed inside the SAME torque budget -- the 0.73 N m cap is the
# real GL40 peak and is deliberately left alone for sim-to-real fidelity.
_WRIST_DAMPING = 2.5                     # -> (0.73 - 0.14 gravity) / 2.5 ~= 0.24 rad/s


def _wrist_damping_override(cfg, damping: float) -> dict:
    """cfg.actuators with the left_wrist group's damping swapped out.

    Copies the dict and replaces the one entry, so BIMANUAL_ARM_CFG's own actuator
    objects are left untouched for every other importer of that config.
    """
    actuators = dict(cfg.actuators)
    actuators["left_wrist"] = actuators["left_wrist"].replace(damping=damping)
    return actuators
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

# The humanoid_vla env USD: GroundPlane + DomeLight/KeyLight + table/block/box,
# and the arm referenced in at /World/new_pioneer_all_cams (z=1.202, floor-stand
# height). The arm is NOT spawned by this script -- it already lives in the USD.
# v2 = v1 with box.usd's flat tray swapped for assets/props/bin.usd, an open-top bin
# (scripts/build_bin_usd.py builds both).
_VLA_ENV_USD = str(Path(__file__).resolve().parents[0] / "assets" / "09-27_vla_env_v2.usd")


# ---------------------------------------------------------------------------
# Scene geometry -- all measured off 09-19_vla_env_v1.usd with a USD BBoxCache (the bin
# comes from scripts/build_bin_usd.py instead),
# world frame, metres. Re-measure if the env USD's props move.
# ---------------------------------------------------------------------------
TABLE_TOP_Z = 0.7507                     # table upper surface
TABLE_X = (0.2802, 0.9025)               # table top extent
TABLE_Y = (-0.6953, 0.6953)
TABLE_MARGIN = 0.02                      # keep props this far inside the rim

# link1L origin == the actuated (left) arm's shoulder. Reach is measured from here,
# not from the robot root, because the root sits at the bottom of the floor stand.
SHOULDER = (0.0216, 0.1633, 1.2625)

# Summed L-chain link lengths, shoulder -> fingertip (link1L..link6l plus link6l->link7l).
# link7l/link8l are the two jaw fingers, so that segment is the jaw gap, not reach.
ARM_CHAIN_LEN = 0.783
# Usable envelope. 0.75 is ~96% of full extension: deliberately permissive so the
# scene's authored cube pose (0.737 from the shoulder) stays inside it. Lower it if
# the IK starts failing near the far edge -- see the note in randomize_props().
REACH_MAX_3D = 0.75

# Props spawn inside 2/3 of the reachable table footprint. The full envelope is where
# the gripper can *touch*, not where it can *grasp*: out at the rim it arrives fully
# extended, with no margin left to orient the jaws, and picks fail. This scales the
# HORIZONTAL budget, not REACH_MAX_3D itself -- 2/3 of the 3D radius is 0.50 m, which is
# less than the 0.51 m shoulder-to-table drop, so it would leave the sampler with no
# valid points at all. At 0.75 the horizontal budget is ~0.548 m, so this gives ~0.365 m.
PROP_REACH_SCALE = 2.0 / 3.0

# Prop half-extents (cube is 0.0508^3; the "tray" is bin.usd, 0.20 x 0.20 x 0.07 --
# keep in sync with OUTER/HEIGHT in scripts/build_bin_usd.py).
CUBE_HALF = 0.0254
TRAY_HALF = 0.10

# Each prop's prim origin is offset from its geometry centre (the offset lives inside
# the payload). Poses are written to the prim, but constraints apply to the geometry,
# so convert with these. OFFSET = geometry_centre - prim_origin.
BLOCK_PRIM_0 = (0.5226, 0.0, 0.7502)
BLOCK_OFFSET = (0.0254, 0.2225, 0.0254)
# bin.usd's origin is its bottom centre, so its offset is only half its height.
TRAY_PRIM_0 = (0.5007, -0.2886, 0.7517)
TRAY_OFFSET = (0.0, 0.0, 0.035)
TRAY_ROT_0 = (1.0, 0.0, 0.0, 0.0)        # identity (w, x, y, z); the bin is square

# The cube spawns in a small disc rather than anywhere in the reach envelope, so keyboard
# teleop starts every episode a short, easy move away. Centre is geometry-centre XY,
# 0.41 m horizontally from SHOULDER -- the same reach as the old (0.42, 0.16) disc, just
# swung out to the left arm's own side (+Y) instead of sitting on the centreline.
BLOCK_SPAWN_CENTRE = (0.42, 0.26)
BLOCK_SPAWN_RADIUS = 0.08

# Table centre, used only for the cube keep-out below.
TABLE_CENTRE = ((TABLE_X[0] + TABLE_X[1]) / 2.0, (TABLE_Y[0] + TABLE_Y[1]) / 2.0)

# Minimum |y - table centreline| for the cube. Keyboard teleop jogs the tip along world
# axes (W/A/S/D/Q/E) and cannot curve the arm around anything, so a cube sitting on the
# table's centreline has to be approached across the body -- awkward at best, and the
# elbow fouls the far side of the bin at worst. Holding it out on the left arm's side
# makes every grasp a straight reach. Rejection-sampled inside the disc rather than
# baked into the centre, so the disc can be nudged back toward the middle later without
# silently losing the guarantee.
BLOCK_MIN_CENTRELINE_OFFSET = 0.15

# Keep the cube out from under/inside the tray, and far enough out that the pick and the
# place stay two distinct moves: the gripper has to clear the cube's start height before
# it is anywhere near the bin rim. This is the free gap between the bin's rim and the
# cube's nearest face; MIN_PROP_SEPARATION is the centre-to-centre distance it implies.
PROP_CLEARANCE = 0.07
MIN_PROP_SEPARATION = TRAY_HALF + CUBE_HALF + PROP_CLEARANCE


def _reach_ok(x: float, y: float, inset: float = 0.0,
              reach_max: float = REACH_MAX_3D,
              xy_scale: float = PROP_REACH_SCALE) -> bool:
    """True if a table-height point at (x, y) is inside the arm's spawn envelope.

    ``inset`` shrinks the horizontal distance before the check: pass the tray's
    half-width so the tray is judged by its near rim (what the gripper actually has
    to reach over) rather than by its centre, which can sit past full extension.

    ``xy_scale`` shrinks the horizontal budget so props land comfortably inside the
    envelope rather than at its rim -- see PROP_REACH_SCALE. Pass 1.0 for the raw
    geometric reach test.
    """
    d_xy = math.hypot(x - SHOULDER[0], y - SHOULDER[1])
    d_xy = max(0.0, d_xy - inset)
    # Horizontal slice of the reach sphere at table height: the table sits
    # SHOULDER[2] - TABLE_TOP_Z below the shoulder, so only this is left for XY travel.
    xy_budget = math.sqrt(max(0.0, reach_max ** 2 - (SHOULDER[2] - TABLE_TOP_Z) ** 2))
    return d_xy <= xy_budget * xy_scale


def _sample_centre(rng, half: float, inset: float, reach_max: float = REACH_MAX_3D,
                   table_margin: float = TABLE_MARGIN, tries: int = 400):
    """Uniform geometry-centre XY that is fully on the table AND within reach.

    Rejection sampling: the table rectangle and the reach disc only partly overlap,
    so there is no closed form. Returns None if no sample landed in the overlap.
    """
    x_lo, x_hi = TABLE_X[0] + half + table_margin, TABLE_X[1] - half - table_margin
    y_lo, y_hi = TABLE_Y[0] + half + table_margin, TABLE_Y[1] - half - table_margin
    for _ in range(tries):
        x, y = rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)
        if _reach_ok(x, y, inset, reach_max):
            return x, y
    return None


def _sample_block_centre(rng, centre=BLOCK_SPAWN_CENTRE, radius: float = BLOCK_SPAWN_RADIUS,
                         min_centreline_offset: float = BLOCK_MIN_CENTRELINE_OFFSET,
                         tries: int = 200):
    """Uniform cube geometry-centre XY inside the BLOCK_SPAWN_* disc.

    Rejects anything within ``min_centreline_offset`` of the table's centreline -- see
    BLOCK_MIN_CENTRELINE_OFFSET. Rejection keeps the surviving region uniform; if the
    disc were clipped instead the cube would pile up along the cut. Falls back to the
    disc centre, which is outside the keep-out by construction, so a reset never stalls.
    """
    for _ in range(tries):
        r = radius * math.sqrt(rng.random())    # sqrt -> uniform over the disc's area
        a = rng.uniform(0.0, 2.0 * math.pi)
        x, y = centre[0] + r * math.cos(a), centre[1] + r * math.sin(a)
        if abs(y - TABLE_CENTRE[1]) >= min_centreline_offset:
            return x, y
    return centre


def sample_prop_poses(rng, reach_max: float = REACH_MAX_3D,
                      table_margin: float = TABLE_MARGIN,
                      min_separation: float = MIN_PROP_SEPARATION):
    """Sample (block_prim_xyz, tray_prim_xyz) for one reset.

    Both are returned as PRIM positions (geometry centre minus the payload offset),
    ready to hand to write_root_pose_to_sim. Falls back to the authored poses if the
    sampler cannot satisfy the constraints, so a reset never leaves props unplaced.

    The cube goes first: its disc is small, so the tray is the one that gives way.
    """
    cube = _sample_block_centre(rng)
    for _ in range(400):
        tray = _sample_centre(rng, TRAY_HALF, TRAY_HALF, reach_max, table_margin)
        if tray is None:
            break
        if math.hypot(cube[0] - tray[0], cube[1] - tray[1]) >= min_separation:
            block_prim = (
                cube[0] - BLOCK_OFFSET[0],
                cube[1] - BLOCK_OFFSET[1],
                BLOCK_PRIM_0[2],
            )
            tray_prim = (
                tray[0] - TRAY_OFFSET[0],
                tray[1] - TRAY_OFFSET[1],
                TRAY_PRIM_0[2],
            )
            return block_prim, tray_prim
    # Only the tray search failed -- the cube's disc is unconditionally valid, so keep
    # the sampled cube and drop just the tray back to its authored pose. Returning both
    # authored poses here would quietly undo the cube randomisation on every near-miss.
    block_prim = (
        cube[0] - BLOCK_OFFSET[0],
        cube[1] - BLOCK_OFFSET[1],
        BLOCK_PRIM_0[2],
    )
    if math.hypot(cube[0] - TRAY_PRIM_0[0] - TRAY_OFFSET[0],
                  cube[1] - TRAY_PRIM_0[1] - TRAY_OFFSET[1]) >= min_separation:
        return block_prim, TRAY_PRIM_0
    return BLOCK_PRIM_0, TRAY_PRIM_0


def randomize_props(scene, rng, reach_max: float = REACH_MAX_3D,
                    table_margin: float = TABLE_MARGIN,
                    min_separation: float = MIN_PROP_SEPARATION,
                    block_key: str = "block", tray_key: str = "tray"):
    """Reposition cube + tray on the table. Call from the teleop reset path.

    Zeroes velocities too -- a rigid body keeps its old velocity across a pose write
    and would otherwise skate off the table on the first step after a reset.
    """
    block_xyz, tray_xyz = sample_prop_poses(rng, reach_max, table_margin, min_separation)
    for key, xyz, rot in ((block_key, block_xyz, (1.0, 0.0, 0.0, 0.0)),
                          (tray_key, tray_xyz, TRAY_ROT_0)):
        obj = scene[key]
        pose = torch.tensor([[*xyz, *rot]], device=scene.device, dtype=torch.float32)
        pose[:, 0:3] += scene.env_origins           # correct for num_envs > 1
        obj.write_root_pose_to_sim(pose)
        obj.write_root_velocity_to_sim(torch.zeros((scene.num_envs, 6), device=scene.device))
    return block_xyz, tray_xyz


# Module-level RNG so an EventTerm (which takes no rng argument) stays reproducible
# via PROP_RNG.seed(...) from the caller.
PROP_RNG = random.Random()


def randomize_props_event(
    env,
    env_ids=None,
    reach_max: float = REACH_MAX_3D,
    table_margin: float = TABLE_MARGIN,
    min_separation: float = MIN_PROP_SEPARATION,
    block_key: str = "block",
    tray_key: str = "tray",
    seed: int | None = None,
):
    """EventTerm-compatible wrapper around randomize_props.

    Every knob is a keyword so it can be driven from EventTerm(params={...}) without
    editing this file. ``seed`` re-seeds PROP_RNG on each call: set it for a
    reproducible layout, leave it None for a fresh one per reset.

    NOTE: EventTerms only fire inside a ManagerBasedRLEnv. A plain teleop script
    (SimulationContext + InteractiveScene + hand-written loop) has no event manager,
    so there call randomize_props(scene, rng) directly from the reset branch.
    """
    if seed is not None:
        PROP_RNG.seed(seed)
    return randomize_props(
        env.scene, PROP_RNG, reach_max, table_margin, min_separation, block_key, tray_key
    )


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """VLA pick-and-place scene, robot included in the env USD."""

    # Declared FIRST: spawning this creates the arm prim that `robot` binds to below, and
    # InteractiveScene builds entities in field order.
    #
    # The `: AssetBaseCfg` annotation is LOAD-BEARING, not decoration. configclass keeps
    # explicitly annotated fields in class-body order but appends auto-annotated (bare
    # `x = ...`) ones after them -- so with `robot` annotated and this one bare, `robot`
    # sorted ahead of `scene_env`, the Articulation was built before the env USD had
    # spawned, and the scene died with "Could not find prim with path
    # /World/envs/env_.*/Scene/new_pioneer_all_cams". Every field here is annotated to
    # pin the order.
    scene_env: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Scene",
        spawn=sim_utils.UsdFileCfg(usd_path=_VLA_ENV_USD),
    )

    # spawn=None -> bind to the arm already inside scene_env rather than spawning one.
    # init_state.pos must match the arm's z inside the USD.
    robot: ArticulationCfg = BIMANUAL_ARM_CFG.replace(
        prim_path=ROBOT_PRIM_IN_SCENE,
        spawn=None,
        init_state=BIMANUAL_ARM_CFG.init_state.replace(pos=(0.0, 0.0, 1.202)),
        actuators=_wrist_damping_override(BIMANUAL_ARM_CFG, _WRIST_DAMPING),
    )

    # The cube and tray already exist inside the env USD (both carry PhysicsRigidBodyAPI
    # via their payloads), so these bind to them with spawn=None -- same trick as `robot`.
    # Without these entries they would only be XFormPrims under `scene_env` and could not
    # be repositioned at runtime.
    block: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/Environment/block",
        spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(pos=BLOCK_PRIM_0),
    )
    tray: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/Environment/box",
        spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(pos=TRAY_PRIM_0, rot=TRAY_ROT_0),
    )

    # --- cameras -------------------------------------------------------------
    # All three are prims that already exist inside the robot USD, so spawn=None
    # binds to them. Declaring them HERE (rather than building Camera sensors at
    # runtime) means scene.update(dt) ticks them for free -- no manual .update()
    # and no _initialize_callback poke, which is what quest teleop has to do for
    # its runtime-created eye cameras.
    #
    # data_types: add "distance_to_image_plane" for depth. Each extra type costs
    # another render pass, so a VLA policy that only consumes RGB should stay at rgb.
    wrist_cam_left: CameraCfg = CameraCfg(
        prim_path=ROBOT_PRIM_IN_SCENE + "/link6l/left_wrist_camera_link/left_wrist_camera",
        spawn=None, width=256, height=256, update_period=0.0, data_types=["rgb"],
    )
    wrist_cam_right: CameraCfg = CameraCfg(
        prim_path=ROBOT_PRIM_IN_SCENE + "/link6/right_wrist_camera_link/right_wrist_camera",
        spawn=None, width=256, height=256, update_period=0.0, data_types=["rgb"],
    )
    # BASE CAM: the RSD455 body is referenced from the Omniverse S3 CDN, so this prim
    # only exists once that resolves (needs network on first load). If it is missing,
    # Camera init raises "Could not find prim with path ..." -- drop this entry or
    # vendor rsd455.usd locally.
    base_cam: CameraCfg = CameraCfg(
        prim_path=ROBOT_PRIM_IN_SCENE + "/base_link/Realsense/RSD455/Camera_OmniVision_OV9782_Color",
        spawn=None, width=256, height=256, update_period=0.0, data_types=["rgb"],
    )


class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )

    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # func is randomize_props_EVENT, not randomize_props: EventTerm calls
    # func(env, env_ids, **params), and randomize_props takes (scene, rng).
    reset_prop_positions = EventTerm(
        func=randomize_props_event,
        mode="reset",
        params={
            "reach_max": REACH_MAX_3D,
            "table_margin": TABLE_MARGIN,
            "min_separation": MIN_PROP_SEPARATION,
            "block_key": "block",
            "tray_key": "tray",
            "seed": None,          # set an int for a reproducible layout
        },
    )

# Kit drops a ViewportWindow the moment its last Python reference goes, so the windows
# have to be parked at module level -- a local in main() would close them all on return.
_OPEN_VIEWPORTS = []


def open_camera_viewports(robot_prim_path: str, names: list[str], size=(480, 360)) -> list:
    """Open one live Kit viewport window per named camera prim.

    A viewport is NOT a second way of reading the CameraCfg sensors in MySceneCfg -- the
    two are independent consumers of the same camera prim. The sensor renders at its cfg
    width/height into a tensor (what recording and a VLA policy read); the viewport renders
    at the window size for the operator's eyes only, and nothing reads it back. So each
    window added here costs an extra render pass per frame on top of the sensors'.

    Cameras are resolved under env_0 only: one window per camera, not per env, since
    num_envs > 1 would otherwise open num_envs identical windows.
    """
    if not names:
        return []
    # Both imports are absent in a build with no viewport extension (and create_viewport_window
    # itself returns None when omni.kit.viewport.window is not loaded, e.g. true headless).
    try:
        from omni.kit.viewport.utility import create_viewport_window
    except ImportError as exc:
        print(f"[WARN] no camera viewports -- omni.kit.viewport.utility unavailable ({exc})", flush=True)
        return []
    import omni.usd

    matches = sim_utils.find_matching_prim_paths(robot_prim_path)
    if not matches:
        print(f"[WARN] no camera viewports -- nothing matched robot prim {robot_prim_path}", flush=True)
        return []
    robot_root = matches[0]
    stage = omni.usd.get_context().get_stage()
    width, height = size

    windows = []
    for i, name in enumerate(names):
        suffix, label = _CAM_VIEWPORT_PRIMS[name]
        cam_path = robot_root + suffix
        # The base RealSense body is a reference to rsd455.usd on the Omniverse S3 CDN, so
        # its camera prim only exists once that has resolved (needs network on first load).
        # Skip rather than raise: a missing preview window must not kill a teleop session.
        if not stage.GetPrimAtPath(cam_path).IsValid():
            print(f"[WARN] skipping '{name}' viewport -- no prim at {cam_path}", flush=True)
            continue
        window = create_viewport_window(
            name=label,
            camera_path=cam_path,
            width=width,
            height=height,
            position_x=20,
            position_y=60 + i * (height + 40),
        )
        if window is None:
            print(f"[WARN] could not create '{name}' viewport window (headless?)", flush=True)
            continue
        # Window size and render-texture size are two different properties: left alone, a
        # 320x240 window still renders 1280x720 and costs the same as a full one. Pin the
        # texture to the window so --cam-viewport-size actually buys back frame time.
        window.viewport_api.resolution = (width, height)
        windows.append(window)
        print(f"[INFO] camera viewport '{label}' -> {cam_path}", flush=True)

    _OPEN_VIEWPORTS.extend(windows)
    return windows


def _joint_ids(robot, names: list[str]) -> list[int]:
    """Articulation joint indices for a list of config joint names."""
    resolved = [resolve_joint_name(robot, n) for n in names]
    return [list(robot.data.joint_names).index(n) for n in resolved]


# Schema image key -> MySceneCfg camera sensor. The keys become observation.images.<key>.
_RECORD_CAMS = {"base": "base_cam", "wrist_left": "wrist_cam_left", "wrist_right": "wrist_cam_right"}
# What the dataset's state/action vectors hold, in order: the actuated arm, then its fingers.
_RECORD_JOINTS = list(RIGHT_ARM_JOINTS) + list(RIGHT_GRIPPER_JOINTS)


def _init_recorder(scene: InteractiveScene, sim_dt: float):
    """(SimLeRobotRecorder, sim steps per recorded frame) from --schema, or (None, 0).

    Built at startup so a bad schema fails before any teleop, but the dataset itself is only
    opened (init_dataset) on the first B press -- a session that never records leaves no
    folder behind. Without humanoid-il installed, teleop still runs with recording disabled.

    The recorder's own start_keyboard() is deliberately NOT used: it listens through pynput,
    system-wide, on S/N/D -- and S/D also drive the arm, so every jog would start or discard
    an episode. run_simulator routes episode keys through its carb subscription instead.
    """
    if str(_IL_PKG) not in sys.path:
        sys.path.insert(0, str(_IL_PKG))
    try:
        from humanoid_il.record_utils import resolve_config_path
        from humanoid_il.schema import enabled_images, load_yaml
        from humanoid_il.sim_recorder import SimLeRobotRecorder
    except ImportError as exc:
        print(f"[WARN] recording disabled -- humanoid-il not importable ({exc}). "
              "Install with: pip install -e src/il[sim]", flush=True)
        return None, 0

    cfg = load_yaml(resolve_config_path(args_cli.schema, anchor=_IL_PKG))

    # The schema is the dataset's contract, the scene is what actually produces it -- fail
    # at startup rather than write a dataset whose names or shapes lie.
    if list(cfg["joint_names"]) != _RECORD_JOINTS:
        raise ValueError(
            f"schema joint_names {cfg['joint_names']} != recorded joints {_RECORD_JOINTS}"
        )
    cameras = {n: {"height": c["height"], "width": c["width"]} for n, c in enabled_images(cfg).items()}
    if set(cameras) != set(_RECORD_CAMS):
        raise ValueError(f"schema images {sorted(cameras)} != recorded cameras {sorted(_RECORD_CAMS)}")
    for name, spec in cameras.items():
        cam_cfg = scene[_RECORD_CAMS[name]].cfg
        if (spec["height"], spec["width"]) != (cam_cfg.height, cam_cfg.width):
            raise ValueError(
                f"schema '{name}' is {spec['height']}x{spec['width']}, "
                f"but sensor {_RECORD_CAMS[name]} renders {cam_cfg.height}x{cam_cfg.width}"
            )

    # Frames are paced in SIM time (every Nth physics step), not wall-clock time: the sim
    # rarely runs at exactly real time, and a wall-clock pacer would stretch or squash the
    # recorded motion relative to the fps written into the dataset.
    fps = int(cfg["fps"])
    record_every = round(1.0 / (sim_dt * fps))
    if abs(record_every * sim_dt * fps - 1.0) > 1e-6:
        raise ValueError(f"schema fps {fps} does not divide the sim rate {1.0 / sim_dt:g} Hz")

    if scene.num_envs > 1:
        print(f"[WARN] --num_envs {scene.num_envs}: only env_0 is recorded", flush=True)

    dataset_root = Path(
        args_cli.dataset_root or (cfg.get("record") or {}).get("root", "datasets/pioneer_vla")
    )
    recorder = SimLeRobotRecorder(
        task_name=args_cli.task_description,
        repo_id=str(cfg["repo_id"]),
        dataset_root=dataset_root,
        fps=fps,
        device=scene.device,
        joint_names=_RECORD_JOINTS,
        cameras=cameras,
        buffer_capacity_s=args_cli.max_episode_s,
        robot_type=str(cfg.get("robot_id", "pioneer_bimanual")),
        rate_limit=False,  # paced by record_every above
    )
    print(f"[RECORD] Dataset: {dataset_root.resolve()} at {fps} fps (every {record_every} sim steps)")
    print(f"[RECORD] Task prompt: '{args_cli.task_description}'")
    print("[RECORD] Keys: B = start recording, N = stop recording, M = delete last recording, "
          "ESC = quit")
    return recorder, record_every


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Keyboard -> differential IK -> joint targets, adapted from keyboard_teleop.py.

    Naming follows this file's import aliases: RIGHT_* is the ACTUATED arm (the
    L-suffixed chain), LEFT_* is the one held at its default pose.
    """
    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()
    recorder, record_every = _init_recorder(scene, sim_dt)

    # populate robot buffers before reading joint names / limits
    scene.update(sim_dt)
    apply_joint_limits(robot)

    arm_names = [resolve_joint_name(robot, n) for n in RIGHT_ARM_JOINTS]
    print(f"[INFO] actuated arm joints: {arm_names}")

    robot_entity_cfg = SceneEntityCfg("robot", joint_names=arm_names, body_names=[RIGHT_EE_BODY])
    robot_entity_cfg.resolve(scene)

    ee_jacobi_idx = (
        robot_entity_cfg.body_ids[0] - 1 if robot.is_fixed_base else robot_entity_cfg.body_ids[0]
    )
    wrist_body_id = robot_entity_cfg.body_ids[0]
    finger_body_ids = resolve_body_ids(robot, RIGHT_FINGER_TIP_BODIES)

    arm_ids = robot_entity_cfg.joint_ids
    gripper_ids = _joint_ids(robot, RIGHT_GRIPPER_JOINTS)
    held_ids = _joint_ids(robot, LEFT_ARM_JOINTS)
    held_default = robot.data.default_joint_pos[:, held_ids].clone()
    record_ids = list(arm_ids) + gripper_ids     # same order as _RECORD_JOINTS

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    n = scene.num_envs
    gripper_open = torch.tensor(
        [[GRIPPER_OPEN[j] for j in RIGHT_GRIPPER_JOINTS]], device=sim.device
    ).repeat(n, 1)
    gripper_closed = torch.tensor(
        [[GRIPPER_CLOSED[j] for j in RIGHT_GRIPPER_JOINTS]], device=sim.device
    ).repeat(n, 1)
    zero_gripper_vel = torch.zeros(n, len(gripper_ids), device=sim.device)

    # Absolute-pose IK against a persistent target, leashed to the tip -- see the long
    # rationale in keyboard_teleop.py: relative mode re-anchors every step and feels mushy.
    diff_ik_controller = DifferentialIKController(
        DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        num_envs=n,
        device=sim.device,
    )
    target = {"pos": None, "quat": None}
    _MAX_LEAD_M = 0.06                   # how far the target may lead the tip, per axis
    # Reach clamp anchor. SHOULDER is authored in world coords; the target lives in the
    # robot base frame, so it gets converted each step off the (fixed) root pose.
    shoulder_w = torch.tensor([SHOULDER], dtype=torch.float32, device=sim.device).repeat(n, 1)

    # Fast by default; hold Shift for fine control (Se3Keyboard has no modifier support,
    # so base sensitivity is high and Shift scales it down via a carb subscription).
    teleop = Se3Keyboard(
        Se3KeyboardCfg(pos_sensitivity=0.011, rot_sensitivity=0.09, gripper_term=True)
    )
    _FINE_SCALE = 0.2
    fine = {"active": False}
    _shift = {carb.input.KeyboardInput.LEFT_SHIFT, carb.input.KeyboardInput.RIGHT_SHIFT}

    # Wrist jog. joint5l and joint6l are driven DIRECTLY, outside the IK: C owns joint5l,
    # V owns joint6l, SHIFT reverses. Steering wrist orientation through the IK pose
    # target did not work -- with the leash gone the orientation target winds past what
    # the arm can reach and the DLS solver just stalls against it. So those two DOFs come
    # off the solver entirely, and the IK's orientation target is pinned to the measured
    # tip each step (see the loop) so it has nothing left to fight over.
    _WRIST_JOG_KEYS = {
        carb.input.KeyboardInput.C: 0,   # -> joint5l
        carb.input.KeyboardInput.V: 1,   # -> joint6l
    }
    # Sized to what the joints can actually deliver, not to what feels fast: joint6l tops
    # out near 0.24 rad/s with _WRIST_DAMPING, joint5l near 0.20 (22 N m / 110 damping).
    # Commanding past that just builds lead the joint never works off.
    _WRIST_JOG_STEP = 0.002              # rad per sim step -> 0.2 rad/s at dt=0.01
    _MAX_WRIST_LEAD = 0.05               # rad the jog target may lead the measured joint
    wrist_held = [False, False]
    # Local indices into arm_ids / joint_pos_des, which follow RIGHT_ARM_JOINTS' order.
    _wrist_local = [RIGHT_ARM_JOINTS.index(j) for j in ("joint5l", "joint6l")]
    _wrist_ids = [arm_ids[i] for i in _wrist_local]
    # apply_joint_limits() ran above, so these are the URDF limits, not the USD defaults.
    _wrist_lo = robot.data.joint_pos_limits[:, _wrist_ids, 0]
    _wrist_hi = robot.data.joint_pos_limits[:, _wrist_ids, 1]
    wrist_target = {"q": None}           # seeded from the measured joints, like target

    # Recording keys, set here and consumed at the top of the loop. None of B/N/M/ESC is bound
    # by Se3Keyboard, and being carb events they only fire with the Isaac window focused.
    #
    # A stopped episode is held back ("pending") in the recorder's GPU buffers instead of
    # being written straight away, so M can still delete it. It is written to disk when the
    # next recording starts or on quit. Episodes already on disk are not deletable from here:
    # LeRobot's delete_episodes rewrites the whole dataset into a new folder, which is not
    # something to run under the recorder's live writer.
    rec = {
        "on": False, "pending": False, "opened": False, "saved": 0, "step": 0,
        "start": False, "stop": False, "delete": False, "quit": False,
    }
    _REC_KEYS = {
        carb.input.KeyboardInput.B: "start",
        carb.input.KeyboardInput.N: "stop",
        carb.input.KeyboardInput.M: "delete",
        carb.input.KeyboardInput.ESCAPE: "quit",
    }

    def _on_kb(event, *_):
        if (
            recorder is not None
            and event.type == carb.input.KeyboardEventType.KEY_PRESS
            and event.input in _REC_KEYS
        ):
            rec[_REC_KEYS[event.input]] = True
        if event.input in _shift:
            fine["active"] = event.type == carb.input.KeyboardEventType.KEY_PRESS
        idx = _WRIST_JOG_KEYS.get(event.input)
        if idx is not None:
            if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                wrist_held[idx] = True
            elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
                wrist_held[idx] = False
        return True

    _kb_sub = carb.input.acquire_input_interface().subscribe_to_keyboard_events(
        omni.appwindow.get_default_app_window().get_keyboard(), _on_kb
    )
    # Se3Keyboard binds these to its own roll/pitch/yaw; unbind so the two schemes
    # cannot both drive the target. Translation (W/A/S/D/Q/E) and K are untouched.
    for _k in ("Z", "X", "T", "G", "C", "V"):
        teleop._INPUT_KEY_MAPPING.pop(_k, None)
    # Q = down, E = up (Se3Keyboard's default is the reverse). Press and release both
    # read the same table, so swapping the entries keeps held keys balanced.
    _km = teleop._INPUT_KEY_MAPPING
    _km["Q"], _km["E"] = _km["E"], _km["Q"]
    should_reset = {"v": False}
    teleop.add_callback("R", lambda: should_reset.update(v=True))
    teleop.reset()
    # Se3Keyboard.__str__ hardcodes its own six-key rotation table (Z/X, T/G, C/V), which
    # no longer matches the bindings above -- print the real ones instead.
    print(
        "[INFO] Click the 3D viewport, then: W/A/S/D move, Q down, E up, C joint5l, "
        "V joint6l, K gripper, R reset. SHIFT = fine on W/A/S/D/Q/E, reverse on C/V."
    )

    while simulation_app.is_running():
        if recorder is not None:
            if rec["quit"]:
                break
            if rec["delete"]:
                rec["delete"] = False
                if rec["on"]:
                    recorder.cancel_recording()
                    rec["on"] = False
                    should_reset["v"] = True
                    print("[RECORD] Deleted the recording in progress.")
                elif rec["pending"]:
                    recorder.cancel_recording()
                    rec["pending"] = False
                    print("[RECORD] Deleted the last recording.")
                else:
                    print("[RECORD] Nothing to delete -- the last recording is already on disk.")
            if rec["stop"]:
                rec["stop"] = False
                if rec["on"]:
                    rec["on"], rec["pending"] = False, True
                    should_reset["v"] = True      # fresh prop layout for the next episode
                    print("[RECORD] Stopped. M = delete it, B = keep it and record the next one.")
            if rec["start"]:
                rec["start"] = False
                if not rec["on"]:
                    if not rec["opened"]:
                        recorder.init_dataset()
                        rec["opened"] = True
                    if rec["pending"]:
                        recorder.save_episode()   # async write; blocks only if 2 are in flight
                        rec["pending"] = False
                        rec["saved"] += 1
                    rec["on"], rec["step"] = True, 0
                    print(f"[RECORD] Recording episode {rec['saved'] + 1}... (N = stop)")

        if should_reset["v"]:
            if rec["on"]:
                # A reset teleports the arm and props; frames either side of it are not one demo.
                recorder.cancel_recording()
                rec["on"] = False
                print("[RECORD] Reset mid-episode -- episode discarded.")
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            randomize_props(scene, PROP_RNG)      # new cube/tray layout each reset
            diff_ik_controller.reset()
            teleop.reset()
            target["pos"] = None                  # re-seed from the measured tip
            wrist_target["q"] = None              # re-seed the wrist jog as well
            should_reset["v"] = False

        cmd = teleop.advance()
        command = cmd[:6].to(dtype=torch.float32, device=sim.device).unsqueeze(0).repeat(n, 1)
        if fine["active"]:
            command = command * _FINE_SCALE
        close_gripper = bool(cmd[6].item() < 0.0) if cmd.numel() > 6 else False

        ee_pose_w = robot.data.body_state_w[:, wrist_body_id, 0:7]
        root_pose_w = robot.data.root_state_w[:, 0:7]
        arm_joint_pos = robot.data.joint_pos[:, arm_ids]

        ee_pos_b, _ = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        tip_pos_b, tip_quat_b = compute_gripper_tip_pose_b(
            robot, root_pose_w, wrist_body_id, finger_body_ids
        )

        if target["pos"] is None:
            target["pos"] = tip_pos_b.clone()
            target["quat"] = tip_quat_b.clone()

        target["pos"] = target["pos"] + command[:, 0:3]
        # Orientation is not commanded through the IK any more -- the wrist jog below owns
        # it. Pinning the target to the measured tip leaves zero orientation error, so the
        # solver spends its DOFs on position and does not undo the jog.
        target["quat"] = tip_quat_b.clone()

        # Guard 1 -- reach sphere. Bounds the target in ABSOLUTE terms, around the
        # shoulder. Purely geometric: it knows nothing about joint limits, so a point on
        # this sphere is not necessarily a pose the arm can actually strike.
        shoulder_b, _ = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], shoulder_w
        )
        off = target["pos"] - shoulder_b
        dist = torch.linalg.vector_norm(off, dim=-1, keepdim=True)
        target["pos"] = shoulder_b + off * (REACH_MAX_3D / dist.clamp_min(1e-9)).clamp(max=1.0)

        # Guard 2 -- leash. Bounds how far the target may lead the MEASURED tip, which is
        # what guard 1 cannot do. Inside _MAX_LEAD_M this is the identity (tip + err ==
        # target), so a commanded pose is still remembered exactly; it only bites when the
        # arm falls behind, i.e. at a joint limit or a singularity -- precisely when the
        # target has to stop running away.
        #
        # Why this is required: DifferentialIKController.compute() returns
        # joint_pos + delta with delta proportional to the pose error and NO clamp and no
        # settling term. A target the arm cannot reach leaves a permanently non-zero
        # error, so the solver re-issues a joint delta every single step and the arm
        # creeps along its limit forever instead of stopping. Bounding the error bounds
        # that command: once the arm stalls, tip stops moving, so target stops moving too.
        target["pos"] = tip_pos_b + (target["pos"] - tip_pos_b).clamp(-_MAX_LEAD_M, _MAX_LEAD_M)

        diff_ik_controller.set_command(
            torch.cat([target["pos"], target["quat"]], dim=-1), ee_quat=tip_quat_b
        )
        jacobian = compute_tip_ik_jacobian(
            robot,
            robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_ids],
            ee_pos_b,
            tip_pos_b,
        )
        joint_pos_des = diff_ik_controller.compute(tip_pos_b, tip_quat_b, jacobian, arm_joint_pos)

        # Wrist jog overrides the solver on joint5l / joint6l. Held-key integration with a
        # limit clamp, so the wrist holds its angle when nothing is pressed and stops dead
        # at the URDF stops instead of winding past them.
        if wrist_target["q"] is None:
            wrist_target["q"] = robot.data.joint_pos[:, _wrist_ids].clone()
        _jog_step = -_WRIST_JOG_STEP if fine["active"] else _WRIST_JOG_STEP
        jog = torch.tensor(
            [[_jog_step if held else 0.0 for held in wrist_held]],
            dtype=torch.float32,
            device=sim.device,
        ).repeat(n, 1)
        wrist_target["q"] = torch.clamp(wrist_target["q"] + jog, _wrist_lo, _wrist_hi)
        # Leash, same idea as the IK target's: these joints are torque-limited, so a free
        # integrator outruns them and a brief tap would command the joint all the way to
        # its stop and leave it crawling there long after the key is up. Bounding the lead
        # means the command stops advancing as soon as the joint stops moving.
        _wrist_now = robot.data.joint_pos[:, _wrist_ids]
        wrist_target["q"] = _wrist_now + (wrist_target["q"] - _wrist_now).clamp(
            -_MAX_WRIST_LEAD, _MAX_WRIST_LEAD
        )
        joint_pos_des[:, _wrist_local] = wrist_target["q"]

        robot.set_joint_position_target(joint_pos_des, joint_ids=arm_ids)

        # Coupled fingers: high stiffness + zero velocity target stops bounce on the move.
        gripper_target = gripper_closed if close_gripper else gripper_open
        robot.set_joint_position_target(gripper_target, joint_ids=gripper_ids)
        robot.set_joint_velocity_target(zero_gripper_vel, joint_ids=gripper_ids)

        # Hold the other arm at its default pose.
        robot.set_joint_position_target(held_default, joint_ids=held_ids)

        # One frame = what the robot saw and measured at this step (state + images from the
        # last scene.update) paired with what it was commanded to do from it (action).
        if rec["on"]:
            if rec["step"] % record_every == 0:
                recorder.push_frame_to_buffer(
                    torch.cat([joint_pos_des[0], gripper_target[0]]),
                    robot.data.joint_pos[0, record_ids],
                    {name: scene[sensor].data.output["rgb"][0, ..., :3] for name, sensor in _RECORD_CAMS.items()},
                )
            rec["step"] += 1

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

    if recorder is not None and rec["opened"]:
        if rec["on"]:
            print("[RECORD] Recording in progress was not stopped with N -- discarded.")
        elif rec["pending"]:
            recorder.save_episode()
        print("[RECORD] Waiting for queued episodes to finish writing...")
        recorder.finalize()
        print(f"[RECORD] {recorder.num_recorded_episodes} episode(s) saved this session under "
              f"{recorder.dataset_root.resolve()}")


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # the viewpoint saved inside 09-19_vla_env_v1.usd
    sim.set_camera_view([1.86, 1.056, 1.185], [-0.232, -0.354, 0.870])

    scene_cfg = MySceneCfg(num_envs=args_cli.num_envs, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)
    print(f"[INFO] scene USD = {_VLA_ENV_USD}")
    print(f"[INFO] robot prim = {scene_cfg.robot.prim_path} (spawn=None, already in the USD)")

    # spawn=None skips BIMANUAL_ARM_CFG's custom spawner, whose only job is writing the
    # Physics Inspector joint limits onto the USD joints. Do it here -- it must land
    # BEFORE sim.reset(), which initialises the articulation and reads them.
    patched = 0
    for path in sim_utils.find_matching_prim_paths(scene_cfg.robot.prim_path):
        patch_joint_pos_limits_on_prim(path)
        patched += 1
    print(f"[INFO] patched joint limits on {patched} robot prim(s)")

    sim.reset()

    # First layout before the loop starts, so episode 0 isn't always the authored pose.
    randomize_props(scene, PROP_RNG)
    scene.write_data_to_sim()

    # After sim.reset(): the camera prims come from the robot's referenced USD layer, and
    # binding a viewport to a path that has not composed yet leaves the window on the
    # default camera with no error.
    open_camera_viewports(scene_cfg.robot.prim_path, _cam_viewport_names, _CAM_VIEWPORT_SIZE)

    print("[INFO] Setup complete. Teleoperating the L-suffixed (left) arm.")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
