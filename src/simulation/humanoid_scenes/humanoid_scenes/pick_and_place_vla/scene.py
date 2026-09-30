"""Pick-and-place VLA scene: the humanoid_vla env USD, robot included.

The env USD (``09-19_vla_env_v1.usd``) already carries the arm at
``/World/new_pioneer_all_cams`` (z=1.202, the floor-stand height), plus its own
GroundPlane, DomeLight and KeyLight -- so this cfg adds no ground and no light.
The arm is bound with ``spawn=None`` in the teleop script; see README notes there.

Table top sits at ~0.75; the block and box start on it.
"""
from __future__ import annotations

from dataclasses import MISSING
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from humanoid_scenes import scene

VLA_ENV_USD = str(
    Path(__file__).resolve().parents[5]
    / "src" / "simulation" / "humanoid_vla" / "assets" / "09-19_vla_env_v1.usd"
)

# the arm prim inside the env USD -- what the teleop script binds to with spawn=None
ROBOT_PRIM_IN_SCENE = "{ENV_REGEX_NS}/Scene/new_pioneer_all_cams"


@scene(
    "pick_and_place_vla",
    robot_pos=(0.0, 0.0, 1.202),  # must match the arm's z inside the env USD
    camera=([1.86, 1.056, 1.185], [-0.232, -0.354, 0.870]),  # the USD's saved view
)

@configclass
class PickAndPlaceVlaSceneCfg(InteractiveSceneCfg):
    # declared first: the robot binds to a prim this spawns
    env = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Scene",
        spawn=sim_utils.UsdFileCfg(usd_path=VLA_ENV_USD),
    )
    robot: ArticulationCfg = MISSING
