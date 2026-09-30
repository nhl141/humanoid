"""The one data transform that is genuinely yours: 8 recorded joints <-> 7.

joint7l and joint8l are a mimic pair -- one physical gripper recorded as two
mirrored numbers (closed is -0.05 / +0.05, open is 0.0 / 0.0). Two output dims
for one degree of freedom wastes capacity and lets a policy emit physically
impossible asymmetric pairs.

Lives in its own file because THREE places need it and they must agree exactly:
  1. recording  (src/il) -- ideally collapse here, so datasets are born 7-dim
  2. training   -- only if your existing datasets are still 8-dim
  3. deployment -- expand 7 -> 8 before joint_command

Best move is option 1. Collapse at record time and there is no preprocessing
step at all -- LeRobot handles everything else (see README.md).
"""

from __future__ import annotations

import numpy as np

# Joint order from dataset_schema_pick_place_bimanual.yaml.
JOINT_ORDER = [
    "joint1L", "joint2l", "joint3l", "joint4l", "joint5l", "joint6l",
    "joint7l", "joint8l",
]
N_ARM_JOINTS = 6
GRIPPER_STROKE_M = 0.05  # LEFT_GRIPPER_CLOSED in pioneer_humanoid.bimanual_arm
# How far the two fingers may disagree, in closure units (1.0 = full stroke).
# 0.02 is 2% of 50 mm, i.e. 1 mm of mechanical slop.
GRIPPER_MIMIC_ATOL = 0.02


def collapse(vec: np.ndarray, *, strict: bool = True) -> np.ndarray:
    """(..., 8) -> (..., 7). The gripper pair becomes one scalar, 0.0 open -> 1.0 closed.

    `vec` is joint positions in JOINT_ORDER: six arm joints in radians, then the
    joint7l / joint8l gripper pair in metres. Works on a single frame (8,), a
    whole episode (T, 8), or a batch of chunks (B, H, 8).

    Call it on `observation.state` and on `action` separately -- different data,
    identical layout.

    With strict=True a finger disagreement raises, because it means the mimic
    coupling isn't holding and the recorded gripper value is not trustworthy.
    Pass strict=False to take the average anyway.
    """
    vec = np.asarray(vec, dtype=np.float32)
    if vec.shape[-1] != len(JOINT_ORDER):
        raise ValueError(
            f"expected last axis {len(JOINT_ORDER)} (JOINT_ORDER), got shape {vec.shape}"
        )

    arm = vec[..., :N_ARM_JOINTS]

    # The fingers travel opposite directions over the same stroke, so each one
    # gives the same closure fraction: joint7l goes 0 -> -0.05, joint8l 0 -> +0.05.
    from_7 = -vec[..., 6] / GRIPPER_STROKE_M
    from_8 = vec[..., 7] / GRIPPER_STROKE_M

    disagreement = np.abs(from_7 - from_8)
    if strict and np.any(disagreement > GRIPPER_MIMIC_ATOL):
        i = int(np.argmax(disagreement))
        raise ValueError(
            f"gripper fingers disagree by {disagreement.flat[i]:.4f} closure units "
            f"at flat index {i}: joint7l={vec[..., 6].flat[i]:.5f} implies "
            f"{from_7.flat[i]:.4f}, joint8l={vec[..., 7].flat[i]:.5f} implies "
            f"{from_8.flat[i]:.4f}. The mimic coupling is not holding -- check the "
            f"recording before training on it (strict=False to average anyway)."
        )

    # Average the two readings rather than trusting one: same answer when they
    # agree, and it halves the sensor noise on the gripper channel.
    closure = 0.5 * (from_7 + from_8)
    return np.concatenate([arm, closure[..., None]], axis=-1)

#create delta action space for preprocessing to convert aboslute joint angles into relative actions
def create_delta_action_space(vec: np.ndarray) -> np.ndarray:
    """Convert absolute joint actions to per-step deltas along the time axis.

    The returned array has the same shape as ``vec``. Its first action in each
    sequence is zero because no preceding action is available.
    """
    vec = np.asarray(vec, dtype=np.float32)
    if vec.shape[-1] != len(JOINT_ORDER):
        raise ValueError(
            f"expected last axis {len(JOINT_ORDER)} (JOINT_ORDER), got shape {vec.shape}"
        )
    if vec.ndim < 2:
        raise ValueError(
            f"expected at least a time and joint axis, got shape {vec.shape}"
        )

    delta = np.empty_like(vec)
    delta[..., 0, :] = 0.0
    np.subtract(vec[..., 1:, :], vec[..., :-1, :], out=delta[..., 1:, :])
    return delta


def expand(vec: np.ndarray) -> np.ndarray:
    """(..., 7) -> (..., 8). Inverse of collapse().

    Runs at deployment on every control step: the policy emits 7, joint_command
    wants 8. Closure is clipped to [0, 1] first -- a policy can and will emit
    1.03, and that would drive the fingers past their mechanical stroke.
    """
    vec = np.asarray(vec, dtype=np.float32)
    if vec.shape[-1] != N_ARM_JOINTS + 1:
        raise ValueError(
            f"expected last axis {N_ARM_JOINTS + 1}, got shape {vec.shape}"
        )

    arm = vec[..., :N_ARM_JOINTS]
    closure = np.clip(vec[..., N_ARM_JOINTS], 0.0, 1.0)
    joint7l = -closure * GRIPPER_STROKE_M
    joint8l = closure * GRIPPER_STROKE_M
    return np.concatenate([arm, joint7l[..., None], joint8l[..., None]], axis=-1)
