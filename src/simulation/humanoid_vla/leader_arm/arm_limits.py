"""Joint travel limits for the Pioneer arm under leader-arm teleop.

Consumed by ``leader_teleop.py`` and ``vla_imitation_record.py``:

* every commanded target is clamped through :func:`limited_target`, and
* the same bounds are written into the USD revolute joints before PhysX
  builds the articulation, so the simulated arm cannot be driven past them.

SCOPE -- these limits govern the SIMULATED arm only. The leader arm itself is
a passive input device with torque permanently disabled, so nothing here can
move a physical actuator. Before any of this is reused to command real
hardware, the numbers below must be re-derived from the arm's actual
mechanical stops, including cable routing and sensor-wiring interference --
encoder travel is not the same thing as safe travel.
"""
from __future__ import annotations

import math

# Total travel per joint, in degrees, as measured on the physical leader arm.
# Each range is split symmetrically about the arm's straight resting pose,
# which is what leader_teleop.py zeroes to on startup.
#
JOINT_TRAVEL_DEG: dict[str, float] = {
    "joint1": 360.0,
    "joint2": 180.0,
    "joint3": 360.0,
    "joint4": 180.0,
    "joint5": 360.0,
    "joint6": 180.0,  # wrist (servo F)
}

# Gripper servo (G) range, in degrees. The claw starts OPEN at GRIPPER_OPEN_DEG
# (its maximum) and closes toward GRIPPER_CLOSED_DEG (its minimum). The leader
# zeroes wherever it starts, so the startup pose is taken to be 41.5 deg -- zero
# the leader with the claw fully open. See gripper_fraction().
GRIPPER_OPEN_DEG = 41.5
GRIPPER_CLOSED_DEG = 0.0

# The leader maps six servos (A-F) onto all six arm joints; G, the gripper, is
# separate (see gripper_fraction). joint_limits_deg() returns exactly this many
# pairs so it stays aligned with ARM_SERVOS in servo_leader.py.
LEADER_JOINT_COUNT = 6


def _symmetric(travel_deg: float) -> tuple[float, float]:
    """Split a total travel range evenly about zero."""
    half = travel_deg / 2.0
    return (-half, half)


# Ordered low-to-high by joint number; index i corresponds to servo axis i.
_ORDERED_JOINTS = tuple(sorted(JOINT_TRAVEL_DEG))
_BOUNDS = tuple(_symmetric(JOINT_TRAVEL_DEG[name]) for name in _ORDERED_JOINTS)


def joint_limits_deg(
    arm: str = "left", joints: int = LEADER_JOINT_COUNT
) -> tuple[tuple[float, float], ...]:
    """``(lo, hi)`` degree bounds for the first ``joints`` leader-driven joints.

    ``joints`` defaults to all six (A-F); pass 5 for the no-wrist leader.

    The left and right chains share one table. That is not an approximation:
    every range is symmetric about zero, and mirroring a symmetric range maps
    it onto itself, so the right and left bounds are genuinely identical. If a
    joint ever gets an asymmetric range, this function has to branch on
    ``arm`` and negate/swap that pair for the right chain.
    """
    if arm not in ("left", "right"):
        raise ValueError(f"arm must be 'left' or 'right', got {arm!r}")
    if not 1 <= joints <= len(_BOUNDS):
        raise ValueError(f"joints must be in 1..{len(_BOUNDS)}, got {joints}")
    return _BOUNDS[:joints]


def limited_target(
    angle_rad: float,
    sign: float,
    scale: float,
    bounds_deg: tuple[float, float],
    offset_rad: float = 0.0,
) -> float:
    """Map one leader angle to a clamped sim joint target, in radians.

    ``angle_rad`` is the leader's startup-relative angle, ``sign`` flips the
    axis direction (see ``--signs``) and ``scale`` is the leader-to-sim gain.
    ``offset_rad`` is the sim joint angle the leader's zero maps to (its home
    pose); the leader's motion is added on top of it before the clamp.
    The result is hard-clamped into ``bounds_deg``; callers rely on that clamp
    rather than checking the range themselves, so it must never be bypassed.
    """
    lo_deg, hi_deg = bounds_deg
    if not (math.isfinite(lo_deg) and math.isfinite(hi_deg)) or lo_deg >= hi_deg:
        raise ValueError(f"invalid bounds {bounds_deg!r}: need finite lo < hi")
    value = offset_rad + angle_rad * sign * scale
    if not math.isfinite(value):
        # A NaN would propagate straight into a position target; hold at zero,
        # which is inside every range in the table.
        return 0.0
    return max(math.radians(lo_deg), min(math.radians(hi_deg), value))


def gripper_fraction(angle_rad: float, sign: float) -> float:
    """Map the gripper servo's reading to a closure fraction: 0 = open, 1 = closed.

    ``angle_rad`` is startup-relative, and startup is the open end, so the claw's
    angle is ``GRIPPER_OPEN_DEG + sign * angle``, clamped into
    [GRIPPER_CLOSED_DEG, GRIPPER_OPEN_DEG]. With ``sign`` = +1 the servo reading
    has to DECREASE to close; if closing the leader claw does nothing in sim,
    flip the gripper's sign. A non-finite reading returns 0 (open) rather than
    propagating into a finger target.
    """
    span = GRIPPER_OPEN_DEG - GRIPPER_CLOSED_DEG
    claw_deg = GRIPPER_OPEN_DEG + sign * math.degrees(angle_rad)
    if not math.isfinite(claw_deg):
        return 0.0
    claw_deg = max(GRIPPER_CLOSED_DEG, min(GRIPPER_OPEN_DEG, claw_deg))
    return (GRIPPER_OPEN_DEG - claw_deg) / span
