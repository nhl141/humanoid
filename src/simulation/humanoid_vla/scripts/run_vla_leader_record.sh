#!/usr/bin/env bash
# Launch vla_imitation_record.py -- the VLA pick-and-place teleop, LEADER-ARM variant (09-27_vla_env_v2 scene, left arm
# driven by differential IK, cube + bin randomized on reset).
#
# On the host:
#   ./watod -t simulation_isaac_dev
# Then inside the container:
#   cd /workspace/humanoid/src/simulation/humanoid_vla/scripts
#   ./run_vla_leader_record.sh
#
# Outside the container, point ISAAC_LAB at a local checkout:
#   ISAAC_LAB=$HOME/IsaacLab ./run_vla_leader_record.sh
#
# Recording is always available from the Isaac window: B = start, N = stop,
# M = delete the last recording, ESC = quit. Extra args pass straight through, e.g.:
#   ./run_vla_leader_record.sh --dataset_root datasets/my_run --task_description "..."
#
# A live viewport window on the base RealSense RGB camera opens by default. To change
# which cameras get a window (each one costs an extra render pass per frame):
#   ./run_vla_leader_record.sh --cam-viewport all
#   ./run_vla_leader_record.sh --cam-viewport base,wrist_left
#   ./run_vla_leader_record.sh --cam-viewport none
#   ./run_vla_leader_record.sh --cam-viewport-size 640x480
#
# The 7-servo leader drives joint1L..joint6l (A-F) and the gripper (G); zero it
# with the gripper open. Flip a joint's direction with --leader-signs (7 values):
#   ./run_vla_leader_record.sh --leader-signs 1,-1,1,1,1,1,1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLA_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${VLA_DIR}/../../.." && pwd)"
ISAAC_LAB="${ISAAC_LAB:-/workspace/isaaclab}"

if [ ! -x "${ISAAC_LAB}/isaaclab.sh" ]; then
  echo "error: no isaaclab.sh under ISAAC_LAB=${ISAAC_LAB}" >&2
  echo "       set ISAAC_LAB to your IsaacLab checkout, e.g. ISAAC_LAB=\$HOME/IsaacLab $0" >&2
  exit 1
fi

# humanoid_scenes provides ROBOT_PRIM_IN_SCENE (and @scene discovery). It is
# editable-installed in the image; this keeps a bare bind-mounted checkout working.
# pioneer_humanoid and src/il are added to sys.path by the script itself.
export PYTHONPATH="${REPO_ROOT}/src/simulation/humanoid_scenes:${VLA_DIR}:${PYTHONPATH:-}"

# --enable_cameras is REQUIRED here: MySceneCfg declares three CameraCfg sensors
# (both wrist cams + the base RealSense). Without the flag Isaac starts with the
# render pipeline disabled and camera init fails.
#
# --device cuda is deliberate, not a default. The box grasp holds on GPU PhysX and
# slips on CPU PhysX -- same finding as quest teleop, identical scene and friction,
# only this flag differing. It costs latency (CPU-side reads of GPU-resident physics
# state sync every step), which is the right trade for a grasp that works.
exec "${ISAAC_LAB}/isaaclab.sh" -p \
  "${VLA_DIR}/vla_imitation_record.py" \
  --device cuda \
  --enable_cameras \
  "$@"
