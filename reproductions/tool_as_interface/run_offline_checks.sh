#!/usr/bin/env bash
set -euo pipefail

: "${TAI_ENV:?Set TAI_ENV to the separate Python environment}"
: "${TAI_UPSTREAM:?Set TAI_UPSTREAM to the pinned upstream checkout}"
: "${TAI_DATASET:?Set TAI_DATASET to the preserved hammer_human directory}"
: "${TAI_OUTPUT:?Set TAI_OUTPUT to a new result directory outside raw data}"

source "$TAI_ENV/bin/activate"
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
export MAX_JOBS="${MAX_JOBS:-2}"

python -m reproductions.tool_as_interface.runner audit \
  --upstream "$TAI_UPSTREAM" --dataset "$TAI_DATASET" --output "$TAI_OUTPUT/audit"
pose_args=()
if [[ -n "${TAI_POSE_ESTIMATES:-}" ]]; then
  pose_args=(--pose-estimates "$TAI_POSE_ESTIMATES")
fi
python -m reproductions.tool_as_interface.runner policy-smoke \
  --upstream "$TAI_UPSTREAM" --dataset "$TAI_DATASET" \
  --config reproductions/tool_as_interface/policy_smoke.json --output "$TAI_OUTPUT/policy" \
  "${pose_args[@]}"
python -m reproductions.tool_as_interface.runner scene-smoke \
  --upstream "$TAI_UPSTREAM" --dataset "$TAI_DATASET" \
  --config reproductions/tool_as_interface/scene_smoke.json --output "$TAI_OUTPUT/scene"
