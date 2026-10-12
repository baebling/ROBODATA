#!/usr/bin/env bash
set -euo pipefail
: "${TAI_ENV:?Set TAI_ENV}"
: "${TAI_UPSTREAM:?Set TAI_UPSTREAM}"
: "${TAI_DATASET:?Set TAI_DATASET}"
: "${TAI_OUTPUT:?Set TAI_OUTPUT to a new directory outside raw data}"
: "${TAI_NATIVE:?Set TAI_NATIVE to the extracted native libraries}"
source "$TAI_ENV/bin/activate"
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
export MAX_JOBS="${MAX_JOBS:-2}"
export LD_LIBRARY_PATH="$TAI_NATIVE/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# The Linux NVIDIA driver installed by a toolkit is not the Windows/WSL driver.
if [[ -d /usr/lib/wsl/lib ]]; then
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
fi

python -m reproductions.tool_as_interface.preprocess_smoke \
  --upstream "$TAI_UPSTREAM" --dataset "$TAI_DATASET" \
  --config "${TAI_PREPROCESS_CONFIG:-reproductions/tool_as_interface/preprocessing_smoke.json}" \
  --output "$TAI_OUTPUT"
