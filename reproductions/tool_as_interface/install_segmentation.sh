#!/usr/bin/env bash
set -euo pipefail
: "${TAI_ENV:?Set TAI_ENV}"
: "${TAI_UPSTREAM:?Set TAI_UPSTREAM}"
source "$TAI_ENV/bin/activate"
export MAX_JOBS="${MAX_JOBS:-2}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export CUDA_HOME="$(python -c 'from torch.utils.cpp_extension import CUDA_HOME; print(CUDA_HOME)')"

python -m pip install --no-deps --no-build-isolation \
  "$TAI_UPSTREAM/third_party/Grounded-Segment-Anything/segment_anything"
python -m pip install --no-deps --no-build-isolation \
  "$TAI_UPSTREAM/third_party/Grounded-Segment-Anything/GroundingDINO"
