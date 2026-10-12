#!/usr/bin/env bash
set -euo pipefail
: "${TAI_ENV:?Set TAI_ENV}"
: "${TAI_UPSTREAM:?Set TAI_UPSTREAM}"
: "${TAI_NATIVE:?Set TAI_NATIVE to an external, dedicated native-library directory}"
source "$TAI_ENV/bin/activate"
export MAX_JOBS="${MAX_JOBS:-2}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export CUDA_HOME="$(python -c 'from torch.utils.cpp_extension import CUDA_HOME; print(CUDA_HOME)')"

# Pin a PyTorch-2.0-compatible renderer instead of upstream's unpinned Git URL.
python -m pip install --no-deps --no-build-isolation \
  git+https://github.com/NVlabs/nvdiffrast.git@c5caf7bdb8a2448acc491a9faa47753972edd380

mkdir -p "$TAI_NATIVE"
cd "$TAI_NATIVE"
apt-get download libeigen3-dev libboost1.74-dev libboost-system1.74-dev \
  libboost-program-options1.74-dev libboost-system1.74.0 libboost-program-options1.74.0
for package in ./*.deb; do
  dpkg-deb -x "$package" .
done

pybind="$(python -m pybind11 --cmakedir)"
cmake -S "$TAI_UPSTREAM/third_party/FoundationPose/mycpp" \
  -B "$TAI_UPSTREAM/third_party/FoundationPose/mycpp/build" \
  -DCMAKE_PREFIX_PATH="$TAI_NATIVE/usr;$pybind" \
  -DBoost_INCLUDE_DIR="$TAI_NATIVE/usr/include" \
  -DBoost_LIBRARY_DIR_RELEASE="$TAI_NATIVE/usr/lib/x86_64-linux-gnu"
cmake --build "$TAI_UPSTREAM/third_party/FoundationPose/mycpp/build" --parallel "$MAX_JOBS"
