#!/usr/bin/env bash
set -euo pipefail
: "${TAI_ENV:?Set TAI_ENV}"
: "${TAI_UPSTREAM:?Set TAI_UPSTREAM}"
source "$TAI_ENV/bin/activate"

download() {
  local url="$1" destination="$2"
  if [[ -e "$destination" ]]; then
    printf 'Preserving existing model: %s\n' "$destination"
    return
  fi
  mkdir -p "$(dirname "$destination")"
  wget -q -c -O "$destination.partial" "$url"
  mv -n -- "$destination.partial" "$destination"
}

download \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth \
  "$TAI_UPSTREAM/ti/utils/Grounded_SAM/groundingdino_swinb_cogcoor.pth"
download \
  https://huggingface.co/lkeab/hq-sam/resolve/67ab82412bc794d5ce2e9799b8b6a3c0a8cfe1d2/sam_hq_vit_h.pth \
  "$TAI_UPSTREAM/ti/utils/Grounded_SAM/sam_hq_vit_h.pth"

gdown --folder https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i \
  -O "$TAI_UPSTREAM/third_party/FoundationPose/weights"
