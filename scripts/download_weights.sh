#!/usr/bin/env bash
set -euo pipefail

mkdir -p weights

temporary_output=""

cleanup() {
  if [[ -n "${temporary_output}" ]]; then
    rm -f -- "${temporary_output}"
  fi
}

trap cleanup EXIT

download_checkpoint() {
  local url="$1"
  local destination="$2"

  temporary_output="$(mktemp "${destination}.tmp.XXXXXX")"
  curl --fail --location --retry 3 \
    "${url}" \
    --output "${temporary_output}"
  if [[ -d "${destination}" ]]; then
    echo "refusing directory checkpoint destination: ${destination}" >&2
    return 1
  fi
  mv -f -- "${temporary_output}" "${destination}"
  temporary_output=""
}

echo "Downloading Pi3 weights..."
download_checkpoint \
  "https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors" \
  "weights/model.safetensors"

echo "Downloading SALAD weights..."
download_checkpoint \
  "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt" \
  "weights/dino_salad.ckpt"

echo "Downloading DINO weights..."
download_checkpoint \
  "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth" \
  "weights/dinov2_vitb14_pretrain.pth"
