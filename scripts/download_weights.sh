#!/usr/bin/env bash
set -euo pipefail

mkdir -p weights

echo "Downloading Pi3 weights..."
curl --fail --location --retry 3 \
  "https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors" \
  --output "weights/model.safetensors"

echo "Downloading SALAD weights..."
curl --fail --location --retry 3 \
  "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt" \
  --output "weights/dino_salad.ckpt"

echo "Downloading DINO weights..."
curl --fail --location --retry 3 \
  "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth" \
  --output "weights/dinov2_vitb14_pretrain.pth"
