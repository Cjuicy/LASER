#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPOSITORY_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
cd "${REPOSITORY_ROOT}"

UTILITY="${SCRIPT_DIR}/preflight_experiments.py"
SUMMARY="${SCRIPT_DIR}/summarize_results.py"
LASER_ENV="${LASER_ENV:-vggt}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${REPOSITORY_ROOT}/outputs/disposable_experiments}"
SEVEN_SCENES_ROOT="${SEVEN_SCENES_ROOT:-${REPOSITORY_ROOT}/data/7-Scenes}"
NRGBD_ROOT="${NRGBD_ROOT:-${REPOSITORY_ROOT}/data/NeuralRGBD}"
KITTI_ROOT="${KITTI_ROOT:-${REPOSITORY_ROOT}/data/KITTI}"
SEVEN_SCENES_MAP="${SEVEN_SCENES_MAP:-${REPOSITORY_ROOT}/datasets/seq-id-maps/7scenes_mv-recon_seq-id-map-kf10.json}"
NRGBD_MAP="${NRGBD_MAP:-${REPOSITORY_ROOT}/datasets/seq-id-maps/NRGBD_mv-recon_seq-id-map-kf10.json}"
PI3_CHECKPOINT="${PI3_CHECKPOINT:-${REPOSITORY_ROOT}/weights/PI3/model.safetensors}"
SALAD_CHECKPOINT="${SALAD_CHECKPOINT:-${REPOSITORY_ROOT}/weights/dino_salad.ckpt}"
DINO_CHECKPOINT="${DINO_CHECKPOINT:-${REPOSITORY_ROOT}/weights/dinov2_vitb14_pretrain.pth}"
MINIMUM_FREE_GB="${MINIMUM_FREE_GB:-20}"
DRY_RUN="${DRY_RUN:-0}"

run_laser_python() {
  local command=()
  if [[ -n "${LASER_PYTHON:-}" ]]; then
    command=("${LASER_PYTHON}" "$@")
  else
    command=(conda run --no-capture-output -n "${LASER_ENV}" python "$@")
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY-RUN'
    printf ' %q' "${command[@]}"
    printf '\n'
  else
    "${command[@]}"
  fi
}

mkdir -p "${EXPERIMENT_ROOT}/logs" "${EXPERIMENT_ROOT}/summary"

printf '[all] phase=preflight\n'
run_laser_python "${UTILITY}" preflight \
  --repository "${REPOSITORY_ROOT}" \
  --seven-root "${SEVEN_SCENES_ROOT}" \
  --nrgbd-root "${NRGBD_ROOT}" \
  --kitti-root "${KITTI_ROOT}" \
  --seven-map "${SEVEN_SCENES_MAP}" \
  --nrgbd-map "${NRGBD_MAP}" \
  --pi3-checkpoint "${PI3_CHECKPOINT}" \
  --salad-checkpoint "${SALAD_CHECKPOINT}" \
  --dino-checkpoint "${DINO_CHECKPOINT}" \
  --minimum-free-gb "${MINIMUM_FREE_GB}" \
  --output "${EXPERIMENT_ROOT}/preflight.json"

printf '[all] phase=pointcloud\n'
bash "${SCRIPT_DIR}/run_pointcloud_all.sh"

printf '[all] phase=ate\n'
bash "${SCRIPT_DIR}/run_kitti_ate_all.sh"

printf '[all] phase=summary\n'
run_laser_python "${SUMMARY}" \
  --metric-root "${EXPERIMENT_ROOT}/metrics" \
  --seven-map "${SEVEN_SCENES_MAP}" \
  --nrgbd-map "${NRGBD_MAP}" \
  --output-dir "${EXPERIMENT_ROOT}/summary"

printf '[all] phase=complete\n'
