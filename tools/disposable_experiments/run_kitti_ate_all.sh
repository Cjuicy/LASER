#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPOSITORY_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
cd "${REPOSITORY_ROOT}"

UTILITY="${SCRIPT_DIR}/preflight_experiments.py"
CONTROL_PYTHON="${CONTROL_PYTHON:-python}"
LASER_ENV="${LASER_ENV:-vggt}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${REPOSITORY_ROOT}/outputs/disposable_experiments}"
WORK_ROOT="${EXPERIMENT_ROOT}/work"
METRIC_ROOT="${EXPERIMENT_ROOT}/metrics/ate/kitti"
LOG_ROOT="${EXPERIMENT_ROOT}/logs/ate/kitti"
ARTIFACT_ROOT="${WORK_ROOT}/ate_artifacts/kitti"
CACHE_ROOT="${WORK_ROOT}/prediction_cache/ate/kitti"
KITTI_ROOT="${KITTI_ROOT:-${REPOSITORY_ROOT}/data/KITTI}"
PI3_CHECKPOINT="${PI3_CHECKPOINT:-${REPOSITORY_ROOT}/weights/PI3/model.safetensors}"
SALAD_CHECKPOINT="${SALAD_CHECKPOINT:-${REPOSITORY_ROOT}/weights/dino_salad.ckpt}"
DINO_CHECKPOINT="${DINO_CHECKPOINT:-${REPOSITORY_ROOT}/weights/dinov2_vitb14_pretrain.pth}"
KEEP_INTERMEDIATES="${KEEP_INTERMEDIATES:-0}"
DRY_RUN="${DRY_RUN:-0}"
DRY_RUN_KITTI_SEQUENCE_LIST="${DRY_RUN_KITTI_SEQUENCE_LIST:-}"

ATE_METHODS=(
  depth-traditional
  depth-corrected
  geometry-corrected
  atomic-original-corrected
  atomic-split-assisted-corrected
  atomic-split-no-assisted-corrected
)

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

result_valid() {
  local result_path="$1"
  local sequence="$2"
  local method="$3"
  [[ "${DRY_RUN}" != "1" && -f "${result_path}" ]] || return 1
  run_laser_python "${UTILITY}" result-ok \
    --path "${result_path}" \
    --evaluation ate \
    --dataset kitti \
    --scene "${sequence}" \
    --method "${method}" >/dev/null 2>&1
}

mkdir -p "${METRIC_ROOT}" "${LOG_ROOT}" "${WORK_ROOT}"
sequences=(00 01 02 03 04 05 06 07 08 09 10)
if [[ -n "${DRY_RUN_KITTI_SEQUENCE_LIST}" ]]; then
  if [[ "${DRY_RUN}" != "1" ]]; then
    printf 'DRY_RUN_KITTI_SEQUENCE_LIST is allowed only with DRY_RUN=1\n' >&2
    exit 2
  fi
  read -r -a sequences <<< "${DRY_RUN_KITTI_SEQUENCE_LIST}"
fi

for sequence in "${sequences[@]}"; do
  [[ "${sequence}" =~ ^(0[0-9]|10)$ ]] || {
    printf 'Invalid KITTI sequence: %s\n' "${sequence}" >&2
    exit 2
  }
  image_dir="${KITTI_ROOT}/${sequence}/image_2"
  ground_truth="${KITTI_ROOT}/${sequence}/poses.txt"
  scene_cache="${CACHE_ROOT}/${sequence}"
  result_paths=()
  printf '[ate] dataset=kitti sequence=%s\n' "${sequence}"

  for method in "${ATE_METHODS[@]}"; do
    result_path="${METRIC_ROOT}/${sequence}/${method}.json"
    result_paths+=("${result_path}")
    if result_valid "${result_path}" "${sequence}" "${method}"; then
      printf '[skip] kitti/%s/%s\n' "${sequence}" "${method}"
      continue
    fi

    method_root="${ARTIFACT_ROOT}/${sequence}/${method}"
    method_overrides=()
    while IFS= read -r override_value; do
      method_overrides+=("${override_value}")
    done < <(
        "${CONTROL_PYTHON}" "${UTILITY}" method-overrides \
          --evaluation ate --method "${method}"
    )
    method_arguments=()
    for override in "${method_overrides[@]}"; do
      method_arguments+=(--set "${override}")
    done
    segmentation=""
    reconstruction=""
    for override in "${method_overrides[@]}"; do
      [[ "${override}" == segmentation.method=* ]] && segmentation="${override#*=}"
      [[ "${override}" == reconstruction.mode=* ]] && reconstruction="${override#*=}"
    done
    artifact_dir="${method_root}/artifact/${segmentation}-${reconstruction}"
    evaluator_dir="${method_root}/evaluation"
    evaluator_result="${evaluator_dir}/trajectory_metrics.json"
    log_path="${LOG_ROOT}/${sequence}/${method}.log"
    mkdir -p "$(dirname "${log_path}")"

    # A failed run is kept for inspection. Starting a later retry discards only
    # that stale campaign-owned method directory so the core CLI can recreate it.
    if [[ "${DRY_RUN}" != "1" && -d "${method_root}" ]]; then
      run_laser_python "${UTILITY}" cleanup-artifact \
        --path "${method_root}" \
        --allowed-root "${ARTIFACT_ROOT}"
    fi

    run_laser_python run_reconstruction.py \
      --config configs/reconstruction/pi3_laser.yaml \
      --set "input.image_dir=${image_dir}" \
      --set input.sample_stride=1 \
      --set "model.checkpoint=${PI3_CHECKPOINT}" \
      --set window.size=75 \
      --set window.overlap=30 \
      --set registration.confidence_keep_ratio=0.5 \
      --set "prediction_cache.root=${scene_cache}" \
      --set prediction_cache.mode=auto \
      --set output.scene_name=artifact \
      --set "output.cache_dir=${method_root}/legacy_cache" \
      --set "output.result_dir=${method_root}" \
      --set "loop.detection.salad_checkpoint=${SALAD_CHECKPOINT}" \
      --set "loop.detection.dino_checkpoint=${DINO_CHECKPOINT}" \
      "${method_arguments[@]}" 2>&1 | tee -a "${log_path}"

    run_laser_python evaluate_ate.py \
      --artifact "${artifact_dir}" \
      --config configs/evaluation/ate.yaml \
      --ground-truth "${ground_truth}" \
      --ground-truth-format replica \
      --output "${evaluator_dir}" 2>&1 | tee -a "${log_path}"

    run_laser_python "${UTILITY}" compact-ate \
      --artifact "${artifact_dir}" \
      --evaluator-result "${evaluator_result}" \
      --output "${result_path}" \
      --evaluation ate \
      --dataset kitti \
      --scene "${sequence}" \
      --method "${method}" 2>&1 | tee -a "${log_path}"

    run_laser_python "${UTILITY}" result-ok \
      --path "${result_path}" \
      --evaluation ate \
      --dataset kitti \
      --scene "${sequence}" \
      --method "${method}" >/dev/null

    if [[ "${KEEP_INTERMEDIATES}" != "1" ]]; then
      run_laser_python "${UTILITY}" cleanup-artifact \
        --path "${method_root}" \
        --allowed-root "${ARTIFACT_ROOT}"
    fi
  done

  if [[ "${KEEP_INTERMEDIATES}" != "1" ]]; then
    cache_arguments=()
    for result_path in "${result_paths[@]}"; do
      cache_arguments+=(--result "${result_path}")
    done
    run_laser_python "${UTILITY}" cleanup-scene-cache \
      --cache-root "${scene_cache}" \
      "${cache_arguments[@]}"
  fi
done
