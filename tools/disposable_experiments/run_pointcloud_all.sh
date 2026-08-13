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
METRIC_ROOT="${EXPERIMENT_ROOT}/metrics/pointcloud"
LOG_ROOT="${EXPERIMENT_ROOT}/logs/pointcloud"
PREPARED_ROOT="${WORK_ROOT}/prepared_pointcloud"
ARTIFACT_ROOT="${WORK_ROOT}/pointcloud_artifacts"
CACHE_ROOT="${WORK_ROOT}/prediction_cache/pointcloud"
SEVEN_SCENES_ROOT="${SEVEN_SCENES_ROOT:-${REPOSITORY_ROOT}/data/7-Scenes}"
NRGBD_ROOT="${NRGBD_ROOT:-${REPOSITORY_ROOT}/data/NeuralRGBD}"
SEVEN_SCENES_MAP="${SEVEN_SCENES_MAP:-${REPOSITORY_ROOT}/datasets/seq-id-maps/7scenes_mv-recon_seq-id-map-kf10.json}"
NRGBD_MAP="${NRGBD_MAP:-${REPOSITORY_ROOT}/datasets/seq-id-maps/NRGBD_mv-recon_seq-id-map-kf10.json}"
PI3_CHECKPOINT="${PI3_CHECKPOINT:-${REPOSITORY_ROOT}/weights/PI3/model.safetensors}"
KEEP_INTERMEDIATES="${KEEP_INTERMEDIATES:-0}"
DRY_RUN="${DRY_RUN:-0}"

POINTCLOUD_METHODS=(
  depth
  geometry
  atomic-original
  atomic-split-assisted
  atomic-split-no-assisted
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
  local dataset="$2"
  local scene="$3"
  local method="$4"
  [[ "${DRY_RUN}" != "1" && -f "${result_path}" ]] || return 1
  run_laser_python "${UTILITY}" result-ok \
    --path "${result_path}" \
    --evaluation pointcloud \
    --dataset "${dataset}" \
    --scene "${scene}" \
    --method "${method}" >/dev/null 2>&1
}

safe_scene_name() {
  local scene="$1"
  printf '%s\n' "${scene//\//__}"
}

load_scenes() {
  local map_path="$1"
  "${CONTROL_PYTHON}" -c \
    'import json,sys; print("\n".join(json.load(open(sys.argv[1])).keys()))' \
    "${map_path}"
}

run_dataset() {
  local dataset="$1"
  local raw_root="$2"
  local sequence_map="$3"
  local scenes=()
  while IFS= read -r scene_name; do
    scenes+=("${scene_name}")
  done < <(load_scenes "${sequence_map}")

  for scene in "${scenes[@]}"; do
    [[ -n "${scene}" ]] || continue
    local safe_scene
    safe_scene="$(safe_scene_name "${scene}")"
    local prepared_dir="${PREPARED_ROOT}/${dataset}/${safe_scene}"
    local image_dir="${prepared_dir}/images"
    local ground_truth="${prepared_dir}/ground_truth.npz"
    local scene_cache="${CACHE_ROOT}/${dataset}/${safe_scene}"
    local result_paths=()

    printf '[pointcloud] dataset=%s scene=%s\n' "${dataset}" "${scene}"
    local all_results_valid=1
    local method
    for method in "${POINTCLOUD_METHODS[@]}"; do
      local result_path="${METRIC_ROOT}/${dataset}/${safe_scene}/${method}.json"
      result_paths+=("${result_path}")
      if ! result_valid "${result_path}" "${dataset}" "${scene}" "${method}"; then
        all_results_valid=0
      fi
    done
    if [[ "${all_results_valid}" == "1" ]]; then
      printf '[skip-scene] %s/%s all methods validated\n' "${dataset}" "${scene}"
      if [[ "${KEEP_INTERMEDIATES}" != "1" ]]; then
        local cache_arguments=()
        local result_path
        for result_path in "${result_paths[@]}"; do
          cache_arguments+=(--result "${result_path}")
        done
        run_laser_python "${UTILITY}" cleanup-scene-cache \
          --cache-root "${scene_cache}" \
          "${cache_arguments[@]}"
        run_laser_python "${UTILITY}" cleanup-prepared \
          --path "${prepared_dir}" \
          --allowed-root "${PREPARED_ROOT}"
      fi
      continue
    fi

    run_laser_python "${UTILITY}" prepare-pointcloud \
      --dataset "${dataset}" \
      --root "${raw_root}" \
      --sequence-map "${sequence_map}" \
      --scene "${scene}" \
      --output-root "${PREPARED_ROOT}"

    result_paths=()
    for method in "${POINTCLOUD_METHODS[@]}"; do
      local result_path="${METRIC_ROOT}/${dataset}/${safe_scene}/${method}.json"
      result_paths+=("${result_path}")
      if result_valid "${result_path}" "${dataset}" "${scene}" "${method}"; then
        printf '[skip] %s/%s/%s\n' "${dataset}" "${scene}" "${method}"
        continue
      fi

      local method_root="${ARTIFACT_ROOT}/${dataset}/${safe_scene}/${method}"
      local method_overrides=()
      while IFS= read -r override_value; do
        method_overrides+=("${override_value}")
      done < <(
          "${CONTROL_PYTHON}" "${UTILITY}" method-overrides \
            --evaluation pointcloud --method "${method}"
      )
      local method_arguments=()
      local override
      for override in "${method_overrides[@]}"; do
        method_arguments+=(--set "${override}")
      done
      local segmentation=""
      local reconstruction=""
      for override in "${method_overrides[@]}"; do
        [[ "${override}" == segmentation.method=* ]] && segmentation="${override#*=}"
        [[ "${override}" == reconstruction.mode=* ]] && reconstruction="${override#*=}"
      done
      local artifact_dir="${method_root}/artifact/${segmentation}-${reconstruction}"
      local log_path="${LOG_ROOT}/${dataset}/${safe_scene}/${method}.log"
      mkdir -p "$(dirname "${log_path}")"

      # A failed run is kept for inspection. Starting a later retry discards only
      # that stale campaign-owned method directory so the core CLI can recreate it.
      if [[ "${DRY_RUN}" != "1" && -d "${method_root}" ]]; then
        run_laser_python "${UTILITY}" cleanup-artifact \
          --path "${method_root}" \
          --allowed-root "${ARTIFACT_ROOT}"
      fi

      run_laser_python run_reconstruction.py \
        --config configs/reconstruction/pi3_laser_no_loop.yaml \
        --set "input.image_dir=${image_dir}" \
        --set input.sample_stride=1 \
        --set "model.checkpoint=${PI3_CHECKPOINT}" \
        --set window.size=20 \
        --set window.overlap=5 \
        --set registration.confidence_keep_ratio=0.5 \
        --set segmentation.confidence_quantile_method=nearest \
        --set "prediction_cache.root=${scene_cache}" \
        --set prediction_cache.mode=auto \
        --set output.scene_name=artifact \
        --set "output.cache_dir=${method_root}/legacy_cache" \
        --set "output.result_dir=${method_root}" \
        "${method_arguments[@]}" 2>&1 | tee -a "${log_path}"

      run_laser_python "${UTILITY}" compact-pointcloud \
        --artifact "${artifact_dir}" \
        --ground-truth "${ground_truth}" \
        --evaluation-config configs/evaluation/pointcloud.yaml \
        --output "${result_path}" \
        --evaluation pointcloud \
        --dataset "${dataset}" \
        --scene "${scene}" \
        --method "${method}" 2>&1 | tee -a "${log_path}"

      run_laser_python "${UTILITY}" result-ok \
        --path "${result_path}" \
        --evaluation pointcloud \
        --dataset "${dataset}" \
        --scene "${scene}" \
        --method "${method}" >/dev/null

      if [[ "${KEEP_INTERMEDIATES}" != "1" ]]; then
        run_laser_python "${UTILITY}" cleanup-artifact \
          --path "${method_root}" \
          --allowed-root "${ARTIFACT_ROOT}"
      fi
    done

    if [[ "${KEEP_INTERMEDIATES}" != "1" ]]; then
      local cache_arguments=()
      local result_path
      for result_path in "${result_paths[@]}"; do
        cache_arguments+=(--result "${result_path}")
      done
      run_laser_python "${UTILITY}" cleanup-scene-cache \
        --cache-root "${scene_cache}" \
        "${cache_arguments[@]}"
      run_laser_python "${UTILITY}" cleanup-prepared \
        --path "${prepared_dir}" \
        --allowed-root "${PREPARED_ROOT}"
    fi
  done
}

mkdir -p "${METRIC_ROOT}" "${LOG_ROOT}" "${WORK_ROOT}"
run_dataset 7scenes "${SEVEN_SCENES_ROOT}" "${SEVEN_SCENES_MAP}"
run_dataset nrgbd "${NRGBD_ROOT}" "${NRGBD_MAP}"
