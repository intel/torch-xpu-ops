#!/usr/bin/env bash
set -euo pipefail

# need pytest-timeout for timeout options
# and pytest-rerunfailures for rerun options
# and pytest-xdist for dist options, worksteal is used to minimize idle time of workers when some tests are slower than others
readonly DEFAULT_PYTEST_ADDOPTS=' --timeout 600 --timeout_method=thread --max-worker-restart 1000000 --dist worksteal '

env_file="${GITHUB_ENV:-}"
output_file="${GITHUB_OUTPUT:-}"
PYTEST_BASE_ARGS="${PYTEST_BASE_ARGS:-${DEFAULT_PYTEST_ADDOPTS}}"
pytest_others_args=""
DETECTED_GPU_COUNT=0
DETECTED_DEVICE_NAMES=""

log() {
  printf '%s\n' "$*" >&2
}

usage() {
  cat <<'EOF'
Usage: detect-devices.sh [options]

Detect online XPU devices and produce pytest sharding environment variables.

Options:
  --pytest-others-args S     Extra PYTEST_ADDOPTS suffix to customize pytest behavior (e.g. for different workflows)
  --help                   Show this message

Environment overrides:
  GITHUB_ENV
  GITHUB_OUTPUT
  NUM_GPUS
  PYTEST_BASE_ARGS

Outputs:
  ZE_AFFINITY_MASK
  NUMACTL_ARGS
  PYTEST_EXTRA_ARGS
  PYTEST_ADDOPTS
  XPU_CPU_COUNT
  XPU_TOTAL_COUNT
  XPU_ONLINE_COUNT
EOF
}

require_command() {
  local cmd="$1"
  local package="$2"
  sudo -E apt-get update -qq
  sudo -E apt-get install -y -qq "${package}"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    log "[Error] Missing required command: ${cmd}"
    exit 1
  fi
}

parse_args() {
  while (($# > 0)); do
    case "$1" in
      --pytest-others-args)
        pytest_others_args="$2"
        shift 2
        ;;
      --help|-h)
        usage
        exit 0
        ;;
      *)
        log "[Error] Unknown argument: $1"
        usage
        exit 1
        ;;
    esac
  done
}

detect_cpu_count() {
  local cores_per_socket
  local sockets

  cores_per_socket="$(lscpu | awk -F: '/Core\(s\) per socket/ {gsub(/ /, "", $2); print $2}')"
  sockets="$(lscpu | awk -F: '/Socket\(s\)/ {gsub(/ /, "", $2); print $2}')"

  if [ -z "${cores_per_socket}" ] || [ -z "${sockets}" ]; then
    log "[Error] Failed to determine CPU topology from lscpu"
    exit 1
  fi

  printf '%d' "$((cores_per_socket * sockets))"
}

get_device_rows() {
  lspci -nn | grep -Ei 'VGA|DISPLAY' | grep -v 'UHD' | grep '8086:' || true
}

detect_device_info() {
  local device_rows="$1"
  local row
  local total_count=0
  local tile_count
  local device_names=()
  local device_name
  local joined_names=""

  if [ -n "${NUM_GPUS:-}" ]; then
    DETECTED_GPU_COUNT="${NUM_GPUS}"
    DETECTED_DEVICE_NAMES=""
    return
  fi

  while IFS= read -r row; do
    [ -z "${row}" ] && continue
    tile_count=1
    if [[ "${row}" =~ \(([0-9]+)[[:space:]]Tile\) ]]; then
      tile_count="${BASH_REMATCH[1]}"
    fi
    total_count=$((total_count + tile_count))
    device_name="$(sed -E 's/^[^]]*\]:[[:space:]]*//; s/[[:space:]]*\[[^]]+\]$//' <<< "${row}")"
    device_names+=("${device_name}")
  done <<< "${device_rows}"

  if ((${#device_names[@]} > 0)); then
    printf -v joined_names '%s; ' "${device_names[@]}"
    joined_names="${joined_names%; }"
  fi

  DETECTED_GPU_COUNT="${total_count}"
  DETECTED_DEVICE_NAMES="${joined_names}"
}

detect_available_gpu_ids() {
  local total_count="$1"
  local gpu_ids=()
  local gpu_id
  local probe_output
  local joined_ids=""

  for ((gpu_id = 0; gpu_id < total_count; gpu_id++)); do
    probe_output="$(ZE_AFFINITY_MASK=${gpu_id} clinfo --list | grep 'Graphics' || true)"
    if [ -n "${probe_output}" ] && [[ "${probe_output}" != *" UHD "* ]]; then
      gpu_ids+=("${gpu_id}")
    fi
  done

  if ((${#gpu_ids[@]} > 0)); then
    printf -v joined_ids '%s,' "${gpu_ids[@]}"
    joined_ids="${joined_ids%,}"
  fi

  printf '%s' "${joined_ids}"
}

count_csv_items() {
  local value="$1"

  if [ -z "${value}" ]; then
    printf '0'
  else
    awk -F, '{print NF}' <<< "${value}"
  fi
}

build_numactl_args() {
  local gpu_list="$1"
  local cpus_per_gpu="$2"
  local gpu_ids=()
  local index
  local gpu_id
  local cpu_start
  local cpu_end
  local numactl_args=""

  if [ -z "${gpu_list}" ]; then
    printf ' numactl -l '
    return
  fi

  IFS=',' read -r -a gpu_ids <<< "${gpu_list}"
  if ((${#gpu_ids[@]} <= 1)); then
    printf ' numactl -l '
    return
  fi

  for index in "${!gpu_ids[@]}"; do
    gpu_id="${gpu_ids[index]}"
    cpu_start=$((index * cpus_per_gpu))
    cpu_end=$((((index + 1) * cpus_per_gpu) - 1))
    numactl_args+=" ZE_AFFINITY_MASK=${gpu_id} OMP_NUM_THREADS=${cpus_per_gpu} numactl -l -C ${cpu_start}-${cpu_end} ;"
  done

  printf '%s' "${numactl_args}"
}

build_pytest_extra_args() {
  local gpu_list="$1"
  local cpus_per_gpu="$2"
  local gpu_ids=()
  local index
  local gpu_id
  local cpu_start
  local cpu_end
  local pytest_args=""

  if [ -z "${gpu_list}" ]; then
    printf ' -n 1 '
    return
  fi

  IFS=',' read -r -a gpu_ids <<< "${gpu_list}"
  if ((${#gpu_ids[@]} <= 1)); then
    printf ' -n 1 '
    return
  fi

  for index in "${!gpu_ids[@]}"; do
    gpu_id="${gpu_ids[index]}"
    cpu_start=$((index * cpus_per_gpu))
    cpu_end=$((((index + 1) * cpus_per_gpu) - 1))
    pytest_args+=" --tx popen//env:ZE_AFFINITY_MASK=${gpu_id}//env:OMP_NUM_THREADS=${cpus_per_gpu}//python=\"numactl -l -C ${cpu_start}-${cpu_end} python\""
  done

  printf '%s' "${pytest_args}"
}

emit_var() {
  local key="$1"
  local value="$2"

  if [ -n "${env_file}" ]; then
    printf '%s=%s\n' "${key}" "${value}" >> "${env_file}"
  fi
  if [ -n "${output_file}" ]; then
    printf '%s=%s\n' "${key}" "${value}" >> "${output_file}"
  fi
  if [ -z "${env_file}" ] && [ -z "${output_file}" ]; then
    printf '%s=%s\n' "${key}" "${value}"
  fi
}

print_summary() {
  local cpu_count="$1"
  local total_xpu_count="$2"
  local online_xpu_count="$3"
  local ze_affinity_mask="$4"
  local cpus_per_xpu="$5"
  local device_names_str="$6"
  local availability_status='OK'
  local formatted_device_names='<none>'

  if [ "${online_xpu_count}" -lt "${total_xpu_count}" ]; then
    availability_status='WARNING: available < detected'
  fi

  if [ -n "${device_names_str}" ]; then
    formatted_device_names="$(printf '%s' "${device_names_str}" | sed 's/; /\n                       - /g')"
    formatted_device_names="- ${formatted_device_names}"
  fi

  {
    printf '[Summary] Device detection completed\n'
    printf '  %-20s %s\n' 'Status:' "${availability_status}"
    printf '  %-20s %s\n' 'CPUs:' "${cpu_count}"
    printf '  %-20s %s\n' 'Detected GPUs:' "${total_xpu_count}"
    printf '  %-20s %s\n' 'Available GPUs:' "${online_xpu_count}"
    printf '  %-20s %s\n' 'CPUs per GPU:' "${cpus_per_xpu}"
    printf '  %-20s %s\n' 'ZE_AFFINITY_MASK:' "${ze_affinity_mask:-<none>}"
    printf '  %-20s %s\n' 'Device names:' "${formatted_device_names}"
  } >&2
}

main() {
  local cpu_count
  local device_rows
  local total_xpu_count
  local online_xpu_count
  local ze_affinity_mask
  local cpus_per_xpu
  local pytest_extra_args
  local pytest_addopts
  local numactl_args
  local device_names

  parse_args "$@"

  require_command clinfo  clinfo
  require_command lspci   pciutils
  require_command numactl numactl

  cpu_count="$(detect_cpu_count)"
  device_rows="$(get_device_rows)"
  detect_device_info "${device_rows}"
  total_xpu_count="${DETECTED_GPU_COUNT}"
  device_names="${DETECTED_DEVICE_NAMES}"
  ze_affinity_mask="$(detect_available_gpu_ids "${total_xpu_count}")"
  online_xpu_count="$(count_csv_items "${ze_affinity_mask}")"

  if [ "${online_xpu_count}" -lt "${total_xpu_count}" ]; then
    log "[Warning] Available GPUs are fewer than detected GPUs: available=${online_xpu_count}, detected=${total_xpu_count}"
  fi

  if [ "${total_xpu_count}" -gt 0 ]; then
    cpus_per_xpu="$((cpu_count / total_xpu_count))"
    if [ "${cpus_per_xpu}" -lt 1 ]; then
      cpus_per_xpu=1
    fi
  else
    cpus_per_xpu=1
  fi

  pytest_extra_args="$(build_pytest_extra_args "${ze_affinity_mask}" "${cpus_per_xpu}")"
  numactl_args="$(build_numactl_args "${ze_affinity_mask}" "${cpus_per_xpu}")"
  pytest_addopts="${PYTEST_ADDOPTS:-}"
  if [ -z "${pytest_addopts}" ]; then
    pytest_addopts="${PYTEST_BASE_ARGS} ${pytest_others_args} ${pytest_extra_args}"
  fi

  emit_var ZE_AFFINITY_MASK "${ze_affinity_mask}"
  emit_var NUMACTL_ARGS "${numactl_args}"
  emit_var PYTEST_EXTRA_ARGS "${pytest_extra_args}"
  emit_var PYTEST_ADDOPTS "${pytest_addopts}"
  emit_var XPU_CPU_COUNT "${cpu_count}"
  emit_var XPU_TOTAL_COUNT "${total_xpu_count}"
  emit_var XPU_ONLINE_COUNT "${online_xpu_count}"

  print_summary "${cpu_count}" "${total_xpu_count}" "${online_xpu_count}" "${ze_affinity_mask}" "${cpus_per_xpu}" "${device_names}"
}

main "$@"
