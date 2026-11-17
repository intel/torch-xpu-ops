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
recover_bdf=""
recover_root_bdf=""
DETECTED_GPU_COUNT=0
DETECTED_DEVICE_NAMES=""
self_path="$(realpath "${BASH_SOURCE[0]}")"
# GPU health events of this job, for the UT report.
health_log="${XPU_HEALTH_LOG:-${GITHUB_WORKSPACE:+${GITHUB_WORKSPACE}/xpu_health.jsonl}}"
readonly BDF_RE='^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$'

log() {
  printf '%s\n' "$*" >&2
}

usage() {
  cat <<'EOF'
Usage: detect-devices.sh [options]

Detect online XPU devices and produce pytest sharding environment variables.
Before that, every Intel GPU on PCI that the compute runtime cannot use is
recovered (driver unbind, PCI reset, rebind) or, failing that, excluded.

Options:
  --pytest-others-args S     Extra PYTEST_ADDOPTS suffix to customize pytest behavior (e.g. for different workflows)
  --recover BDF              Only recover the GPU at PCI address BDF (needs root or passwordless sudo,
                             and a writable /sys); prints "RESULT <status> <detail>" and exits
  --help                   Show this message

Environment overrides:
  GITHUB_ENV
  GITHUB_OUTPUT
  NUM_GPUS
  PYTEST_BASE_ARGS
  XPU_HEALTH_LOG

Outputs:
  ZE_AFFINITY_MASK
  NUMACTL_ARGS
  PYTEST_EXTRA_ARGS
  PYTEST_ADDOPTS
  XPU_CPU_COUNT
  XPU_TOTAL_COUNT
  NUM_GPUS
  XPU_RECOVER_CMD
  XPU_HEALTH_LOG
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
      --recover)
        recover_bdf="${2,,}"
        shift 2
        ;;
      --recover-as-root)
        recover_root_bdf="$2"
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

health_event() {
  local event="$1"
  local card="$2"
  local detail="$3"

  [ -n "${health_log}" ] || return 0
  detail="${detail//\\/\\\\}"
  detail="${detail//\"/\\\"}"
  detail="${detail//$'\n'/ }"
  printf '{"time": %d, "machine": "%s", "card": "%s", "event": "%s", "detail": "%s", "source": "detect-devices"}\n' \
    "$(date +%s)" "${RUNNER_NAME:-$(hostname)}" "${card}" "${event}" "${detail}" >> "${health_log}" || true
}

# Intel GPUs on PCI (same filter as get_device_rows), as full BDFs.
pci_cards() {
  lspci -D -nn 2>/dev/null | grep -Ei 'VGA|DISPLAY' | grep -v 'UHD' | grep '8086:' | awk '{print $1}' | sort || true
}

# Intel GPUs the compute runtime can use, as full BDFs.
runtime_cards() {
  timeout -k 5 60 clinfo 2>/dev/null \
    | awk '/Platform Name/ {gpu = /Intel/ && /Graphics/} gpu && /PCI bus info \(KHR\)/ {print tolower($NF)}' \
    | sort -u || true
}

runtime_device_count() {
  timeout -k 5 60 clinfo -l 2>/dev/null \
    | awk '/^Platform/ {gpu = /Intel/ && /Graphics/} gpu && /Device #/ {n++} END {print n + 0}' || echo 0
}

# Write a sysfs file in the background: a driver stuck in the kernel must not hang us.
sysfs_write() {
  local path="$1"
  local value="$2"
  local seconds="$3"
  local pid
  local i

  ( printf '%s' "${value}" > "${path}" ) 2>/dev/null &
  pid=$!
  for ((i = 0; i < seconds; i++)); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      wait "${pid}"
      return
    fi
    sleep 1
  done
  return 124
}

recover_result() {
  printf 'RESULT %s %s\n' "$1" "$2"
  [ "$1" = ok ]
  exit
}

card_holders() {
  local node

  for node in "$@"; do
    find /proc/[0-9]*/fd -lname "${node}" 2>/dev/null
  done | cut -d/ -f3 | sort -u | grep -vx "$$" || true
}

# Runs as root: kill what holds the card, unbind it, PCI reset, bind again.
recover_as_root() {
  local bdf="$1"
  local dev="/sys/bus/pci/devices/$1"
  local drv=""
  local nodes=()
  local node
  local pids
  local i

  [[ "${bdf}" =~ ${BDF_RE} ]] && [ -e "${dev}" ] || recover_result refused "no PCI device ${bdf}"
  if [ "$(cat "${dev}/vendor")" != 0x8086 ] || [[ "$(cat "${dev}/class")" != 0x03* ]]; then
    recover_result refused "${bdf} is not an Intel GPU"
  fi
  if (( $(cat /proc/sys/kernel/tainted) & 128 )); then
    recover_result reboot-required "kernel oopsed (taint D): unbinding ${bdf} would hang, reboot the node"
  fi
  if [ -e "${dev}/driver" ]; then
    drv="$(basename "$(readlink -f "${dev}/driver")")"
  elif [ -d /sys/bus/pci/drivers/xe ]; then
    drv=xe
  else
    drv=i915
  fi
  case "${drv}" in
    xe|i915) ;;
    *) recover_result refused "${bdf} is bound to unexpected driver ${drv}" ;;
  esac
  [ -w "/sys/bus/pci/drivers/${drv}/bind" ] \
    || recover_result no-privilege "/sys is read-only here (container without --privileged?)"

  for node in "${dev}"/drm/card* "${dev}"/drm/renderD*; do
    [ -e "${node}" ] && nodes+=("/dev/dri/$(basename "${node}")")
  done
  if ((${#nodes[@]} > 0)); then
    pids="$(card_holders "${nodes[@]}")"
    if [ -n "${pids}" ]; then
      log "[Recover] ${bdf}: killing holders $(tr '\n' ' ' <<< "${pids}")"
      kill -9 ${pids} 2>/dev/null || true
    fi
    for ((i = 0; i < 30; i++)); do
      [ -z "$(card_holders "${nodes[@]}")" ] && break
      sleep 1
    done
  fi
  if [ -e "${dev}/driver" ]; then
    sysfs_write "/sys/bus/pci/drivers/${drv}/unbind" "${bdf}" 120 \
      || recover_result reboot-required "unbinding ${bdf} from ${drv} hung or failed: driver stuck, reboot the node"
  fi
  if [ -e "${dev}/reset" ] && ! sysfs_write "${dev}/reset" 1 60; then
    log "[Recover] ${bdf}: PCI reset failed, binding anyway"
  fi
  sysfs_write "/sys/bus/pci/drivers/${drv}/bind" "${bdf}" 180 \
    || recover_result failed "binding ${bdf} to ${drv} failed"
  for ((i = 0; i < 60; i++)); do
    compgen -G "${dev}/drm/renderD*" > /dev/null && break
    sleep 1
  done
  if runtime_cards | grep -qx "${bdf}"; then
    recover_result ok "${bdf} rebound to ${drv}"
  fi
  recover_result failed "${bdf} rebound to ${drv} but the compute runtime does not list it"
}

recover_card() {
  local bdf="$1"

  if ! [[ "${bdf}" =~ ${BDF_RE} ]]; then
    printf 'RESULT refused bad PCI address %s\n' "${bdf}"
    return 1
  fi
  if [ "$(id -u)" -eq 0 ]; then
    bash "${self_path}" --recover-as-root "${bdf}"
  elif sudo -n true 2>/dev/null; then
    sudo -n bash "${self_path}" --recover-as-root "${bdf}"
  else
    printf 'RESULT no-privilege neither root nor passwordless sudo\n'
    return 1
  fi
}

# Recover every GPU on PCI the runtime cannot use; what stays broken is left out
# of ZE_AFFINITY_MASK by detect_available_gpu_ids.
check_cards() {
  local pci
  local runtime
  local bdf
  local out
  local status
  local detail

  pci="$(pci_cards)"
  [ -n "${pci}" ] || return 0
  runtime="$(runtime_cards)"
  if [ -z "${runtime}" ] && [ "$(runtime_device_count)" -gt 0 ]; then
    log "[Warning] clinfo does not report PCI addresses, skipping the GPU card check"
    return 0
  fi
  for bdf in $(comm -23 <(printf '%s\n' "${pci}") <(printf '%s\n' "${runtime}")); do
    log "[Warning] GPU ${bdf} is on PCI but the compute runtime cannot use it, recovering it"
    health_event unhealthy "${bdf}" "on PCI but not enumerated by the compute runtime"
    out="$(recover_card "${bdf}" 2>&1 | tail -n 1 || true)"
    status="$(awk '{print $2}' <<< "${out}")"
    detail="${out#RESULT * }"
    if [ "${status}" = ok ]; then
      log "[Info] GPU ${bdf} recovered: ${detail}"
      health_event recovered "${bdf}" "${detail}"
    else
      log "[Warning] GPU ${bdf} excluded from this run: ${status:-failed}: ${detail}"
      health_event "${status:-failed}" "${bdf}" "${detail}"
      health_event excluded "${bdf}" "not usable before the tests started"
    fi
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
  local pci_count

  parse_args "$@"

  if [ -n "${recover_root_bdf}" ]; then
    recover_as_root "${recover_root_bdf}"
  fi
  if [ -n "${recover_bdf}" ]; then
    recover_card "${recover_bdf}"
    return
  fi

  require_command clinfo  clinfo
  require_command lspci   pciutils
  require_command numactl numactl

  if [ -n "${health_log}" ]; then
    : > "${health_log}"
  fi
  check_cards

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
  emit_var NUM_GPUS "${online_xpu_count}"
  emit_var XPU_RECOVER_CMD "bash ${self_path} --recover"
  pci_count="$(pci_cards | wc -l)"
  if [ "${pci_count}" -gt 0 ] && [ $((total_xpu_count % pci_count)) -eq 0 ]; then
    emit_var XPU_TILES_PER_CARD "$((total_xpu_count / pci_count))"
  fi
  if [ -n "${health_log}" ]; then
    emit_var XPU_HEALTH_LOG "${health_log}"
  fi

  print_summary "${cpu_count}" "${total_xpu_count}" "${online_xpu_count}" "${ze_affinity_mask}" "${cpus_per_xpu}" "${device_names}"
}

main "$@"
