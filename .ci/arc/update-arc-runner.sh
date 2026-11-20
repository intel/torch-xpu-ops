#!/usr/bin/env bash
# Keep the ARC runner image and the ARC controller/scale sets on their latest
# releases, recover broken scale sets, and write a health summary to STATUS_FILE.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"

RUNNER_NAMESPACE="arc-runners"
CONTROLLER_NAMESPACE="arc-systems"
RUNNER_RELEASE=""
RUNNER_IMAGE=""
RUNNER_VERSION="latest"
GH_VERSION="latest"
FORCE=0
INSTALL_TIMER=0
# Hourly bounds outages; a run with nothing to do only checks versions and phase.
TIMER_SCHEDULE="hourly"
VERSION_LABEL="arc.runner.version"
GH_VERSION_LABEL="arc.gh.version"
RELEASES=()
ARC_VERSION="latest"
ARC_DRAIN_AFTER_HOURS=24
CHART_REPO="oci://ghcr.io/actions/actions-runner-controller-charts"
STATUS_FILE="${SCRIPT_DIR}/current.status"
STATE_DIR="${SCRIPT_DIR}/.state"
ISSUES=()
WARNINGS=()
ARC_CURRENT=""
ARC_LATEST=""
RUNNER_LATEST=""
GH_LATEST=""

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Rebuild the ARC runner image when a newer actions/runner or gh is released,
import it into K3s, upgrade ARC when a newer chart is published, recreate
Outdated scale sets, and write a health summary to the status file.

Options:
  --runner-version VER       Runner version to use (default: latest)
  --gh-version VER           GitHub CLI version to use (default: latest)
  --arc-version VER|off      ARC chart version to keep (default: latest; off disables upgrades)
  --arc-drain-after HOURS    Block new jobs after an ARC upgrade waited this long (default: ${ARC_DRAIN_AFTER_HOURS})
  --status-file PATH         Health summary file (default: ${STATUS_FILE})
  --runner-namespace NS      Runner scale set namespace (default: ${RUNNER_NAMESPACE})
  --controller-namespace NS  ARC controller namespace (default: ${CONTROLLER_NAMESPACE})
  --runner-release NAME      Helm release (default: all releases in the namespace)
  --image IMAGE              Runner image (default: image used by each scale set)
  --force                    Rebuild even if the image is already on the target version
  --install-timer            Install a systemd timer that runs this script periodically
  --schedule SPEC            systemd OnCalendar value for the timer (default: ${TIMER_SCHEDULE})
  -h, --help                 Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --runner-version) RUNNER_VERSION=$2; shift 2 ;;
        --gh-version) GH_VERSION=$2; shift 2 ;;
        --arc-version) ARC_VERSION=$2; shift 2 ;;
        --arc-drain-after) ARC_DRAIN_AFTER_HOURS=$2; shift 2 ;;
        --status-file) STATUS_FILE=$2; shift 2 ;;
        --runner-namespace) RUNNER_NAMESPACE=$2; shift 2 ;;
        --controller-namespace) CONTROLLER_NAMESPACE=$2; shift 2 ;;
        --runner-release) RUNNER_RELEASE=$2; shift 2 ;;
        --image) RUNNER_IMAGE=$2; shift 2 ;;
        --force) FORCE=1; shift ;;
        --install-timer) INSTALL_TIMER=1; shift ;;
        --schedule) TIMER_SCHEDULE=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

export KUBECONFIG=${KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}

log() {
    printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

issue() {
    ISSUES+=("$*")
    log "ERROR: $*"
}

warn() {
    WARNINGS+=("$*")
    log "WARN: $*"
}

k() {
    kubectl --request-timeout=30s "$@"
}

if [[ $(id -u) -eq 0 ]]; then
    SUDO=()
else
    SUDO=(sudo -n)
fi

install_timer() {
    local user unit_dir=/etc/systemd/system
    user=$(id -un)
    log "Installing systemd timer arc-runner-update.timer (${TIMER_SCHEDULE}) as ${user}"
    "${SUDO[@]}" tee "${unit_dir}/arc-runner-update.service" >/dev/null <<EOF
[Unit]
Description=Update and health-check ARC (runner image, gh, ARC chart)
Wants=network-online.target
After=network-online.target k3s.service docker.service

[Service]
Type=oneshot
User=${user}
Environment=KUBECONFIG=${KUBECONFIG}
# A hung build would otherwise block every later timer run.
TimeoutStartSec=2h
ExecStart=${SCRIPT_PATH} --runner-namespace ${RUNNER_NAMESPACE} --controller-namespace ${CONTROLLER_NAMESPACE}${RUNNER_RELEASE:+ --runner-release ${RUNNER_RELEASE}}
EOF
    "${SUDO[@]}" tee "${unit_dir}/arc-runner-update.timer" >/dev/null <<EOF
[Unit]
Description=Periodically update the ARC runner image

[Timer]
OnCalendar=${TIMER_SCHEDULE}
RandomizedDelaySec=30m
Persistent=true

[Install]
WantedBy=timers.target
EOF
    "${SUDO[@]}" systemctl daemon-reload
    "${SUDO[@]}" systemctl enable --now arc-runner-update.timer
}

latest_release() {
    local requested=$1 repo=$2 url
    if [[ "${requested}" != "latest" ]]; then
        echo "${requested#v}"
        return
    fi
    url=$(curl -fsSL --retry 3 --max-time 60 -o /dev/null -w '%{url_effective}' \
        "https://github.com/${repo}/releases/latest" || true)
    url="${url##*/v}"
    if [[ ! "${url}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "Could not resolve latest ${repo} release (got '${url}')" >&2
        return 1
    fi
    echo "${url}"
}

detect_releases() {
    if [[ -n "${RUNNER_RELEASE}" ]]; then
        RELEASES=("${RUNNER_RELEASE}")
        return
    fi
    mapfile -t RELEASES < <(helm list -n "${RUNNER_NAMESPACE}" -q)
    if [[ ${#RELEASES[@]} -eq 0 ]]; then
        issue "No Helm releases found in ${RUNNER_NAMESPACE}"
        return 1
    fi
}

release_image() {
    if [[ -n "${RUNNER_IMAGE}" ]]; then
        echo "${RUNNER_IMAGE}"
        return
    fi
    helm get values -n "${RUNNER_NAMESPACE}" "$1" -o json \
        | python3 -c 'import json, sys; print(json.load(sys.stdin)["template"]["spec"]["containers"][0]["image"])'
}

image_label() {
    docker image inspect "$1" --format "{{ index .Config.Labels \"$2\" }}" 2>/dev/null || true
}

# K3s may GC the local-only image, or hold a stale copy after a failed import.
sync_k3s_image() {
    local image=$1 ref docker_id k3s_id
    ref="${image}"
    if [[ "${ref}" != */* ]]; then
        ref="docker.io/library/${ref}"
    elif [[ "${ref%%/*}" != *[.:]* && "${ref%%/*}" != localhost ]]; then
        ref="docker.io/${ref}"
    fi
    docker_id=$(docker image inspect "${image}" --format '{{.Id}}' 2>/dev/null || true)
    k3s_id=$("${SUDO[@]}" k3s ctr images ls 2>/dev/null | awk -v r="${ref}" '$1 == r {print $3}')
    if [[ -n "${docker_id}" && "${docker_id}" != "${k3s_id}" ]]; then
        log "K3s copy of ${image} is ${k3s_id:-missing}; importing ${docker_id}"
        docker save "${image}" | "${SUDO[@]}" k3s ctr images import -
    fi
}

build_and_import() {
    local image=$1 version=$2 gh_version=$3
    log "Building ${image} with actions/runner ${version}, gh ${gh_version}"
    docker build \
        --build-arg ACTIONS_RUNNER_VERSION="${version}" \
        --build-arg GH_VERSION="${gh_version}" \
        --label "${VERSION_LABEL}=${version}" \
        --label "${GH_VERSION_LABEL}=${gh_version}" \
        -t "${image}" \
        -f "${SCRIPT_DIR}/Dockerfile.arc-runner-ubuntu-24.04" \
        "${SCRIPT_DIR}" || return 1

    log "Importing ${image} into K3s"
    docker save "${image}" | "${SUDO[@]}" k3s ctr images import - || return 1

    log "Pruning superseded images"
    docker image prune -f >/dev/null
    docker builder prune -f --filter until=168h >/dev/null
    # Only untagged images: the tagged runner image must stay local for IfNotPresent.
    "${SUDO[@]}" k3s crictl images -q --filter dangling=true 2>/dev/null \
        | xargs -r "${SUDO[@]}" k3s crictl --timeout 120s rmi >/dev/null 2>&1 || true
}

release_scale_set() {
    kubectl -n "${RUNNER_NAMESPACE}" get autoscalingrunnerset -o jsonpath=\
"{.items[?(@.metadata.annotations.meta\.helm\.sh/release-name==\"$1\")].metadata.name}" 2>/dev/null || true
}

recover_outdated_scale_set() {
    local release=$1 name phase chart values_file
    name=$(release_scale_set "${release}")
    if [[ -z "${name}" ]]; then
        phase="Missing"
    else
        phase=$(kubectl -n "${RUNNER_NAMESPACE}" get autoscalingrunnerset "${name}" -o jsonpath='{.status.phase}')
    fi
    if [[ "${phase}" != "Outdated" && "${phase}" != "Missing" ]]; then
        log "Scale set ${name} phase=${phase:-Running}; no recovery needed"
        return
    fi

    log "Scale set phase=${phase}; recreating it from Helm release ${release}"
    if [[ -n "${name}" ]]; then
        kubectl -n "${RUNNER_NAMESPACE}" delete autoscalingrunnerset "${name}" --wait --timeout=5m || return 1
    fi

    chart=$(helm list -n "${RUNNER_NAMESPACE}" -f "^${release}$" -o json \
        | python3 -c 'import json, sys; print(json.load(sys.stdin)[0]["chart"].rsplit("-", 1)[1])') || return 1
    values_file=$(mktemp --suffix=.yaml)
    helm get values -n "${RUNNER_NAMESPACE}" "${release}" -o yaml > "${values_file}" || return 1

    helm upgrade --install "${release}" \
        oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set \
        --version "${chart}" \
        --namespace "${RUNNER_NAMESPACE}" \
        -f "${values_file}" \
        --timeout 10m || { rm -f "${values_file}"; return 1; }
    rm -f "${values_file}"

    name=$(release_scale_set "${release}")
    log "Waiting for listener of scale set ${name}"
    local _
    for _ in $(seq 60); do
        if kubectl -n "${CONTROLLER_NAMESPACE}" get pods \
            -l "actions.github.com/scale-set-name=${name}" \
            -o jsonpath='{.items[*].status.containerStatuses[*].ready}' 2>/dev/null | grep -q true; then
            log "Listener is ready"
            return
        fi
        sleep 5
    done
    echo "Listener for ${name} did not become ready" >&2
    return 1
}

listener_ready() {
    kubectl -n "${CONTROLLER_NAMESPACE}" get pods \
        -l "actions.github.com/scale-set-name=$1" \
        -o jsonpath='{.items[*].status.containerStatuses[*].ready}' 2>/dev/null | grep -q true
}

# ARC leaves EphemeralRunners that exhausted their retries in phase Failed.
delete_failed_runners() {
    local failed
    failed=$(kubectl -n "${RUNNER_NAMESPACE}" get ephemeralrunners \
        -o jsonpath='{range .items[?(@.status.phase=="Failed")]}{.metadata.name}{"\n"}{end}' 2>/dev/null || true)
    if [[ -n "${failed}" ]]; then
        log "Deleting failed EphemeralRunners: ${failed//$'\n'/ }"
        xargs -r kubectl -n "${RUNNER_NAMESPACE}" delete ephemeralrunner --wait=false <<<"${failed}" || true
    fi
}

semver_ok() {
    [[ "$1" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]
}

controller_release() {
    helm list -n "${CONTROLLER_NAMESPACE}" -o json \
        | python3 -c 'import json, sys
for r in json.load(sys.stdin):
    if r["chart"].startswith("gha-runner-scale-set-controller-"):
        print(r["name"]); break'
}

release_chart_version() {
    helm list -n "$1" -f "^$2$" -o json \
        | python3 -c 'import json, sys; print(json.load(sys.stdin)[0]["chart"].rsplit("-", 1)[1])'
}

latest_arc_version() {
    if [[ "${ARC_VERSION}" != "latest" ]]; then
        echo "${ARC_VERSION#v}"
        return
    fi
    local v
    v=$(timeout 120 helm show chart "${CHART_REPO}/gha-runner-scale-set-controller" 2>/dev/null \
        | awk '/^version:/ {print $2}')
    semver_ok "${v}" || return 1
    echo "${v}"
}

busy_runners() {
    local out
    # An API error must never look like an idle node.
    out=$(k -n "${RUNNER_NAMESPACE}" get ephemeralrunners --no-headers 2>/dev/null) || { echo 999; return; }
    if [[ -z "${out}" ]]; then echo 0; else wc -l <<<"${out}"; fi
}

all_scale_sets() {
    k -n "${RUNNER_NAMESPACE}" get autoscalingrunnersets -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}'
}

# Setting maxRunners=0 stops new jobs from landing here so the node drains.
drain_scale_sets() {
    local name max
    for name in $(all_scale_sets); do
        [[ -f "${STATE_DIR}/drained-${name}" ]] && continue
        max=$(k -n "${RUNNER_NAMESPACE}" get autoscalingrunnerset "${name}" -o jsonpath='{.spec.maxRunners}')
        echo "${max}" > "${STATE_DIR}/drained-${name}"
        k -n "${RUNNER_NAMESPACE}" patch autoscalingrunnerset "${name}" --type merge \
            -p '{"spec":{"maxRunners":0}}' >/dev/null
        log "Drained ${name} (maxRunners ${max} -> 0)"
    done
}

undrain_scale_sets() {
    local file name max
    for file in "${STATE_DIR}"/drained-*; do
        [[ -e "${file}" ]] || continue
        name=${file##*/drained-}
        max=$(<"${file}")
        if k -n "${RUNNER_NAMESPACE}" get autoscalingrunnerset "${name}" >/dev/null 2>&1; then
            k -n "${RUNNER_NAMESPACE}" patch autoscalingrunnerset "${name}" --type merge \
                -p "{\"spec\":{\"maxRunners\":${max}}}" >/dev/null || { issue "Could not restore maxRunners=${max} on ${name}"; continue; }
            log "Restored ${name} maxRunners=${max}"
        fi
        rm -f "${file}"
    done
}

wait_listeners() {
    local release name _
    for release in "$@"; do
        name=""
        for _ in $(seq 60); do
            name=$(release_scale_set "${release}")
            if [[ -n "${name}" ]] && listener_ready "${name}"; then
                break
            fi
            name=""
            sleep 5
        done
        [[ -n "${name}" ]] || { log "Listener for release ${release} did not become ready"; return 1; }
    done
}

# Helm does not upgrade CRDs, so an ARC version change is a full reinstall.
install_arc() {
    local version=$1 backup=$2 controller=$3 release
    shift 3
    for release in "$@"; do
        helm uninstall "${release}" -n "${RUNNER_NAMESPACE}" --wait --timeout 10m 2>/dev/null || true
    done
    helm uninstall "${controller}" -n "${CONTROLLER_NAMESPACE}" --wait --timeout 10m 2>/dev/null || true
    k get crd -o name | grep '\.actions\.github\.com$' | xargs -r kubectl delete --timeout=5m || return 1

    helm install "${controller}" "${CHART_REPO}/gha-runner-scale-set-controller" \
        --version "${version}" --namespace "${CONTROLLER_NAMESPACE}" --create-namespace \
        -f "${backup}/${controller}.yaml" --wait --timeout 10m || return 1
    for release in "$@"; do
        helm install "${release}" "${CHART_REPO}/gha-runner-scale-set" \
            --version "${version}" --namespace "${RUNNER_NAMESPACE}" \
            -f "${backup}/${release}.yaml" --timeout 10m || return 1
    done
    wait_listeners "$@"
}

upgrade_arc() {
    local from=$1 to=$2 controller backup release f
    local -a scale_releases=()
    controller=$(controller_release)
    [[ -n "${controller}" ]] || { issue "ARC controller release not found; upgrade aborted"; return 1; }
    for release in $(helm list -n "${RUNNER_NAMESPACE}" -q); do
        [[ "$(helm list -n "${RUNNER_NAMESPACE}" -f "^${release}$" -o json)" == *'"gha-runner-scale-set-'* ]] \
            && scale_releases+=("${release}")
    done

    backup="${STATE_DIR}/arc-backup-$(date -u +%Y%m%dT%H%M%SZ)"
    mkdir -p "${backup}"
    for release in "${scale_releases[@]}"; do
        helm get values "${release}" -n "${RUNNER_NAMESPACE}" -o yaml > "${backup}/${release}.yaml" \
            || { issue "Backing up values of ${release} failed; ARC upgrade aborted"; return 1; }
    done
    helm get values "${controller}" -n "${CONTROLLER_NAMESPACE}" -o yaml > "${backup}/${controller}.yaml" \
        || { issue "Backing up values of ${controller} failed; ARC upgrade aborted"; return 1; }
    for f in "${backup}"/*.yaml; do
        [[ "$(<"${f}")" == "null" ]] && echo '{}' > "${f}"
    done
    log "Upgrading ARC ${from} -> ${to} (backup: ${backup})"

    if install_arc "${to}" "${backup}" "${controller}" "${scale_releases[@]}"; then
        log "ARC upgraded to ${to}"
        rm -f "${STATE_DIR}"/drained-*
        return 0
    fi
    issue "ARC upgrade ${from} -> ${to} failed; rolling back to ${from}"
    if install_arc "${from}" "${backup}" "${controller}" "${scale_releases[@]}"; then
        rm -f "${STATE_DIR}"/drained-*
        warn "ARC rolled back to ${from}; values backup kept in ${backup}"
    else
        issue "ARC rollback to ${from} FAILED; reinstall manually from ${backup} (see README 'ARC version')"
    fi
    return 1
}

check_arc_upgrade() {
    local controller pending_file="${STATE_DIR}/arc-upgrade-pending" since busy age_h
    controller=$(controller_release)
    if [[ -z "${controller}" ]]; then
        issue "ARC controller Helm release not found in ${CONTROLLER_NAMESPACE}"
        return 1
    fi
    ARC_CURRENT=$(release_chart_version "${CONTROLLER_NAMESPACE}" "${controller}")
    if [[ "${ARC_VERSION}" == "off" ]]; then
        undrain_scale_sets
        return 0
    fi
    if ! ARC_LATEST=$(latest_arc_version); then
        warn "ARC chart version lookup failed; ARC upgrade check skipped this run"
        return 1
    fi
    if [[ "${ARC_CURRENT}" == "${ARC_LATEST}" ]]; then
        log "ARC ${ARC_CURRENT} is up to date"
        rm -f "${pending_file}"
        undrain_scale_sets
        return 0
    fi
    if [[ $(k get autoscalingrunnersets -A --no-headers 2>/dev/null | awk -v ns="${RUNNER_NAMESPACE}" '$1 != ns' | wc -l) -gt 0 ]]; then
        issue "ARC ${ARC_LATEST} available but scale sets exist outside ${RUNNER_NAMESPACE}; upgrade manually"
        return 1
    fi

    [[ -f "${pending_file}" ]] || date +%s > "${pending_file}"
    since=$(<"${pending_file}")
    age_h=$(( ($(date +%s) - since) / 3600 ))
    busy=$(busy_runners)

    if [[ ${busy} -eq 0 ]]; then
        drain_scale_sets
        sleep 15
        busy=$(busy_runners)
        if [[ ${busy} -eq 0 ]]; then
            upgrade_arc "${ARC_CURRENT}" "${ARC_LATEST}" || return 1
            rm -f "${pending_file}"
            ARC_CURRENT=$(release_chart_version "${CONTROLLER_NAMESPACE}" "${controller}")
            return 0
        fi
        [[ ${age_h} -ge ${ARC_DRAIN_AFTER_HOURS} ]] || undrain_scale_sets
    elif [[ ${age_h} -ge ${ARC_DRAIN_AFTER_HOURS} ]]; then
        drain_scale_sets
    fi

    if compgen -G "${STATE_DIR}/drained-*" >/dev/null; then
        warn "ARC upgrade ${ARC_CURRENT} -> ${ARC_LATEST}: DRAINING (no new jobs), waiting for ${busy} running runner(s) to finish"
    else
        warn "ARC upgrade ${ARC_CURRENT} -> ${ARC_LATEST} pending ${age_h}h: waiting for ${busy} busy runner(s); drain starts after ${ARC_DRAIN_AFTER_HOURS}h"
    fi
}

write_status() {
    local rc=$1 state="OK" tmp name release image line next
    set +e
    if [[ ${rc} -ne 0 && ${#ISSUES[@]} -eq 0 ]]; then
        ISSUES+=("Update script exited unexpectedly (exit ${rc}); see: journalctl -u arc-runner-update -n 100")
    fi
    if [[ ${#ISSUES[@]} -gt 0 ]]; then
        state="FAILED"
    elif [[ ${#WARNINGS[@]} -gt 0 ]]; then
        state="WARNING"
    fi
    next=$(systemctl show arc-runner-update.timer -p NextElapseUSecRealtime --value 2>/dev/null)
    tmp=$(mktemp "${STATUS_FILE}.XXXXXX")
    {
        echo "ARC status: ${state}"
        echo "Checked:    $(date -u '+%Y-%m-%d %H:%M:%S UTC') (exit ${rc})"
        echo "Next check: ${next:-unknown}"
        echo
        echo "Versions (current / latest):"
        echo "  ARC chart:      ${ARC_CURRENT:-unknown} / ${ARC_LATEST:-unknown}"
        for image in $(for release in "${RELEASES[@]}"; do release_image "${release}"; done | sort -u); do
            echo "  ${image}:"
            echo "    actions/runner: $(image_label "${image}" "${VERSION_LABEL}") / ${RUNNER_LATEST:-unknown}"
            echo "    gh:             $(image_label "${image}" "${GH_VERSION_LABEL}") / ${GH_LATEST:-unknown}"
            echo "    image id:       $(docker image inspect "${image}" --format '{{.Id}}' 2>/dev/null | cut -c8-19)"
        done
        echo
        echo "Scale sets:"
        for release in "${RELEASES[@]}"; do
            name=$(release_scale_set "${release}")
            if [[ -z "${name}" ]]; then
                echo "  ${release}: MISSING"
                continue
            fi
            printf '  %-20s phase=%-8s listener=%-9s runners=%s/%s\n' "${name}" \
                "$(k -n "${RUNNER_NAMESPACE}" get autoscalingrunnerset "${name}" -o jsonpath='{.status.phase}' 2>/dev/null)" \
                "$(listener_ready "${name}" && echo Ready || echo NOT-READY)" \
                "$(k -n "${RUNNER_NAMESPACE}" get pods -l "actions.github.com/scale-set-name=${name}" --no-headers 2>/dev/null | wc -l)" \
                "$(k -n "${RUNNER_NAMESPACE}" get autoscalingrunnerset "${name}" -o jsonpath='{.spec.maxRunners}' 2>/dev/null)"
        done
        echo
        echo "Issues (${#ISSUES[@]}):"
        for line in "${ISSUES[@]}"; do echo "  - ${line}"; done
        echo "Warnings (${#WARNINGS[@]}):"
        for line in "${WARNINGS[@]}"; do echo "  - ${line}"; done
        for release in "${RELEASES[@]}"; do
            name=$(release_scale_set "${release}")
            if [[ -n "${name}" ]] && ! listener_ready "${name}"; then
                echo
                echo "Recent listener log for ${name}:"
                k -n "${CONTROLLER_NAMESPACE}" logs -l "actions.github.com/scale-set-name=${name}" --tail=10 2>&1 | sed 's/^/  /'
            fi
        done
        echo
        echo "Details: journalctl -u arc-runner-update --since today --no-pager"
    } > "${tmp}"
    chmod 0644 "${tmp}"
    mv -f "${tmp}" "${STATUS_FILE}"
}

main() {
    if [[ ${INSTALL_TIMER} -eq 1 ]]; then
        install_timer
    fi

    # Read-only open so root and non-root runs share the same lock.
    exec 9<"${SCRIPT_PATH}"
    if ! flock -n 9; then
        log "Another update is already running; skipping"
        exit 0
    fi
    mkdir -p "${STATE_DIR}"
    trap 'write_status $?' EXIT

    if ! k get --raw /readyz >/dev/null 2>&1; then
        issue "Kubernetes API unreachable; check: systemctl status k3s"
        return 1
    fi
    if ! k -n "${CONTROLLER_NAMESPACE}" get pods -l app.kubernetes.io/component=controller-manager \
        -o jsonpath='{.items[*].status.containerStatuses[*].ready}' 2>/dev/null | grep -q true; then
        issue "ARC controller pod is not ready; check: kubectl -n ${CONTROLLER_NAMESPACE} get pods"
    fi

    detect_releases || return 1

    local release image name
    local -A images=()
    for release in "${RELEASES[@]}"; do
        images["$(release_image "${release}")"]=1
    done

    if RUNNER_LATEST=$(latest_release "${RUNNER_VERSION}" actions/runner) \
        && GH_LATEST=$(latest_release "${GH_VERSION}" cli/cli); then
        for image in "${!images[@]}"; do
            local current gh_current
            current=$(image_label "${image}" "${VERSION_LABEL}")
            gh_current=$(image_label "${image}" "${GH_VERSION_LABEL}")
            log "Image ${image}: runner ${current:-unknown}->${RUNNER_LATEST}, gh ${gh_current:-unknown}->${GH_LATEST}"
            if [[ ${FORCE} -eq 1 || "${current}" != "${RUNNER_LATEST}" || "${gh_current}" != "${GH_LATEST}" ]]; then
                if ! build_and_import "${image}" "${RUNNER_LATEST}" "${GH_LATEST}"; then
                    issue "Build of ${image} failed; still using the previous image"
                fi
            else
                log "Image ${image} is up to date"
            fi
        done
    else
        warn "actions/runner or gh version lookup failed; image rebuild skipped this run"
    fi

    for image in "${!images[@]}"; do
        sync_k3s_image "${image}" || issue "Importing ${image} into K3s failed"
    done

    check_arc_upgrade || true

    delete_failed_runners

    for release in "${RELEASES[@]}"; do
        recover_outdated_scale_set "${release}" || issue "Recovering scale set of release ${release} failed"
        name=$(release_scale_set "${release}")
        if [[ -n "${name}" ]] && ! listener_ready "${name}"; then
            issue "Listener for ${name} is not ready; check: kubectl -n ${CONTROLLER_NAMESPACE} logs -l actions.github.com/scale-set-name=${name}"
        fi
    done
    log "Done (issues=${#ISSUES[@]}, warnings=${#WARNINGS[@]})"
    [[ ${#ISSUES[@]} -eq 0 ]]
}

main
