#!/usr/bin/env bash
# Keep the ARC runner image on the latest actions/runner release and recover
# scale sets that ARC tore down after GitHub rejected an outdated runner.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"

RUNNER_NAMESPACE="arc-runners"
CONTROLLER_NAMESPACE="arc-systems"
RUNNER_RELEASE=""
RUNNER_IMAGE=""
RUNNER_VERSION="latest"
FORCE=0
INSTALL_TIMER=0
# Hourly bounds outages; a run with nothing to do only checks versions and phase.
TIMER_SCHEDULE="hourly"
VERSION_LABEL="arc.runner.version"
values_file=""

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Rebuild the ARC runner image when a newer actions/runner is released, import it
into K3s, and recreate the runner scale set if ARC marked it Outdated.

Options:
  --runner-version VER       Runner version to use (default: latest)
  --runner-namespace NS      Runner scale set namespace (default: ${RUNNER_NAMESPACE})
  --controller-namespace NS  ARC controller namespace (default: ${CONTROLLER_NAMESPACE})
  --runner-release NAME      Helm release (default: only release in the namespace)
  --image IMAGE              Runner image (default: image used by the scale set)
  --force                    Rebuild even if the image is already on the target version
  --install-timer            Install a systemd timer that runs this script periodically
  --schedule SPEC            systemd OnCalendar value for the timer (default: ${TIMER_SCHEDULE})
  -h, --help                 Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --runner-version) RUNNER_VERSION=$2; shift 2 ;;
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
Description=Update ARC runner image to the latest actions/runner release
Wants=network-online.target
After=network-online.target k3s.service docker.service

[Service]
Type=oneshot
User=${user}
Environment=KUBECONFIG=${KUBECONFIG}
ExecStart=${SCRIPT_PATH} --runner-namespace ${RUNNER_NAMESPACE} --controller-namespace ${CONTROLLER_NAMESPACE}
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

resolve_runner_version() {
    if [[ "${RUNNER_VERSION}" != "latest" ]]; then
        echo "${RUNNER_VERSION#v}"
        return
    fi
    local url
    url=$(curl -fsSL -o /dev/null -w '%{url_effective}' https://github.com/actions/runner/releases/latest)
    echo "${url##*/v}"
}

detect_release() {
    if [[ -n "${RUNNER_RELEASE}" ]]; then
        return
    fi
    mapfile -t releases < <(helm list -n "${RUNNER_NAMESPACE}" -q)
    if [[ ${#releases[@]} -ne 1 ]]; then
        echo "Expected one Helm release in ${RUNNER_NAMESPACE}, found ${#releases[@]}; pass --runner-release." >&2
        exit 1
    fi
    RUNNER_RELEASE=${releases[0]}
}

detect_image() {
    if [[ -n "${RUNNER_IMAGE}" ]]; then
        return
    fi
    RUNNER_IMAGE=$(helm get values -n "${RUNNER_NAMESPACE}" "${RUNNER_RELEASE}" -o json \
        | python3 -c 'import json, sys; print(json.load(sys.stdin)["template"]["spec"]["containers"][0]["image"])')
}

image_version() {
    docker image inspect "${RUNNER_IMAGE}" --format "{{ index .Config.Labels \"${VERSION_LABEL}\" }}" 2>/dev/null || true
}

build_and_import() {
    local version=$1
    log "Building ${RUNNER_IMAGE} with actions/runner ${version}"
    docker build \
        --build-arg ACTIONS_RUNNER_VERSION="${version}" \
        --label "${VERSION_LABEL}=${version}" \
        -t "${RUNNER_IMAGE}" \
        -f "${SCRIPT_DIR}/Dockerfile.arc-runner-ubuntu-24.04" \
        "${SCRIPT_DIR}"

    log "Importing ${RUNNER_IMAGE} into K3s"
    docker save "${RUNNER_IMAGE}" | "${SUDO[@]}" k3s ctr images import -

    log "Pruning superseded images"
    docker image prune -f >/dev/null
    docker builder prune -f --filter until=168h >/dev/null
    # Only untagged images: the tagged runner image must stay local for IfNotPresent.
    "${SUDO[@]}" k3s crictl images -q --filter dangling=true 2>/dev/null \
        | xargs -r "${SUDO[@]}" k3s crictl --timeout 120s rmi >/dev/null 2>&1 || true
}

recover_outdated_scale_set() {
    local name phase chart
    name=$(kubectl -n "${RUNNER_NAMESPACE}" get autoscalingrunnerset \
        -l "app.kubernetes.io/instance" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
    if [[ -z "${name}" ]]; then
        phase="Missing"
    else
        phase=$(kubectl -n "${RUNNER_NAMESPACE}" get autoscalingrunnerset "${name}" -o jsonpath='{.status.phase}')
    fi
    if [[ "${phase}" != "Outdated" && "${phase}" != "Missing" ]]; then
        log "Scale set ${name} phase=${phase:-Running}; no recovery needed"
        return
    fi

    log "Scale set phase=${phase}; recreating it from Helm release ${RUNNER_RELEASE}"
    if [[ -n "${name}" ]]; then
        kubectl -n "${RUNNER_NAMESPACE}" delete autoscalingrunnerset "${name}" --wait --timeout=5m
    fi

    chart=$(helm list -n "${RUNNER_NAMESPACE}" -f "^${RUNNER_RELEASE}$" -o json \
        | python3 -c 'import json, sys; print(json.load(sys.stdin)[0]["chart"].rsplit("-", 1)[1])')
    values_file=$(mktemp --suffix=.yaml)
    trap 'rm -f "${values_file}"' EXIT
    helm get values -n "${RUNNER_NAMESPACE}" "${RUNNER_RELEASE}" -o yaml > "${values_file}"

    helm upgrade --install "${RUNNER_RELEASE}" \
        oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set \
        --version "${chart}" \
        --namespace "${RUNNER_NAMESPACE}" \
        -f "${values_file}" \
        --timeout 10m

    name=$(kubectl -n "${RUNNER_NAMESPACE}" get autoscalingrunnerset \
        -l "app.kubernetes.io/instance" -o jsonpath='{.items[0].metadata.name}')
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
    exit 1
}

main() {
    if [[ ${INSTALL_TIMER} -eq 1 ]]; then
        install_timer
    fi

    detect_release
    detect_image

    local target current
    target=$(resolve_runner_version)
    current=$(image_version)
    log "Runner image ${RUNNER_IMAGE}: current=${current:-unknown} target=${target}"

    if [[ ${FORCE} -eq 1 || "${current}" != "${target}" ]]; then
        build_and_import "${target}"
    else
        log "Runner image already on ${target}"
    fi

    recover_outdated_scale_set
    log "Done"
}

main
