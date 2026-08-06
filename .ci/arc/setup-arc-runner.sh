#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  setup-arc-runner.sh --github-token TOKEN [options]

Required:
  --github-token TOKEN        GitHub token with permission to register repo runners.

Options:
    --repo-url URL              GitHub repository URL. Default: detected from git remote
    --node-name NAME            Kubernetes node name for runner pods. Default: detected from Kubernetes or hostname
    --max-runners N             Maximum ARC runners. Default: computed from host CPU and memory
  --min-runners N             Minimum ARC runners. Default: 0
    --runner-label LABEL        ARC runner scale set name. Default: <host-os>-<host-version>-<hostname>
    --extra-labels CSV          Extra ARC labels. Default: detected host OS label, plus ubuntu-latest on Ubuntu
  --runner-namespace NAME     Runner namespace. Default: arc-runners
  --controller-namespace NAME Controller namespace. Default: arc-systems
  --controller-release NAME   Controller Helm release. Default: arc
    --runner-release NAME       Runner scale set Helm release. Default: sanitized runner label
    --image NAME                Local runner image name. Default: derived from host OS
    --runner-cpu N              CPU request and limit per runner. Default: 4
    --runner-memory SIZE        Memory request and limit per runner. Default: 16Gi
    --reserve-cpu N             Physical CPU cores reserved for host/controller. Default: 4
    --cache-root PATH           Host cache root. Default: /var/cache/arc/<runner-label>
  --skip-k8s-install          Do not install local Kubernetes if kubectl is already configured.
  --help                      Show this help.

The token is never printed by this script. Passing secrets as command arguments can
still expose them briefly through shell history or process listings. Use a short-lived
token and clear shell history according to local policy.
USAGE
}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || printf '%s\n' "${SCRIPT_DIR}")
OS_ID="ubuntu"
OS_VERSION_ID="24.04"
if [[ -r /etc/os-release ]]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    OS_ID=${ID:-${OS_ID}}
    OS_VERSION_ID=${VERSION_ID:-${OS_VERSION_ID}}
fi

REPO_URL=""
NODE_NAME=""
MAX_RUNNERS=""
MIN_RUNNERS="0"
RUNNER_LABEL="${OS_ID}-${OS_VERSION_ID}-$(hostname -s)"
RUNNER_NAMESPACE="arc-runners"
CONTROLLER_NAMESPACE="arc-systems"
CONTROLLER_RELEASE="arc"
RUNNER_RELEASE=""
RUNNER_IMAGE="arc-runner:${OS_ID}-${OS_VERSION_ID}-tools"
RUNNER_CPU="4"
RUNNER_MEMORY="16Gi"
RESERVE_CPU="4"
CACHE_ROOT=""
EXTRA_LABELS=""
GITHUB_TOKEN=""
SKIP_K8S_INSTALL="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --github-token)
            GITHUB_TOKEN=${2:-}
            shift 2
            ;;
        --repo-url)
            REPO_URL=${2:-}
            shift 2
            ;;
        --node-name)
            NODE_NAME=${2:-}
            shift 2
            ;;
        --max-runners)
            MAX_RUNNERS=${2:-}
            shift 2
            ;;
        --min-runners)
            MIN_RUNNERS=${2:-}
            shift 2
            ;;
        --runner-label)
            RUNNER_LABEL=${2:-}
            shift 2
            ;;
        --extra-labels)
            EXTRA_LABELS=${2:-}
            shift 2
            ;;
        --runner-namespace)
            RUNNER_NAMESPACE=${2:-}
            shift 2
            ;;
        --controller-namespace)
            CONTROLLER_NAMESPACE=${2:-}
            shift 2
            ;;
        --controller-release)
            CONTROLLER_RELEASE=${2:-}
            shift 2
            ;;
        --runner-release)
            RUNNER_RELEASE=${2:-}
            shift 2
            ;;
        --image)
            RUNNER_IMAGE=${2:-}
            shift 2
            ;;
        --runner-cpu)
            RUNNER_CPU=${2:-}
            shift 2
            ;;
        --runner-memory)
            RUNNER_MEMORY=${2:-}
            shift 2
            ;;
        --reserve-cpu)
            RESERVE_CPU=${2:-}
            shift 2
            ;;
        --cache-root)
            CACHE_ROOT=${2:-}
            shift 2
            ;;
        --skip-k8s-install)
            SKIP_K8S_INSTALL="true"
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "${GITHUB_TOKEN}" ]]; then
    echo "Missing required --github-token argument." >&2
    usage >&2
    exit 2
fi

if [[ ${EUID} -eq 0 ]]; then
    SUDO=()
else
    SUDO=(sudo)
fi
DOCKER=(docker)

log() {
    printf '[arc-setup] %s\n' "$*"
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1
}

sanitize_name() {
    printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//'
}

detect_repo_url() {
    if [[ -n "${REPO_URL}" ]]; then
        return
    fi

    REPO_URL=$(git -C "${REPO_ROOT}" config --get remote.origin.url 2>/dev/null || true)
    if [[ -z "${REPO_URL}" ]]; then
        echo "Unable to detect repository URL. Pass --repo-url." >&2
        exit 2
    fi

    case "${REPO_URL}" in
        git@github.com:*)
            REPO_URL="https://github.com/${REPO_URL#git@github.com:}"
            REPO_URL="${REPO_URL%.git}"
            ;;
        https://github.com/*.git)
            REPO_URL="${REPO_URL%.git}"
            ;;
    esac
}

physical_cpu_cores() {
    if require_cmd lscpu; then
        local cores sockets
        cores=$(lscpu -p=Core,Socket 2>/dev/null | grep -v '^#' | sort -u | wc -l)
        if [[ "${cores}" =~ ^[0-9]+$ ]] && [[ "${cores}" -gt 0 ]]; then
            printf '%s\n' "${cores}"
            return
        fi

        cores=$(lscpu | awk -F: '/Core\(s\) per socket/ {gsub(/ /, "", $2); print $2; exit}')
        sockets=$(lscpu | awk -F: '/Socket\(s\)/ {gsub(/ /, "", $2); print $2; exit}')
        if [[ "${cores}" =~ ^[0-9]+$ ]] && [[ "${sockets}" =~ ^[0-9]+$ ]]; then
            printf '%s\n' "$((cores * sockets))"
            return
        fi
    fi

    nproc
}

memory_gib() {
    awk '/MemTotal/ {print int($2 / 1024 / 1024)}' /proc/meminfo
}

memory_to_gib() {
    local value=$1
    case "${value}" in
        *Gi) printf '%s\n' "${value%Gi}" ;;
        *G) printf '%s\n' "${value%G}" ;;
        *Mi) printf '%s\n' "$(( ${value%Mi} / 1024 ))" ;;
        *M) printf '%s\n' "$(( ${value%M} / 1024 ))" ;;
        *) printf '%s\n' "${value}" ;;
    esac
}

detect_capacity_defaults() {
    if [[ -z "${MAX_RUNNERS}" ]]; then
        local total_cpu usable_cpu cpu_capacity total_mem runner_mem mem_capacity computed
        total_cpu=$(physical_cpu_cores)
        usable_cpu=$((total_cpu - RESERVE_CPU))
        if [[ "${usable_cpu}" -lt "${RUNNER_CPU}" ]]; then
            usable_cpu=${RUNNER_CPU}
        fi

        cpu_capacity=$((usable_cpu / RUNNER_CPU))
        total_mem=$(memory_gib)
        runner_mem=$(memory_to_gib "${RUNNER_MEMORY}")
        mem_capacity=$((total_mem / runner_mem))
        computed=${cpu_capacity}
        if [[ "${mem_capacity}" -lt "${computed}" ]]; then
            computed=${mem_capacity}
        fi
        if [[ "${computed}" -lt 1 ]]; then
            computed=1
        fi
        MAX_RUNNERS=${computed}
    fi

    if [[ -z "${RUNNER_RELEASE}" ]]; then
        RUNNER_RELEASE=$(sanitize_name "${RUNNER_LABEL}")
    fi

    if [[ -z "${CACHE_ROOT}" ]]; then
        CACHE_ROOT="/var/cache/arc/${RUNNER_LABEL}"
    fi

    if [[ -z "${EXTRA_LABELS}" ]]; then
        EXTRA_LABELS="${OS_ID}-${OS_VERSION_ID}"
        if [[ "${OS_ID}" == "ubuntu" ]]; then
            EXTRA_LABELS="${EXTRA_LABELS},ubuntu-latest"
        fi
    fi
}

detect_node_name() {
    if [[ -n "${NODE_NAME}" ]]; then
        return
    fi

    NODE_NAME=$(kubectl get node -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
    if [[ -z "${NODE_NAME}" ]]; then
        NODE_NAME=$(hostname)
    fi
}

install_base_packages() {
    log "Installing base Ubuntu packages"
    "${SUDO[@]}" apt-get update
    "${SUDO[@]}" apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        docker.io \
        gnupg \
        jq \
        tar

    if require_cmd systemctl; then
        "${SUDO[@]}" systemctl enable --now docker >/dev/null 2>&1 || true
    fi
}

configure_docker_command() {
    if docker info >/dev/null 2>&1; then
        DOCKER=(docker)
    else
        DOCKER=("${SUDO[@]}" docker)
    fi
}

install_k3s_if_needed() {
    if [[ "${SKIP_K8S_INSTALL}" == "true" ]]; then
        log "Skipping local Kubernetes install"
        return
    fi

    if require_cmd kubectl && kubectl version --client >/dev/null 2>&1 && [[ -n "${KUBECONFIG:-}" ]]; then
        log "kubectl is available and KUBECONFIG is set; skipping local Kubernetes install"
        return
    fi

    if ! require_cmd k3s; then
        log "Installing local Kubernetes distribution"
        curl -sfL https://get.k3s.io | "${SUDO[@]}" sh -s - --write-kubeconfig-mode 0644
    else
        log "Local Kubernetes distribution already installed"
    fi

    export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
}

install_helm_if_needed() {
    if require_cmd helm; then
        log "Helm already installed"
        return
    fi

    log "Installing Helm"
    local tmp_dir
    tmp_dir=$(mktemp -d)
    curl -fsSL https://get.helm.sh/helm-v3.21.3-linux-amd64.tar.gz -o "${tmp_dir}/helm.tar.gz"
    tar -xzf "${tmp_dir}/helm.tar.gz" -C "${tmp_dir}"
    "${SUDO[@]}" install -m 0755 "${tmp_dir}/linux-amd64/helm" /usr/local/bin/helm
    rm -rf "${tmp_dir}"
}

wait_for_kubernetes() {
    log "Waiting for Kubernetes API"
    for _ in $(seq 1 60); do
        if kubectl get nodes >/dev/null 2>&1; then
            kubectl get nodes
            return
        fi
        sleep 2
    done

    echo "Kubernetes API did not become ready in time." >&2
    exit 1
}

prepare_cache_dirs() {
    local host_uid host_gid
    host_uid=$(id -u)
    host_gid=$(id -g)

    log "Preparing host cache directories"
    "${SUDO[@]}" mkdir -p \
        "${CACHE_ROOT}/home-cache" \
        "${CACHE_ROOT}/tool-cache"
        "${SUDO[@]}" chown -R "${host_uid}:${host_gid}" "${CACHE_ROOT}"
}

render_values() {
        local values_file=$1
        local label labels_yaml
        labels_yaml=""

        IFS=',' read -ra labels <<<"${EXTRA_LABELS}"
        for label in "${labels[@]}"; do
        label=$(printf '%s' "${label}" | xargs)
        if [[ -n "${label}" ]]; then
            labels_yaml+="  - ${label}"$'\n'
        fi
        done

        cat >"${values_file}" <<EOF
githubConfigUrl: ${REPO_URL}
githubConfigSecret: github-token

runnerScaleSetName: ${RUNNER_LABEL}
scaleSetLabels:
${labels_yaml}
minRunners: ${MIN_RUNNERS}
maxRunners: ${MAX_RUNNERS}

template:
    spec:
        automountServiceAccountToken: false
        nodeSelector:
            kubernetes.io/hostname: ${NODE_NAME}
        securityContext:
            seccompProfile:
                type: RuntimeDefault
        containers:
            - name: runner
                image: ${RUNNER_IMAGE}
                imagePullPolicy: IfNotPresent
                command:
                    - /usr/local/bin/arc-runner-entrypoint.sh
                env:
                    - name: ACTIONS_RUNNER_TOOL_CACHE
                        value: /opt/hostedtoolcache
                    - name: ARC_RUNNER_UID
                        value: "$(id -u)"
                    - name: ARC_RUNNER_GID
                        value: "$(id -g)"
                securityContext:
                    privileged: false
                    allowPrivilegeEscalation: false
                    capabilities:
                        drop:
                            - ALL
                        add:
                            - CHOWN
                            - FOWNER
                            - SETGID
                            - SETUID
                resources:
                    requests:
                        cpu: "${RUNNER_CPU}"
                        memory: ${RUNNER_MEMORY}
                    limits:
                        cpu: "${RUNNER_CPU}"
                        memory: ${RUNNER_MEMORY}
                volumeMounts:
                    - name: runner-home-cache
                        mountPath: /home/runner/.cache
                    - name: runner-tool-cache
                        mountPath: /opt/hostedtoolcache
        volumes:
            - name: runner-home-cache
                hostPath:
                    path: ${CACHE_ROOT}/home-cache
                    type: DirectoryOrCreate
            - name: runner-tool-cache
                hostPath:
                    path: ${CACHE_ROOT}/tool-cache
                    type: DirectoryOrCreate
EOF
}

build_and_import_image() {
    log "Building runner image"
    "${DOCKER[@]}" build \
        -t "${RUNNER_IMAGE}" \
        -f "${SCRIPT_DIR}/Dockerfile.arc-runner-ubuntu-24.04" \
        "${SCRIPT_DIR}"

    log "Importing runner image into local Kubernetes runtime"
    local image_tar
    image_tar=$(mktemp --suffix=.tar)
    "${DOCKER[@]}" save -o "${image_tar}" "${RUNNER_IMAGE}"
    if require_cmd k3s; then
        "${SUDO[@]}" k3s ctr images import "${image_tar}"
    elif require_cmd ctr; then
        "${SUDO[@]}" ctr images import "${image_tar}"
    else
        echo "No supported local container runtime import command found. Expected k3s or ctr." >&2
        rm -f "${image_tar}"
        exit 1
    fi
    rm -f "${image_tar}"
}

install_arc_controller() {
    log "Installing or updating ARC controller"
    helm upgrade --install "${CONTROLLER_RELEASE}" \
        oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set-controller \
        --namespace "${CONTROLLER_NAMESPACE}" \
        --create-namespace \
        --timeout 10m
}

create_runner_secret() {
    log "Creating ARC GitHub token secret"
    kubectl create namespace "${RUNNER_NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -
    kubectl create secret generic github-token \
        --namespace "${RUNNER_NAMESPACE}" \
        --from-literal=github_token="${GITHUB_TOKEN}" \
        --dry-run=client -o yaml | kubectl apply -f -
    unset GITHUB_TOKEN
}

install_runner_scale_set() {
    local values_file
    values_file=$(mktemp --suffix=.yaml)
    render_values "${values_file}"

    log "Installing or updating ARC runner scale set"
    helm upgrade --install "${RUNNER_RELEASE}" \
        oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set \
        --namespace "${RUNNER_NAMESPACE}" \
        --create-namespace \
        -f "${values_file}" \
        --set controllerServiceAccount.name=arc-gha-rs-controller \
        --set controllerServiceAccount.namespace="${CONTROLLER_NAMESPACE}" \
        --timeout 10m
    rm -f "${values_file}"
}

verify_install() {
    log "Verifying non-secret ARC state"
    kubectl -n "${CONTROLLER_NAMESPACE}" get pods -o wide
    kubectl -n "${RUNNER_NAMESPACE}" get autoscalingrunnersets,autoscalinglisteners,ephemeralrunnersets,ephemeralrunners,pods -o wide
}

main() {
    detect_repo_url
    install_base_packages
    install_k3s_if_needed
    install_helm_if_needed

    if [[ -z "${KUBECONFIG:-}" ]]; then
        export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
    fi

    wait_for_kubernetes
    detect_node_name
    detect_capacity_defaults
    prepare_cache_dirs
    configure_docker_command
    build_and_import_image
    install_arc_controller
    create_runner_secret
    install_runner_scale_set
    verify_install
    log "ARC setup complete"
}

main
