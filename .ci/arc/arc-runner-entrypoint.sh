#!/usr/bin/env bash
set -euo pipefail

RUNNER_USER=${RUNNER_USER:-runner}
RUNNER_GROUP=${RUNNER_GROUP:-${RUNNER_USER}}
RUNNER_HOME=${RUNNER_HOME:-/home/runner}
TOOL_CACHE=${ACTIONS_RUNNER_TOOL_CACHE:-/opt/hostedtoolcache}
OWNER_FILE=${ARC_RUNNER_OWNER_FILE:-/tmp/arc-runner-host-owner}

detect_host_owner() {
    for path in "${RUNNER_HOME}/.cache" "${TOOL_CACHE}"; do
        if [[ -e "${path}" ]]; then
            local uid
            local gid
            uid=$(stat -c '%u' "${path}")
            gid=$(stat -c '%g' "${path}")
            if [[ "${uid}:${gid}" != "0:0" ]]; then
                printf '%s:%s\n' "${uid}" "${gid}"
                return
            fi
        fi
    done

    printf '%s:%s\n' "${ARC_RUNNER_UID:-1000}" "${ARC_RUNNER_GID:-1000}"
}

cleanup() {
    /usr/local/bin/arc-runner-cleanup.sh || true
}

forward_signal() {
    if [[ -n "${runner_pid:-}" ]]; then
        kill -TERM -- "-${runner_pid}" 2>/dev/null || kill -TERM "${runner_pid}" 2>/dev/null || true
        wait "${runner_pid}" 2>/dev/null || true
    fi
}

start_runner_direct() {
    export HOME="${RUNNER_HOME}" USER="${RUNNER_USER}" LOGNAME="${RUNNER_USER}"
    exec "${RUNNER_HOME}/run.sh"
}

# Pod runs directly as the host UID/GID (no root). Skip all root-only setup.
if [[ "$(id -u)" != "0" ]]; then
    mkdir -p "${RUNNER_HOME}/.cache" "${TOOL_CACHE}" 2>/dev/null || true
    start_runner_direct
fi

# Fallback: container started as root. Align runner identity to the host cache
# owner, fix ownership, then drop privileges to that UID/GID.
host_owner=$(detect_host_owner)
case "${host_owner}" in
    *[!0-9:]* | *:*:* | :* | *: | 0:* | *:0)
        echo "Invalid ARC runner host owner: ${host_owner}" >&2
        exit 1
        ;;
esac

host_uid=${host_owner%%:*}
host_gid=${host_owner##*:}
printf '%s:%s\n' "${host_uid}" "${host_gid}" >"${OWNER_FILE}"

if ! getent group "${host_gid}" >/dev/null; then
    groupmod --non-unique --gid "${host_gid}" "${RUNNER_GROUP}"
else
    host_group=$(getent group "${host_gid}" | cut -d: -f1)
    usermod --gid "${host_group}" "${RUNNER_USER}"
fi

if [[ "$(id -u "${RUNNER_USER}")" != "${host_uid}" ]]; then
    usermod --non-unique --uid "${host_uid}" "${RUNNER_USER}"
fi

mkdir -p "${RUNNER_HOME}/.cache" "${TOOL_CACHE}"
chown -R "${host_uid}:${host_gid}" "${RUNNER_HOME}" "${TOOL_CACHE}"

trap cleanup EXIT
trap forward_signal TERM INT

setsid env \
    HOME="${RUNNER_HOME}" \
    USER="${RUNNER_USER}" \
    LOGNAME="${RUNNER_USER}" \
    setpriv --reuid "${host_uid}" --regid "${host_gid}" --init-groups "${RUNNER_HOME}/run.sh" &
runner_pid=$!
wait "${runner_pid}"
