#!/usr/bin/env bash
set -euo pipefail

RUNNER_HOME=${RUNNER_HOME:-/home/runner}
TOOL_CACHE=${ACTIONS_RUNNER_TOOL_CACHE:-/opt/hostedtoolcache}
OWNER_FILE=${ARC_RUNNER_OWNER_FILE:-/tmp/arc-runner-host-owner}

detect_owner() {
    if [[ -s "${OWNER_FILE}" ]]; then
        cat "${OWNER_FILE}"
        return
    fi

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

owner=$(detect_owner)
case "${owner}" in
    *[!0-9:]* | *:*:* | :* | *: | 0:* | *:0)
        echo "Invalid ARC runner cache owner: ${owner}" >&2
        exit 1
        ;;
esac

paths=()
[[ -e "${RUNNER_HOME}/.cache" ]] && paths+=("${RUNNER_HOME}/.cache")
[[ -e "${TOOL_CACHE}" ]] && paths+=("${TOOL_CACHE}")

if [[ ${#paths[@]} -eq 0 ]]; then
    exit 0
fi

if [[ ${EUID} -eq 0 ]]; then
    chown -R "${owner}" "${paths[@]}"
else
    sudo -n chown -R "${owner}" "${paths[@]}"
fi
