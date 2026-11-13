#!/usr/bin/env bash
# Shared CI caches (pip, uv, HF hub, torch hub, torchbench weights) and CA certificates.
# Every command is best effort: a cache or cert problem must never fail the job.
#   env                     export cache locations to GITHUB_ENV
#   certs                   refresh system CAs + certifi, export a CA bundle without expired certs
#   torchbench-restore DIR  copy cached model/input data into a torchbench checkout
#   torchbench-save DIR     mirror the checkout's model/input data back into the cache
#   prune                   drop superseded revisions and entries unused for CI_CACHE_MAX_AGE_DAYS
set -uo pipefail

CACHE_ROOT="${XDG_CACHE_HOME:-${HOME}/.cache}"
TB_CACHE="${CACHE_ROOT}/torchbench"
MAX_AGE_DAYS="${CI_CACHE_MAX_AGE_DAYS:-14}"

log() { echo "[ci_cache] $*"; }
warn() { echo "::warning::[ci_cache] $*"; }
set_env() {
    export "$1=$2"
    [ -n "${GITHUB_ENV:-}" ] && echo "$1=$2" >> "${GITHUB_ENV}"
}
privileged() {
    if [ "$(id -u)" = 0 ]; then "$@"
    elif sudo -n true 2>/dev/null; then sudo -n "$@"
    else return 1
    fi
}
# Run under an exclusive lock so concurrent jobs on one host don't race on the same cache dir.
with_lock() {
    local lock="$1"; shift
    mkdir -p "$(dirname "${lock}")"
    if command -v flock >/dev/null; then flock -w 1800 "${lock}" "$@"; else "$@"; fi
}
# Age-based pruning relies on atime; with noatime every entry looks stale.
atime_usable() {
    ! findmnt -no OPTIONS -T "${CACHE_ROOT}" 2>/dev/null | grep -qw noatime
}

cmd_env() {
    mkdir -p "${CACHE_ROOT}"/{pip,uv,torch} "${TB_CACHE}"
    set_env PIP_CACHE_DIR "${CACHE_ROOT}/pip"
    set_env UV_CACHE_DIR "${CACHE_ROOT}/uv"
    # cache and venv live on different mounts, hardlinking always falls back
    set_env UV_LINK_MODE copy
    set_env HF_HOME "${HF_HOME:-${CACHE_ROOT}/huggingface}"
    set_env TORCH_HOME "${TORCH_HOME:-${CACHE_ROOT}/torch}"
    # pip as root ignores a cache dir it doesn't own; job cleanup chowns it back to 1000
    chown "$(id -u):$(id -g)" "${PIP_CACHE_DIR}" 2>/dev/null || true
}

cmd_certs() {
    if command -v apt-get >/dev/null; then
        local pkgs=(ca-certificates)
        command -v rsync >/dev/null || pkgs+=(rsync)
        command -v openssl >/dev/null || pkgs+=(openssl)
        if privileged env DEBIAN_FRONTEND=noninteractive timeout 600 sh -c \
            "apt-get -o DPkg::Lock::Timeout=120 update -qq && \
             apt-get -o DPkg::Lock::Timeout=120 install -y -qq --no-install-recommends ${pkgs[*]}" >/dev/null; then
            privileged update-ca-certificates >/dev/null 2>&1 || true
        else
            warn "could not refresh system packages: ${pkgs[*]}"
        fi
    fi
    if command -v uv >/dev/null && [ -n "${VIRTUAL_ENV:-}" ]; then
        uv pip install -q -U certifi || warn "could not upgrade certifi"
    fi

    local bundle="${RUNNER_TEMP:-/tmp}/ci-ca-bundle.pem" srcs=() certifi_pem split
    [ -s /etc/ssl/certs/ca-certificates.crt ] && srcs+=(/etc/ssl/certs/ca-certificates.crt)
    certifi_pem="$(python -c 'import certifi; print(certifi.where())' 2>/dev/null)"
    [ -s "${certifi_pem}" ] && srcs+=("${certifi_pem}")
    if [ ${#srcs[@]} -eq 0 ]; then
        warn "no CA bundle found"; return 0
    fi
    split="$(mktemp -d)"
    cat "${srcs[@]}" | awk -v d="${split}" \
        '/BEGIN CERTIFICATE/{n++; f=sprintf("%s/%05d.pem", d, n)} f{print > f} /END CERTIFICATE/{close(f); f=""}'
    : > "${bundle}"
    for c in "${split}"/*.pem; do
        # an expired root kept next to its renewal can still be picked for the chain
        if ! command -v openssl >/dev/null || openssl x509 -noout -checkend 0 -in "${c}" >/dev/null 2>&1; then
            cat "${c}" >> "${bundle}"
        fi
    done
    rm -rf "${split}"
    log "CA bundle ${bundle}: $(grep -c 'BEGIN CERTIFICATE' "${bundle}") certs from ${srcs[*]}"
    set_env SSL_CERT_FILE "${bundle}"
    set_env REQUESTS_CA_BUNDLE "${bundle}"
    set_env CURL_CA_BUNDLE "${bundle}"

    local url
    for url in https://pypi.org/simple/ https://huggingface.co https://download.pytorch.org/whl/ \
               https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth; do
        python - "${url}" <<'EOF' || warn "TLS check failed for ${url} (host time: $(date -u))"
import sys, urllib.request
urllib.request.urlopen(urllib.request.Request(sys.argv[1], method="HEAD"), timeout=30)
EOF
    done
}

# torchbench keeps checkpoints in models/<m>/.data and S3 inputs in torchbenchmark/data/.data
tb_pairs() {
    local repo="$1" d m
    echo "${repo}/torchbenchmark/data/.data ${TB_CACHE}/data"
    for d in "${repo}"/torchbenchmark/models/*/; do
        m="$(basename "${d}")"
        echo "${d}.data ${TB_CACHE}/models/${m}"
    done
}

cmd_torchbench_restore() {
    local repo="$1" src dst n=0
    command -v rsync >/dev/null || { warn "rsync missing, skip torchbench restore"; return 0; }
    while read -r dst src; do
        [ -d "${src}" ] || continue
        mkdir -p "${dst}"
        rsync -a "${src}/" "${dst}/" && n=$((n + 1))
        touch "${src}"
    done < <(tb_pairs "${repo}")
    log "restored torchbench data for ${n} entries from ${TB_CACHE}"
}

tb_save() {
    local repo="$1" src dst n=0
    while read -r src dst; do
        [ -d "${src}" ] && [ -n "$(ls -A "${src}" 2>/dev/null)" ] || continue
        mkdir -p "${dst}"
        # --delete keeps only the files the current pin produces; --delay-updates swaps them in at the end
        rsync -a --delete --delay-updates "${src}/" "${dst}/" && n=$((n + 1))
        touch "${dst}"
    done < <(tb_pairs "${repo}")
    log "saved torchbench data for ${n} entries to ${TB_CACHE}"
}

cmd_torchbench_save() {
    command -v rsync >/dev/null || { warn "rsync missing, skip torchbench save"; return 0; }
    with_lock "${TB_CACHE}/.lock" bash -c "$(declare -f log tb_pairs tb_save); TB_CACHE='${TB_CACHE}' tb_save '$1'"
}

prune_all() {
    local stamp="${CACHE_ROOT}/.ci_cache_pruned" age="${MAX_AGE_DAYS}"
    if [ -n "$(find "${stamp}" -mmin -720 2>/dev/null)" ]; then
        log "pruned within the last 12h, skip"; return 0
    fi
    atime_usable || { warn "${CACHE_ROOT} is mounted noatime, only pruning superseded versions"; age=0; }
    log "before: $(du -sh "${CACHE_ROOT}" 2>/dev/null | cut -f1)"

    command -v uv >/dev/null && UV_CACHE_DIR="${CACHE_ROOT}/uv" timeout 600 uv cache prune 2>&1 | tail -1

    if [ "${age}" -gt 0 ]; then
        local d
        for d in "${CACHE_ROOT}/pip" "${TORCH_HOME:-${CACHE_ROOT}/torch}/hub/checkpoints"; do
            [ -d "${d}" ] || continue
            find "${d}" -type f -atime +"${age}" -delete
            find "${d}" -mindepth 1 -type d -empty -delete
        done
        # cached model dirs are touched on every restore/save
        [ -d "${TB_CACHE}/models" ] && find "${TB_CACHE}/models" -mindepth 1 -maxdepth 1 -type d \
            -mtime +"${age}" -print -exec rm -rf {} +
    fi

    HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}" python - "${age}" <<'EOF'
import sys, time
try:
    from huggingface_hub import scan_cache_dir
    info = scan_cache_dir()
except Exception as e:  # hub not installed or no cache yet
    print(f"[ci_cache] skip HF prune: {e}")
    sys.exit(0)
now, age = time.time(), int(sys.argv[1])
stale = []
for repo in info.repos:
    if age and repo.last_accessed < now - age * 86400:
        stale += [r.commit_hash for r in repo.revisions]
    else:
        # a revision no ref points to anymore was superseded by a newer upstream commit
        stale += [r.commit_hash for r in repo.revisions if not r.refs and r.last_modified < now - 86400]
if stale:
    strategy = info.delete_revisions(*stale)
    print(f"[ci_cache] HF: deleting {len(stale)} revisions, {strategy.expected_freed_size_str}")
    strategy.execute()
EOF
    touch "${stamp}"
    log "after: $(du -sh "${CACHE_ROOT}" 2>/dev/null | cut -f1)"
}

cmd_prune() {
    with_lock "${CACHE_ROOT}/.ci_cache.lock" bash -c \
        "$(declare -f log warn atime_usable prune_all); CACHE_ROOT='${CACHE_ROOT}' TB_CACHE='${TB_CACHE}' MAX_AGE_DAYS='${MAX_AGE_DAYS}' prune_all"
}

case "${1:-}" in
    env) cmd_env ;;
    certs) cmd_certs ;;
    torchbench-restore) cmd_torchbench_restore "$2" ;;
    torchbench-save) cmd_torchbench_save "$2" ;;
    prune) cmd_prune ;;
    *) echo "usage: $0 {env|certs|torchbench-restore DIR|torchbench-save DIR|prune}" >&2; exit 2 ;;
esac
exit 0
