#!/usr/bin/env bash
#
# Step -1 - Environment Setup
#
# Updates pytorch_opencode_env with the latest PyTorch XPU nightly build, then
# aligns the local pytorch and torch-xpu-ops source checkouts to the exact
# commits that produced the installed wheel.
#
# Usage:
#   bash update_env_from_nightly.sh
#
# Steps:
#   1. pip install --pre --upgrade torch pytorch-triton-xpu from PyPI XPU nightly index
#   2. Read torch.version.git_version -> pytorch SHA
#   3. Read pytorch/third_party/xpu.txt -> torch-xpu-ops SHA
#   4. Stash + checkout both source trees to their respective SHAs
#   5. Verify torch.xpu.is_available() and write session provenance JSON

set -euo pipefail

ENV_NAME="${ENV_NAME:-pytorch_opencode_env}"
INDEX_URL="${INDEX_URL:-https://download.pytorch.org/whl/nightly/xpu}"
PACKAGES="${PACKAGES:-torch pytorch-triton-xpu}"
PROVENANCE_FILE="${PROVENANCE_FILE:-${HOME}/.claude_classify_ut_session_provenance.json}"
PYTORCH_SRC="${PYTORCH_SRC:-/home/daisyden/upstream/pytorch}"
XPU_OPS_SRC="${XPU_OPS_SRC:-${PYTORCH_SRC}/third_party/torch-xpu-ops}"

CONDA_BASE="${CONDA_BASE:-/home/daisyden/miniforge3}"
if [[ -f "${CONDA_BASE}/bin/activate" ]]; then
    set +u
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/bin/activate" "${ENV_NAME}"
    set -u
else
    echo "ERROR: cannot find ${CONDA_BASE}/bin/activate to activate ${ENV_NAME}." >&2
    exit 3
fi

echo "[env  ] active: $(python -c 'import sys; print(sys.prefix)')"

echo "[pip  ] index=${INDEX_URL}"
echo "[pip  ] packages=${PACKAGES}"
PIP_EXTRA_FLAGS=()
if [[ "${PIP_FORCE_REINSTALL:-0}" == "1" ]]; then
    PIP_EXTRA_FLAGS+=(--force-reinstall --no-deps)
    echo "[pip  ] PIP_FORCE_REINSTALL=1 -> --force-reinstall --no-deps"
fi
pushd /tmp >/dev/null
python -m pip install --pre --upgrade --index-url "${INDEX_URL}" "${PIP_EXTRA_FLAGS[@]}" ${PACKAGES}
popd >/dev/null

echo "[verify] importing torch from /tmp..."
TORCH_INFO_JSON="$(mktemp)"
pushd /tmp >/dev/null
python - "${TORCH_INFO_JSON}" <<'PYEOF'
import json, sys, torch
info = {
    "torch_version": torch.__version__,
    "torch_git_version": getattr(torch.version, "git_version", None),
    "torch_file": torch.__file__,
    "xpu_available": torch.xpu.is_available(),
    "xpu_device_count": torch.xpu.device_count() if torch.xpu.is_available() else 0,
}
with open(sys.argv[1], "w") as f:
    json.dump(info, f)
print(json.dumps(info, indent=2))
if not torch.xpu.is_available():
    print("WARN: XPU not available after install.", file=sys.stderr)
PYEOF
popd >/dev/null

PYTORCH_SHA="$(python3 -c "import json; print(json.load(open('${TORCH_INFO_JSON}')).get('torch_git_version') or '')")"
TORCH_VERSION="$(python3 -c "import json; print(json.load(open('${TORCH_INFO_JSON}')).get('torch_version') or '')")"
XPU_AVAILABLE="$(python3 -c "import json; print(json.load(open('${TORCH_INFO_JSON}')).get('xpu_available'))")"
rm -f "${TORCH_INFO_JSON}"

if [[ -z "${PYTORCH_SHA}" ]]; then
    echo "ERROR: torch.version.git_version is empty; cannot align source tree." >&2
    exit 4
fi

if [[ ! -d "${PYTORCH_SRC}/.git" ]] && [[ ! -f "${PYTORCH_SRC}/.git" ]]; then
    echo "ERROR: ${PYTORCH_SRC} is not a git checkout." >&2
    exit 5
fi

align_repo() {
    local repo="$1" sha="$2" label="$3"
    echo "[align] ${label}: ${repo} -> ${sha}"
    pushd "${repo}" >/dev/null

    if ! git rev-parse --verify --quiet "${sha}^{commit}" >/dev/null; then
        echo "[align] ${label}: fetching origin (sha not present locally)..."
        git fetch --quiet origin || true
    fi
    if ! git rev-parse --verify --quiet "${sha}^{commit}" >/dev/null; then
        echo "ERROR: ${label}: SHA ${sha} not found after fetch." >&2
        popd >/dev/null
        return 6
    fi

    if [[ -n "$(git status --porcelain)" ]]; then
        local stash_msg="classify_ut auto-stash $(date -u +%Y%m%dT%H%M%SZ)"
        echo "[align] ${label}: dirty tree -> git stash --include-untracked -m '${stash_msg}'"
        git stash push --include-untracked --message "${stash_msg}" >/dev/null
        echo "[align] ${label}: stashed; restore later with: git -C ${repo} stash pop"
    fi

    local current
    current="$(git rev-parse HEAD)"
    if [[ "${current}" == "${sha}" ]]; then
        echo "[align] ${label}: already at ${sha}"
    else
        git checkout --quiet --detach "${sha}"
        echo "[align] ${label}: HEAD now $(git rev-parse HEAD)"
    fi
    popd >/dev/null
}

echo "[align] pytorch SHA = ${PYTORCH_SHA}"
align_repo "${PYTORCH_SRC}" "${PYTORCH_SHA}" "pytorch"

XPU_TXT="${PYTORCH_SRC}/third_party/xpu.txt"
if [[ ! -f "${XPU_TXT}" ]]; then
    echo "ERROR: ${XPU_TXT} missing after pytorch checkout." >&2
    exit 7
fi
XPU_OPS_SHA="$(tr -d '[:space:]' < "${XPU_TXT}")"
if [[ -z "${XPU_OPS_SHA}" ]]; then
    echo "ERROR: ${XPU_TXT} is empty." >&2
    exit 8
fi
echo "[align] torch-xpu-ops SHA (from ${XPU_TXT}) = ${XPU_OPS_SHA}"

if [[ ! -d "${XPU_OPS_SRC}/.git" ]] && [[ ! -f "${XPU_OPS_SRC}/.git" ]]; then
    echo "ERROR: ${XPU_OPS_SRC} is not a git checkout." >&2
    exit 9
fi
align_repo "${XPU_OPS_SRC}" "${XPU_OPS_SHA}" "torch-xpu-ops"

mkdir -p "$(dirname "${PROVENANCE_FILE}")"
python3 - "${TORCH_VERSION}" "${PYTORCH_SHA}" "${XPU_OPS_SHA}" "${INDEX_URL}" "${PACKAGES}" "${PROVENANCE_FILE}" "${XPU_AVAILABLE}" <<'PYEOF'
import json, sys, datetime, subprocess
torch_version, pytorch_sha, xpu_ops_sha, index_url, packages, out, xpu_avail = sys.argv[1:]
def _ver(pkg):
    try:
        for line in subprocess.check_output(["python","-m","pip","show",pkg], text=True).splitlines():
            if line.startswith("Version:"):
                return line.split(":",1)[1].strip()
    except Exception:
        pass
    return None
record = {
    "recorded_at": datetime.datetime.utcnow().isoformat() + "Z",
    "pypi_index_url": index_url,
    "packages_installed": packages.split(),
    "versions": {
        "torch": _ver("torch") or torch_version,
        "pytorch-triton-xpu": _ver("pytorch-triton-xpu"),
    },
    "source_alignment": {
        "pytorch_sha": pytorch_sha,
        "torch_xpu_ops_sha": xpu_ops_sha,
        "xpu_txt_path": "third_party/xpu.txt",
    },
    "xpu_available": xpu_avail == "True",
}
with open(out, "w") as f:
    json.dump(record, f, indent=2)
print(f"[prov ] wrote {out}")
print(json.dumps(record, indent=2))
PYEOF

echo "[done ] env + source-tree alignment complete. classify_ut may now begin."
