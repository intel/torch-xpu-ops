#!/usr/bin/env bash
# ci_env_setup.sh — Stage 3 CI environment alignment for fix/reproduce
#
# Finds the latest pytorch/pytorch xpu workflow run where all linux build jobs
# succeeded, downloads the wheel artifacts, pulls the matching CI docker image,
# and drops you into a container ready to run the failing test.
#
# Called from fix/reproduce Stage 3 after a pytorch checkout already exists
# from Stage 2. pytorch_dir is passed as the first positional argument.
#
# Usage:
#   bash ci_env_setup.sh <pytorch_dir> [--py 3.10] [--outdir agent_space_xpu/ci_env] [--container-name pytorch_xpu_ci]
#
# Requirements: gh CLI (authenticated), docker, curl, python3

set -euo pipefail

# --- first positional arg: pytorch_dir ---
if [[ $# -lt 1 || "$1" == --* ]]; then
    echo "Usage: bash ci_env_setup.sh <pytorch_dir> [--py 3.10] [--outdir <dir>] [--container-name <name>]" >&2
    exit 1
fi
PYTORCH_DIR="$1"; shift

if [[ ! -d "$PYTORCH_DIR" ]]; then
    echo "ERROR: pytorch_dir '$PYTORCH_DIR' does not exist." >&2
    exit 1
fi

# Reject whitespace in path to keep the generated docker run command valid.
if [[ "$PYTORCH_DIR" =~ [[:space:]] ]]; then
    echo "ERROR: pytorch_dir path contains whitespace; not supported: '$PYTORCH_DIR'" >&2
    exit 1
fi

# --- defaults ---
PY_VERSION="3.10"
OUTDIR="agent_space_xpu/ci_env"
CONTAINER_NAME="pytorch_xpu_ci_$(date +%s)"
CONTAINER_NAME_IS_DEFAULT=1

# xpu.yml (as of pytorch/pytorch main) only defines py3.10 linux builds.
# Each entry maps a build-environment name to its docker-image-name suffix.
# If xpu.yml adds more build envs (e.g. py3.11), extend this table.
declare -A BUILD_ENV_IMAGE=(
    ["linux-noble-xpu-n-py3.10"]="pytorch-linux-noble-xpu-n-py3"
    ["linux-noble-xpu-n-py3.10-client"]="pytorch-linux-noble-xpu-n-py3-client"
    ["linux-jammy-xpu-n-1-py3.10"]="pytorch-linux-jammy-xpu-n-1-py3"
)

# --- parse args ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --py) PY_VERSION="$2"; shift 2 ;;
        --outdir) OUTDIR="$2"; shift 2 ;;
        --container-name) CONTAINER_NAME="$2"; CONTAINER_NAME_IS_DEFAULT=0; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ "$OUTDIR" =~ [[:space:]] ]]; then
    echo "ERROR: --outdir path contains whitespace; not supported: '$OUTDIR'" >&2
    exit 1
fi

mkdir -p "$OUTDIR"

# --- Resolve workflow ID dynamically ---
# Looks up the "xpu" workflow by name at runtime so the script stays valid
# even if the workflow is recreated or renamed.
# Manual lookup: gh api repos/pytorch/pytorch/actions/workflows --jq '.workflows[] | select(.name=="xpu") | {id,name,path}'
echo "[0/4] Resolving xpu workflow ID..."
WORKFLOW_ID=$(gh api "repos/pytorch/pytorch/actions/workflows?per_page=100" --paginate \
    --jq '.workflows[] | select(.name == "xpu") | .id' 2>/dev/null | head -1)
if [[ -z "$WORKFLOW_ID" ]]; then
    echo "ERROR: Could not find a workflow named 'xpu' in pytorch/pytorch." >&2
    echo "  Check: gh api repos/pytorch/pytorch/actions/workflows --paginate --jq '.workflows[].name'" >&2
    exit 1
fi
echo "  Workflow ID: $WORKFLOW_ID"

# --- Step 1: find latest run where all linux build jobs succeeded ---
# Real job name pattern in the xpu workflow is "<build-env> / build"
# (e.g. "linux-noble-xpu-n-py3.10 / build"). "linux-*-build-osdc" is not a
# real pattern; do NOT reintroduce it.
echo "[1/4] Searching for latest xpu workflow run with successful linux builds..."

RUN_ID=""
PAGE=1
while [[ -z "$RUN_ID" ]]; do
    # Emit one TSV row per run: id\tsha\tcreated_at. Avoids per-row python3.
    RUNS_TSV=$(gh api \
        "repos/pytorch/pytorch/actions/workflows/${WORKFLOW_ID}/runs?per_page=20&page=${PAGE}&status=completed" \
        --jq '.workflow_runs[] | [.id, .head_sha, .created_at] | @tsv')

    if [[ -z "$RUNS_TSV" ]]; then
        echo "ERROR: No more completed runs found." >&2
        exit 1
    fi

    while IFS=$'\t' read -r run_id run_sha run_date; do
        [[ -z "$run_id" ]] && continue
        # Check that all linux */ build jobs succeeded.
        build_conclusions=$(gh api \
            "repos/pytorch/pytorch/actions/runs/${run_id}/jobs?per_page=100" \
            --paginate \
            --jq '.jobs[] | select(.name | test("^linux.*/ build$")) | .conclusion' 2>/dev/null || true)

        if [[ -z "$build_conclusions" ]]; then
            continue
        fi

        # All must be "success".
        failed=$(echo "$build_conclusions" | grep -v "^success$" || true)
        if [[ -z "$failed" ]]; then
            echo "  Found: run $run_id ($run_date, sha ${run_sha:0:12})"
            RUN_ID="$run_id"
            RUN_SHA="$run_sha"
            break
        fi
    done <<< "$RUNS_TSV"

    PAGE=$((PAGE + 1))
    if [[ $PAGE -gt 10 ]]; then
        echo "ERROR: Could not find a qualifying run in the last 200 runs." >&2
        exit 1
    fi
done

echo "  Run ID: $RUN_ID  SHA: $RUN_SHA"

# --- Step 2: compute docker image tag suffix (ci-docker-hash) ---
# The hash is git rev-parse HEAD:.ci/docker of the run's commit. Match
# _runner-determinator.yml's "Compute .ci/docker tree hash" step exactly.
echo "[2/4] Computing CI docker image tag..."

ROOT_TREE_SHA=$(gh api "repos/pytorch/pytorch/git/commits/${RUN_SHA}" \
    --jq '.tree.sha')
CI_TREE_SHA=$(gh api "repos/pytorch/pytorch/git/trees/${ROOT_TREE_SHA}" \
    --jq '.tree[] | select(.path == ".ci") | .sha')
DOCKER_HASH=$(gh api "repos/pytorch/pytorch/git/trees/${CI_TREE_SHA}" \
    --jq '.tree[] | select(.path == "docker") | .sha')

echo "  ci-docker-hash: $DOCKER_HASH"

# --- Step 3: download wheel artifacts + verify docker image per build env ---
# S3 bucket pattern used by pytorch/pytorch GHA artifact uploads.
# Format: https://gha-artifacts.s3.amazonaws.com/pytorch/pytorch/<run_id>/<build-env>/artifacts.zip
# Confirmed from a real xpu build job log; if artifacts move, check the
# Actions run page on GitHub for the updated URL.
S3_BASE="https://gha-artifacts.s3.amazonaws.com/pytorch/pytorch/${RUN_ID}"
WHEELS_DIR="${OUTDIR}/wheels"
mkdir -p "$WHEELS_DIR"

echo "[3/4] Downloading wheel artifacts from run $RUN_ID..."

DOCKER_IMAGE=""
DOWNLOADED=0
# Iterate keys in sorted order for a deterministic DOCKER_IMAGE selection.
# Otherwise Bash 4 associative arrays iterate in unspecified order and the
# same input can pick different build envs across runs.
mapfile -t SORTED_ENVS < <(printf '%s\n' "${!BUILD_ENV_IMAGE[@]}" | sort)
for build_env in "${SORTED_ENVS[@]}"; do
    # Filter to requested python version (e.g. "3.10" matches "py3.10" but
    # not "py3.11", and stops false-matching if we extend the table later).
    if [[ "$build_env" != *"py${PY_VERSION}"* ]]; then
        continue
    fi
    # Skip client build for the "full runtime" wheel selection - its wheel
    # is a subset. Users needing the client image should extend as needed.
    if [[ "$build_env" == *-client ]]; then
        continue
    fi

    url="${S3_BASE}/${build_env}/artifacts.zip"
    http_status=$(curl -s -o /dev/null -w "%{http_code}" -I -L "$url")
    if [[ "$http_status" != "200" ]]; then
        echo "  Skip: $build_env (S3 returned $http_status)"
        continue
    fi

    echo "  Downloading: $build_env..."
    zipfile="${OUTDIR}/${build_env}.zip"
    curl -sL --progress-bar "$url" -o "$zipfile"

    before_count=$(find "${WHEELS_DIR}" -maxdepth 1 -name "*.whl" | wc -l)
    # Try dist/*.whl first (upstream layout). unzip exit code 11 means
    # "no files matched pattern" - that is the only case where the
    # fallback should run; any other non-zero exit indicates a real
    # unzip error and must abort.
    set +e
    unzip -o -j "$zipfile" 'dist/*.whl' -d "${WHEELS_DIR}"
    unzip_rc=$?
    set -e
    if [[ $unzip_rc -eq 11 ]]; then
        # dist/ path not present; extract any *.whl.
        unzip -o -j "$zipfile" '*.whl' -d "${WHEELS_DIR}"
    elif [[ $unzip_rc -ne 0 ]]; then
        echo "ERROR: unzip failed for $zipfile (exit $unzip_rc)" >&2
        exit 1
    fi
    rm -f "$zipfile"

    after_count=$(find "${WHEELS_DIR}" -maxdepth 1 -name "*.whl" | wc -l)
    if [[ "$after_count" -le "$before_count" ]]; then
        echo "ERROR: unzip completed for $build_env but produced no .whl in $WHEELS_DIR" >&2
        exit 1
    fi
    echo "  Extracted wheel(s) to ${WHEELS_DIR}/ (total so far: $after_count)"
    DOWNLOADED=$((DOWNLOADED + 1))

    # Docker image tag for this build env. All qualifying builds in xpu.yml
    # share the same ci-docker-hash for a given run, so recording the first
    # (in sorted order) is enough for the container we're about to launch.
    if [[ -z "$DOCKER_IMAGE" ]]; then
        image_suffix="${BUILD_ENV_IMAGE[$build_env]}"
        DOCKER_IMAGE="ghcr.io/pytorch/ci-image:${image_suffix}-${DOCKER_HASH}"
    fi
done

if [[ $DOWNLOADED -eq 0 ]]; then
    echo "ERROR: No wheels downloaded. Check python version filter or S3 availability." >&2
    exit 1
fi

if [[ -z "$DOCKER_IMAGE" ]]; then
    echo "ERROR: Did not resolve a docker image tag; downloaded wheels but no matching build env." >&2
    exit 1
fi
echo "  Total wheels downloaded: $DOWNLOADED"
echo "  Image: $DOCKER_IMAGE"

# Verify image exists on the registry before printing the run command.
if ! docker manifest inspect "$DOCKER_IMAGE" &>/dev/null; then
    echo "ERROR: Docker image not found on ghcr: $DOCKER_IMAGE" >&2
    exit 1
fi

# --- Step 4: pull image and print run command ---
echo "[4/4] Checking docker image..."
if docker image inspect "$DOCKER_IMAGE" &>/dev/null; then
    echo "  Already present locally, skipping pull."
else
    echo "  Not found locally, pulling (may take a while)..."
    docker pull "$DOCKER_IMAGE"
fi

# If a container with the requested name already exists AND we are using
# the auto-generated default name, remove it so the next `docker run
# --name` does not collide. When the user supplied `--container-name`
# explicitly, do NOT touch a pre-existing container - abort with a clear
# error instead, since removing it may destroy user state.
if docker container inspect "$CONTAINER_NAME" &>/dev/null; then
    if [[ "$CONTAINER_NAME_IS_DEFAULT" -eq 1 ]]; then
        echo "  Removing pre-existing container: $CONTAINER_NAME"
        docker rm -f "$CONTAINER_NAME" >/dev/null
    else
        echo "ERROR: Container '$CONTAINER_NAME' already exists." >&2
        echo "  Remove it manually (docker rm -f $CONTAINER_NAME) or pick a different --container-name." >&2
        exit 1
    fi
fi

echo ""
echo "================================================================"
echo "  CI environment ready"
echo "  Run ID    : $RUN_ID"
echo "  SHA       : $RUN_SHA"
echo "  Image     : $DOCKER_IMAGE"
echo "  Wheels    : $WHEELS_DIR"
echo "  Container : $CONTAINER_NAME"
echo "================================================================"
echo ""
echo "To start the container:"
echo ""
echo "  docker run -it \\"
echo "    --name \"${CONTAINER_NAME}\" \\"
echo "    --device=/dev/dri \\"
echo "    -v \"${WHEELS_DIR}:/workspace/wheels\" \\"
echo "    -v \"${PYTORCH_DIR}:/workspace/pytorch\" \\"
echo "    \"${DOCKER_IMAGE}\" \\"
echo "    /bin/bash"
echo ""
echo "Inside the container, install the wheel:"
echo "  pip install /workspace/wheels/*.whl --pre"
echo ""
echo "Then run your failing test from /workspace/pytorch."

# Write the run command to a file for easy copy-paste. Quote every
# interpolated path so whitespace in the paths would not break the
# generated command (defence-in-depth; script already rejects such paths).
RUN_SCRIPT="${OUTDIR}/run_container.sh"
cat > "$RUN_SCRIPT" <<EOF
#!/usr/bin/env bash
# Generated by ci_env_setup.sh — run $RUN_ID
docker run -it \\
  --name "${CONTAINER_NAME}" \\
  --device=/dev/dri \\
  -v "${WHEELS_DIR}:/workspace/wheels" \\
  -v "${PYTORCH_DIR}:/workspace/pytorch" \\
  "${DOCKER_IMAGE}" \\
  /bin/bash
EOF
chmod +x "$RUN_SCRIPT"
echo "Run command saved to: $RUN_SCRIPT"
