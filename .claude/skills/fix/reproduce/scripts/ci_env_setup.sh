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
#   bash ci_env_setup.sh <pytorch_dir> [--py 3.10] [--outdir /tmp/ci_env] [--container-name pytorch_xpu_ci]
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

# --- defaults ---
PY_VERSION="3.10"
OUTDIR="${HOME}/ci_env_artifacts"
CONTAINER_NAME="pytorch_xpu_ci"

BUILD_ENVS=(
    "linux-noble-xpu-n-py3.10"
    "linux-noble-xpu-n-py3.10-client"
    "linux-noble-xpu-n-py3.11"
    "linux-noble-xpu-n-py3.11-client"
)

# --- parse args ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --py) PY_VERSION="$2"; shift 2 ;;
        --outdir) OUTDIR="$2"; shift 2 ;;
        --container-name) CONTAINER_NAME="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUTDIR"

# --- Resolve workflow ID dynamically ---
# Looks up the "xpu" workflow by name at runtime so the script stays valid
# even if the workflow is recreated or renamed.
# Manual lookup: gh api repos/pytorch/pytorch/actions/workflows --jq '.workflows[] | select(.name=="xpu") | {id,name,path}'
echo "[0/4] Resolving xpu workflow ID..."
WORKFLOW_ID=$(gh api repos/pytorch/pytorch/actions/workflows \
    --jq '.workflows[] | select(.name == "xpu") | .id' 2>/dev/null | head -1)
if [[ -z "$WORKFLOW_ID" ]]; then
    echo "ERROR: Could not find a workflow named 'xpu' in pytorch/pytorch." >&2
    echo "  Check: gh api repos/pytorch/pytorch/actions/workflows --jq '.workflows[].name'" >&2
    exit 1
fi
echo "  Workflow ID: $WORKFLOW_ID"

# --- Step 1: find latest run where all linux build-osdc jobs succeeded ---
echo "[1/4] Searching for latest xpu workflow run with successful linux builds..."

RUN_ID=""
PAGE=1
while [[ -z "$RUN_ID" ]]; do
    RUNS=$(gh api \
        "repos/pytorch/pytorch/actions/workflows/${WORKFLOW_ID}/runs?per_page=20&page=${PAGE}&status=completed" \
        --jq '.workflow_runs[] | {id: .id, sha: .head_sha, created_at: .created_at}')

    if [[ -z "$RUNS" ]]; then
        echo "ERROR: No more completed runs found." >&2
        exit 1
    fi

    while IFS= read -r run_json; do
        run_id=$(echo "$run_json" | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['id'])")
        run_sha=$(echo "$run_json" | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['sha'])")
        run_date=$(echo "$run_json" | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['created_at'])")

        # Check that all linux-*-build-osdc jobs succeeded
        build_conclusions=$(gh api \
            "repos/pytorch/pytorch/actions/runs/${run_id}/jobs?per_page=100" \
            --jq '.jobs[] | select(.name | test("linux.*build-osdc")) | .conclusion' 2>/dev/null || true)

        if [[ -z "$build_conclusions" ]]; then
            continue
        fi

        # All must be "success"
        failed=$(echo "$build_conclusions" | grep -v "^success$" || true)
        if [[ -z "$failed" ]]; then
            echo "  Found: run $run_id ($run_date, sha ${run_sha:0:12})"
            RUN_ID="$run_id"
            RUN_SHA="$run_sha"
            break
        fi
    done < <(echo "$RUNS" | python3 -c "
import json, sys
data = sys.stdin.read().strip()
# gh --jq returns one JSON object per line
for line in data.splitlines():
    line = line.strip()
    if line:
        print(line)
")

    PAGE=$((PAGE + 1))
    if [[ $PAGE -gt 10 ]]; then
        echo "ERROR: Could not find a qualifying run in the last 200 runs." >&2
        exit 1
    fi
done

echo "  Run ID: $RUN_ID  SHA: $RUN_SHA"

# --- Step 2: derive docker image tag ---
echo "[2/4] Computing CI docker image tag..."

# RUN_SHA is a commit SHA; get the commit's root tree SHA first
ROOT_TREE_SHA=$(gh api "repos/pytorch/pytorch/git/commits/${RUN_SHA}" \
    --jq '.tree.sha')
CI_TREE_SHA=$(gh api "repos/pytorch/pytorch/git/trees/${ROOT_TREE_SHA}" \
    --jq '.tree[] | select(.path == ".ci") | .sha')
DOCKER_HASH=$(gh api "repos/pytorch/pytorch/git/trees/${CI_TREE_SHA}" \
    --jq '.tree[] | select(.path == "docker") | .sha')

# Pick image name matching the requested python version and noble
# Default: noble py3 (covers py3.10 and py3.11 builds)
IMAGE_BASE="pytorch-linux-noble-xpu-n-py3"
DOCKER_IMAGE="ghcr.io/pytorch/ci-image:${IMAGE_BASE}-${DOCKER_HASH}"
echo "  Image: $DOCKER_IMAGE"

# Verify image exists
if ! docker manifest inspect "$DOCKER_IMAGE" &>/dev/null; then
    echo "ERROR: Docker image not found on ghcr: $DOCKER_IMAGE" >&2
    exit 1
fi

# --- Step 3: download wheel artifacts ---
echo "[3/4] Downloading wheel artifacts from run $RUN_ID..."

# S3 bucket pattern used by pytorch/pytorch GHA artifact uploads.
# Format: https://gha-artifacts.s3.amazonaws.com/<owner>/<repo>/<run_id>/<job_name>/artifacts.zip
# This is a well-known pytorch CI convention; if artifacts move, check the
# Actions run page on GitHub for updated URLs.
S3_BASE="https://gha-artifacts.s3.amazonaws.com/pytorch/pytorch/${RUN_ID}"
WHEELS_DIR="${OUTDIR}/wheels"
mkdir -p "$WHEELS_DIR"

DOWNLOADED=0
for build in "${BUILD_ENVS[@]}"; do
    # Filter to requested python version (e.g. 3.10 -> py3.10 in build name)
    if [[ "$build" != *"py${PY_VERSION}"* ]]; then
        continue
    fi

    url="${S3_BASE}/${build}/artifacts.zip"
    http_status=$(curl -s -o /dev/null -w "%{http_code}" -I -L "$url")
    if [[ "$http_status" != "200" ]]; then
        echo "  Skip: $build (S3 returned $http_status)"
        continue
    fi

    echo "  Downloading: $build..."
    zipfile="${OUTDIR}/${build}.zip"
    curl -sL --progress-bar "$url" -o "$zipfile"

    unzip -o -j "$zipfile" 'dist/*.whl' -d "${WHEELS_DIR}" 2>/dev/null || true
    rm -f "$zipfile"

    whl_count=$(find "${WHEELS_DIR}" -maxdepth 1 -name "*.whl" 2>/dev/null | wc -l)
    echo "  Extracted wheel(s) to ${WHEELS_DIR}/ (total so far: $whl_count)"
    DOWNLOADED=$((DOWNLOADED + 1))
done

if [[ $DOWNLOADED -eq 0 ]]; then
    echo "ERROR: No wheels downloaded. Check python version filter or S3 availability." >&2
    exit 1
fi

echo "  Total wheels downloaded: $DOWNLOADED"

# --- Step 4: pull image and print run command ---
echo "[4/4] Checking docker image..."
if docker image inspect "$DOCKER_IMAGE" &>/dev/null; then
    echo "  Already present locally, skipping pull."
else
    echo "  Not found locally, pulling (may take a while)..."
    docker pull "$DOCKER_IMAGE"
fi

echo ""
echo "================================================================"
echo "  CI environment ready"
echo "  Run ID  : $RUN_ID"
echo "  SHA     : $RUN_SHA"
echo "  Image   : $DOCKER_IMAGE"
echo "  Wheels  : $WHEELS_DIR"
echo "================================================================"
echo ""
echo "To start the container:"
echo ""

MOUNT_ARGS="-v ${WHEELS_DIR}:/workspace/wheels -v ${PYTORCH_DIR}:/workspace/pytorch"

echo "  docker run -it \\"
echo "    --name ${CONTAINER_NAME} \\"
echo "    --device=/dev/dri \\"
echo "    ${MOUNT_ARGS} \\"
echo "    ${DOCKER_IMAGE} \\"
echo "    /bin/bash"
echo ""
echo "Inside the container, install the wheel:"
echo "  pip install /workspace/wheels/*.whl --pre"
echo ""
echo "Then run your failing test from /workspace/pytorch."

# Write the run command to a file for easy copy-paste
RUN_SCRIPT="${OUTDIR}/run_container.sh"
cat > "$RUN_SCRIPT" <<EOF
#!/usr/bin/env bash
# Generated by ci_env_setup.sh — run $RUN_ID
docker run -it \\
  --name ${CONTAINER_NAME} \\
  --device=/dev/dri \\
  -v ${WHEELS_DIR}:/workspace/wheels \\
  -v ${PYTORCH_DIR}:/workspace/pytorch \\
  ${DOCKER_IMAGE} \\
  /bin/bash
EOF
chmod +x "$RUN_SCRIPT"
echo "Run command saved to: $RUN_SCRIPT"
