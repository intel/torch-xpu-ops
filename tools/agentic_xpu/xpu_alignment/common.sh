#!/usr/bin/env bash
# common.sh — sourced by daily_scan.sh and batch_scan.sh.
# Requires ENTRY_DIR to be set before sourcing.
#
# Sets:   SOURCE_SKILL_DIR  OPENCODE_BIN  WORKSPACE  XPU_PYTHON
# Defines: run_post_scan <run_dir> [rc]

# --- Locate source skill dir (prefer .github/skills/xpu-alignment/ in repo root, fallback to ENTRY_DIR) ---
_repo_root_skill="$(git -C "$ENTRY_DIR" rev-parse --show-toplevel 2>/dev/null || echo "")"
if [[ -n "$_repo_root_skill" && -f "$_repo_root_skill/.github/skills/xpu-alignment/SKILL.md" ]]; then
  SOURCE_SKILL_DIR="$_repo_root_skill/.github/skills/xpu-alignment"
else
  SOURCE_SKILL_DIR="${SOURCE_SKILL_DIR:-$ENTRY_DIR}"
fi
unset _repo_root_skill
[[ -f "$SOURCE_SKILL_DIR/SKILL.md" ]] || { echo "ERROR: SKILL.md not found in $SOURCE_SKILL_DIR" >&2; exit 1; }

# --- Resolve opencode ---
OPENCODE_BIN="${OPENCODE_BIN:-$(command -v opencode 2>/dev/null || echo "")}"
[[ -n "$OPENCODE_BIN" ]] || { echo "ERROR: opencode not found. Set OPENCODE_BIN or add to PATH." >&2; exit 1; }

# --- Load .env and token ---
ENV_FILE="${ENV_FILE:-$ENTRY_DIR/../.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi
[[ -z "${GH_TOKEN:-}" && -n "${GITHUB_TOKEN:-}" ]] && export GH_TOKEN="$GITHUB_TOKEN"

# --- Detect workspace (repo root via git, fallback to ENTRY_DIR) ---
if [[ -z "${WORKSPACE:-}" ]]; then
  WORKSPACE="$(git -C "$ENTRY_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$ENTRY_DIR")"
fi

# --- Refresh workspace-local XPU nightly ---
LOCAL_XPU_ENV_DIR_NAME=".conda-xpu-fix-alignment"
_venv="$WORKSPACE/$LOCAL_XPU_ENV_DIR_NAME"
_py="$_venv/bin/python"
_boot="$(command -v python3 2>/dev/null || true)"
[[ -x "$_boot" ]] || { echo "ERROR: python3 not found in PATH." >&2; exit 1; }
if [[ -n "${XPU_PYTHON:-}" && "$XPU_PYTHON" != "$_py" ]]; then
  echo "Ignoring externally provided XPU_PYTHON; using the workspace-local XPU interpreter"
fi
[[ -x "$_py" ]] || { echo "Creating XPU env at $_venv"; "$_boot" -m venv "$_venv"; }
echo "Refreshing XPU nightly in $_venv"
"$_py" -m pip install --upgrade pip -q
"$_py" -m pip install --upgrade --pre -q \
  --index-url https://download.pytorch.org/whl/nightly/xpu \
  --extra-index-url https://pypi.org/simple \
  torch torchvision torchaudio
"$_py" -c "import torch; print('torch:', torch.__version__); print('xpu:', torch.xpu.is_available())"
XPU_PYTHON="$_py"
unset _venv _py _boot

# --- Sync skill into workspace (.opencode/skills) so opencode can load it by name ---
_skill="$WORKSPACE/.opencode/skills/pytorch-cuda-fix-xpu-alignment"
mkdir -p "$_skill"
cp "$SOURCE_SKILL_DIR/SKILL.md" "$_skill/SKILL.md"
unset _skill

# --- Post-run: render report + audit ---
run_post_scan() {
  local run_dir="$1" rc="${2:-0}"
  # Reports are written under the run directory by the render scripts.
  local _run_name
  _run_name="$(basename "$run_dir")"
  local _report_path="$run_dir/reports/full_scan.md"

  if [[ -f "$run_dir/artifacts/candidate_ledger.jsonl" ]]; then
    "$XPU_PYTHON" "$ENTRY_DIR/render_issue_ready_report.py" "$run_dir"
    "$XPU_PYTHON" "$ENTRY_DIR/render_issue_drafts.py" "$run_dir"
  fi
  if [[ -f "$_report_path" ]]; then
    echo "Report: $_report_path"
    _audit="$ENTRY_DIR/audit_scan_report.sh"
    if [[ ! -x "$_audit" ]]; then
      echo "STATUS: PASSED"
    elif bash "$_audit" "$_report_path" "$run_dir/artifacts/candidate_ledger.jsonl"; then
      echo "STATUS: PASSED"
    else
      echo "STATUS: INCOMPLETE"
      rc=1
    fi
    unset _audit
  else
    echo "STATUS: NO REPORT GENERATED"
    rc=1
  fi
  return "$rc"
}
