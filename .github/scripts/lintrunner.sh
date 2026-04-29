#!/usr/bin/env bash
# Run lintrunner with consistent setup, structured output, and a GitHub step summary.
#
# Environment variables (all optional):
#   LINT_PYTHON_VERSION       Python interpreter for the lint venv (default: 3.10)
#   LINT_VENV_DIR             Directory for the venv                (default: lint)
#   LINT_CACHE_DIR            Cache for downloaded linter binaries  (default: /tmp/.lintbin)
#   ADDITIONAL_LINTRUNNER_ARGS  Extra args forwarded to `lintrunner`
#   CLANG                     If "1", run clang-tidy build-file generator
#   GITHUB_STEP_SUMMARY       Auto-set in GitHub Actions; appended to if present.

set -euo pipefail

LINT_PYTHON_VERSION="${LINT_PYTHON_VERSION:-3.10}"
LINT_VENV_DIR="${LINT_VENV_DIR:-lint}"
LINT_CACHE_DIR="${LINT_CACHE_DIR:-/tmp/.lintbin}"
ADDITIONAL_LINTRUNNER_ARGS="${ADDITIONAL_LINTRUNNER_ARGS:-}"

log()   { printf '\033[1;36m[lint]\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33m[lint][warn]\033[0m %s\n' "$*" >&2; }
error() { printf '\033[1;31m[lint][error]\033[0m %s\n' "$*" >&2; }

# Append to the GitHub step summary if available.
summary() {
    if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
        printf '%s\n' "$*" >> "${GITHUB_STEP_SUMMARY}"
    fi
}

# ── Bootstrap uv ─────────────────────────────────────────────────────────────
if ! command -v uv > /dev/null 2>&1; then
    log "Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi

# ── Create venv ──────────────────────────────────────────────────────────────
log "Creating lint venv (python=${LINT_PYTHON_VERSION}) at ${LINT_VENV_DIR}"
uv venv "${LINT_VENV_DIR}" --python "${LINT_PYTHON_VERSION}" --clear
# shellcheck disable=SC1091
source "${LINT_VENV_DIR}/bin/activate"

log "Installing base dependencies"
uv pip install --quiet -U pip 'setuptools==81.0.0' wheel pyyaml typing-extensions ruamel.yaml

if ! command -v lintrunner > /dev/null 2>&1; then
    log "Installing lintrunner"
    uv pip install --quiet lintrunner
fi

# ── Restore linter binary cache (best-effort) ───────────────────────────────
if [[ -d "${LINT_CACHE_DIR}" ]]; then
    log "Restoring linter cache from ${LINT_CACHE_DIR}"
    cp -r "${LINT_CACHE_DIR}" .lintbin || warn "failed to restore lint cache (will redownload)"
fi

# Suppress noisy shellcheck rules across the repository.
export SHELLCHECK_OPTS="${SHELLCHECK_OPTS:--e SC2154 -e SC2086 -e SC1091 -e SC2046 -e SC2076 -e SC2034 -e SC2190}"

# ── Initialize linters ──────────────────────────────────────────────────────
log "Running 'lintrunner init'"
if ! lintrunner init; then
    error "lintrunner init failed"
    summary "❌ \`lintrunner init\` failed. Check job logs."
    exit 1
fi

# ── clang-tidy build prep (optional) ────────────────────────────────────────
if [[ "${CLANG:-0}" == "1" ]]; then
    gen_script="third_party/torch-xpu-ops/tools/linter/clang_tidy/generate_build_files.py"
    if [[ -e "${gen_script}" ]]; then
        log "Generating clang-tidy build files"
        python3 "${gen_script}"
    else
        warn "CLANG=1 but ${gen_script} not found; run from the pytorch source folder"
    fi
fi

# ── Run lintrunner ──────────────────────────────────────────────────────────
rm -f lint.json

log "Running lintrunner ${ADDITIONAL_LINTRUNNER_ARGS}"
RC=0
# shellcheck disable=SC2086
lintrunner --force-color --tee-json=lint.json ${ADDITIONAL_LINTRUNNER_ARGS} || RC=$?

if [[ ${RC} -ne 0 ]]; then
    cat <<'EOF' >&2

────────────────────────────────────────────────────────────────────────
Lint check failed. To reproduce locally:
  lintrunner -m origin/main
To auto-fix:
  lintrunner -a
See https://github.com/pytorch/pytorch/wiki/lintrunner for setup.
────────────────────────────────────────────────────────────────────────
EOF
fi

# ── Emit GitHub annotations from lint.json ──────────────────────────────────
if [[ -s lint.json ]] && command -v jq > /dev/null 2>&1; then
    jq --raw-output '
        "::\(if .severity == "advice" or .severity == "disabled" then "warning" else .severity end) " +
        "file=\(.path),line=\(.line),col=\(.char),title=\(.code) \(.name)::" +
        (.description | gsub("\\n"; "%0A"))
    ' lint.json || true
fi

# ── Step summary ────────────────────────────────────────────────────────────
if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    summary "## Lint results"
    summary ""
    if [[ -s lint.json ]] && command -v jq > /dev/null 2>&1; then
        total=$(wc -l < lint.json | tr -d ' ')
        errors=$(jq -r 'select(.severity == "error") | .code' lint.json 2>/dev/null | wc -l | tr -d ' ')
        warnings=$(jq -r 'select(.severity == "warning" or .severity == "advice") | .code' lint.json 2>/dev/null | wc -l | tr -d ' ')

        summary "| Status | Count |"
        summary "|--------|-------|"
        summary "| Errors | ${errors} |"
        summary "| Warnings | ${warnings} |"
        summary "| Total findings | ${total} |"
        summary ""

        if [[ "${total}" != "0" ]]; then
            summary "<details><summary>Top findings</summary>"
            summary ""
            summary '| Severity | File | Line | Code | Message |'
            summary '|----------|------|------|------|---------|'
            jq -r --arg sep '|' '
                [.severity, .path, (.line|tostring), .code, ((.name // "") + ": " + ((.description // "") | gsub("\\n"; " ") | .[0:120]))]
                | "| " + join(" | ") + " |"
            ' lint.json 2>/dev/null | head -n 50 >> "${GITHUB_STEP_SUMMARY}" || true
            summary ""
            summary "</details>"
        fi
    else
        summary "No lint findings."
    fi

    if [[ ${RC} -eq 0 ]]; then
        summary ""
        summary "✅ Lint passed."
    else
        summary ""
        summary "❌ Lint failed — see annotations and the **Top findings** table above."
    fi
fi

exit ${RC}
