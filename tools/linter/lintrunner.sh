#!/usr/bin/env bash
# Run the lintrunner formatters and linters for one language group.
#
#   LINT_GROUP  all | general | python | cpp | cmake | shell | config   (default: all)
#   LINT_SCOPE  full | incremental                                      (default: full)
#   LINT_BASE   revision to diff against when LINT_SCOPE=incremental   (default: origin/main)
#   LINT_TAKE   comma-separated linter codes; overrides LINT_GROUP
#   LINT_APPLY  1 to apply formatter patches (lintrunner -a)
#   PYTORCH_ROOT  PyTorch source tree, required by CLANGTIDY (cpp group)
set -euo pipefail

declare -A LINT_GROUPS=(
    [general]="NEWLINE,SPACES,TABS,EXEC,COPYRIGHT,LINTRUNNER_VERSION"
    [python]="RUFF,PYFMT,TYPEIGNORE,NOQA,ROOT_LOGGING,TESTOWNERS,TEST_HAS_MAIN"
    [cpp]="CLANGFORMAT,CLANGTIDY,INCLUDE,C10_UNUSED,C10_NODISCARD,PYBIND11_INCLUDE,PYBIND11_SPECIALIZATION,CALL_ONCE,ONCE_FLAG,ATEN_CPU_GPU_AGNOSTIC,CUBINCLUDE,RAWCUDA,RAWCUDADEVICE"
    [cmake]="CMAKE"
    [shell]="SHELLCHECK,SHFMT"
    [config]="SYNTAX,GHA,NO_WORKFLOWS_ON_FORK"
)
LINT_GROUP="${LINT_GROUP:-all}"
LINT_SCOPE="${LINT_SCOPE:-full}"
LINT_BASE="${LINT_BASE:-origin/main}"
# Linter binaries, venv and report all live in the git-ignored .lintbin directory.
LINT_DIR="${PWD}/.lintbin"
LINT_JSON="${LINT_DIR}/lint.json"
mkdir -p "${LINT_DIR}"

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${PATH}:${HOME}/.local/bin"
fi
uv venv "${LINT_DIR}/venv" --python 3.12 --clear
export VIRTUAL_ENV="${LINT_DIR}/venv"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
uv pip install lintrunner pyyaml ruamel.yaml typing-extensions "setuptools<81" wheel

# Every linter must belong to exactly one group, otherwise CI would silently never run it.
mapfile -t ALL_CODES < <(python3 -c 'import tomllib; print("\n".join(l["code"] for l in tomllib.load(open(".lintrunner.toml", "rb"))["linter"]))')
GROUPED=",$(
    IFS=,
    echo "${LINT_GROUPS[*]}"
),"
for code in "${ALL_CODES[@]}"; do
    if [[ "${GROUPED}" != *",${code},"* ]]; then
        echo "::error file=tools/linter/lintrunner.sh::Linter ${code} is not assigned to a LINT_GROUPS entry"
        exit 1
    fi
done

ARGS=(--force-color "--tee-json=${LINT_JSON}")
if [[ -n "${LINT_TAKE:-}" ]]; then
    ARGS+=(--take "${LINT_TAKE}")
elif [[ "${LINT_GROUP}" != "all" ]]; then
    if [[ -z "${LINT_GROUPS[${LINT_GROUP}]+x}" ]]; then
        echo "Unknown LINT_GROUP '${LINT_GROUP}'. Valid: all ${!LINT_GROUPS[*]}" >&2
        exit 2
    fi
    ARGS+=(--take "${LINT_GROUPS[${LINT_GROUP}]}")
fi
case "${LINT_SCOPE}" in
    full) ARGS+=(--all-files) ;;
    incremental) ARGS+=(--merge-base-with "${LINT_BASE}") ;;
    *)
        echo "Unknown LINT_SCOPE '${LINT_SCOPE}'. Valid: full incremental" >&2
        exit 2
        ;;
esac
if [[ "${LINT_APPLY:-0}" == "1" ]]; then
    ARGS+=(-a)
fi

if [[ -d /tmp/.lintbin ]]; then
    cp -rn /tmp/.lintbin/. "${LINT_DIR}/" || true
fi
lintrunner init

rm -f "${LINT_JSON}"
RC=0
if ! lintrunner "${ARGS[@]}"; then
    REPRO="lintrunner ${ARGS[*]:2}"
    echo
    echo -e "\e[1m\e[36mReproduce locally with: ${REPRO/--merge-base-with ${LINT_BASE}/-m origin/main}\e[0m"
    echo -e "\e[1m\e[36mApply formatter patches with: lintrunner -a ${REPRO#lintrunner }\e[0m"
    RC=1
fi

# Surface findings as GitHub annotations.
jq --raw-output --arg root "${PWD}/" \
    '"::\(if .severity == "advice" or .severity == "disabled" then "warning" else .severity end) file=\((.path // "") | ltrimstr($root)),line=\(.line // 1),col=\(.char // 1),title=\(.code) \(.name)::" + ((.description // "") | gsub("\\n"; "%0A"))' \
    "${LINT_JSON}" || true

exit "${RC}"
