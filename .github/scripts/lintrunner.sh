#!/usr/bin/env bash
set -ex

# Creat a venv for lint check
if ! uv --help > /dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$PATH:$HOME/.local/bin"
fi
uv venv lint --python 3.12 --clear
source lint/bin/activate
uv pip install -U pip setuptools wheel

# Use uv to speed up lintrunner init
uv pip install ruamel.yaml

CACHE_DIRECTORY="/tmp/.lintbin"
# Try to recover the cached binaries
if [[ -d "${CACHE_DIRECTORY}" ]]; then
    # It's ok to fail this as lintrunner init would download these binaries
    # again if they do not exist
    cp -r "${CACHE_DIRECTORY}" . || true
fi

# if lintrunner is not installed, install it
if ! command -v lintrunner &> /dev/null; then
    uv pip install lintrunner
fi

# Ignoring errors in one specific run
export SHELLCHECK_OPTS="-e SC2154 -e SC2086 -e SC1091 -e SC2046 -e SC2076 -e SC2034 -e SC2190"

# This has already been cached in the docker image
lintrunner init 2> /dev/null

# Do build steps necessary for linters
if [[ "${CLANG}" == "1" ]]; then
    if [[ -e "third_party/torch-xpu-ops/tools/linter/clang_tidy/generate_build_files.py" ]];then
        python3 third_party/torch-xpu-ops/tools/linter/clang_tidy/generate_build_files.py
    else
        echo "Please run the checker under pytorch source code folder"
    fi
fi
#uv tools.generate_torch_version --is_debug=false
#uv tools.pyi.gen_pyi \
#    --native-functions-path aten/src/ATen/native/native_functions.yaml \
#    --tags-path aten/src/ATen/native/tags.yaml \
#    --deprecated-functions-path "tools/autograd/deprecated.yaml"
#python3 torch/utils/data/datapipes/gen_pyi.py

RC=0
# Run lintrunner on all files
if ! lintrunner --force-color --tee-json=lint.json ${ADDITIONAL_LINTRUNNER_ARGS} 2> /dev/null; then
    echo ""
    echo -e "\e[1m\e[36mYou can reproduce these results locally by using \`lintrunner -m origin/main\`. (If you don't get the same results, run \'lintrunner init\' to update your local linter)\e[0m"
    echo -e "\e[1m\e[36mSee https://github.com/pytorch/pytorch/wiki/lintrunner for setup instructions. To apply suggested patches automatically, use the -a flag. Before pushing another commit,\e[0m"
    echo -e "\e[1m\e[36mplease verify locally and ensure everything passes.\e[0m"
    RC=1
fi

# Use jq to massage the JSON lint output into GitHub Actions workflow commands.
jq --raw-output \
    '"::\(if .severity == "advice" or .severity == "disabled" then "warning" else .severity end) file=\(.path),line=\(.line),col=\(.char),title=\(.code) \(.name)::" + (.description | gsub("\\n"; "%0A"))' \
    lint.json || true

exit $RC
