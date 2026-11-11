#!/bin/bash
# PyTorch XPU tests
# Usage: ./run_ut.sh [group1 group2 ...]

set +e -x # not exit on error

# -------------------- Configuration --------------------
# PYTORCH_ROOT_DIR must be set in the environment
if [ -z "$PYTORCH_ROOT_DIR" ]; then
    echo "ERROR: PYTORCH_ROOT_DIR environment variable is not set."
    echo "Please set it to the root of your PyTorch repository before running this script."
    exit 1
fi
XML_OUTPUT_DIR="${XML_OUTPUT_DIR:-./test-reports}"        # where JUnit XML files go
XML_PREFIX="${XML_PREFIX:-op_p0_with_}"                   # prefix for XML filenames
mkdir -p "$XML_OUTPUT_DIR"

# Keep skip policy separate from test execution; resolve it independently of cwd.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || exit 1
SKIP_CONFIG="$SCRIPT_DIR/ut_skips.yaml"
if ! deselect_text="$(python - "$SKIP_CONFIG" <<'PY'
import sys

import yaml

with open(sys.argv[1], encoding="utf-8") as f:
    config = yaml.safe_load(f)

if not isinstance(config, dict) or set(config) != {"pytest_deselects"}:
    sys.exit("ERROR: P0 skip config must contain only pytest_deselects")
nodes = config["pytest_deselects"]
if not isinstance(nodes, list):
    sys.exit("ERROR: pytest_deselects must be a list")
for node in nodes:
    if (
        not isinstance(node, str)
        or not node.strip()
        or any(c in node for c in "\r\n\0")
    ):
        sys.exit(f"ERROR: Invalid P0 deselection: {node!r}")
for node in nodes:
    print(node)
PY
)"; then
    echo "ERROR: Could not load P0 skip config: $SKIP_CONFIG" >&2
    exit 1
fi
PYTEST_DESELECTS=()
if [[ -n "$deselect_text" ]]; then
    mapfile -t PYTEST_DESELECTS <<< "$deselect_text"
fi

# -------------------- Helper Functions --------------------
# Run pytest with a unique JUnit XML filename
# Arguments: test_file [optional: extra pytest args]
run_pytest() {
    local test_file="$1"
    shift
    local extra_args=("$@")
    local deselect test_name xml_file node_prefix
    node_prefix=$(realpath --relative-to="$PYTORCH_ROOT_DIR" "$PWD") || return 1
    if [[ "$node_prefix" == "." ]]; then
        node_prefix=""
    else
        node_prefix+="/"
    fi
    for deselect in "${PYTEST_DESELECTS[@]}"; do
        if [[ "$deselect" == "$test_file"::* ]]; then
            extra_args+=("--deselect=${node_prefix}${deselect}")
        fi
    done
    test_name=$(basename "$test_file" .py)
    xml_file="${XML_OUTPUT_DIR}/${XML_PREFIX}${test_name}_$(date +%s).xml"
    echo "Running pytest $test_file -> $xml_file"
    # A fixed root keeps node IDs consistent for tests in the nested XPU repo.
    pytest "$test_file" --junit-xml="$xml_file" "${extra_args[@]}" --rootdir="$PYTORCH_ROOT_DIR"
}

# -------------------- Test Groups --------------------
# Each group is an array of (directory, test_file, extra_args) tuples.
# For simplicity we use associative arrays with space-separated values.

declare -A inductor_tests=(
    ["test/inductor/test_aot_inductor.py"]="$PYTORCH_ROOT_DIR"
    ["test/inductor/test_aot_inductor_package.py"]="$PYTORCH_ROOT_DIR"
    ["test/inductor/test_kernel_benchmark.py"]="$PYTORCH_ROOT_DIR"
    ["test/inductor/test_mkldnn_pattern_matcher.py"]="$PYTORCH_ROOT_DIR"
    ["test/inductor/test_torchinductor.py"]="$PYTORCH_ROOT_DIR"
    ["test/inductor/test_torchinductor_opinfo.py"]="$PYTORCH_ROOT_DIR"
    ["test/inductor/test_triton_heuristics.py"]="$PYTORCH_ROOT_DIR"
    ["test/inductor/test_triton_kernels.py"]="$PYTORCH_ROOT_DIR"
    ["test/inductor/test_triton_syntax.py"]="$PYTORCH_ROOT_DIR"
    ["test/inductor/test_triton_wrapper.py"]="$PYTORCH_ROOT_DIR"
)

declare -A non_inductor_tests=(
    ["test/test_view_ops.py"]="$PYTORCH_ROOT_DIR"
    ["test/xpu/test_conv.py"]="$PYTORCH_ROOT_DIR"
    ["test/xpu/test_fusion.py"]="$PYTORCH_ROOT_DIR"
    ["test/xpu/test_gemm.py"]="$PYTORCH_ROOT_DIR"
    ["test/xpu/test_binary_ufuncs_xpu.py"]="$PYTORCH_ROOT_DIR/third_party/torch-xpu-ops"
    ["test/xpu/test_masked_xpu.py"]="$PYTORCH_ROOT_DIR/third_party/torch-xpu-ops"
    ["test/xpu/test_ops_xpu.py"]="$PYTORCH_ROOT_DIR/third_party/torch-xpu-ops"
    ["test/xpu/test_optim_xpu.py"]="$PYTORCH_ROOT_DIR/third_party/torch-xpu-ops"
    ["test/xpu/test_shape_ops_xpu.py"]="$PYTORCH_ROOT_DIR/third_party/torch-xpu-ops"
    ["test/xpu/test_unary_ufuncs_xpu.py"]="$PYTORCH_ROOT_DIR/third_party/torch-xpu-ops"
)

declare -A profiling_pytest_tests=(
    ["test/profiler/test_cpp_thread.py"]="$PYTORCH_ROOT_DIR"
    ["test/profiler/test_execution_trace.py"]="$PYTORCH_ROOT_DIR"
    ["test/profiler/test_memory_profiler.py"]="$PYTORCH_ROOT_DIR"
    ["test/profiler/test_profiler_tree.py"]="$PYTORCH_ROOT_DIR"
    ["test/profiling/correlation_id_mixed.py"]="$PYTORCH_ROOT_DIR/third_party/torch-xpu-ops"
    ["test/profiling/profile_partial_runtime_ops.py"]="$PYTORCH_ROOT_DIR/third_party/torch-xpu-ops"
    ["test/profiling/reproducer.missing.gpu.kernel.time.py"]="$PYTORCH_ROOT_DIR/third_party/torch-xpu-ops"
    ["test/profiling/time_precision_in_profile.py"]="$PYTORCH_ROOT_DIR/third_party/torch-xpu-ops"
    ["test/profiling/triton_xpu_ops_time.py"]="$PYTORCH_ROOT_DIR/third_party/torch-xpu-ops"
)

# Profiling scripts that are not pytest (run with python)
profiling_python_scripts=(
    "$PYTORCH_ROOT_DIR/third_party/torch-xpu-ops/test/profiling/rn50.py"
    "$PYTORCH_ROOT_DIR/third_party/torch-xpu-ops/test/profiling/llama.py"
)

declare -A distributed_tests=(
    ["test/distributed/_composable/fsdp/test_fully_shard_state_dict.py"]="$PYTORCH_ROOT_DIR"
    ["test/distributed/_composable/fsdp/test_fully_shard_frozen.py"]="$PYTORCH_ROOT_DIR"
    ["test/distributed/_composable/test_checkpoint.py"]="$PYTORCH_ROOT_DIR"
    ["test/distributed/_composable/test_contract.py"]="$PYTORCH_ROOT_DIR"
    ["test/distributed/_tools/test_fsdp2_mem_tracker.py"]="$PYTORCH_ROOT_DIR"
    ["test/distributed/_tools/test_mem_tracker.py"]="$PYTORCH_ROOT_DIR"
    ["test/distributed/_tools/test_memory_tracker.py"]="$PYTORCH_ROOT_DIR"
    ["test/distributed/fsdp/test_fsdp_apply.py"]="$PYTORCH_ROOT_DIR"
    ["test/distributed/fsdp/test_fsdp_checkpoint.py"]="$PYTORCH_ROOT_DIR"
    ["test/distributed/fsdp/test_fsdp_clip_grad_norm.py"]="$PYTORCH_ROOT_DIR"
    ["test/distributed/fsdp/test_fsdp_comm.py"]="$PYTORCH_ROOT_DIR"
    ["test/distributed/fsdp/test_fsdp_comm_hooks.py"]="$PYTORCH_ROOT_DIR"
    ["test/distributed/fsdp/test_fsdp_exec_order.py"]="$PYTORCH_ROOT_DIR"
    ["test/distributed/fsdp/test_fsdp_fine_tune.py"]="$PYTORCH_ROOT_DIR"
    ["test/distributed/fsdp/test_fsdp_flatten_params.py"]="$PYTORCH_ROOT_DIR"
    ["test/distributed/fsdp/test_fsdp_fx.py"]="$PYTORCH_ROOT_DIR"
    ["test/distributed/fsdp/test_fsdp_input.py"]="$PYTORCH_ROOT_DIR"
    ["test/distributed/fsdp/test_fsdp_misc.py"]="$PYTORCH_ROOT_DIR"
    ["test/distributed/fsdp/test_fsdp_multiple_forward.py"]="$PYTORCH_ROOT_DIR"
    ["test/distributed/fsdp/test_fsdp_multiple_wrapping.py"]="$PYTORCH_ROOT_DIR"
    ["test/distributed/fsdp/test_fsdp_sharded_grad_scaler.py"]="$PYTORCH_ROOT_DIR"
    ["test/distributed/fsdp/test_fsdp_uneven.py"]="$PYTORCH_ROOT_DIR"
    ["test/distributed/fsdp/test_fsdp_unshard_params.py"]="$PYTORCH_ROOT_DIR"
    ["test/distributed/fsdp/test_utils.py"]="$PYTORCH_ROOT_DIR"
    ["test/distributed/fsdp/test_wrap.py"]="$PYTORCH_ROOT_DIR"
    ["test/distributed/pipelining/test_backward.py"]="$PYTORCH_ROOT_DIR"
    ["test/distributed/pipelining/test_microbatch.py"]="$PYTORCH_ROOT_DIR"
    ["test/distributed/tensor/test_math_ops.py"]="$PYTORCH_ROOT_DIR"
    ["test/distributed/test_functional_api.py"]="$PYTORCH_ROOT_DIR"
    ["test/xpu/distributed/test_c10d_ops_xccl.py"]="$PYTORCH_ROOT_DIR/third_party/torch-xpu-ops"
    ["test/xpu/distributed/test_c10d_xccl.py"]="$PYTORCH_ROOT_DIR/third_party/torch-xpu-ops"
)

# -------------------- Execution --------------------
# Show help if requested
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    echo "Usage: $0 [group1 group2 ...]"
    echo "Available groups: inductor, non-inductor, profiling, distributed"
    echo "If no groups given, all groups are run."
    echo "All groups use the platform-independent skip config: $SKIP_CONFIG"
    exit 0
fi

# Determine which groups to run
run_all=false
if [ $# -eq 0 ]; then
    run_all=true
fi

run_group() {
    local group_name="$1"
    case $group_name in
        inductor)
            echo "=== Running Inductor tests ==="
            for test_file in "${!inductor_tests[@]}"; do
                (cd "${inductor_tests[$test_file]}" && run_pytest "$test_file")
            done
            ;;
        non-inductor)
            echo "=== Running Non-Inductor tests ==="
            for test_file in "${!non_inductor_tests[@]}"; do
                (cd "${non_inductor_tests[$test_file]}" && run_pytest "$test_file")
            done
            ;;
        profiling)
            echo "=== Running Profiling tests (pytest) ==="
            for test_file in "${!profiling_pytest_tests[@]}"; do
                (cd "${profiling_pytest_tests[$test_file]}" && run_pytest "$test_file")
            done
            echo "=== Running Profiling scripts (python) ==="
            for script in "${profiling_python_scripts[@]}"; do
                echo "Running $script"
                python "$script"
            done
            ;;
        distributed)
            echo "=== Running Distributed tests ==="
            # First show topology (if needed)
            xpu-smi topology -m
            for test_file in "${!distributed_tests[@]}"; do
                (cd "${distributed_tests[$test_file]}" && run_pytest "$test_file")
            done
            ;;
        *)
            echo "Unknown group: $group_name"
            exit 1
            ;;
    esac
}

if $run_all; then
    run_group inductor
    run_group non-inductor
    run_group profiling
    run_group distributed
else
    for group in "$@"; do
        run_group "$group"
    done
fi

echo "All requested test groups completed."
