#!/bin/bash
# Verification script for pytorch-agent issues
# Runs all verifications and saves results

# Setup env
source ~/intel/oneapi/setvars.sh 2>/dev/null
cd ~/pytorch
source .venv/bin/activate

RESULTS_DIR=~/torch-xpu-ops/verification_results
mkdir -p "$RESULTS_DIR"

ORIG_XPU_TXT=$(cat third_party/xpu.txt)
echo "Original xpu.txt: $ORIG_XPU_TXT"

restore_xpu() {
    echo "$ORIG_XPU_TXT" > ~/pytorch/third_party/xpu.txt
    cd ~/pytorch && git submodule update --init third_party/torch-xpu-ops 2>&1 | tail -3
}

run_test() {
    local issue=$1
    local desc=$2
    local cmd=$3

    echo ""
    echo "================================================================"
    echo "=== Issue #$issue: $desc ==="
    echo "=== Command: $cmd ==="
    echo "================================================================"

    cd ~/pytorch
    local outfile="$RESULTS_DIR/issue_${issue}.log"
    echo "Command: $cmd" > "$outfile"
    echo "Time: $(date)" >> "$outfile"
    echo "---" >> "$outfile"

    eval "$cmd" >> "$outfile" 2>&1
    local rc=$?

    if [ $rc -eq 0 ]; then
        echo "RESULT: PASS (rc=$rc)"
        echo "RESULT: PASS" >> "$outfile"
    else
        echo "RESULT: FAIL (rc=$rc)"
        echo "RESULT: FAIL (rc=$rc)" >> "$outfile"
    fi
    tail -10 "$outfile"
}

switch_copilot_branch() {
    local branch=$1
    local issue=$2
    echo "--- Switching to copilot branch: $branch for #$issue ---"
    cd ~/torch-xpu-ops
    local sha=$(git rev-parse "origin/$branch" 2>&1)
    if [ $? -ne 0 ]; then
        echo "ERROR: branch origin/$branch not found"
        return 1
    fi
    echo "$sha" > ~/pytorch/third_party/xpu.txt
    cd ~/pytorch
    git submodule update --init third_party/torch-xpu-ops 2>&1 | tail -3
    echo "xpu.txt now: $(cat third_party/xpu.txt)"
    return 0
}

check_cpp_rebuild() {
    local branch=$1
    cd ~/torch-xpu-ops
    local cpp_changes=$(git diff --name-only origin/main..."origin/$branch" -- '*.cpp' '*.h' '*.hpp' '*.sycl' 2>/dev/null)
    if [ -n "$cpp_changes" ]; then
        echo "C++ changes detected in $branch:"
        echo "$cpp_changes"
        echo "--- Rebuilding PyTorch ---"
        cd ~/pytorch
        USE_XPU=1 python setup.py develop 2>&1 | tail -5
    else
        echo "No C++ changes, skipping rebuild"
    fi
}

echo ""
echo "========================================"
echo "=== COPILOT PR VERIFICATIONS ==="
echo "========================================"

# --- #1951 ---
if switch_copilot_branch "copilot/fix-functionality-issues-testcommon" 1951; then
    check_cpp_rebuild "copilot/fix-functionality-issues-testcommon"
    run_test 1951 "BatchNorm test_out (copilot #3665)" \
        'python -m pytest -v third_party/torch-xpu-ops/test/xpu/test_ops_xpu.py::TestCommonXPU::test_out_triangular_solve_xpu_float32 third_party/torch-xpu-ops/test/xpu/test_ops_xpu.py::TestCommonXPU::test_out_cholesky_inverse_xpu_float32 third_party/torch-xpu-ops/test/xpu/test_ops_xpu.py::TestCommonXPU::test_out_geqrf_xpu_float32 third_party/torch-xpu-ops/test/xpu/test_ops_xpu.py::TestCommonXPU::test_out_narrow_copy_xpu_float32 third_party/torch-xpu-ops/test/xpu/test_ops_xpu.py::TestCommonXPU::test_out_ormqr_xpu_float32 third_party/torch-xpu-ops/test/xpu/test_ops_xpu.py::TestCommonXPU::test_out__native_batch_norm_legit_xpu_float32'
    restore_xpu
fi

# --- #2015 ---
if switch_copilot_branch "copilot/fix-transformerencoderlayer-inf-nan" 2015; then
    check_cpp_rebuild "copilot/fix-transformerencoderlayer-inf-nan"
    run_test 2015 "TransformerEncoderLayer (copilot #3666)" \
        'PYTORCH_TEST_WITH_SLOW=1 python -m pytest -v third_party/torch-xpu-ops/test/xpu/test_nn_xpu.py -k "test_transformerencoderlayer_gelu_xpu_float16 or test_transformerencoderlayer_xpu_float16 or test_transformerencoderlayer_xpu_float32 or test_transformerencoderlayer_gelu_xpu_float32"'
    restore_xpu
fi

# --- #2512 ---
if switch_copilot_branch "copilot/fix-histc-error-integer-input" 2512; then
    check_cpp_rebuild "copilot/fix-histc-error-integer-input"
    run_test 2512 "histc deterministic (copilot #3607)" \
        'PYTORCH_TEST_WITH_SLOW=1 python -m pytest -v third_party/torch-xpu-ops/test/xpu/test_torch_xpu.py -k test_nondeterministic_alert_histc_xpu_float32'
    restore_xpu
fi

# --- #2518 ---
if switch_copilot_branch "copilot/fix-tensor-subclass-issue" 2518; then
    check_cpp_rebuild "copilot/fix-tensor-subclass-issue"
    run_test 2518 "tensor subclass (copilot #3667)" \
        'PYTORCH_TEST_WITH_SLOW=1 python -m pytest -v third_party/torch-xpu-ops/test/xpu/test_torch_xpu.py -k test_as_subclass'
    restore_xpu
fi

# --- #2615 ---
if switch_copilot_branch "copilot/fix-unsupported-dtype-half" 2615; then
    check_cpp_rebuild "copilot/fix-unsupported-dtype-half"
    run_test 2615 "dtype Half FFT (copilot #3668)" \
        'python -m pytest -v third_party/torch-xpu-ops/test/xpu/test_decomp_xpu.py -k "test_comprehensive_fft_fft2_xpu_float16 or test_comprehensive_fft_fft_xpu_float16 or test_comprehensive_fft_fftn_xpu_float16 or test_comprehensive_fft_hfft2_xpu_float16"'
    restore_xpu
fi

# --- #2953 ---
if switch_copilot_branch "copilot/fix-runtimeerror-huggingface-models" 2953; then
    check_cpp_rebuild "copilot/fix-runtimeerror-huggingface-models"
    # #2953 is a benchmark test, not a unit test - check what the PR actually changed
    cd ~/torch-xpu-ops
    echo "PR #3669 changes:" 
    git diff --name-only origin/main...origin/copilot/fix-runtimeerror-huggingface-models
    # The fill_kernel fix - let's test with a simple reproducer if possible
    run_test 2953 "TrOCR/XGLM fill_kernel (copilot #3669)" \
        'python -c "import torch; t = torch.zeros(1, device=\"xpu\", dtype=torch.float16); t.fill_(1.0); print(\"fill_ float16 OK:\", t)"'
    restore_xpu
fi

echo ""
echo "========================================"
echo "=== OPENCODE PR VERIFICATIONS ==="
echo "========================================"

# --- #2609 ---
cd ~/pytorch
git fetch review 2>&1 | tail -3 || echo "fetch review failed"
if git rev-parse review/agent/issue-2609 >/dev/null 2>&1; then
    git stash 2>/dev/null || true
    git checkout review/agent/issue-2609 2>&1 | tail -3 || git checkout -b local-agent-2609 review/agent/issue-2609
    run_test 2609 "CppCompileError inductor (opencode chuanqi129#16)" \
        'PYTORCH_TEST_WITH_SLOW=1 python -m pytest -v test/inductor/test_aot_inductor_custom_ops.py -k test_custom_op_square_xpu'
    git checkout main 2>&1 | tail -3 || true
else
    echo "SKIP #2609: review/agent/issue-2609 branch not found" | tee "$RESULTS_DIR/issue_2609.log"
fi

# --- #2712 ---
if git rev-parse review/agent/issue-2712 >/dev/null 2>&1; then
    git stash 2>/dev/null || true
    git checkout review/agent/issue-2712 2>&1 | tail -3 || git checkout -b local-agent-2712 review/agent/issue-2712
    run_test 2712 "weakref swap (opencode chuanqi129#17)" \
        'PYTORCH_TEST_WITH_SLOW=1 python -m pytest -v test/test_fake_tensor.py -k test_module_to'
    git checkout main 2>&1 | tail -3 || true
else
    echo "SKIP #2712: review/agent/issue-2712 branch not found" | tee "$RESULTS_DIR/issue_2712.log"
fi

echo ""
echo "========================================"
echo "=== REPRODUCER-ONLY (on main) ==="
echo "========================================"

# Restore to original state
restore_xpu
cd ~/pytorch

# --- #2253 ---
echo "SKIP #2253: No reproducer or failed tests in issue body" | tee "$RESULTS_DIR/issue_2253.log"

# --- #2359 ---
run_test 2359 "backward not reentrant (reproducer-only on main)" \
    'PYTORCH_TEST_WITH_SLOW=1 python -m pytest -v third_party/torch-xpu-ops/test/xpu/test_ops_gradients_xpu.py -k "test_fn_gradgrad_index_reduce_mean_xpu_float64 or test_inplace_gradgrad_index_reduce_mean_xpu_float64 or test_inplace_gradgrad_index_reduce_prod_xpu_float64"'

# --- #2436 ---
run_test 2436 "NoneType clone (reproducer-only on main)" \
    'python -m pytest -v third_party/torch-xpu-ops/test/xpu/test_expanded_weights_xpu.py -k "test_Conv3d_circular_stride2_pad2_xpu_double_xpu or test_Conv1d_pad2_xpu_double_xpu or test_Conv1d_xpu_double_xpu"'

# --- #2554 ---
run_test 2554 "AssertionError not raised (reproducer-only on main)" \
    'PYTORCH_TEST_WITH_SLOW=1 python -m pytest -v test/inductor/test_cuda_repro.py -k "test_truediv_base_not_bitwise_equivalent or test_emulate_precision_casts_min_pow_chain or test_selecsls42b_misaligned_address"'

# --- #2693 ---
run_test 2693 "scalars not equal (reproducer-only on main)" \
    'PYTORCH_TEST_WITH_SLOW=1 python -m pytest -v test/inductor/test_cuda_repro.py -k test_flash_attention_dynamic'

echo ""
echo "========================================"
echo "=== ALL VERIFICATIONS COMPLETE ==="
echo "========================================"
echo "Results in: $RESULTS_DIR/"
ls -la "$RESULTS_DIR/"

# Summary
echo ""
echo "=== SUMMARY ==="
for f in "$RESULTS_DIR"/issue_*.log; do
    issue=$(basename "$f" .log | sed 's/issue_//')
    result=$(grep "^RESULT:" "$f" 2>/dev/null | tail -1 || echo "UNKNOWN")
    echo "#$issue: $result"
done
