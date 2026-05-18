#!/bin/bash
# Phase 1: Python-only copilot PRs + reproducer-only tests (no C++ rebuild)
source ~/intel/oneapi/setvars.sh 2>/dev/null
cd ~/pytorch
source .venv/bin/activate

RESULTS_DIR=~/torch-xpu-ops/verification_results
mkdir -p "$RESULTS_DIR"
ORIG_SHA=d67a87783002c17786e9501c8c67b360479f6bb7

switch_branch() {
    local branch=$1 issue=$2
    echo "--- Switch torch-xpu-ops to origin/$branch (#$issue) ---"
    cd ~/pytorch/third_party/torch-xpu-ops
    git checkout "origin/$branch" -- . 2>&1 | tail -3
    git log --oneline -1 "origin/$branch"
}

restore() {
    cd ~/pytorch/third_party/torch-xpu-ops
    git checkout "$ORIG_SHA" -- . 2>&1 | tail -3
}

run_test() {
    local issue=$1 desc=$2 cmd=$3
    echo ""
    echo "================================================================"
    echo "=== #$issue: $desc ==="
    echo "================================================================"
    cd ~/pytorch
    local outfile="$RESULTS_DIR/issue_${issue}.log"
    echo "Command: $cmd" > "$outfile"
    echo "Time: $(date)" >> "$outfile"
    echo "---" >> "$outfile"
    eval "$cmd" >> "$outfile" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "RESULT: PASS"
        echo "RESULT: PASS" >> "$outfile"
    else
        echo "RESULT: FAIL (rc=$rc)"
        echo "RESULT: FAIL (rc=$rc)" >> "$outfile"
    fi
    tail -15 "$outfile"
}

echo "=== PYTHON-ONLY COPILOT PRs ==="

# #2015: test_nn_xpu.py change only
switch_branch "copilot/fix-transformerencoderlayer-inf-nan" 2015
run_test 2015 "TransformerEncoderLayer (copilot #3666)" \
    'PYTORCH_TEST_WITH_SLOW=1 python -m pytest -v third_party/torch-xpu-ops/test/xpu/test_nn_xpu.py -k "test_transformerencoderlayer_gelu_xpu_float16 or test_transformerencoderlayer_xpu_float16 or test_transformerencoderlayer_xpu_float32 or test_transformerencoderlayer_gelu_xpu_float32"'
restore

# #2518: test repro only
switch_branch "copilot/fix-tensor-subclass-issue" 2518
run_test 2518 "tensor subclass (copilot #3667)" \
    'PYTORCH_TEST_WITH_SLOW=1 python -m pytest -v third_party/torch-xpu-ops/test/xpu/test_torch_xpu.py -k test_as_subclass'
restore

# #2615: test_decomp_xpu.py change only
switch_branch "copilot/fix-unsupported-dtype-half" 2615
run_test 2615 "dtype Half FFT (copilot #3668)" \
    'python -m pytest -v third_party/torch-xpu-ops/test/xpu/test_decomp_xpu.py -k "test_comprehensive_fft_fft2_xpu_float16 or test_comprehensive_fft_fft_xpu_float16 or test_comprehensive_fft_fftn_xpu_float16 or test_comprehensive_fft_hfft2_xpu_float16"'
restore

echo ""
echo "=== OPENCODE PRs (pytorch changes) ==="

# #2609: chuanqi129/pytorch PR#16
cd ~/pytorch
git fetch review 2>&1 | tail -3 || echo "fetch review failed"
if git rev-parse review/agent/issue-2609 >/dev/null 2>&1; then
    # Stash any local changes
    git stash 2>/dev/null || true
    git checkout review/agent/issue-2609 2>&1 | tail -3 || git checkout -b local-2609 review/agent/issue-2609 2>&1 | tail -3
    run_test 2609 "CppCompileError inductor (opencode chuanqi129#16)" \
        'PYTORCH_TEST_WITH_SLOW=1 python -m pytest -v test/inductor/test_aot_inductor_custom_ops.py -k test_custom_op_square_xpu'
    git checkout main 2>&1 | tail -3
else
    echo "SKIP #2609: branch not found" | tee "$RESULTS_DIR/issue_2609.log"
fi

# #2712: chuanqi129/pytorch PR#17
if git rev-parse review/agent/issue-2712 >/dev/null 2>&1; then
    git stash 2>/dev/null || true
    git checkout review/agent/issue-2712 2>&1 | tail -3 || git checkout -b local-2712 review/agent/issue-2712 2>&1 | tail -3
    run_test 2712 "weakref swap (opencode chuanqi129#17)" \
        'PYTORCH_TEST_WITH_SLOW=1 python -m pytest -v test/test_fake_tensor.py -k test_module_to'
    git checkout main 2>&1 | tail -3
else
    echo "SKIP #2712: branch not found" | tee "$RESULTS_DIR/issue_2712.log"
fi

echo ""
echo "=== REPRODUCER-ONLY (confirm bugs still exist on main) ==="

cd ~/pytorch
# Restore submodule
cd third_party/torch-xpu-ops && git checkout "$ORIG_SHA" 2>&1 | tail -1
cd ~/pytorch

echo "SKIP #2253: No reproducer in issue body" | tee "$RESULTS_DIR/issue_2253.log"

run_test 2359 "backward not reentrant (reproducer on main)" \
    'PYTORCH_TEST_WITH_SLOW=1 python -m pytest -v third_party/torch-xpu-ops/test/xpu/test_ops_gradients_xpu.py -k "test_fn_gradgrad_index_reduce_mean_xpu_float64 or test_inplace_gradgrad_index_reduce_mean_xpu_float64 or test_inplace_gradgrad_index_reduce_prod_xpu_float64"'

run_test 2436 "NoneType clone (reproducer on main)" \
    'python -m pytest -v third_party/torch-xpu-ops/test/xpu/test_expanded_weights_xpu.py -k "test_Conv3d_circular_stride2_pad2_xpu_double_xpu or test_Conv1d_pad2_xpu_double_xpu or test_Conv1d_xpu_double_xpu"'

run_test 2554 "AssertionError not raised (reproducer on main)" \
    'PYTORCH_TEST_WITH_SLOW=1 python -m pytest -v test/inductor/test_cuda_repro.py -k "test_truediv_base_not_bitwise_equivalent or test_emulate_precision_casts_min_pow_chain or test_selecsls42b_misaligned_address"'

run_test 2693 "scalars not equal (reproducer on main)" \
    'PYTORCH_TEST_WITH_SLOW=1 python -m pytest -v test/inductor/test_cuda_repro.py -k test_flash_attention_dynamic'

echo ""
echo "=== PHASE 1 COMPLETE ==="
for f in "$RESULTS_DIR"/issue_*.log; do
    issue=$(basename "$f" .log | sed 's/issue_//')
    result=$(grep "^RESULT:" "$f" 2>/dev/null | tail -1 || echo "UNKNOWN")
    echo "#$issue: $result"
done
