#!/bin/bash
# Phase 2: C++ copilot PRs (need rebuild)
source ~/intel/oneapi/setvars.sh 2>/dev/null
cd ~/pytorch
source .venv/bin/activate

RESULTS_DIR=~/torch-xpu-ops/verification_results
ORIG_SHA=d67a87783002c17786e9501c8c67b360479f6bb7

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

# For C++ changes we need to:
# 1. Checkout the branch in the submodule
# 2. Rebuild pytorch
# 3. Run test
# 4. Restore and rebuild

test_cpp_branch() {
    local issue=$1 branch=$2 desc=$3 cmd=$4
    
    echo ""
    echo "========================================"
    echo "=== C++ PR: #$issue ($branch) ==="
    echo "========================================"
    
    cd ~/pytorch/third_party/torch-xpu-ops
    git checkout "origin/$branch" 2>&1 | tail -3
    echo "Submodule now at: $(git log --oneline -1)"
    
    echo "--- Rebuilding PyTorch (incremental) ---"
    cd ~/pytorch
    USE_XPU=1 python setup.py develop 2>&1 | tail -10
    
    run_test "$issue" "$desc" "$cmd"
    
    echo "--- Restoring submodule ---"
    cd ~/pytorch/third_party/torch-xpu-ops
    git checkout "$ORIG_SHA" 2>&1 | tail -3
}

# #1951: BatchNormKernels.cpp
test_cpp_branch 1951 "copilot/fix-functionality-issues-testcommon" \
    "BatchNorm test_out (copilot #3665)" \
    'python -m pytest -v third_party/torch-xpu-ops/test/xpu/test_ops_xpu.py::TestCommonXPU::test_out_triangular_solve_xpu_float32 third_party/torch-xpu-ops/test/xpu/test_ops_xpu.py::TestCommonXPU::test_out_cholesky_inverse_xpu_float32 third_party/torch-xpu-ops/test/xpu/test_ops_xpu.py::TestCommonXPU::test_out_geqrf_xpu_float32 third_party/torch-xpu-ops/test/xpu/test_ops_xpu.py::TestCommonXPU::test_out_narrow_copy_xpu_float32'

# Need to rebuild back to original before next branch
echo "--- Rebuilding PyTorch with original submodule ---"
cd ~/pytorch
USE_XPU=1 python setup.py develop 2>&1 | tail -10

# #2512: SummaryOps.cpp (histc)
test_cpp_branch 2512 "copilot/fix-histc-error-integer-input" \
    "histc deterministic (copilot #3607)" \
    'PYTORCH_TEST_WITH_SLOW=1 python -m pytest -v third_party/torch-xpu-ops/test/xpu/test_torch_xpu.py -k test_nondeterministic_alert_histc_xpu_float32'

# Rebuild again
echo "--- Rebuilding PyTorch with original submodule ---"
cd ~/pytorch
USE_XPU=1 python setup.py develop 2>&1 | tail -10

# #2953: FillKernel.cpp
test_cpp_branch 2953 "copilot/fix-runtimeerror-huggingface-models" \
    "TrOCR/XGLM fill_kernel (copilot #3669)" \
    'python -c "
import torch
# Test fill_ with various dtypes that could overflow
for dtype in [torch.float16, torch.bfloat16, torch.float32]:
    t = torch.zeros(10, device=\"xpu\", dtype=dtype)
    t.fill_(1.0)
    assert t.sum().item() == 10.0, f\"fill_ failed for {dtype}\"
    print(f\"fill_ {dtype}: OK\")
# Also test with large values
t = torch.zeros(10, device=\"xpu\", dtype=torch.float16)
t.fill_(65504.0)  # max float16
print(f\"fill_ max float16: {t[0].item()}\")
print(\"All fill_ tests passed\")
"'

# Final restore
echo "--- Final rebuild with original submodule ---"
cd ~/pytorch
USE_XPU=1 python setup.py develop 2>&1 | tail -10

echo ""
echo "=== PHASE 2 COMPLETE ==="
for f in "$RESULTS_DIR"/issue_*.log; do
    issue=$(basename "$f" .log | sed 's/issue_//')
    result=$(grep "^RESULT:" "$f" 2>/dev/null | tail -1 || echo "UNKNOWN")
    echo "#$issue: $result"
done
