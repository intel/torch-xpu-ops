#!/bin/bash
PYTORCH_DIR="$HOME/pytorch"
TXO_DIR="$HOME/torch-xpu-ops"
RESULTS_DIR="$TXO_DIR/verification_results/round2"
mkdir -p "$RESULTS_DIR"

set +eu
source ~/intel/oneapi/setvars.sh 2>/dev/null
set -eu

# Activate venv
export PATH="$PYTORCH_DIR/.venv/bin:$PATH"
cd "$PYTORCH_DIR"

log() { echo "$(date '+%H:%M:%S') $1"; }

run_test() {
    local issue=$1 test_cmd=$2 logfile="$RESULTS_DIR/issue_${1}.log"
    log ">>> Testing issue #$issue: $test_cmd"
    set +e
    eval "$test_cmd" > "$logfile" 2>&1
    local rc=$?
    set -e
    if [ $rc -eq 0 ]; then
        log "  ✅ #$issue PASS (rc=$rc)"
        echo "PASS" > "$RESULTS_DIR/issue_${issue}.result"
    elif [ $rc -eq 5 ]; then
        log "  ⚠️ #$issue NO TESTS COLLECTED (rc=$rc)"
        echo "NO_TESTS" > "$RESULTS_DIR/issue_${issue}.result"
    else
        log "  ❌ #$issue FAIL (rc=$rc)"
        echo "FAIL" > "$RESULTS_DIR/issue_${issue}.result"
        tail -20 "$logfile"
    fi
}

apply_pr() {
    local repo=$1 pr=$2
    log "  Fetching PR #$pr from $repo..."
    gh pr diff "$pr" --repo "$repo" > /tmp/pr_${pr}.patch 2>/dev/null
    cd "$PYTORCH_DIR"
    git apply --check /tmp/pr_${pr}.patch 2>/dev/null && git apply /tmp/pr_${pr}.patch || {
        log "  ⚠️ Patch failed to apply cleanly, trying with --3way"
        git apply --3way /tmp/pr_${pr}.patch 2>/dev/null || {
            log "  ❌ Cannot apply PR #$pr"
            return 1
        }
    }
}

revert_pr() {
    local pr=$1
    cd "$PYTORCH_DIR"
    git checkout -- . 2>/dev/null || true
}

########################################
# PHASE 1: Python-only pytorch PRs
########################################
log "========== PHASE 1: Python-only pytorch PRs =========="

# --- #2795: histc integer + deterministic ---
log "--- #2795: histc integer + deterministic ---"
apply_pr chuanqi129/pytorch 8
# Write inline reproducer
cat > /tmp/test_2795.py << 'EOF'
import torch
torch.use_deterministic_algorithms(True)
x = torch.randint(0, 10, (100,), device='xpu')
# Integer histc should NOT raise under deterministic mode
result = torch.histc(x.float(), bins=10, min=0, max=9)  # float should raise
print("Float histc raised? No - this is expected to fail")
EOF
cat > /tmp/test_2795_int.py << 'EOF'
import torch
torch.use_deterministic_algorithms(True)
x = torch.randint(0, 10, (100,), device='xpu', dtype=torch.int64)
try:
    result = torch.histc(x, bins=10, min=0, max=9)
    print("PASS: Integer histc works under deterministic mode")
except RuntimeError as e:
    if "deterministic" in str(e):
        print(f"FAIL: Integer histc wrongly raises determinism error: {e}")
        exit(1)
    raise
EOF
run_test 2795 "python /tmp/test_2795_int.py"
revert_pr 8

# --- #3361: test_dropout CUDA not available ---
log "--- #3361: test_dropout CUDA not available ---"
apply_pr chuanqi129/pytorch 9
run_test 3361 "python -m pytest test/dynamo/test_higher_order_ops.py -k test_dropout -x -v --timeout=120"
revert_pr 9

# --- #3388: stream_index None ---
log "--- #3388: Dynamo stream_index None ---"
apply_pr chuanqi129/pytorch 10
run_test 3388 "python -m pytest third_party/torch-xpu-ops/test/xpu/dynamo/test_ctx_manager_xpu.py -k test_cuda_event_method -x -v --timeout=120"
revert_pr 10

# --- #2560: iter.device(arg).is_xpu() ---
log "--- #2560: addcmul use_cpu_scalar ---"
apply_pr chuanqi129/pytorch 11
# The issue is about addcmul with CPU scalar on XPU
cat > /tmp/test_2560.py << 'EOF'
import torch
a = torch.randn(10, device='xpu')
b = torch.randn(10, device='xpu')
c = torch.tensor(0.5)  # CPU scalar
try:
    result = torch.addcmul(a, b, c, value=1.0)
    print("PASS: addcmul with CPU scalar works")
except RuntimeError as e:
    print(f"FAIL: {e}")
    exit(1)
EOF
run_test 2560 "python /tmp/test_2560.py"
revert_pr 11

# --- #2715: Attempted to inline skipped function ---
log "--- #2715: inline skipped function ---"
apply_pr chuanqi129/pytorch 12
run_test 2715 "python -m pytest test/dynamo/test_ctx_manager.py -k test_cuda_device -x -v --timeout=120"
revert_pr 12

########################################
# PHASE 2: C++ pytorch PR (#1969 Event.cpp)
########################################
log "========== PHASE 2: C++ pytorch PR =========="

# --- #1969: weak reference to torch.Event ---
log "--- #1969: weak reference to torch.Event (C++ rebuild needed) ---"
apply_pr chuanqi129/pytorch 13
log "  Building pytorch with Event.cpp change..."
cd "$PYTORCH_DIR"
USE_XPU=1 python setup.py develop 2>&1 | tail -5
run_test 1969 "python -m pytest test/dynamo/test_ctx_manager.py -k test_gpu_event_across_graph_break -x -v --timeout=120"
revert_pr 13
log "  Rebuilding pytorch to revert Event.cpp..."
USE_XPU=1 python setup.py develop 2>&1 | tail -5

########################################
# PHASE 3: torch-xpu-ops C++ PRs
########################################
log "========== PHASE 3: torch-xpu-ops C++ PRs =========="

ORIG_SHA=$(cat "$PYTORCH_DIR/third_party/xpu.txt")
log "Original xpu.txt SHA: $ORIG_SHA"

# --- #2512: histc deterministic (torch-xpu-ops #3607) ---
log "--- #2512: histc deterministic (C++ rebuild) ---"
cd "$TXO_DIR"
HISTC_SHA=$(gh pr view 3607 --repo intel/torch-xpu-ops --json headRefOid -q '.headRefOid')
log "  PR #3607 head SHA: $HISTC_SHA"
echo "$HISTC_SHA" > "$PYTORCH_DIR/third_party/xpu.txt"
cd "$PYTORCH_DIR"
git submodule update --init third_party/torch-xpu-ops 2>&1 | tail -3
USE_XPU=1 python setup.py develop 2>&1 | tail -5

cat > /tmp/test_2512.py << 'EOF'
import torch
torch.use_deterministic_algorithms(True)
# Integer histc should work (deterministic for integers)
x = torch.randint(0, 10, (100,), device='xpu', dtype=torch.int64)
try:
    result = torch.histc(x, bins=10, min=0, max=9)
    print("PASS: Integer histc works under deterministic mode")
except RuntimeError as e:
    if "deterministic" in str(e):
        print(f"FAIL: {e}")
        exit(1)
    raise

# Float histc should raise
try:
    result = torch.histc(torch.randn(100, device='xpu'), bins=10)
    print("FAIL: Float histc should have raised determinism error")
    exit(1)
except RuntimeError as e:
    if "deterministic" in str(e):
        print("PASS: Float histc correctly raises determinism error")
    else:
        raise
EOF
run_test 2512 "python /tmp/test_2512.py"

# Restore
echo "$ORIG_SHA" > "$PYTORCH_DIR/third_party/xpu.txt"
cd "$PYTORCH_DIR"
git submodule update --init third_party/torch-xpu-ops 2>&1 | tail -3

# --- #3390: Atomics.h fix (torch-xpu-ops #3603) ---
log "--- #3390: Atomics.h atomic load fix (C++ rebuild) ---"
cd "$TXO_DIR"
ATOMICS_SHA=$(gh pr view 3603 --repo intel/torch-xpu-ops --json headRefOid -q '.headRefOid')
log "  PR #3603 head SHA: $ATOMICS_SHA"
echo "$ATOMICS_SHA" > "$PYTORCH_DIR/third_party/xpu.txt"
cd "$PYTORCH_DIR"
git submodule update --init third_party/torch-xpu-ops 2>&1 | tail -3
USE_XPU=1 python setup.py develop 2>&1 | tail -5

# Atomics fix is a safety/correctness fix - verify it compiles and basic atomic ops work
cat > /tmp/test_3390.py << 'EOF'
import torch
# Basic atomic operations that exercise the Atomics.h codepath
x = torch.zeros(10, dtype=torch.int32, device='xpu')
indices = torch.randint(0, 10, (1000,), device='xpu')
# scatter_add exercises atomic adds
src = torch.ones(1000, dtype=torch.int32, device='xpu')
x.scatter_add_(0, indices, src)
expected = torch.zeros(10, dtype=torch.int32)
for i in indices.cpu():
    expected[i.item()] += 1
if torch.equal(x.cpu(), expected):
    print("PASS: Atomic scatter_add works correctly")
else:
    print(f"FAIL: scatter_add mismatch\n  got: {x.cpu()}\n  expected: {expected}")
    exit(1)

# Also test with float16 (exercises 2-byte atomic path)
x_fp16 = torch.zeros(10, dtype=torch.float16, device='xpu')
src_fp16 = torch.ones(1000, dtype=torch.float16, device='xpu')
x_fp16.scatter_add_(0, indices, src_fp16)
if torch.allclose(x_fp16.cpu().float(), expected.float(), atol=1):
    print("PASS: Float16 atomic scatter_add works correctly")
else:
    print(f"FAIL: fp16 scatter_add mismatch")
    exit(1)
EOF
run_test 3390 "python /tmp/test_3390.py"

# Restore
echo "$ORIG_SHA" > "$PYTORCH_DIR/third_party/xpu.txt"
cd "$PYTORCH_DIR"
git submodule update --init third_party/torch-xpu-ops 2>&1 | tail -3
USE_XPU=1 python setup.py develop 2>&1 | tail -5

# --- #1856: Performance issue, PR CLOSED, skip ---
log "--- #1856: SKIP (performance issue, PR #3602 CLOSED, repro-only) ---"
echo "SKIP" > "$RESULTS_DIR/issue_1856.result"

log "========== DONE =========="
log "Results:"
for f in "$RESULTS_DIR"/issue_*.result; do
    issue=$(basename "$f" .result | sed 's/issue_//')
    result=$(cat "$f")
    log "  #$issue: $result"
done
