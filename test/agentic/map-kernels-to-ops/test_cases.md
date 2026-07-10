# Test Cases for map-kernels-to-ops

## Case 1: Basic kernel-to-op mapping
Input: `test/agentic/map-kernels-to-ops/fixtures/trace.json` + `test/agentic/map-kernels-to-ops/fixtures/unitrace.json`
Task: Run with `--top 10`

### Expected results
- Expected: PASS - Script exits with code 0
- Expected: PASS - 3 kernel pairs matched (trace has 3 kernels, unitrace has 3)
- Expected: PASS - aten::addmm (ext_id=1) gets unitrace duration 4800 us
- Expected: PASS - aten::_softmax (ext_id=2) gets unitrace duration 1900 us
- Expected: PASS - aten::addmm (ext_id=3) gets unitrace duration 3800 us
- Expected: PASS - Total unitrace GPU time is 10500 us
- Expected: PASS - SIMD suffix stripped for name verification (no name mismatch warning)

## Case 2: Name mismatch detection (default mode - warn)
Input: `test/agentic/map-kernels-to-ops/fixtures/trace.json` + `test/agentic/map-kernels-to-ops/fixtures/unitrace_mismatch.json`
Task: Run with `--top 10` (no --strict)

### Expected results
- Expected: PASS - Script exits with code 0 (warning only, not fatal)
- Expected: PASS - WARNING message emitted about name mismatch at position 0
- Expected: PASS - Mismatch identifies trace name "gemm_kernel" vs unitrace name "wrong_kernel_name..."
- Expected: PASS - Despite mismatch, mapping still proceeds and produces results
- Expected: PASS - Total unitrace GPU time still reported as 10500 us

## Case 3: Name mismatch detection (strict mode)
Input: `test/agentic/map-kernels-to-ops/fixtures/trace.json` + `test/agentic/map-kernels-to-ops/fixtures/unitrace_mismatch.json`
Task: Run with `--top 10 --strict`

### Expected results
- Expected: PASS - Script reports name mismatch
- Expected: PASS - Mismatch between "gemm_kernel" and "wrong_kernel_name" is flagged
- Expected: PASS - Results are still produced (strict mode warns but does not abort in current implementation)
