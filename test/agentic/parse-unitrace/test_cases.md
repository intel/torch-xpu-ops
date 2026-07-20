# Test Cases for parse-unitrace

## Case 1: Basic unitrace with ze* runtime filtering
Input: `test/agentic/parse-unitrace/fixtures/basic_unitrace.json`
Task: Parse the unitrace with `--top 10 --sort-by gpu_time`

### Expected results
- Expected: PASS - Script exits with code 0
- Expected: PASS - All ze* events are excluded (zeCommandListAppendLaunchKernel, zeCommandListAppendMemoryCopy, zeFenceHostSynchronize)
- Expected: PASS - gemm_kernel<float> is the top kernel by GPU time (total 9000 us from 2 invocations)
- Expected: PASS - SoftMaxForwardKernel<float> has GPU time of 2000 us with count 1
- Expected: PASS - mul_kernel<float> has GPU time of 800 us with count 1
- Expected: PASS - Total GPU time is 11800 us
- Expected: PASS - gemm_kernel<float> shows count of 2

## Case 2: Unitrace with SIMD suffix in kernel names
Input: `test/agentic/parse-unitrace/fixtures/unitrace_with_simd_suffix.json`
Task: Parse the unitrace with `--top 10 --sort-by gpu_time`

### Expected results
- Expected: PASS - Script exits with code 0
- Expected: PASS - Kernels with [SIMDxx {...} {...}] suffix are treated as valid GPU kernels (not filtered)
- Expected: PASS - gemm_kernel<float>[SIMD32 ...] entries are aggregated together (total 9500 us, count 2)
- Expected: PASS - Non-duration events (ph: "M", ph: "i") are ignored
- Expected: PASS - Total GPU time is 12300 us
- Expected: PASS - 3 unique kernel names reported

## Case 3: Timeline mode
Input: `test/agentic/parse-unitrace/fixtures/basic_unitrace.json`
Task: Parse the unitrace with `--timeline`

### Expected results
- Expected: PASS - Script exits with code 0
- Expected: PASS - Kernels appear in chronological order (gemm first, then SoftMax, then gemm, then mul)
- Expected: PASS - Timestamps are shown for each kernel entry
- Expected: PASS - ze* events do not appear in timeline output
