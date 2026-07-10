# Test Cases for parse-pytorch-trace

## Case 1: Basic trace with distinct ops
Input: `test/agentic/parse-pytorch-trace/fixtures/basic_trace.json`
Task: Parse the trace with `--top 10 --sort-by gpu_time`

### Expected results
- Expected: PASS - Script exits with code 0
- Expected: PASS - aten::addmm is the top op by GPU time (total 9000 us from two kernels)
- Expected: PASS - aten::_softmax has GPU time of 2000 us
- Expected: PASS - aten::mul has GPU time of 800 us
- Expected: PASS - aten::addmm shows 2 kernel invocations
- Expected: PASS - Total GPU time across all ops is 11800 us

## Case 2: Nested ops (aten::linear containing aten::addmm)
Input: `test/agentic/parse-pytorch-trace/fixtures/nested_ops_trace.json`
Task: Parse the trace with `--top 10 --sort-by gpu_time`

### Expected results
- Expected: PASS - Script exits with code 0
- Expected: PASS - aten::linear appears as a top-level op (merges child aten::addmm)
- Expected: PASS - aten::addmm is NOT listed separately (merged into parent aten::linear)
- Expected: PASS - aten::linear total GPU time includes the gemm_kernel durations (10500 us)
- Expected: PASS - aten::gelu has GPU time of 1500 us
- Expected: PASS - Total reported GPU time is 12000 us (only attributed kernels)
