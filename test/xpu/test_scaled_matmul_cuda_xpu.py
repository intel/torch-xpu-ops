# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests

try:
    from xpu_test_utils import XPUPatchForImport
except Exception:
    from .xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_scaled_matmul_cuda import TestFP8Matmul, e4m3_type, e5m2_type, scaled_mm_wrap


# Override fast_accum tests: upstream raises SkipTest("XPU does not support fast accum
# yet")/uses @skipIfXpu decorators. Both tests in fact run on XPU; enable to track.
def _xpu_test_float8_scale_fast_accum(self, device) -> None:
    size = (16, 16)
    x = torch.full(size, 0.5, device=device, dtype=e4m3_type)
    y_type = e5m2_type
    y = torch.full(size, 0.5, device=device, dtype=y_type).t()
    scale_a = torch.tensor(1.5, device=device)
    scale_b = torch.tensor(0.66, device=device)
    out_fp8 = scaled_mm_wrap(x, y, scale_a, scale_b, out_dtype=e4m3_type, use_fast_accum=True)
    self.assertEqual(out_fp8.to(torch.float), torch.full(size, 4.0, device=device))
    out_fp8_s = scaled_mm_wrap(
        x, y, scale_a=scale_a, scale_b=scale_b, out_dtype=e4m3_type, use_fast_accum=True
    )
    self.assertEqual(out_fp8, out_fp8_s)


TestFP8Matmul.test_float8_scale_fast_accum = _xpu_test_float8_scale_fast_accum


def _xpu_test_float8_rowwise_scaling_sanity(self, device, use_fast_accum: bool) -> None:
    M, K, N = (1024, 512, 2048)
    fill_value = 0.5
    x = torch.full((M, K), fill_value, device=device)
    y = torch.full((N, K), fill_value, device=device)

    x_scales = torch.ones((x.shape[0], 1), device=device, dtype=torch.float32)
    y_scales = torch.ones((1, y.shape[0]), device=device, dtype=torch.float32)

    x_fp8 = x.to(e4m3_type)
    y_fp8 = y.to(e4m3_type).t()

    out_fp8 = scaled_mm_wrap(
        x_fp8,
        y_fp8,
        scale_a=x_scales,
        scale_b=y_scales,
        out_dtype=torch.bfloat16,
        use_fast_accum=use_fast_accum,
    )
    self.assertEqual(
        out_fp8.to(torch.float32),
        torch.full((M, N), K * (fill_value**2), device=device),
    )


TestFP8Matmul.test_float8_rowwise_scaling_sanity = parametrize(
    "use_fast_accum", [True, False]
)(_xpu_test_float8_rowwise_scaling_sanity)


# NOTE: test_scaled_mm_deepseek_error_messages_* is left at upstream's CUDA-12.9 skip:
# overriding the body works (skip bypassed), but the asserted error pattern
# (NotImplementedError "DeepSeek...only supported in CUDA for SM90") is CUDA-specific.
# XPU's scaled_mm raises ValueError("Invalid scaling configuration") for blockwise scales,
# so the upstream assertRaisesRegex cannot pass on XPU without changing what it asserts.


instantiate_device_type_tests(
    TestFP8Matmul, globals(), only_for="xpu", allow_xpu=True
)


if __name__ == "__main__":
    run_tests()
