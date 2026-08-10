# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]

"""Pass/fail wrapper around test/profiling/llama.py.

Kept in its own file because it needs `transformers` and a downloadable model,
so CI runs it as a separate step. The per-iteration profiler tables are still
printed: .github/scripts/llama_summary.py parses them out of the same log.
"""

import os
import sys

from torch.testing._internal.common_utils import run_tests, TestCase

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from profiling_test_utils import (
    assert_common,
    kernel_summary,
    load_script_module,
)

EXPECTED_GEMM_CALLS = int(os.environ.get("LLAMA_EXPECTED_GEMM_CALLS", "226"))
GEMM_KERNEL_PATTERN = os.environ.get("LLAMA_GEMM_KERNEL_PATTERN", "gemm_kernel")


class TestLlamaProfiling(TestCase):
    def test_gemm_kernel_calls_per_iteration(self):
        try:
            script = load_script_module("llama.py")
        except ImportError as exc:
            self.skipTest(f"transformers is not available: {exc}")

        calls_per_iteration = {}
        try:
            for i, prof in script.run_profile():
                events = assert_common(
                    self, prof, f"llama[iter={i}]", sort_by=script.SORT_BY
                )
                calls = sum(
                    count
                    for name, (count, _) in kernel_summary(events).items()
                    if GEMM_KERNEL_PATTERN in name
                )
                calls_per_iteration[i] = calls
                print(f"[llama] iter={i} {GEMM_KERNEL_PATTERN} calls={calls}")
        except OSError as exc:
            # Gated/unavailable model or no network: this is an environment
            # problem, not a profiler regression.
            self.skipTest(f"llama model is not available: {exc}")

        self.assertTrue(calls_per_iteration, "llama produced no profiling iterations")

        distinct = set(calls_per_iteration.values())
        self.assertEqual(
            len(distinct),
            1,
            f"'{GEMM_KERNEL_PATTERN}' call count is not stable across iterations: "
            f"{calls_per_iteration}",
        )
        self.assertEqual(
            distinct.pop(),
            EXPECTED_GEMM_CALLS,
            f"'{GEMM_KERNEL_PATTERN}' call count per iteration is "
            f"{calls_per_iteration}, expected {EXPECTED_GEMM_CALLS} "
            "(override with LLAMA_EXPECTED_GEMM_CALLS)",
        )


if __name__ == "__main__":
    run_tests()
