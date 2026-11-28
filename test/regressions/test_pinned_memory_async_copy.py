# Copyright 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
"""
Regression test for non_blocking host-to-device copies out of pinned memory.

A sub-64-KiB `sycl::queue::memcpy` is served by the Level Zero runtime with a
host-side copy that first drains the in-order queue, so the transfer had
already completed by the time `copy_()` returned and the caching host
allocator handed the pinned block straight back to the next `pin_memory()`.
The sizes below sit on either side of that threshold. See
intel/torch-xpu-ops#5403.
"""

import torch
from torch.testing._internal.common_utils import (
    get_cycles_per_ms,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


class TestPinnedMemoryAsyncCopy(TestCase):
    @parametrize("numel", [1, 64 * 1024])
    def test_pinned_block_held_until_copy_completes(self, numel):
        cycles_per_ms = get_cycles_per_ms(device="xpu")

        # Two things have to happen before the timed section. Submitting a
        # SYCL kernel for the first time in a process costs tens of
        # milliseconds of kernel creation, and that perturbs the spin kernel
        # torch.xpu._sleep builds its delay out of, ending the delay early.
        # Dropping the cached pinned blocks afterwards then leaves the block
        # under test as the only reuse candidate, so the assertion below
        # cannot pass just because some other free block was handed out.
        warmup = torch.ones(numel).pin_memory()
        torch.zeros(numel, device="xpu").copy_(warmup, non_blocking=True)
        torch.xpu.synchronize()
        del warmup
        torch.accelerator.empty_host_cache()

        src = torch.ones(numel).pin_memory()
        ptr = src.data_ptr()
        dst = torch.zeros(numel, device="xpu")

        torch.xpu._sleep(int(1000 * cycles_per_ms))
        dst.copy_(src, non_blocking=True)

        del src
        other = torch.full((numel,), 2.0).pin_memory()
        self.assertNotEqual(other.data_ptr(), ptr, msg="allocation reused too soon")
        self.assertEqual(dst, torch.ones(numel, device="xpu"))


instantiate_parametrized_tests(TestPinnedMemoryAsyncCopy)


if __name__ == "__main__":
    run_tests()
