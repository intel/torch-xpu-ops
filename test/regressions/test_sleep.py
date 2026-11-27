# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
import torch
from torch.testing._internal.common_utils import get_cycles_per_ms, run_tests, TestCase


class TestSleep(TestCase):
    def _elapsed_ms(self, cycles):
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        start.record()
        torch.xpu._sleep(cycles)
        end.record()
        end.synchronize()
        return start.elapsed_time(end)

    def test_sleep_is_async(self):
        cycles = int(get_cycles_per_ms("xpu") * 500)
        torch.xpu.synchronize()
        self.assertTrue(torch.xpu.current_stream().query())
        torch.xpu._sleep(cycles)
        self.assertFalse(torch.xpu.current_stream().query())
        torch.xpu.synchronize()
        self.assertTrue(torch.xpu.current_stream().query())

    def test_sleep_duration(self):
        cycles_per_ms = get_cycles_per_ms("xpu")
        short_ms = self._elapsed_ms(int(cycles_per_ms * 20))
        long_ms = self._elapsed_ms(int(cycles_per_ms * 80))
        # The sleep duration is not guaranteed to be exact, but generally
        # the long sleep should take longer than the short sleep.
        self.assertGreater(long_ms, short_ms)


if __name__ == "__main__":
    run_tests()
