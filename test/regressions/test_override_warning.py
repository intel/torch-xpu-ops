# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
import subprocess
import sys

from torch.testing._internal.common_utils import run_tests, TestCase


class TestKernelOverrideWarning(TestCase):
    def test_import_torch_kernel_override_warning(self):
        """Ensure 'import torch' does not emit kernel override warnings."""
        result = subprocess.run(
            [sys.executable, "-W", "all", "-c", "import torch"],
            capture_output=True,
            text=True,
        )
        self.assertNotIn(
            "Overriding a previously registered kernel for the same operator "
            "and the same dispatch key",
            result.stderr,
        )


if __name__ == "__main__":
    run_tests()
