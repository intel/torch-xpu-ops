# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

import sys

import pytest
import torch


# On Windows, memory held by the UR kernel creation path is only released on
# explicit synchronization, so without it a long test run eventually OOMs.
@pytest.fixture(autouse=True)
def _xpu_synchronize_on_windows():
    yield
    if sys.platform == "win32" and torch.xpu.is_available():
        torch.xpu.current_stream().synchronize()
