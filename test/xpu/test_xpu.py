# Owner(s): ["module: intel"]

import os
import sys

import torch
from torch.testing._internal.common_device_type import onlyXPU

# Monkey patch until we have a fancy way
instantiate_device_type_tests = torch.testing._internal.common_device_type.instantiate_device_type_tests
torch.testing._internal.common_device_type.instantiate_device_type_tests = lambda *args, **kwargs: None
torch.testing._internal.common_device_type.onlyCUDA = onlyXPU
torch.testing._internal.common_device_type.onlyNativeDeviceTypes = onlyXPU

pytorch_test_source_dir = os.path.dirname(os.path.abspath(__file__)) + "/../../../../test"
sys.path.append(pytorch_test_source_dir)
