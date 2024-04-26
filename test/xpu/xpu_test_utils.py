# Owner(s): ["module: intel"]

import os
import sys

import torch
from torch.testing._internal.common_device_type import onlyXPU
from torch.testing._internal.common_device_type import instantiate_device_type_tests

class XPUTestPatch:
    def __enter__(self):
        # Monkey patch until we have a fancy way
        self.onlyCUDA_fn = torch.testing._internal.common_device_type.onlyCUDA
        self.onlyNativeDeviceTypes_fn = torch.testing._internal.common_device_type.onlyNativeDeviceTypes
        self.instantiate_device_type_tests_fn = torch.testing._internal.common_device_type.instantiate_device_type_tests
        torch.testing._internal.common_device_type.onlyCUDA = onlyXPU
        torch.testing._internal.common_device_type.onlyNativeDeviceTypes = onlyXPU
        torch.testing._internal.common_device_type.instantiate_device_type_tests = lambda *args, **kwargs: None
        # Config python searching path for PyTorch test cases
        self.pytorch_test_source_dir = os.path.dirname(os.path.abspath(__file__)) + "/../../../../test"
        self.pytorch_test_nn_source_dir = os.path.dirname(os.path.abspath(__file__)) + "/../../../../test/nn"
        sys.path.append(self.pytorch_test_source_dir)
        sys.path.append(self.pytorch_test_nn_source_dir)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        torch.testing._internal.common_device_type.onlyCUDA = self.onlyCUDA_fn
        torch.testing._internal.common_device_type.onlyNativeDeviceTypes = self.onlyNativeDeviceTypes_fn
        torch.testing._internal.common_device_type.instantiate_device_type_tests = self.instantiate_device_type_tests_fn
        sys.path.remove(self.pytorch_test_source_dir)
        sys.path.remove(self.pytorch_test_nn_source_dir)
