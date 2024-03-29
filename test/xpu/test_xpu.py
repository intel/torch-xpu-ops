# Owner(s): ["module: intel"]

import os
import sys

import torch
from torch.testing._internal.common_device_type import onlyXPU

# Monkey patch until we have a fancy way
torch.testing._internal.common_device_type.onlyCUDA = onlyXPU
torch.testing._internal.common_device_type.onlyNativeDeviceTypes = onlyXPU

# Copy the python file from PyTorch and patch it with removing suite instantiate at runtime.
# TODO: Propose lazy instantiate to PyTorch
pytorch_test_source_dir = os.path.dirname(os.path.abspath(__file__)) + "/../../../../test"

def create_template(from_, to_):
    pytorch_test_py = from_ + ".py"
    pytorch_test_template_py = to_ + ".py"
    copy_test = "cp " + pytorch_test_source_dir + "/" + pytorch_test_py + " ./" + pytorch_test_template_py
    remove_suite_instantiate = "sed -i '/^instantiate_device_type_tests/d' " + pytorch_test_template_py
    print(copy_test)
    print(remove_suite_instantiate)
    
    os.system(copy_test)
    os.system(remove_suite_instantiate)
