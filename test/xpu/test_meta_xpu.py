import os
import sys
import yaml
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests
from torchgen.yaml_utils import YamlLoader


try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    from .xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_meta import (
        TestMetaConverter,
        TestMeta,
        meta_function_device_expected_failures,
        meta_function_device_skips,
        meta_dispatch_device_expected_failures,
        meta_dispatch_device_skips,
        print_op_str_if_not_supported,
        )
meta_function_device_expected_failures["xpu"]=meta_function_device_expected_failures["cuda"]
meta_function_device_skips["xpu"]=meta_function_device_skips["cuda"]
meta_dispatch_device_expected_failures["xpu"]=meta_dispatch_device_expected_failures["cuda"]
meta_dispatch_device_skips["xpu"]=meta_dispatch_device_skips["cuda"]
instantiate_device_type_tests(TestMeta, globals(),only_for=("xpu",),allow_xpu=True)

if __name__ == '__main__':
    COMPARE_XLA = os.getenv('PYTORCH_COMPARE_XLA', None)
    if COMPARE_XLA is not None:
        with open(COMPARE_XLA) as f:
            d = yaml.load(f, Loader=YamlLoader)
            ops = d.get("full_codegen", []) + d.get("supported", []) + d.get("autograd", [])
            for op_str in ops:
                print_op_str_if_not_supported(op_str)
        sys.exit(0)

    COMPARE_TEXT = os.getenv('PYTORCH_COMPARE_TEXT', None)
    if COMPARE_TEXT is not None:
        with open(COMPARE_TEXT) as f:
            for op_str in f:
                print_op_str_if_not_supported(op_str.strip())
        sys.exit(0)

    run_tests()


