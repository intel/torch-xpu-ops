# Owner(s): ["module: intel"]


import torch
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    OpDTypes,
    ops,
)
from torch.testing._internal.common_methods_invocations import foreach_binary_op_db
from torch.testing._internal.common_utils import run_tests

try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    from .xpu_test_utils import XPUPatchForImport


def get_device_capability(device=None):
    return (9, 0)


torch.cuda.get_device_capability = get_device_capability

with XPUPatchForImport(False):
    from test_foreach import TestForeach


@ops(
    filter(lambda op: op.supports_out, foreach_binary_op_db),
    dtypes=OpDTypes.supported,
)
def _test_binary_op_list_slow_path(self, device, dtype, op):
    torch.manual_seed(20240607)
    foreach_op, native_op, foreach_op_, native_op_ = self._get_funcs(op)
    # 0-strides

    tensor1 = make_tensor((10, 10), dtype=dtype, device=device)
    tensor2 = make_tensor((1,), device=device, dtype=dtype).expand_as(tensor1)
    inputs = ([tensor1], [tensor2])
    self._binary_test(
        dtype,
        foreach_op,
        native_op,
        inputs,
        is_fastpath=False,
        is_inplace=False,
        alpha=None,
        scalar_self_arg=False,
    )
    self._binary_test(
        dtype,
        foreach_op_,
        native_op_,
        inputs,
        is_fastpath=False,
        is_inplace=True,
        alpha=None,
        scalar_self_arg=False,
    )

    # different strides

    tensor1 = torch.zeros(10, 10, device=device, dtype=dtype)
    tensor2 = torch.ones(10, 10, device=device, dtype=dtype)
    inputs = ([tensor1], [tensor2.t()])
    self._binary_test(
        dtype,
        foreach_op,
        native_op,
        inputs,
        is_fastpath=False,
        is_inplace=False,
        alpha=None,
        scalar_self_arg=False,
    )
    self._binary_test(
        dtype,
        foreach_op_,
        native_op_,
        inputs,
        is_fastpath=False,
        is_inplace=True,
        alpha=None,
        scalar_self_arg=False,
    )

    # non contiguous

    tensor1 = make_tensor((5, 2, 1, 3), device=device, dtype=dtype, noncontiguous=True)
    tensor2 = make_tensor((5, 2, 1, 3), device=device, dtype=dtype, noncontiguous=True)
    self.assertFalse(tensor1.is_contiguous())
    self.assertFalse(tensor2.is_contiguous())
    inputs = ([tensor1], [tensor2])
    self._binary_test(
        dtype,
        foreach_op,
        native_op,
        inputs,
        is_fastpath=False,
        is_inplace=False,
        alpha=None,
        scalar_self_arg=False,
    )
    self._binary_test(
        dtype,
        foreach_op_,
        native_op_,
        inputs,
        is_fastpath=False,
        is_inplace=True,
        alpha=None,
        scalar_self_arg=False,
    )

    # sliced tensor

    tensor1 = make_tensor((5, 2, 1, 3), device=device, dtype=dtype)
    tensor2 = make_tensor((5, 2, 1, 3 * 7), device=device, dtype=dtype)[:, :, :, ::7]
    inputs = ([tensor1], [tensor2])
    self._binary_test(
        dtype,
        foreach_op,
        native_op,
        inputs,
        is_fastpath=False,
        is_inplace=False,
        alpha=None,
        scalar_self_arg=False,
    )
    self._binary_test(
        dtype,
        foreach_op_,
        native_op_,
        inputs,
        is_fastpath=False,
        is_inplace=True,
        alpha=None,
        scalar_self_arg=False,
    )


TestForeach.test_binary_op_list_slow_path = _test_binary_op_list_slow_path


def _test_0dim_tensor_overload_cpu_ok(self):
    tensors = [torch.ones((), device="xpu", dtype=torch.float32) for _ in range(2)]
    scalar_cpu_tensor = torch.tensor(4.0, device="cpu")

    # For mul and div, the scalar is allowed to be on CPU too

    actual = torch._foreach_mul(tensors, scalar_cpu_tensor)
    self.assertEqual(actual, [t.mul(scalar_cpu_tensor) for t in tensors])
    actual = torch._foreach_div(tensors, scalar_cpu_tensor)
    self.assertEqual(actual, [t.div(scalar_cpu_tensor) for t in tensors])


TestForeach.test_0dim_tensor_overload_cpu_ok = _test_0dim_tensor_overload_cpu_ok


def _test_div_reciprocal(self):
    expect_m, expect_e = torch.frexp(torch.div(torch.tensor(0.1, device="xpu"), 10.0))
    actual_m, actual_e = torch.frexp(
        torch._foreach_div([torch.tensor(0.1, device="xpu")], [10.0])[0]
    )
    self.assertEqual(expect_m, actual_m)
    self.assertEqual(expect_e, actual_e)


TestForeach.test_div_reciprocal = _test_div_reciprocal


def _test_0dim_tensor_overload_exception(self):
    # check exceptions of fast path

    tensors = [make_tensor((2, 2), dtype=torch.float, device="xpu") for _ in range(2)]
    with self.assertRaisesRegex(RuntimeError, "scalar tensor expected to be on"):
        torch._foreach_add(tensors, torch.tensor(1.0, device="cpu"), alpha=1.0)
    tensors = [make_tensor((2, 2), dtype=torch.float, device=d) for d in ("cpu", "xpu")]
    with self.assertRaisesRegex(RuntimeError, "scalar tensor expected to be 0 dim but"):
        torch._foreach_mul(tensors, torch.tensor([1.0, 1.0], device="xpu"))
    with self.assertRaisesRegex(RuntimeError, "scalar tensor expected to be 0 dim but"):
        torch._foreach_add(tensors, torch.tensor([1.0, 1.0], device="xpu"))


TestForeach.test_0dim_tensor_overload_exception = _test_0dim_tensor_overload_exception

instantiate_device_type_tests(TestForeach, globals(), only_for="xpu", allow_xpu=True)


if __name__ == "__main__":
    run_tests()
