# Owner(s): ["module: intel"]

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, skipIfCrossRef

try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    from .xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    import test_decomp
    from test_decomp import TestDecomp,DecompOneOffTests, _getDefaultRtolAndAtol

def _op_assert_ref(test_case, op, test_dtype, i, orig, decomp, ref, args, kwargs):
    assert orig.dtype == decomp.dtype, f"{i} Operation:  {op}"
    if orig.numel() == 0 or decomp.numel() == 0:
        assert orig.numel() == decomp.numel()
        return
    assert orig.shape == decomp.shape, f"{i} Operation:  {op}"
    tol_table = {
        (torch.bfloat16, torch.ops.aten.native_layer_norm.default): 1e-5,
        (torch.float16, torch.ops.aten.native_layer_norm.default): 1e-5,
        (torch.float16, torch.ops.aten.native_layer_norm_backward.default): 1e-3,
        (torch.bfloat16, torch.ops.aten.native_layer_norm_backward.default): 2e-2,
        (torch.bfloat16, torch.ops.aten.native_batch_norm.default): 1e-5,
        (torch.float16, torch.ops.aten.native_batch_norm.default): 1e-5,
        (torch.bfloat16, torch.ops.aten._native_batch_norm_legit.default): 1e-5,
        (torch.bfloat16, torch.ops.aten._native_batch_norm_legit.no_stats): 1e-5,
        (torch.float16, torch.ops.aten._native_batch_norm_legit.default): 1e-5,
        (torch.float16, torch.ops.aten._native_batch_norm_legit.no_stats): 1e-5,
        (torch.bfloat16, torch.ops.aten.linalg_vector_norm.default): 1e-4,
        (torch.float16, torch.ops.aten.linalg_vector_norm.default): 1e-4,
        (torch.bfloat16, torch.ops.aten.var_mean.correction): 5e-7,
        (torch.float16, torch.ops.aten.var_mean.correction): 5e-7,
        (torch.bfloat16, torch.ops.aten.var_mean.dim): 5e-7,
        (torch.float16, torch.ops.aten.var_mean.dim): 5e-7,
        (torch.float16, torch.ops.aten.nll_loss_forward.default): 1e-2,
        (torch.bfloat16, torch.ops.aten.nll_loss_forward.default): 1e-1,
        (torch.float16, torch.ops.aten.nll_loss2d_forward.default): 1e-2,
        (torch.bfloat16, torch.ops.aten.nll_loss2d_forward.default): 2e-1,
        (torch.float16, torch.ops.aten.hardswish.default): 2e-7,
        (torch.bfloat16, torch.ops.aten.hardswish.default): 2e-7,
        (torch.float16, torch.ops.aten.multi_margin_loss.default): 3e-2,
        (torch.bfloat16, torch.ops.aten.multi_margin_loss.default): 5e-2,
        (torch.float16, torch.ops.aten.multilabel_margin_loss_forward.default): 3e-2,
        (torch.bfloat16, torch.ops.aten.multilabel_margin_loss_forward.default): 3e-2,
        (torch.float16, torch.ops.aten.reflection_pad1d_backward.default): 5e-3,
        (torch.bfloat16, torch.ops.aten.reflection_pad1d_backward.default): 5e-3,
        (torch.float16, torch.ops.aten.reflection_pad2d_backward.default): 5e-3,
        (torch.bfloat16, torch.ops.aten.reflection_pad2d_backward.default): 7e-3, # adjust tolerance for xpu, so hook this func
        (torch.float16, torch.ops.aten.reflection_pad3d_backward.default): 5e-3,
        (torch.bfloat16, torch.ops.aten.reflection_pad3d_backward.default): 5e-2,
        # see https://github.com/pytorch/pytorch/pull/96264
        (torch.float16, torch.ops.aten.mv.default): 1e-5,
        (torch.bfloat16, torch.ops.aten.mv.default): 1e-5,
        (torch.float16, torch.ops.aten.log_sigmoid_backward.default): 2e-5,
        (torch.float16, torch.ops.aten._batch_norm_with_update.default): 2e-7, # adjust tolerance for xpu, so hook this func
        (torch.bfloat16, torch.ops.aten._batch_norm_with_update.default): 2e-7, # adjust tolerance for xpu, so hook this func
    }
    if ref.is_floating_point():
        orig_diff = (orig - ref).abs().max()
        decomp_diff = (decomp - ref).abs().max()
        atol = tol_table.get((test_dtype, op), 1e-7)
        if decomp_diff > orig_diff + atol:
            raise RuntimeError(
                f"Difference from float64 is larger with decomposition {op.__name__}"
                f" than original on output {i}. Original max diff: {orig_diff}, Decomp max diff: {decomp_diff}\n"
                f"atol = {atol}\n"
                f"args = {args}\n"
                f"kwargs = {kwargs}"
            )
    else:
        test_case.assertEqual(
            orig, decomp, msg=f"{op.__name__}\nargs = {args}\nkwargs = {kwargs}"
        )
test_decomp.op_assert_ref=_op_assert_ref

def _op_assert_equal(test_case, op, test_dtype, orig, decomp, args, kwargs):
    test_case.assertEqual(
        orig.dtype,
        decomp.dtype,
        f"Operation: {op}, orig.dtype: {orig.dtype}, decomp.dtype: {decomp.dtype}, {args}, {kwargs}",
    )
    # Before adding an entry to this table, make sure your decomposition is right :)
    tol_table = {
        # Due to strange epsilon behaviors, see https://github.com/pytorch/pytorch/issues/73161
        (torch.float32, torch.ops.aten.native_layer_norm.default): (1e-3, 1e-3),
        (torch.float32, torch.ops.aten.native_layer_norm_backward.default): (
            1e-3,
            1e-3,
        ),
        (torch.float64, torch.ops.aten.native_layer_norm.default): (1e-6, 1e-6),
        # This exceeds default tolerances only on CPU, on CUDA it's fine
        (torch.float32, torch.ops.aten.grid_sampler_2d.default): (7e-6, 3e-5),
        # Exceeds tolerances on CUDA, likely due to fma
        (torch.float32, torch.ops.aten.mv.default): (1e-5, 3e-5),
        (torch.complex64, torch.ops.aten.mv.default): (5e-5, 5e-5),
        (torch.float64, torch.ops.aten.upsample_bicubic2d.vec): (1e-5, 5e-4),
        (torch.float64, torch.ops.aten.upsample_bicubic2d.default): (1e-5, 5e-4),
        # The decomposition is TOO correct. It computes everything in int64, so sometimes
        # there's an off-by-one error. See
        # https://github.com/pytorch/pytorch/issues/81996
        # https://github.com/pytorch/pytorch/issues/82230
        (torch.int8, torch.ops.aten.linspace.default): (0, 1),
        (torch.uint8, torch.ops.aten.linspace.default): (0, 1),
        (torch.int16, torch.ops.aten.linspace.default): (0, 1),
        (torch.int32, torch.ops.aten.linspace.default): (0, 1),
        (torch.int64, torch.ops.aten.linspace.default): (0, 1),
        (torch.int8, torch.ops.aten.linspace.Tensor_Tensor): (0, 1),
        (torch.uint8, torch.ops.aten.linspace.Tensor_Tensor): (0, 1),
        (torch.int16, torch.ops.aten.linspace.Tensor_Tensor): (0, 1),
        (torch.int32, torch.ops.aten.linspace.Tensor_Tensor): (0, 1),
        (torch.int64, torch.ops.aten.linspace.Tensor_Tensor): (0, 1),
        (torch.int8, torch.ops.aten.linspace.Tensor_Scalar): (0, 1),
        (torch.uint8, torch.ops.aten.linspace.Tensor_Scalar): (0, 1),
        (torch.int16, torch.ops.aten.linspace.Tensor_Scalar): (0, 1),
        (torch.int32, torch.ops.aten.linspace.Tensor_Scalar): (0, 1),
        (torch.int64, torch.ops.aten.linspace.Tensor_Scalar): (0, 1),
        (torch.int8, torch.ops.aten.linspace.Scalar_Tensor): (0, 1),
        (torch.uint8, torch.ops.aten.linspace.Scalar_Tensor): (0, 1),
        (torch.int16, torch.ops.aten.linspace.Scalar_Tensor): (0, 1),
        (torch.int32, torch.ops.aten.linspace.Scalar_Tensor): (0, 1),
        (torch.int64, torch.ops.aten.linspace.Scalar_Tensor): (0, 1),
        (torch.float64,torch.ops.aten._native_batch_norm_legit.default):(3e-7,5e-7), # adjust tolerance for xpu, so hook this func
    }
    if (decomp.dtype, op) in tol_table:
        rtol, atol = tol_table[(decomp.dtype, op)]
    else:
        rtol, atol = _getDefaultRtolAndAtol(orig.dtype, decomp.dtype)
    test_case.assertEqual(
        orig,
        decomp,
        rtol=rtol,
        atol=atol,
        msg=f"{op.__name__}\nargs = {args}\nkwargs = {kwargs}",
    )
test_decomp.op_assert_equal=_op_assert_equal

@skipIfCrossRef
def _test_amp_batch_norm_backward(self):
    device = "xpu"
    grad_out = torch.randn((1, 2, 16, 16), dtype=torch.float16, device=device)
    x = torch.randn((1, 2, 16, 16), dtype=torch.float16, device=device)
    weight = torch.randn((2,), dtype=torch.float32, device=device)
    rmean = torch.randn((2,), dtype=torch.float32, device=device)
    rvar = torch.randn((2,), dtype=torch.float32, device=device)
    mean = torch.randn((0,), dtype=torch.float32, device=device)

    ref = torch.ops.aten.native_batch_norm_backward(
        grad_out,
        x,
        weight,
        rmean,
        rvar,
        mean,
        mean,
        False,
        1e-05,
        [True, True, True],
    )
    res = torch._decomp.decompositions.native_batch_norm_backward(
        grad_out,
        x,
        weight,
        rmean,
        rvar,
        mean,
        mean,
        False,
        1e-05,
        [True, True, True],
    )
    for a, b in zip(ref, res):
        self.assertEqual(a.stride(), b.stride())
        self.assertEqual(a.dtype, b.dtype)
DecompOneOffTests.test_amp_batch_norm_backward=_test_amp_batch_norm_backward

instantiate_device_type_tests(TestDecomp, globals(), only_for="xpu", allow_xpu=True)
instantiate_device_type_tests(DecompOneOffTests, globals(), only_for="xpu", allow_xpu=True)


if __name__ == "__main__":
    run_tests()
