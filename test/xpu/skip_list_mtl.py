skip_dict = {
    "test_torch_xpu":(
        # segment fault
        "test_to",
        "test_to_with_tensor",
        # AssertionError: Tensor-likes are not close!
        # Mismatched elements: 119 / 128 (93.0%)
        # Greatest absolute difference: 0.04191987216472626 at index (2, 0, 13) (up to 0.001 allowed)
        # Greatest relative difference: 0.2167036384344101 at index (2, 0, 21) (up to 0 allowed)
        "test_cdist_cuda_backward_xpu",
    ),
    "test_ops_xpu.py":(
        # AssertionError: Tensor-likes are not close!
        # Mismatched elements: 14 / 176 (8.0%)
        # Greatest absolute difference: 1.1348890364718799e-05 at index (0, 1, 7) (up to 1e-07 allowed)
        # Greatest relative difference: 4.11529848906818e-07 at index (0, 0, 7) (up to 1e-07 allowed)
        "test_numpy_ref_nn_functional_conv_transpose1d_xpu_complex128",
        # AssertionError: Tensor-likes are not close!
        # Mismatched elements: 1 / 18 (5.6%)
        # Greatest absolute difference: 4.182552473253054e-07 at index (0, 1, 2) (up to 1e-07 allowed)
        # Greatest relative difference: 4.5110217125666474e-07 at index (0, 1, 2) (up to 1e-07 allowed)
        "test_numpy_ref_nn_functional_conv_transpose1d_xpu_float64",
    ),
    "test_modules_xpu":(
        # torch.autograd.gradcheck.GradcheckError: Jacobian mismatch for output 0 with respect to input 0,
        # numerical:tensor(0.0484, device='xpu:0', dtype=torch.float64)
        # analytical:tensor(-0.0020, device='xpu:0', dtype=torch.float64)
        "test_grad_nn_ConvTranspose1d_xpu_float64",
        # torch.autograd.gradcheck.GradcheckError: Jacobian mismatch for output 0 with respect to input 0,
        # numerical:tensor(0.0303, device='xpu:0', dtype=torch.float64)
        # analytical:tensor(0.0121, device='xpu:0', dtype=torch.float64)
        "test_grad_nn_ConvTranspose2d_xpu_float64",
        # torch.autograd.gradcheck.GradcheckError: Jacobian mismatch for output 0 with respect to input 2,
        # numerical:tensor(16.9583, device='xpu:0', dtype=torch.float64)
        # analytical:tensor(17.0157, device='xpu:0', dtype=torch.float64)
        "test_grad_nn_ConvTranspose3d_xpu_float64",
        # torch.autograd.gradcheck.GradcheckError: Jacobian mismatch for output 0 with respect to input 0,
        # numerical:tensor(0.0944, device='xpu:0', dtype=torch.float64)
        # analytical:tensor(0.1954, device='xpu:0', dtype=torch.float64)
        "test_grad_nn_LazyConvTranspose1d_xpu_float64",
        # torch.autograd.gradcheck.GradcheckError: Jacobian mismatch for output 0 with respect to input 0,
        # numerical:tensor(0.0103, device='xpu:0', dtype=torch.float64)
        # analytical:tensor(-0.0196, device='xpu:0', dtype=torch.float64)
        "test_grad_nn_LazyConvTranspose2d_xpu_float64",
        # torch.autograd.gradcheck.GradcheckError: Jacobian mismatch for output 0 with respect to input 0,
        # numerical:tensor(-0.0240, device='xpu:0', dtype=torch.float64)
        # analytical:tensor(-0.0562, device='xpu:0', dtype=torch.float64)
        "test_grad_nn_LazyConvTranspose3d_xpu_float64",
        # torch.autograd.gradcheck.GradcheckError: Jacobian mismatch for output 0 with respect to input 1,
        # numerical:tensor(0.9227, device='xpu:0', dtype=torch.float64)
        # analytical:tensor(0.9253, device='xpu:0', dtype=torch.float64)
        "test_gradgrad_nn_Conv1d_xpu_float64",
        # torch.autograd.gradcheck.GradcheckError: Jacobian mismatch for output 0 with respect to input 1,
        # numerical:tensor(0.9752, device='xpu:0', dtype=torch.float64)
        # analytical:tensor(0.9773, device='xpu:0', dtype=torch.float64)
        "test_gradgrad_nn_Conv2d_xpu_float64",
        # torch.autograd.gradcheck.GradcheckError: Jacobian mismatch for output 0 with respect to input 2,
        # numerical:tensor(0.0410, device='xpu:0', dtype=torch.float64)
        # analytical:tensor(0.0387, device='xpu:0', dtype=torch.float64)
        "test_gradgrad_nn_Conv3d_xpu_float64",
        # torch.autograd.gradcheck.GradcheckError: Jacobian mismatch for output 0 with respect to input 1,
        # numerical:tensor(0.4862, device='xpu:0', dtype=torch.float64)
        # analytical:tensor(0.4834, device='xpu:0', dtype=torch.float64)
        "test_gradgrad_nn_LazyConv1d_xpu_float64",
        # torch.autograd.gradcheck.GradcheckError: Jacobian mismatch for output 0 with respect to input 1,
        # numerical:tensor(0.6418, device='xpu:0', dtype=torch.float64)
        # analytical:tensor(0.6452, device='xpu:0', dtype=torch.float64)
        "test_gradgrad_nn_LazyConv2d_xpu_float64",
        # torch.autograd.gradcheck.GradcheckError: Jacobian mismatch for output 0 with respect to input 3,
        # numerical:tensor(0.0641, device='xpu:0', dtype=torch.float64)
        # analytical:tensor(0.0695, device='xpu:0', dtype=torch.float64)
        "test_gradgrad_nn_LazyConv3d_xpu_float64",
    ),
    "test_indexing_xpu.py": (
        "test_index_put_accumulate_large_tensor_xpu",
    ),
    "test_nn_xpu.py": (
        "test_grid_sample_large_xpu",
    ),
    "test_tensor_creation_ops_xpu.py": (
        "test_float_to_int_conversion_finite_xpu_int64",
    ),
    "test_autograd_xpu": (
        # AssertionError: RuntimeError not raised
        "test_increment_version",
    ),
    "test_linalg_xpu": (
        # AssertionError: tensor(False, device='xpu:0') is not true
        "test_addmm_baddbmm_overflow_xpu_float16",
        # Mismatched elements: 50 / 50 (100.0%)
        # Greatest absolute difference: 0.0546875 at index (0,) (up to 0.001 allowed)
        # Greatest relative difference: 0.0113983154296875 at index (0,) (up to 0.001 allowed)
        "test_addmv_xpu_float16",
        # AssertionError: False is not true
        "test_hipblaslt_corner_cases_rocm_xpu_float16",
        # Mismatched elements: 7976517 / 31719908 (25.1%)
        # Greatest absolute difference: 0.109375 at index (56096, 0, 3) (up to 1e-05 allowed)
        # Greatest relative difference: 0.005680084228515625 at index (56096, 0, 3) (up to 0.001 allowed)
        "test_matmul_45724_xpu",
    ),
    "test_ops_fwd_gradients_xpu":(
        # torch.autograd.gradcheck.GradcheckError: While considering the imaginary part of complex inputs only, Jacobian computed with forward mode mismatch for output 0 with respect to input 1,
        # The max per-element difference (slow mode) is: 0.7384582141529449.
        "test_fn_fwgrad_bwgrad_nn_functional_conv1d_xpu_complex128",
        # torch.autograd.gradcheck.GradcheckError: Jacobian computed with forward mode mismatch for output 0 with respect to input 1,
        # The max per-element difference (slow mode) is: 0.2821628302335739.
        "test_fn_fwgrad_bwgrad_nn_functional_conv1d_xpu_float64",
        # torch.autograd.gradcheck.GradcheckError: While considering the imaginary part of complex inputs only, Jacobian computed with forward mode mismatch for output 0 with respect to input 1,
        # The max per-element difference (slow mode) is: 1.1286938618955917.
        "test_fn_fwgrad_bwgrad_nn_functional_conv2d_xpu_complex128",
        # torch.autograd.gradcheck.GradcheckError: Jacobian computed with forward mode mismatch for output 0 with respect to input 1,
        # The max per-element difference (slow mode) is: 0.7601092010736465.
        "test_fn_fwgrad_bwgrad_nn_functional_conv2d_xpu_float64",
        # torch.autograd.gradcheck.GradcheckError: While considering the imaginary part of complex inputs only, Jacobian computed with forward mode mismatch for output 0 with respect to input 0,
        # The max per-element difference (slow mode) is: 9.149260912865309.
        "test_forward_mode_AD_nn_functional_conv_transpose1d_xpu_complex128",
        # torch.autograd.gradcheck.GradcheckError: Jacobian computed with forward mode mismatch for output 0 with respect to input 0,
        # The max per-element difference (slow mode) is: 2.635165214538574.
        "test_forward_mode_AD_nn_functional_conv_transpose1d_xpu_float64",
    ),
    "test_ops_gradients_xpu": (
        # torch.autograd.gradcheck.GradcheckError: While considering the imaginary part of complex outputs only, Jacobian mismatch for output 0 with respect to input 0,
        # numerical:tensor(0.0295-9.1728j, device='xpu:0', dtype=torch.complex128)
        # analytical:tensor(2.4351-9.0580j, device='xpu:0', dtype=torch.complex128)
        "test_fn_grad_nn_functional_conv_transpose1d_xpu_complex128",
        # torch.autograd.gradcheck.GradcheckError: Jacobian mismatch for output 0 with respect to input 0,
        # numerical:tensor(4.8470, device='xpu:0', dtype=torch.float64)
        # analytical:tensor(3.8449, device='xpu:0', dtype=torch.float64)
        "test_fn_grad_nn_functional_conv_transpose1d_xpu_float64",
        # torch.autograd.gradcheck.GradcheckError: While considering the imaginary part of complex outputs only, Jacobian mismatch for output 0 with respect to input 1,
        # numerical:tensor(-0.0944+0.3117j, device='xpu:0', dtype=torch.complex128)
        # analytical:tensor(-0.0929+0.2638j, device='xpu:0', dtype=torch.complex128)
        "test_fn_gradgrad_nn_functional_conv1d_xpu_complex128",
        # torch.autograd.gradcheck.GradcheckError: Jacobian mismatch for output 0 with respect to input 1,
        # numerical:tensor(0.1454, device='xpu:0', dtype=torch.float64)
        # analytical:tensor(-0.0454, device='xpu:0', dtype=torch.float64)
        "test_fn_gradgrad_nn_functional_conv1d_xpu_float64",
        # torch.autograd.gradcheck.GradcheckError: While considering the imaginary part of complex outputs only, Jacobian mismatch for output 0 with respect to input 1,
        # numerical:tensor(0.0952+0.1113j, device='xpu:0', dtype=torch.complex128)
        # analytical:tensor(0.0329-0.0138j, device='xpu:0', dtype=torch.complex128)
        "test_fn_gradgrad_nn_functional_conv2d_xpu_complex128",
        # torch.autograd.gradcheck.GradcheckError: Jacobian mismatch for output 0 with respect to input 1,
        # numerical:tensor(0.0979, device='xpu:0', dtype=torch.float64)
        # analytical:tensor(-0.0408, device='xpu:0', dtype=torch.float64)
        "test_fn_gradgrad_nn_functional_conv2d_xpu_float64",
    ),
    "test_dynamic_shapes_xpu": (
        # AttributeError: 'DimConstraints' object has no attribute 'remove_redundant_dynamic_results'
        "test_dim_constraints_solve_full",
    )
}
