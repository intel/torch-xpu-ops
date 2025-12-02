import os
import sys

# Cases in the file is too slow to run all suites on CPU. So add white list.


def launch_test(test_case, skip_list=None, exe_list=None):
    os.environ["PYTORCH_TEST_WITH_SLOW"] = "1"
    if skip_list is not None:
        skip_options = ' -k "not ' + skip_list[0]
        for skip_case in skip_list[1:]:
            skip_option = " and not " + skip_case
            skip_options += skip_option
        skip_options += '"'
        test_command = (
            "pytest --junit-xml=./op_ut_with_only.xml " + test_case + skip_options
        )
        return os.system(test_command)
    elif exe_list is not None:
        exe_options = ' -k "' + exe_list[0]
        for exe_case in exe_list[1:]:
            exe_option = " or " + exe_case
            exe_options += exe_option
        exe_options += '"'
        test_command = (
            "pytest --junit-xml=./op_ut_with_only.xml " + test_case + exe_options
        )
        return os.system(test_command)
    else:
        test_command = "pytest --junit-xml=./op_ut_with_only.xml " + test_case
        return os.system(test_command)


res = 0

# test_decomp
# full skip_list is in Issue #470
execute_list = (
    "test_comprehensive_nn_functional_cross_entropy_xpu",
    "test_comprehensive_nn_functional_nll_loss_xpu_bfloat16",
    "test_comprehensive_nn_functional_nll_loss_xpu_float32",
    "test_comprehensive_nn_functional_nll_loss_xpu_float64",
    "bincount",
)
skip_list = (
    "test_comprehensive_baddbmm_xpu_float64",
    "test_comprehensive_logspace_tensor_overload_xpu_int16",
    "test_comprehensive_logspace_tensor_overload_xpu_int32",
    "test_comprehensive_logspace_tensor_overload_xpu_int64",
    "test_comprehensive_logspace_xpu_int16",
    "test_comprehensive_logspace_xpu_int32",
    "test_comprehensive_logspace_xpu_int64",
    "test_comprehensive_nn_functional_conv_transpose2d_xpu_bfloat16",
    "test_comprehensive_nn_functional_conv_transpose2d_xpu_complex128",
    "test_comprehensive_nn_functional_conv_transpose2d_xpu_complex32",
    "test_comprehensive_nn_functional_conv_transpose2d_xpu_complex64",
    "test_comprehensive_nn_functional_conv_transpose2d_xpu_float16",
    "test_comprehensive_nn_functional_conv_transpose2d_xpu_float32",
    "test_comprehensive_nn_functional_conv_transpose2d_xpu_float64",
    "test_comprehensive_nn_functional_conv_transpose3d_xpu_bfloat16",
    "test_comprehensive_nn_functional_conv_transpose3d_xpu_complex128",
    "test_comprehensive_nn_functional_conv_transpose3d_xpu_complex32",
    "test_comprehensive_nn_functional_conv_transpose3d_xpu_complex64",
    "test_comprehensive_nn_functional_conv_transpose3d_xpu_float16",
    "test_comprehensive_nn_functional_conv_transpose3d_xpu_float32",
    "test_comprehensive_nn_functional_conv_transpose3d_xpu_float64",
    "test_comprehensive_nn_functional_instance_norm_xpu_float64",
    "test_comprehensive_nn_functional_nll_loss_xpu_float16",
    "test_comprehensive_nn_functional_pad_reflect_xpu_bfloat16",
    "test_comprehensive_torch_ops_aten__flash_attention_forward_xpu_float16",
    "test_comprehensive_vdot_xpu_complex128",
    "test_comprehensive_vdot_xpu_complex64",
    "test_quick_addmm_xpu_float64",
    "test_quick_baddbmm_xpu_float64",
    "test_quick_core_backward_baddbmm_xpu_float64",
    "test_quick_core_backward_mv_xpu_float64",
    "test_quick_logspace_tensor_overload_xpu_int16",
    "test_quick_logspace_tensor_overload_xpu_int32",
    "test_quick_logspace_tensor_overload_xpu_int64",
    "test_quick_logspace_xpu_int16",
    "test_quick_logspace_xpu_int32",
    "test_quick_logspace_xpu_int64",
    "test_quick_vdot_xpu_complex128",
    "test_quick_vdot_xpu_complex64",
    "test_exponential_non_inf_xpu",
    "test_aten_core_operators",
    "test_has_decomposition",
    "test_comprehensive_diff_xpu_complex128",
    "test_comprehensive_ormqr_xpu_complex128",
    "test_quick_var_mean_xpu_float64",
    "test_comprehensive_diff_xpu_complex64",
    "test_comprehensive_ormqr_xpu_complex64",
    "test_quick_mean_xpu_complex128",
    "test_comprehensive_grid_sampler_2d_xpu_bfloat16",
)
# res += launch_test("test_decomp_xpu.py", exe_list=execute_list)
res += launch_test("test_decomp.py", skip_list=skip_list)

if os.name == "nt":
    sys.exit(res)
else:
    exit_code = os.WEXITSTATUS(res)
    sys.exit(exit_code)
