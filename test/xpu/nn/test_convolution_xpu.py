# Owner(s): ["module: intel"]

import torch
import unittest
import torch.autograd.forward_ad as fwAD
from functools import wraps
from itertools import product
from torch import nn
import torch.nn.functional as F
from torch.testing._internal.common_device_type import (
    disablecuDNN,
    disableMkldnn,
    dtypes,
    largeTensorTest,
    instantiate_device_type_tests,
    onlyXPU,
    skipIf,
)
from torch.testing._internal.common_dtype import (
    floating_and_complex_types_and,
)
from torch.testing._internal.common_utils import (
    dtype2prec_DONTUSE,
    gradcheck,
    GRADCHECK_NONDET_TOL,
    gradgradcheck,
    instantiate_parametrized_tests,
    parametrize as parametrize_test,
    run_tests,
    subtest,
)

# Skips a test on XPU if the condition is true.
class skipXPUIf(skipIf):

    def __init__(self, dep, reason):
        super().__init__(dep, reason, device_type='xpu')

# Skips a test on XPU if mkldnn is not available.
def skipXPUIfNoMkldnn(fn):
    return skipXPUIf(not torch.backends.mkldnn.is_available(), "PyTorch is built without mkldnn support")(fn)

def onlyNativeDeviceTypes(fn):
    @wraps(fn)
    def only_fn(self, *args, **kwargs):
        if self.device_type not in ("xpu", "cpu"):
            reason = f"onlyNativeDeviceTypes: doesn't run on {self.device_type}"
            raise unittest.SkipTest(reason)

        return fn(self, *args, **kwargs)

    return only_fn

try:
    from .xpu_test_utils import XPUPatchForImport
except Exception as e:
    from ..xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_convolution import TestConvolutionNN, TestConvolutionNNDeviceType

    @dtypes(torch.float, torch.double, torch.half)
    # Very similar to test_Conv2d_naive_groups but with special care to handle
    # the number of groups == number of input channels
    def conv2d_depthwise_naive_groups(self, device, dtype):
        for depth_multiplier in [1, 2]:
            m = torch.nn.Conv2d(2, 2 * depth_multiplier, kernel_size=3, groups=2).to(
                device, dtype
            )
            i = (
                torch.randn(2, 2, 6, 6, device=device, dtype=dtype)
                .div_(2)
                .requires_grad_()
            )
            output = m(i)
            grad_output = (
                torch.randn(2, 2 * depth_multiplier, 4, 4, device=device, dtype=dtype)
                / 2
            )
            output.backward(grad_output)

            offset = 1 * depth_multiplier

            m1 = torch.nn.Conv2d(1, 1 * depth_multiplier, kernel_size=3).to(device, dtype)
            m1.weight.data = m.weight.data[:offset].clone()
            m1.bias.data = m.bias.data[:offset].clone()
            i1 = i.detach()[:, :1].clone().requires_grad_()
            output1 = m1(i1)
            output1.backward(grad_output[:, :offset].contiguous())

            m2 = torch.nn.Conv2d(1, 1 * depth_multiplier, kernel_size=3).to(device, dtype)
            m2.weight.data.copy_(m.weight.data[offset:])
            m2.bias.data.copy_(m.bias.data[offset:])
            i2 = i.detach()[:, 1:].clone().requires_grad_()
            output2 = m2(i2)
            output2.backward(grad_output[:, offset:].contiguous())

            self.assertEqual(
                output,
                torch.cat([output1, output2], 1),
                atol=dtype2prec_DONTUSE[dtype],
                rtol=0,
            )
            self.assertEqual(
                i.grad.data,
                torch.cat([i1.grad.data, i2.grad.data], 1),
                atol=dtype2prec_DONTUSE[dtype],
                rtol=0,
            )
            self.assertEqual(
                m.bias.grad.data,
                torch.cat([m1.bias.grad.data, m2.bias.grad.data], 0),
                atol=dtype2prec_DONTUSE[dtype],
                rtol=0,
            )
            self.assertEqual(
                m.weight.grad.data,
                torch.cat([m1.weight.grad.data, m2.weight.grad.data], 0),
                atol=dtype2prec_DONTUSE[dtype],
                rtol=0,
            )

    @dtypes(torch.float, torch.double, torch.half)
    def conv3d_depthwise_naive_groups(self, device, dtype):
        for depth_multiplier in [1, 2]:
            m = nn.Conv3d(2, 2 * depth_multiplier, kernel_size=3, groups=2).to(
                device, dtype
            )
            i = (
                torch.randn(2, 2, 6, 6, 6, device=device, dtype=dtype)
                .div_(2)
                .requires_grad_()
            )
            output = m(i)
            grad_output = (
                torch.randn(
                    2, 2 * depth_multiplier, 4, 4, 4, device=device, dtype=dtype
                )
                / 2
            )
            output.backward(grad_output)

            offset = 1 * depth_multiplier

            m1 = nn.Conv3d(1, 1 * depth_multiplier, kernel_size=3).to(device, dtype)
            m1.weight.data = m.weight.data[:offset].clone()
            m1.bias.data = m.bias.data[:offset].clone()
            i1 = i.detach()[:, :1].clone().requires_grad_()
            output1 = m1(i1)
            output1.backward(grad_output[:, :offset].contiguous())

            m2 = nn.Conv3d(1, 1 * depth_multiplier, kernel_size=3).to(device, dtype)
            m2.weight.data.copy_(m.weight.data[offset:])
            m2.bias.data.copy_(m.bias.data[offset:])
            i2 = i.detach()[:, 1:].clone().requires_grad_()
            output2 = m2(i2)
            output2.backward(grad_output[:, offset:].contiguous())
            is_cuda_sm86 = device.startswith(
                "cuda"
            ) and torch.cuda.get_device_capability(0) == (8, 6)
            atol, rtol = (
                (3e-4, 3e-2)
                if dtype == torch.float32 and is_cuda_sm86
                else (dtype2prec_DONTUSE[dtype], 0)
            )

            self.assertEqual(
                output, torch.cat([output1, output2], 1), atol=atol, rtol=rtol
            )
            self.assertEqual(
                i.grad.data,
                torch.cat([i1.grad.data, i2.grad.data], 1),
                atol=dtype2prec_DONTUSE[dtype],
                rtol=0,
            )
            self.assertEqual(
                m.bias.grad.data,
                torch.cat([m1.bias.grad.data, m2.bias.grad.data], 0),
                atol=dtype2prec_DONTUSE[dtype],
                rtol=0,
            )
            self.assertEqual(
                m.weight.grad.data,
                torch.cat([m1.weight.grad.data, m2.weight.grad.data], 0),
                atol=atol,
                rtol=rtol,
            )

    @dtypes(torch.half, torch.float)
    def convTranspose2d_large_output_padding(self, device, dtype):
        net1 = torch.nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
        ).to(device=device, dtype=dtype)
        net2 = torch.nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
        ).to(device=device, dtype=dtype)
        net3 = torch.nn.ConvTranspose2d(
            32, 3, kernel_size=3, stride=2, padding=1, output_padding=1
        ).to(device=device, dtype=dtype)
        x = torch.rand(1, 128, 6, 6, device=device, dtype=dtype, requires_grad=True)
        x = net1(x)
        x = net2(x)
        x = net3(x)
        x.backward(torch.randn_like(x))
        torch.xpu.synchronize()

    @parametrize_test(
        "input_shape,transposed,dilated,groups,layout,backend_expected",
        [
            # === slow ===
            subtest(
                (
                    (2, 6, 7),
                    False,
                    False,
                    3,
                    torch.strided,
                    torch._C._ConvBackend.Overrideable,
                ),
                decorators=[onlyNativeDeviceTypes, disableMkldnn, disablecuDNN],
                name="slow1d",
            ),
            subtest(
                (
                    (2, 6, 7),
                    True,
                    False,
                    3,
                    torch.strided,
                    torch._C._ConvBackend.Overrideable,
                ),
                decorators=[onlyNativeDeviceTypes, disableMkldnn, disablecuDNN],
                name="slow1d_transposed",
            ),
            subtest(
                (
                    (2, 6, 7),
                    False,
                    True,
                    3,
                    torch.strided,
                    torch._C._ConvBackend.Overrideable,
                ),
                decorators=[onlyNativeDeviceTypes, disableMkldnn, disablecuDNN],
                name="slow1d_dilated",
            ),
            subtest(
                (
                    (2, 6, 7),
                    True,
                    True,
                    3,
                    torch.strided,
                    torch._C._ConvBackend.Overrideable,
                ),
                decorators=[onlyNativeDeviceTypes, disableMkldnn, disablecuDNN],
                name="slow1d_dilated_transposed",
            ),
            subtest(
                (
                    (2, 6, 7, 8),
                    False,
                    False,
                    3,
                    torch.strided,
                    torch._C._ConvBackend.Overrideable,
                ),
                decorators=[onlyNativeDeviceTypes, disableMkldnn, disablecuDNN],
                name="slow2d",
            ),
            subtest(
                (
                    (2, 6, 7, 8),
                    True,
                    False,
                    3,
                    torch.strided,
                    torch._C._ConvBackend.Overrideable,
                ),
                decorators=[onlyNativeDeviceTypes, disableMkldnn, disablecuDNN],
                name="slow2d_transposed",
            ),
            subtest(
                (
                    (2, 6, 7, 8),
                    False,
                    True,
                    3,
                    torch.strided,
                    torch._C._ConvBackend.Overrideable,
                ),
                decorators=[onlyNativeDeviceTypes, disableMkldnn, disablecuDNN],
                name="slow2d_dilated",
            ),
            subtest(
                (
                    (2, 6, 7, 8),
                    True,
                    True,
                    3,
                    torch.strided,
                    torch._C._ConvBackend.Overrideable,
                ),
                decorators=[onlyNativeDeviceTypes, disableMkldnn, disablecuDNN],
                name="slow2d_dilated_transposed",
            ),
            # XPU doesn't have a slow 3D implementation, so it goes to the dilated 3D implementation instead
            subtest(
                (
                    (2, 6, 7, 8, 9),
                    False,
                    False,
                    3,
                    torch.strided,
                    torch._C._ConvBackend.Overrideable,
                ),
                decorators=[onlyXPU, disableMkldnn],
                name="slow3d_xpu",
            ),
            # FIXME: RuntimeError: CUDA out of memory.
            # subtest(((2, 6, 7, 8, 9), True, False, 3, torch.strided, torch._C._ConvBackend.SlowTranspose3d),
            #         decorators=[onlyNativeDeviceTypes, disableMkldnn, disablecuDNN], name='slow3d_transposed'),
            subtest(
                (
                    (2, 6, 7, 8, 9),
                    False,
                    True,
                    3,
                    torch.strided,
                    torch._C._ConvBackend.Overrideable,
                ),
                decorators=[onlyNativeDeviceTypes, disableMkldnn, disablecuDNN],
                name="slow3d_dilated",
            ),
            # FIXME: RuntimeError: CUDA out of memory.
            # subtest(((2, 6, 7, 8, 9), True, True, 3, torch.strided, torch._C._ConvBackend.SlowTranspose3d),
            #         decorators=[onlyNativeDeviceTypes, disableMkldnn, disablecuDNN], name='slow3d_dilated_transposed'),
            subtest(
                (
                    (0, 6, 7),
                    False,
                    False,
                    3,
                    torch.strided,
                    torch._C._ConvBackend.Empty,
                ),
                decorators=[onlyNativeDeviceTypes, disableMkldnn],
                name="empty_batch1d",
            ),
            subtest(
                (
                    (2, 0, 7),
                    False,
                    False,
                    3,
                    torch.strided,
                    torch._C._ConvBackend.Empty,
                ),
                decorators=[onlyNativeDeviceTypes, disableMkldnn],
                name="empty_channel1d",
            ),
            subtest(
                (
                    (0, 0, 7),
                    False,
                    False,
                    3,
                    torch.strided,
                    torch._C._ConvBackend.Empty,
                ),
                decorators=[onlyNativeDeviceTypes, disableMkldnn],
                name="empty_batch_channel1d",
            ),
            subtest(
                (
                    (0, 6, 7, 8),
                    False,
                    False,
                    3,
                    torch.strided,
                    torch._C._ConvBackend.Empty,
                ),
                decorators=[onlyNativeDeviceTypes, disableMkldnn],
                name="empty_batch2d",
            ),
            subtest(
                (
                    (2, 0, 7, 8),
                    False,
                    False,
                    3,
                    torch.strided,
                    torch._C._ConvBackend.Empty,
                ),
                decorators=[onlyNativeDeviceTypes, disableMkldnn],
                name="empty_channel2d",
            ),
            subtest(
                (
                    (0, 0, 7, 8),
                    False,
                    False,
                    3,
                    torch.strided,
                    torch._C._ConvBackend.Empty,
                ),
                decorators=[onlyNativeDeviceTypes, disableMkldnn],
                name="empty_batch_channel2d",
            ),
            subtest(
                (
                    (0, 6, 7, 8, 9),
                    False,
                    False,
                    3,
                    torch.strided,
                    torch._C._ConvBackend.Empty,
                ),
                decorators=[onlyNativeDeviceTypes, disableMkldnn],
                name="empty_batch3d",
            ),
            subtest(
                (
                    (2, 0, 7, 8, 9),
                    False,
                    False,
                    3,
                    torch.strided,
                    torch._C._ConvBackend.Empty,
                ),
                decorators=[onlyNativeDeviceTypes, disableMkldnn],
                name="empty_channel3d",
            ),
            subtest(
                (
                    (0, 0, 7, 8, 9),
                    False,
                    False,
                    3,
                    torch.strided,
                    torch._C._ConvBackend.Empty,
                ),
                decorators=[onlyNativeDeviceTypes, disableMkldnn],
                name="empty_batch_channel3d",
            ),
            subtest(
                (
                    (2, 6, 7),
                    False,
                    False,
                    6,
                    torch.strided,
                    torch._C._ConvBackend.Overrideable,
                ),
                decorators=[onlyXPU, disableMkldnn],
                name="xpu_depthwise1d",
            ),
            subtest(
                (
                    (2, 6, 7, 8),
                    False,
                    False,
                    6,
                    torch.strided,
                    torch._C._ConvBackend.Overrideable,
                ),
                decorators=[onlyXPU, disableMkldnn],
                name="xpu_depthwise2d",
            ),
            subtest(
                (
                    (2, 6, 7, 8, 9),
                    False,
                    False,
                    6,
                    torch.strided,
                    torch._C._ConvBackend.Overrideable,
                ),
                decorators=[onlyXPU, disableMkldnn],
                name="xpu_depthwise3d",
            ),
            # === xpu ===
            subtest(
                (
                    (2, 6, 7),
                    False,
                    False,
                    3,
                    torch.strided,
                    torch._C._ConvBackend.Overrideable,
                ),
                decorators=[onlyXPU, skipXPUIfNoMkldnn],
                name="xpu1d",
            ),
            subtest(
                (
                    (2, 6, 7, 8),
                    False,
                    False,
                    3,
                    torch.strided,
                    torch._C._ConvBackend.Overrideable,
                ),
                decorators=[onlyXPU, skipXPUIfNoMkldnn],
                name="xpu2d",
            ),
            subtest(
                (
                    (2, 6, 7, 8, 9),
                    False,
                    False,
                    3,
                    torch.strided,
                    torch._C._ConvBackend.Overrideable,
                ),
                decorators=[onlyXPU, skipXPUIfNoMkldnn],
                name="xpu3d",
            ),
            subtest(
                (
                    (2, 6, 7),
                    True,
                    False,
                    3,
                    torch.strided,
                    torch._C._ConvBackend.Overrideable,
                ),
                decorators=[onlyXPU, skipXPUIfNoMkldnn],
                name="xpu1d_transposed",
            ),
            subtest(
                (
                    (2, 6, 7, 8),
                    True,
                    False,
                    3,
                    torch.strided,
                    torch._C._ConvBackend.Overrideable,
                ),
                decorators=[onlyXPU, skipXPUIfNoMkldnn],
                name="xpu2d_transposed",
            ),
            # Note: Tests for mobile backends are not currently supported. This comprises
            # NnpackSpatial, Winograd3x3Depthwise, and Xnnpack2d backends. Testing these
            # requires the ability to gate tests by whether PyTorch is built with USE_MOBILE=1.
        ],
    )
    # Test with both bias and no bias.
    @parametrize_test("has_bias", [False, True])
    # Test with both stride=1 and stride>1 cases.
    @parametrize_test("strided", [False, True])
    # Test with both contiguous and non-contiguous inputs.
    @parametrize_test("contiguous", [False, True])
    def conv_backend(
        self,
        device,
        input_shape,
        has_bias,
        strided,
        contiguous,
        transposed,
        dilated,
        groups,
        layout,
        backend_expected,
    ):
        # Build up inputs.
        dtype = torch.float32
        C_in, C_out, dim, kernel_size = input_shape[1], 12, len(input_shape) - 2, 3
        x = torch.randn(*input_shape, device=device, dtype=dtype, requires_grad=True)
        weight = torch.randn(
            C_in if transposed else C_out,
            C_out // groups if transposed else C_in // groups,
            *[kernel_size for _ in range(dim)],
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        bias = (
            torch.randn(C_out, device=device, dtype=dtype, requires_grad=True)
            if has_bias
            else None
        )

        def _make_noncontiguous(inp):
            if inp is None:
                return None
            old_requires_grad = inp.requires_grad
            inp = torch.repeat_interleave(inp, 2, dim=-1)
            inp = inp[..., ::2].detach().requires_grad_(old_requires_grad)
            return inp

        if not contiguous:
            x = _make_noncontiguous(x)
            weight = _make_noncontiguous(weight)
            bias = _make_noncontiguous(bias)

        if layout is torch._mkldnn:
            x = x.to_mkldnn()
            # Note that weight and bias are not supported as mkldnn tensors during training.

        stride = (2,) * dim if strided else (1,) * dim
        padding = (0,) * dim
        dilation = (2,) * dim if dilated else (1,) * dim
        output_padding = (0,) * dim
        inputs = [
            x,
            weight,
            bias,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
        ]

        # Ensure correct backend is selected.
        backend_actual = torch._C._select_conv_backend(*inputs)
        self.assertEqual(backend_actual, backend_expected)

        # Ensure backward call succeeds.
        convolution = torch.ops.aten.convolution
        output = convolution(*inputs)
        grad_output = torch.randn(output.shape, device=device, dtype=dtype)
        if not contiguous:
            grad_output = _make_noncontiguous(grad_output)
        if layout is torch._mkldnn:
            grad_output = grad_output.to_mkldnn()
        output.backward(grad_output)

        # mkldnn doesn't support gradcheck :(
        if layout is torch._mkldnn:
            return

        if backend_actual != torch._C._ConvBackend.Empty:  # FIXME: forward AD fails
            # Forward AD and forward-over-reverse AD smoke test in float32
            # TODO: remove this if we introduce per-op gradient tests for float32
            with fwAD.dual_level():
                dual_inputs = [
                    (
                        fwAD.make_dual(i, torch.rand_like(i))
                        if isinstance(i, torch.Tensor)
                        else i
                    )
                    for i in inputs
                ]
                # Forward AD
                output = convolution(*dual_inputs)
                # Forward over reverse AD
                grad_output_d = fwAD.make_dual(
                    torch.rand_like(output), torch.rand_like(output)
                )
                if has_bias:
                    torch.autograd.grad(output, [x, weight, bias], grad_output_d)
                else:
                    torch.autograd.grad(output, [x, weight], grad_output_d)

        # Convert to float64 for gradcheck.
        x = x.to(torch.float64).detach().requires_grad_(True)
        weight = weight.to(torch.float64).detach().requires_grad_(True)
        if bias is not None:
            bias = bias.to(torch.float64).detach().requires_grad_(True)
        inputs = [
            x,
            weight,
            bias,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
        ]

        # Set some backend-specific validation settings.
        gradcheck_nondet_tol = 0.0
        if torch.backends.mkldnn.is_available():
            # mkldnn introduces non-determinism
            gradcheck_nondet_tol = GRADCHECK_NONDET_TOL

        self.assertTrue(gradcheck(convolution, inputs, nondet_tol=gradcheck_nondet_tol))

        # double backward doesn't support bias gradients
        if bias is not None:
            bias.requires_grad_(False)
        self.assertTrue(
            gradgradcheck(convolution, inputs, nondet_tol=gradcheck_nondet_tol)
        )

    def _test_conv_xpu_nhwc_nchw(self, layer, n, c, h, w, k, filter_size, device):
        data = torch.randint(1, 10, (n, c, h, w), dtype=torch.float32, device=device)
        ref_input = data.clone().contiguous().requires_grad_(True)
        ref_conv = layer(c, k, filter_size).float().to(device)
        ref_out = ref_conv(ref_input)
        grad = torch.randint(1, 10, ref_out.size(), dtype=torch.float32, device="xpu")
        ref_out.backward(grad)

        for w_f in [torch.contiguous_format, torch.channels_last]:
            for g_f in [torch.contiguous_format, torch.channels_last]:
                for input_format in [torch.contiguous_format, torch.channels_last]:
                    output_format = torch.contiguous_format
                    if input_format == torch.channels_last:
                        output_format = torch.channels_last
                    # This is because we have N111 weight that cannot handle
                    # the ambiguous memory_format
                    if w_f == torch.channels_last:
                        if layer == nn.Conv2d and filter_size * c != 1:
                            output_format = torch.channels_last
                        if layer == nn.ConvTranspose2d and filter_size * k != 1:
                            output_format = torch.channels_last
                    self._run_conv(
                        layer,
                        device,
                        data,
                        grad,
                        ref_conv,
                        ref_input,
                        ref_out,
                        input_format,
                        w_f,
                        g_f,
                        output_format,
                    )

    @dtypes(torch.half, torch.float)
    def conv_xpu_ndhwc(self, device, dtype):
        def helper(n, c, d, h, w, out_channels, kernel_size, groups):
            input = torch.randint(
                -2, 2, (n, c, d, h, w), dtype=dtype, device=device
            ).to(memory_format=torch.channels_last_3d)
            input.requires_grad_()
            conv = nn.Conv3d(c, out_channels, kernel_size, groups=groups).to(
                device="xpu", dtype=dtype, memory_format=torch.channels_last_3d
            )
            for p in conv.parameters():
                p.data = torch.randint_like(p, -2, 2)

            # use FP64 channels-first conv as reference
            ref_input = input.detach().clone().contiguous().double().requires_grad_()
            ref_conv = nn.Conv3d(c, out_channels, kernel_size, groups=groups)
            # load_state_dict will restore the stride & memory_layout on ref_conv.weight.
            ref_conv.load_state_dict(conv.state_dict())
            ref_conv = ref_conv.to(
                device="xpu", dtype=torch.double, memory_format=torch.contiguous_format
            )

            out = conv(input)
            ref_out = ref_conv(ref_input)

            grad = torch.randint_like(out, -2, 2)
            ref_grad = grad.detach().clone().double().contiguous()

            out.backward(grad)
            ref_out.backward(ref_grad)

            self.assertTrue(out.is_contiguous(memory_format=torch.channels_last_3d))
            self.assertTrue(
                input.grad.is_contiguous(memory_format=torch.channels_last_3d)
            )
            self.assertTrue(
                conv.weight.grad.is_contiguous(memory_format=torch.channels_last_3d)
            )

            self.assertTrue(ref_out.is_contiguous())
            self.assertTrue(ref_input.grad.is_contiguous())
            self.assertTrue(ref_conv.weight.grad.is_contiguous())

            self.assertEqual(out, ref_out, exact_dtype=False)
            self.assertEqual(conv.weight.grad, ref_conv.weight.grad, exact_dtype=False)
            self.assertEqual(conv.bias.grad, ref_conv.bias.grad, exact_dtype=False)
            self.assertEqual(input.grad, ref_input.grad, exact_dtype=False)

        helper(2, 8, 4, 4, 4, out_channels=4, kernel_size=3, groups=1)
        helper(2, 8, 4, 4, 4, out_channels=8, kernel_size=3, groups=8)
        helper(1, 16, 18, 18, 18, out_channels=16, kernel_size=3, groups=1)
        helper(1, 16, 18, 18, 18, out_channels=16, kernel_size=3, groups=16)

    @dtypes(torch.float, torch.double)
    def conv_xpu_nhwc_support(self, device, dtype):
        input = torch.randn(
            (1, 16, 1, 1), dtype=dtype, device="xpu", requires_grad=True
        )
        weight = torch.randn(
            (8, 16, 3, 3), dtype=dtype, device="xpu", requires_grad=True
        )
        weight = weight.to(memory_format=torch.channels_last)
        o = torch.conv2d(input, weight, None, (2, 1), (1, 1), (1, 1), 1)
        self.assertTrue(o.is_contiguous(memory_format=torch.channels_last))
        o.sum().backward()

    @dtypes(torch.half, torch.float)
    def conv_xpu_nhwc(self, device, dtype):
        def helper(n, c, h, w, out_channels, kernel_size, groups):
            input = torch.randint(-3, 3, (n, c, h, w), dtype=dtype, device=device).to(
                memory_format=torch.channels_last
            )
            input.requires_grad_()
            conv = nn.Conv2d(c, out_channels, kernel_size, groups=groups).to(
                device=device, dtype=dtype, memory_format=torch.channels_last
            )
            for p in conv.parameters():
                p.data = torch.randint_like(p, -3, 3)

            # use FP64 channels-first conv as reference
            ref_input = input.detach().clone().contiguous().double().requires_grad_()
            ref_conv = nn.Conv2d(c, out_channels, kernel_size, groups=groups)
            # load_state_dict will restore the stride & memory_layout on ref_conv.weight.
            ref_conv.load_state_dict(conv.state_dict())
            ref_conv = ref_conv.to(
                device=device, dtype=torch.double, memory_format=torch.contiguous_format
            )

            out = conv(input)
            ref_out = ref_conv(ref_input)

            grad = torch.randint_like(out, -3, 3)
            ref_grad = grad.detach().clone().double().contiguous()

            out.backward(grad)
            ref_out.backward(ref_grad)

            self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
            self.assertTrue(input.grad.is_contiguous(memory_format=torch.channels_last))
            self.assertTrue(
                conv.weight.grad.is_contiguous(memory_format=torch.channels_last)
            )

            self.assertTrue(ref_out.is_contiguous())
            self.assertTrue(ref_input.grad.is_contiguous())
            self.assertTrue(ref_conv.weight.grad.is_contiguous())

            self.assertEqual(out, ref_out, exact_dtype=False)
            self.assertEqual(conv.weight.grad, ref_conv.weight.grad, exact_dtype=False)
            self.assertEqual(conv.bias.grad, ref_conv.bias.grad, exact_dtype=False)
            self.assertEqual(input.grad, ref_input.grad, exact_dtype=False)

        helper(2, 8, 4, 4, out_channels=4, kernel_size=3, groups=1)
        helper(2, 8, 4, 4, out_channels=8, kernel_size=3, groups=8)
        helper(1, 16, 56, 56, out_channels=16, kernel_size=3, groups=1)
        helper(1, 16, 56, 56, out_channels=16, kernel_size=3, groups=16)

    def run_conv_double_back_test(
        kern,
        stride,
        padding,
        chan_in,
        chan_out,
        batch_size,
        inp_size,
        dilation,
        no_weight,
        groups=1,
        use_xpu=False,
        use_bias=True,
        dtype=torch.double,
    ):
        if use_xpu:
            device = torch.device("xpu")
        else:
            device = torch.device("cpu")

        x = torch.randn(
            batch_size,
            chan_in,
            inp_size,
            inp_size,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        weight = torch.randn(
            chan_out,
            chan_in // groups,
            kern,
            kern,
            device=device,
            dtype=dtype,
            requires_grad=not no_weight,
        )
        if use_bias:
            bias = torch.randn(chan_out, device=device, dtype=dtype, requires_grad=True)
        else:
            bias = None

        def func(*inputs):
            if use_bias:
                lx, lweight, lbias = inputs
            else:
                lx, lweight = inputs
                lbias = None
            # We disable mkldnn during forward to avoid finite difference imprecision issues
            with torch.backends.mkldnn.flags(enabled=False):
                out = F.conv2d(lx, lweight, lbias, stride, padding, dilation, groups)
            return out

        if use_bias:
            inputs = x, weight, bias
        else:
            inputs = x, weight

        dummy_out = func(*inputs)
        grad_y = torch.randn_like(
            dummy_out, device=device, dtype=dtype, requires_grad=True
        )

        # Issue #15353: test mkldnn double backward, don't run gradgradcheck due
        # to imprecision issues
        if dtype == torch.float:
            (g,) = torch.autograd.grad(dummy_out.sum(), x, create_graph=True)
            return g.requires_grad

        return gradgradcheck(func, inputs, (grad_y,))

    @dtypes(torch.double)
    def conv_double_backward(self, device, dtype):
        with torch.backends.mkldnn.flags(enabled=True, deterministic=True):
            # Double backward only runs with DoubleTensor due to precision reason
            batch_size = 1
            for kern, inp_size, dilations in [(3, 5, [1, 2]), (4, 9, [1])]:
                for stride, padding, chan_in, chan_out, dilation in product(
                    [1], [2], [2], [3], dilations
                ):
                    no_weight = stride == 2
                    result = run_conv_double_back_test(
                        kern,
                        stride,
                        padding,
                        chan_in,
                        chan_out,
                        batch_size,
                        inp_size,
                        dilation,
                        no_weight,
                        use_xpu=True,
                        dtype=dtype,
                    )
                    self.assertTrue(
                        result,
                        "Conv double backward test failed with parameters:"
                        + "\nkern: "
                        + str(kern)
                        + "\nstride: "
                        + str(stride)
                        + "\npadding: "
                        + str(padding)
                        + "\nchan_in: "
                        + str(chan_in)
                        + "\nchan_out: "
                        + str(chan_out)
                        + "\nbatch_size: "
                        + str(batch_size)
                        + "\ninp_size: "
                        + str(inp_size)
                        + "\ndilation: "
                        + str(dilation),
                    )

    def conv2d_inconsistent_types_on_GPU_with_mkldnn(self):
        inputs = torch.randn(4, 1, 7, 7, dtype=torch.float, device="xpu")
        weights = torch.randn(1, 1, 3, 3, dtype=torch.double, device="xpu")
        bias = torch.randn(1, dtype=torch.double, device="xpu")

        with torch.backends.mkldnn.flags(enabled=True):
            # inconsistent types should raise an exception
            self.assertRaises(
                RuntimeError, lambda: nn.functional.conv2d(inputs, weights)
            )
            self.assertRaises(
                RuntimeError,
                lambda: nn.functional.conv2d(inputs, weights.float(), bias),
            )

            # but it should work with the same type
            nn.functional.conv2d(inputs.float(), weights.float(), bias.float())

    def conv2d_inconsistent_types_on_GPU_without_mkldnn(self):
        inputs = torch.randn(4, 1, 7, 7, dtype=torch.float, device="xpu")
        weights = torch.randn(1, 1, 3, 3, dtype=torch.double, device="xpu")
        bias = torch.randn(1, dtype=torch.double, device="xpu")

        with torch.backends.mkldnn.flags(enabled=False):
            # inconsistent types should raise an exception
            self.assertRaises(
                RuntimeError, lambda: nn.functional.conv2d(inputs, weights)
            )
            self.assertRaises(
                RuntimeError,
                lambda: nn.functional.conv2d(inputs, weights.float(), bias),
            )

            # but it should work with the same type
            nn.functional.conv2d(inputs.float(), weights.float(), bias.float())

    def convTranspose2d_half_gemm(self):
        with torch.backends.mkldnn.flags(enabled=False):
            inputs = torch.randn(1, 1, 16, 16, device="xpu", dtype=torch.half)
            deconv = (
                nn.ConvTranspose2d(1, 1, 3, stride=2, padding=1, output_padding=1)
                .to("xpu")
                .half()
            )
            output = deconv(inputs)
            output.mean().backward()

    @unittest.expectedFailure
    def conv_mkldnn_memory_layout_dominance(self):
        # desired behavior here is to have the memory_layout of conv.weight to
        # dominante the layout of output.
        # which is not the same as current behavior, we'll fix this in
        # following up PRs and remove the `expectedFailure` tag
        input = torch.randint(
            1, 10, (2, 8, 4, 4), dtype=torch.float32, device="xpu", requires_grad=True
        )
        conv = nn.Conv2d(8, 4, 3, device="xpu").float()

        out = conv(input)
        self.assertTrue(out.is_contiguous())

        input = input.contiguous(memory_format=torch.channels_last)
        out = conv(input)
        self.assertTrue(out.is_contiguous())

        conv.weight.data = conv.weight.contiguous(memory_format=torch.channels_last)
        out = conv(input)
        self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))

        input = input.contiguous()
        out = conv(input)
        self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))

    def mkldnn_non_contiguous(self):
        x = torch.randn(192, 16, 50, device="xpu")
        x = x.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        m = torch.nn.Conv1d(
            in_channels=16, out_channels=32, kernel_size=2, bias=True, device="xpu"
        )
        result = m(x)

    def mkldnn_noncontiguous_weight(self):
        # Noncontiguous weights must be contiguous() before being
        # passed to MKLDNN
        input = torch.tensor([1, 1, 1], dtype=torch.double, device="xpu").view(1, 1, 3)
        weights1 = torch.tensor([1], dtype=torch.double, device="xpu").expand(1, 1, 2)
        weights2 = (
            torch.tensor([1], dtype=torch.double, device="xpu")
            .expand(1, 1, 2)
            .contiguous()
        )
        self.assertEqual(
            F.conv1d(input, weights1, bias=None, stride=2, dilation=2),
            F.conv1d(input, weights2, bias=None, stride=2, dilation=2),
        )

    def grouped_conv_mkldnn_nhwc_support(self):
        # in order to catch the hols in grouped convolution in nhwc support for earlier cudnn version
        input = torch.randn((16, 16, 8, 8), dtype=torch.float16, device="xpu").to(
            memory_format=torch.channels_last
        )
        weight = torch.randn((8, 4, 3, 3), dtype=torch.float16, device="xpu").to(
            memory_format=torch.channels_last
        )
        out = torch.convolution(
            input, weight, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 4
        )
        input = torch.randn((16, 8, 8, 8), dtype=torch.float16, device="xpu").to(
            memory_format=torch.channels_last
        )
        out_transpose = torch.convolution(
            input, weight, None, (1, 1), (1, 1), (1, 1), True, (0, 0), 4
        )

    def thnn_conv_strided_padded_dilated(self):
        for convfn, dims, transposed in (
            (torch.nn.functional.conv2d, 2, False),
            (torch.nn.functional.conv_transpose2d, 2, True),
            (torch.nn.functional.conv3d, 3, False),
            (torch.nn.functional.conv_transpose3d, 3, True),
        ):
            for stride, padding, dilation in (
                (2, 0, 1),
                (1, 1, 1),
                (2, 1, 1),
                (1, 0, 2),
            ):
                kwargs = {"stride": stride, "padding": padding, "dilation": dilation}
                inp_shape = (1, 2) + dims * (4,)
                weight_shape = (2, 2) + dims * (1,)
                inputs = torch.randn(
                    inp_shape, dtype=torch.double, device="xpu", requires_grad=True
                )
                weight = torch.randn(
                    weight_shape, dtype=torch.double, device="xpu", requires_grad=True
                )
                bias = torch.randn(
                    2, dtype=torch.double, device="xpu", requires_grad=True
                )
                with torch.backends.mkldnn.flags(enabled=False):
                    res = convfn(inputs, weight, bias, **kwargs)
                res_cpu = convfn(inputs.cpu(), weight.cpu(), bias.cpu(), **kwargs)
                self.assertEqual(res, res_cpu)
                with torch.backends.mkldnn.flags(enabled=False):
                    torch.autograd.gradcheck(
                        lambda x, w, b: convfn(x, w, b, **kwargs),
                        (inputs, weight, bias),
                    )
                    torch.autograd.gradcheck(
                        lambda x, w, b: convfn(x, w, b, **kwargs),
                        (inputs.cpu(), weight.cpu(), bias.cpu()),
                    )

    @largeTensorTest("24GB", "xpu")
    def conv3d_64bit_indexing(self, device):
        x = torch.rand(1, 32, 512, 512, 256)
        m = torch.nn.Conv3d(32, 1, kernel_size=1, padding=0, stride=1, bias=False)
        yref = m(x)
        y = m.to(device=device)(x.to(device=device))
        self.assertEqual(yref, y)

    @largeTensorTest("20GB", "xpu")
    def conv3d_large_batch_1(self, device):
        x = torch.rand(1, 32, 512, 512, 256)
        m = torch.nn.Conv3d(32, 1, kernel_size=1, padding=0, stride=1, bias=False)
        yref = m(x)
        y = m.to(device=device)(x.to(device=device))
        self.assertEqual(yref, y.cpu())

    @largeTensorTest("60GB", "xpu")
    def conv_large_batch_1(self, device):
        in_channels = 514
        dim = 2048
        out_channels = 1
        kernel_size = 3
        stride = 1
        padding = 1

        input_tensor = torch.ones(1, in_channels, dim, dim, device="xpu").half()
        model = (
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, device="xpu")
            .half()
        )
        output = model(input_tensor)
        model_cpu = model.cpu().float()
        output_cpu = model(input_tensor.float().cpu())
        self.assertEqual(output.cpu().float(), output_cpu, atol=1e-3, rtol=1e-3)

    @largeTensorTest("12GB", "xpu")
    def conv_large_nosplit(self, device):
        # Here we just test the convolution correctly route to the fallback implementation
        # that is, it does not crash. The correctness of fallback implementation should be
        # covered in other tests
        dtype = torch.half if self.device_type == "xpu" else torch.float
        conv1 = nn.Conv2d(2, 2, 8, 8).to(device).to(dtype)
        input_large = torch.randn(1, 2, 1024, 1024 * 1024, dtype=dtype, device=device)
        conv1(input_large)
        conv2 = torch.nn.Conv2d(1, 1024, 1, 1).to(device).to(dtype)
        input_large = torch.randn(1, 1, 2048, 1024, dtype=dtype, device=device)
        conv2(input_large)

    @largeTensorTest("12GB")
    def conv_transposed_large(self, device):
        dtype = torch.half if self.device_type == "xpu" else torch.float
        conv = nn.ConvTranspose2d(1, 1, 1, 1, bias=False).to(device).to(dtype)
        input_large = torch.randn(4096, 1, 512, 1024, dtype=dtype, device=device)
        # forward
        ret = conv(input_large)
        maxdiff0 = (
            (ret.narrow(0, 0, 1024) - conv(input_large.narrow(0, 0, 1024)))
            .abs_()
            .max()
            .item()
        )
        maxdiff1 = (
            (ret.narrow(0, 1024, 1024) - conv(input_large.narrow(0, 1024, 1024)))
            .abs_()
            .max()
            .item()
        )
        maxdiff2 = (
            (ret.narrow(0, 2048, 1024) - conv(input_large.narrow(0, 2048, 1024)))
            .abs_()
            .max()
            .item()
        )
        maxdiff3 = (
            (ret.narrow(0, 3072, 1024) - conv(input_large.narrow(0, 3072, 1024)))
            .abs_()
            .max()
            .item()
        )
        if self.device_type == "xpu":
            # cuDNN may use algorithms such as FFT that don't guarantee a diff of 0
            self.assertEqual(maxdiff0, 0, atol=2e-3, rtol=1e-5)
            self.assertEqual(maxdiff1, 0, atol=2e-3, rtol=1e-5)
            self.assertEqual(maxdiff2, 0, atol=2e-3, rtol=1e-5)
            self.assertEqual(maxdiff3, 0, atol=2e-3, rtol=1e-5)
        else:
            self.assertEqual(maxdiff0, 0)
            self.assertEqual(maxdiff1, 0)
            self.assertEqual(maxdiff2, 0)
            self.assertEqual(maxdiff3, 0)

    def Conv2d_groups_nobias(self):
        dev_dtypes = [("xpu", torch.float), ("xpu", torch.half)]
        for device, dtype in dev_dtypes:
            m = nn.Conv2d(4, 4, kernel_size=3, groups=2, bias=False).to(device, dtype)
            i = torch.randn(2, 4, 6, 6, device=device, dtype=dtype, requires_grad=True)
            output = m(i)
            grad_output = torch.randn(2, 4, 4, 4, device=device, dtype=dtype)
            output.backward(grad_output)

            m1 = nn.Conv2d(2, 2, kernel_size=3, bias=False).to(device, dtype)
            m1.weight.data.copy_(m.weight.data[:2])
            i1 = i.data[:, :2].contiguous().requires_grad_(True)
            output1 = m1(i1)
            output1.backward(grad_output[:, :2].contiguous())

            m2 = nn.Conv2d(2, 2, kernel_size=3, bias=False).to(device, dtype)
            m2.weight.data.copy_(m.weight.data[2:])
            i2 = i.data[:, 2:].contiguous().requires_grad_(True)
            output2 = m2(i2)
            output2.backward(grad_output[:, 2:].contiguous())

            self.assertEqual(output, torch.cat([output1, output2], 1))
            self.assertEqual(
                i.grad.data,
                torch.cat([i1.grad.data, i2.grad.data], 1),
                atol=dtype2prec_DONTUSE[dtype],
                rtol=0,
            )
            self.assertEqual(
                m.weight.grad.data,
                torch.cat([m1.weight.grad.data, m2.weight.grad.data], 0),
                atol=1e-1 if dtype == torch.half else dtype2prec_DONTUSE[dtype],
                rtol=0,
            )

    def Conv2d_groups_nobias_v2(self):
        torch.manual_seed(123)
        dev_dtypes = [("xpu", torch.float), ("xpu", torch.half)]
        for device, dtype in dev_dtypes:
            m = nn.Conv2d(4, 16, kernel_size=3, groups=2, bias=False).to(device, dtype)
            i = torch.randn(2, 4, 6, 6, device=device, dtype=dtype, requires_grad=True)
            output = m(i)
            grad_output = torch.randn(2, 16, 4, 4, device=device, dtype=dtype)
            output.backward(grad_output)

            m1 = nn.Conv2d(2, 8, kernel_size=3, bias=False).to(device, dtype)
            m1.weight.data.copy_(m.weight.data[:8])
            i1 = i.data[:, :2].contiguous().requires_grad_(True)
            output1 = m1(i1)
            output1.backward(grad_output[:, :8].contiguous())

            m2 = nn.Conv2d(2, 8, kernel_size=3, bias=False).to(device, dtype)
            m2.weight.data.copy_(m.weight.data[8:])
            i2 = i.data[:, 2:].contiguous().requires_grad_(True)
            output2 = m2(i2)
            output2.backward(grad_output[:, 8:].contiguous())

            self.assertEqual(output, torch.cat([output1, output2], 1))
            self.assertEqual(
                i.grad.data,
                torch.cat([i1.grad.data, i2.grad.data], 1),
                atol=dtype2prec_DONTUSE[dtype],
                rtol=0,
            )
            self.assertEqual(
                m.weight.grad.data,
                torch.cat([m1.weight.grad.data, m2.weight.grad.data], 0),
                atol=1e-1 if dtype == torch.half else dtype2prec_DONTUSE[dtype],
                rtol=0,
            )
    @dtypes(
        *floating_and_complex_types_and(
            torch.half, torch.bfloat16
        )
    )
    def conv2d_deterministic_cudnn(self, device, dtype):
        inputs = torch.randn(2, 3, 5, 5, device=device, dtype=dtype, requires_grad=True)
        with torch.backends.mkldnn.flags(enabled=False, deterministic=True):
            conv1 = torch.nn.Conv2d(3, 3, 3).to(device, dtype)
            conv2 = torch.nn.Conv2d(3, 3, 3).to(device, dtype)
            conv2.bias.data.copy_(conv1.bias.data)
            conv2.weight.data.copy_(conv1.weight.data)
            out1 = conv1(inputs)
            out2 = conv2(inputs)
            self.assertEqual(out1, out2, atol=0.0, rtol=0)
            y = torch.randn(out1.size(), device=device, dtype=dtype)
            out1.backward(y)
            out2.backward(y)
            self.assertEqual(
                conv1.bias.grad.data, conv2.bias.grad.data, atol=0.0, rtol=0
            )
            self.assertEqual(
                conv1.weight.grad.data, conv2.weight.grad.data, atol=0.0, rtol=0
            )


    TestConvolutionNNDeviceType.test_Conv2d_depthwise_naive_groups = conv2d_depthwise_naive_groups
    TestConvolutionNNDeviceType.test_Conv3d_depthwise_naive_groups = conv3d_depthwise_naive_groups
    TestConvolutionNNDeviceType.test_ConvTranspose2d_large_output_padding = convTranspose2d_large_output_padding
    TestConvolutionNNDeviceType.test_conv_backend = conv_backend
    TestConvolutionNNDeviceType._test_conv_cudnn_nhwc_nchw = _test_conv_xpu_nhwc_nchw
    TestConvolutionNNDeviceType.test_conv_cudnn_ndhwc = conv_xpu_ndhwc
    TestConvolutionNNDeviceType.test_conv_cudnn_nhwc_support = conv_xpu_nhwc_support
    TestConvolutionNNDeviceType.test_conv_cudnn_nhwc = conv_xpu_nhwc
    TestConvolutionNNDeviceType.test_conv_double_backward = conv_double_backward
    TestConvolutionNNDeviceType.test_conv3d_64bit_indexing = conv3d_64bit_indexing
    TestConvolutionNNDeviceType.test_conv3d_large_batch_1 = conv3d_large_batch_1
    TestConvolutionNNDeviceType.test_conv_large_batch_1 = conv_large_batch_1
    TestConvolutionNNDeviceType.test_conv_large_nosplit = conv_large_nosplit
    TestConvolutionNNDeviceType.test_conv_transposed_large = conv_transposed_large
    TestConvolutionNNDeviceType.test_Conv2d_deterministic_cudnn = conv2d_deterministic_cudnn
    TestConvolutionNN.test_Conv2d_inconsistent_types_on_GPU_with_cudnn = conv2d_inconsistent_types_on_GPU_with_mkldnn
    TestConvolutionNN.test_Conv2d_inconsistent_types_on_GPU_without_cudnn = conv2d_inconsistent_types_on_GPU_without_mkldnn
    TestConvolutionNN.test_ConvTranspose2d_half_cublas_gemm = convTranspose2d_half_gemm
    TestConvolutionNN.test_conv_cudnn_memory_layout_dominance = conv_mkldnn_memory_layout_dominance
    TestConvolutionNN.test_cudnn_non_contiguous = mkldnn_non_contiguous
    TestConvolutionNN.test_cudnn_noncontiguous_weight = mkldnn_noncontiguous_weight
    TestConvolutionNN.test_grouped_conv_cudnn_nhwc_support = grouped_conv_mkldnn_nhwc_support
    TestConvolutionNN.test_thnn_conv_strided_padded_dilated = thnn_conv_strided_padded_dilated
    TestConvolutionNN.test_Conv2d_groups_nobias=Conv2d_groups_nobias
    TestConvolutionNN.test_Conv2d_groups_nobias_v2=Conv2d_groups_nobias_v2
instantiate_device_type_tests(TestConvolutionNNDeviceType, globals(), only_for="xpu", allow_xpu=True)
instantiate_parametrized_tests(TestConvolutionNN)


if __name__ == "__main__":
    run_tests()

