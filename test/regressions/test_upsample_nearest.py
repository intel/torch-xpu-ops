# Owner(s): ["module: intel"]
import torch
from torch.autograd import Variable
from torch.testing._internal.common_utils import TestCase

device = torch.device("xpu")
cpu_device = torch.device("cpu")


class TestTorchMethod(TestCase):
    def test_upsample_nearest(self):
        test_dtypes = [torch.float, torch.bfloat16, torch.float16]

        def test_upsample_nearest_exact(dtype):
            # #### upsample nearest exact 1D #####
            input_cpu = torch.randn((2, 3, 5), dtype=dtype, device=cpu_device)
            input_xpu = input_cpu.to("xpu")
            scales = [6]
            input_cpu.requires_grad = True
            input_xpu.requires_grad = True
            rsf = False

            output_cpu = torch.nn.functional.interpolate(
                input_cpu,
                scale_factor=scales,
                mode="nearest-exact",
                recompute_scale_factor=rsf,
            )
            output_xpu = torch.nn.functional.interpolate(
                input_xpu,
                scale_factor=scales,
                mode="nearest-exact",
                recompute_scale_factor=rsf,
            )
            self.assertEqual(output_cpu, output_xpu.cpu())

            grad_out_cpu = torch.ones_like(output_cpu)
            grad_out_xpu = grad_out_cpu.to("xpu")
            grad_out_cpu = Variable(grad_out_cpu, requires_grad=True)
            grad_out_xpu = Variable(grad_out_xpu, requires_grad=True)

            output_cpu.backward(grad_out_cpu)
            output_xpu.backward(grad_out_xpu)
            grad_cpu = input_cpu.grad
            grad_xpu = input_xpu.grad
            self.assertEqual(grad_cpu, grad_xpu.cpu())

            # #### upsample nearest exact 2D #####
            input_cpu = torch.randn(
                (2, 3, 5, 5), dtype=torch.float32, device=cpu_device
            )
            input_dpcpp = input_cpu.to("xpu")
            scales = [6, 8]
            input_cpu.requires_grad = True
            input_dpcpp.requires_grad = True
            alc = False

            output_cpu = torch.nn.functional.interpolate(
                input_cpu,
                scale_factor=scales,
                mode="nearest-exact",
                recompute_scale_factor=rsf,
            )
            output_dpcpp = torch.nn.functional.interpolate(
                input_dpcpp,
                scale_factor=scales,
                mode="nearest-exact",
                recompute_scale_factor=rsf,
            )
            self.assertEqual(output_cpu, output_dpcpp.cpu())

            grad_out_cpu = torch.randn(
                (2, 3, 30, 40), dtype=torch.float32, device=cpu_device
            )
            grad_out_dpcpp = grad_out_cpu.to("xpu")
            grad_out_cpu = Variable(grad_out_cpu, requires_grad=True)
            grad_out_dpcpp = Variable(grad_out_dpcpp, requires_grad=True)

            output_cpu.backward(grad_out_cpu)
            output_dpcpp.backward(grad_out_dpcpp)
            grad_cpu = input_cpu.grad
            grad_dpcpp = input_dpcpp.grad
            self.assertEqual(grad_cpu, grad_dpcpp.cpu())

        def test_upsamle_nearest(dtype):
            # #### upsample nearest 1D #####
            input_cpu = torch.randn((2, 3, 5), dtype=torch.float32, device=cpu_device)
            input_xpu = input_cpu.to("xpu")
            scales = [6]
            input_cpu.requires_grad = True
            input_xpu.requires_grad = True
            rsf = False

            output_cpu = torch.nn.functional.interpolate(
                input_cpu,
                scale_factor=scales,
                mode="nearest",
                recompute_scale_factor=rsf,
            )
            output_xpu = torch.nn.functional.interpolate(
                input_xpu,
                scale_factor=scales,
                mode="nearest",
                recompute_scale_factor=rsf,
            )
            self.assertEqual(output_cpu, output_xpu.cpu())

            grad_out_cpu = torch.randn(
                (2, 3, 30), dtype=torch.float32, device=cpu_device
            )
            grad_out_xpu = grad_out_cpu.to("xpu")
            grad_out_cpu = Variable(grad_out_cpu, requires_grad=True)
            grad_out_xpu = Variable(grad_out_xpu, requires_grad=True)

            output_cpu.backward(grad_out_cpu)
            output_xpu.backward(grad_out_xpu)
            grad_cpu = input_cpu.grad
            grad_xpu = input_xpu.grad
            self.assertEqual(grad_cpu, grad_xpu.cpu())

            # #### upsample nearest 2D #####
            input_cpu = torch.randn(
                (2, 3, 5, 5), dtype=torch.float32, device=cpu_device
            )
            input_dpcpp = input_cpu.to("xpu")
            scales = [6, 8]
            input_cpu.requires_grad = True
            input_dpcpp.requires_grad = True
            alc = False

            output_cpu = torch.nn.functional.interpolate(
                input_cpu,
                scale_factor=scales,
                mode="nearest",
                recompute_scale_factor=rsf,
            )
            output_dpcpp = torch.nn.functional.interpolate(
                input_dpcpp,
                scale_factor=scales,
                mode="nearest",
                recompute_scale_factor=rsf,
            )
            self.assertEqual(output_cpu, output_dpcpp.cpu())

            grad_out_cpu = torch.randn(
                (2, 3, 30, 40), dtype=torch.float32, device=cpu_device
            )
            grad_out_dpcpp = grad_out_cpu.to("xpu")
            grad_out_cpu = Variable(grad_out_cpu, requires_grad=True)
            grad_out_dpcpp = Variable(grad_out_dpcpp, requires_grad=True)

            output_cpu.backward(grad_out_cpu)
            output_dpcpp.backward(grad_out_dpcpp)
            grad_cpu = input_cpu.grad
            grad_dpcpp = input_dpcpp.grad
            self.assertEqual(grad_cpu, grad_dpcpp.cpu())

        for dtype in test_dtypes:
            test_upsamle_nearest(dtype)
            test_upsample_nearest_exact(dtype)
