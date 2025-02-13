# Owner(s): ["module: intel"]
import numpy as np
import torch
from torch.testing._internal.common_utils import TestCase

np.set_printoptions(threshold=np.inf)

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_index_and_index_put(self, dtype=torch.float):
        x_cpu = torch.randn([3, 3], dtype=torch.float, device=cpu_device)
        y_cpu = torch.randn([3, 3], dtype=torch.float, device=cpu_device)
        mask_cpu = y_cpu.gt(0)

        # xpu part
        x_xpu = x_cpu.to("xpu")
        mask_xpu = mask_cpu.to("xpu")
        self.assertEqual(mask_cpu.nonzero(), mask_xpu.to(cpu_device).nonzero())
        self.assertEqual(x_cpu[mask_cpu], x_xpu[mask_xpu].to(cpu_device))

        # index put
        input = torch.ones([1], dtype=torch.float, device=cpu_device)
        indcies = torch.tensor([0, 0])
        x_cpu[indcies] = input
        x_cpu.index_put_([indcies], input, True)

        input = input.to("xpu")
        indcies = indcies.to("xpu")
        x_xpu[indcies] = input
        x_xpu.index_put_([indcies], input, True)
        self.assertEqual(x_cpu, x_xpu.to(cpu_device))

    def test_index_put(self, dtype=torch.bfloat16):
        cpu_device = torch.device("cpu")
        xpu_device = torch.device("xpu")

        accumulate = True
        x_cpu = torch.zeros([4, 512, 128], dtype=dtype, device=cpu_device)
        y_cpu = torch.ones([4, 15000, 128], dtype=dtype, device=cpu_device)
        x_xpu = x_cpu.to(xpu_device)
        y_xpu = y_cpu.to(xpu_device)
        index_cpu = [
            torch.ones([4, 15000, 128], device=cpu_device).to(torch.long),
            torch.ones([4, 15000, 128], device=cpu_device).to(torch.long),
            torch.ones([4, 15000, 128], device=cpu_device).to(torch.long),
        ]
        index_xpu = [
            torch.ones([4, 15000, 128], device=xpu_device).to(torch.long),
            torch.ones([4, 15000, 128], device=xpu_device).to(torch.long),
            torch.ones([4, 15000, 128], device=xpu_device).to(torch.long),
        ]

        z_cpu = x_cpu.index_put_(index_cpu, y_cpu, accumulate)
        z_xpu = x_xpu.index_put_(index_xpu, y_xpu, accumulate)
        self.assertEqual(z_cpu, z_xpu.cpu())

    def test_index_put_outer_inner(self, dtype=torch.long):
        # XXX using long to avoid accumulate error caused by order of combiniation
        torch.use_deterministic_algorithms(True)
        batch = 15  # outer
        stride = 33  # inner
        numel = 17
        a = torch.randint(
            0, 5, (batch, numel, stride), dtype=dtype, device=torch.device("xpu")
        )
        b = torch.randint(
            0, 5, (batch, numel, stride), dtype=dtype, device=torch.device("xpu")
        )
        idx = a < b
        idx_ = torch.nonzero(idx, as_tuple=True)
        nonzero = torch.nonzero(idx)
        idx_ = (None, idx_[1], None)
        values = torch.randint(
            0,
            5,
            (batch, nonzero.shape[0], stride),
            dtype=dtype,
            device=torch.device("xpu"),
        )
        a_cpu = a.cpu()
        idx_cpu = (None, idx_[1].cpu(), None)
        values_cpu = values.cpu()

        torch.ops.aten._index_put_impl_(a, idx_, values, True)
        torch.ops.aten._index_put_impl_(a_cpu, idx_cpu, values_cpu, True)
        self.assertEqual(a_cpu, a.cpu())
        torch.use_deterministic_algorithms(False)

    def test_index_put_with_zero_shape_dim(self, dtype=torch.bfloat16):
        torch.use_deterministic_algorithms(True)
        a = torch.randn([10, 0], dtype=dtype, device=torch.device("xpu"))
        b = torch.randn([5, 0], dtype=dtype, device=torch.device("xpu"))
        a[:5, :] = a[:5, :] * 2 + b
        torch.use_deterministic_algorithms(False)
