# Owner(s): ["module: intel"]
import torch
from torch.testing._internal.common_utils import TestCase

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")


class TestSimpleResize(TestCase):
    def test_resize(self, dtype=torch.float):
        x = torch.ones([2, 2, 4, 3], device=xpu_device, dtype=dtype)
        x.resize_(1, 2, 3, 4)

        y = torch.ones([2, 2, 4, 3], device=cpu_device, dtype=dtype)
        y.resize_(1, 2, 3, 4)

        self.assertEqual(y, x.cpu())

    def test_view(self, dtype=torch.float):
        a_cpu = torch.randn(2, 3, 4, dtype=dtype)
        a_xpu = a_cpu.to(xpu_device)

        a_xpu = a_xpu.view(4, 3, 2)
        b_xpu = torch.full_like(a_xpu, 1)
        c_cpu = torch.ones([4, 3, 2])

        assert b_xpu.shape[0] == 4
        assert b_xpu.shape[1] == 3
        assert b_xpu.shape[2] == 2

        self.assertEqual(c_cpu, b_xpu.to(cpu_device))

    def test_view_as_real(self, dtype=torch.cfloat):
        a_cpu = torch.randn(2, 3, 4, dtype=dtype)
        a_xpu = a_cpu.to(xpu_device)
        b_cpu = torch.view_as_real(a_cpu)
        b_xpu = torch.view_as_real(a_xpu)
        self.assertEqual(b_cpu, b_xpu.to(cpu_device))

    def test_view_as_complex(self, dtype=torch.float):
        a_cpu = torch.randn(109, 2, dtype=dtype)
        a_xpu = a_cpu.to(xpu_device)
        b_cpu = torch.view_as_complex(a_cpu)
        b_xpu = torch.view_as_complex(a_xpu)
        self.assertEqual(b_cpu, b_xpu.to(cpu_device))

    def test_tensor_set_storage(self, dtype=torch.float):
        t1 = torch.tensor([], dtype=dtype).to(xpu_device)
        t2 = torch.randn(3, 4, 9, 10, dtype=dtype).to(xpu_device)
        size = torch.Size([9, 3, 4, 10])
        stride = (10, 360, 90, 1)

        # 1. case when source is storage
        t1.set_(source=t2.storage())
        self.assertEqual(t1.storage()._cdata, t2.storage()._cdata)
        # 2 case when source is storage, and other args also specified
        t1.set_(source=t2.storage(), storage_offset=0, size=size, stride=stride)
        self.assertEqual(t1.size(), size)
        self.assertEqual(t1.stride(), stride)

    def test_unfold(self, dtype=torch.float):
        a_cpu = torch.randn(2, 3, 16, dtype=dtype)
        a_xpu = a_cpu.to(xpu_device)
        b_xpu = a_xpu.unfold(2, 2, 2)
        b_cpu = a_cpu.unfold(2, 2, 2)
        assert b_xpu.shape == b_cpu.shape
        self.assertEqual(b_cpu, b_xpu.to(cpu_device))
