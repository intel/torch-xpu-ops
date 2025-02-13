# Owner(s): ["module: intel"]
import torch
from torch.testing._internal.common_utils import TestCase

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")


class TestSimpleBinary(TestCase):
    def test_add(self, dtype=torch.float):
        a_cpu = torch.randn(2, 3)
        b_cpu = torch.randn(2, 3)
        a_xpu = a_cpu.to(xpu_device)
        b_xpu = b_cpu.to(xpu_device)
        c_xpu = a_xpu + b_xpu
        c_cpu = a_cpu + b_cpu
        self.assertEqual(c_cpu, c_xpu.to(cpu_device))

    def test_add_scalar(self, dtype=torch.float):
        a_cpu = torch.randn(2, 3)
        b = 1.11
        a_xpu = a_cpu.to(xpu_device)
        c_xpu = a_xpu + b
        c_cpu = a_cpu + b
        self.assertEqual(c_cpu, c_xpu.to(cpu_device))

    def test_sub(self, dtype=torch.float):
        a_cpu = torch.randn(2, 3)
        b_cpu = torch.randn(2, 3)
        a_xpu = a_cpu.to(xpu_device)
        b_xpu = b_cpu.to(xpu_device)
        c_xpu = a_xpu - b_xpu
        c_cpu = a_cpu - b_cpu
        self.assertEqual(c_cpu, c_xpu.to(cpu_device))

    def test_mul(self, dtype=torch.float):
        a_cpu = torch.randn(2, 3)
        b_cpu = torch.randn(2, 3)
        a_xpu = a_cpu.to(xpu_device)
        b_xpu = b_cpu.to(xpu_device)
        c_xpu = a_xpu * b_xpu
        c_cpu = a_cpu * b_cpu
        self.assertEqual(c_cpu, c_xpu.to(cpu_device))

    def test_div(self, dtype=torch.float):
        a_cpu = torch.randn(2, 3)
        b_cpu = torch.randn(2, 3)
        a_xpu = a_cpu.to(xpu_device)
        b_xpu = b_cpu.to(xpu_device)
        c_xpu = a_xpu / b_xpu
        c_cpu = a_cpu / b_cpu
        self.assertEqual(c_cpu, c_xpu.to(cpu_device))

    def test_div_int(self, dtype=torch.float):
        a_cpu = torch.randint(2, 3, [8, 8])
        b_cpu = torch.randint(2, 3, [8, 8])
        a_xpu = a_cpu.to(xpu_device)
        b_xpu = b_cpu.to(xpu_device)
        c_xpu = a_xpu / b_xpu
        c_cpu = a_cpu / b_cpu
        self.assertEqual(c_cpu.dtype, c_xpu.dtype)  # assume float
        self.assertEqual(c_cpu, c_xpu.to(cpu_device))

    def test_binary_div_channels_last(self, dtype=torch.float):
        shapes = [
            (1, 2, 3, 4),
            (2, 2, 3, 3),
            (4, 4, 4, 4),
            (4, 4, 1, 1),
            (4, 1, 4, 4),
            (4, 1, 4, 1),
            (4, 1, 1, 4),
            (1, 4, 1, 4),
            (1, 4, 4, 1),
            (4, 1, 1, 1),
        ]
        for shape in shapes:
            N, C, H, W = shape[0], shape[1], shape[2], shape[3]
            a_cpu = torch.randn(N, C, H, W)
            b_cpu = torch.randn(N, C, H, W)
            a_xpu = a_cpu.to("xpu").to(memory_format=torch.contiguous_format)
            b_xpu = b_cpu.to("xpu").to(memory_format=torch.contiguous_format)

            a_cpu.div_(b_cpu)
            a_xpu.div_(b_xpu)
            self.assertEqual(a_cpu, a_xpu.cpu())
            if 1 == C or (1 == H and 1 == W):
                self.assertEqual(a_xpu.is_contiguous(), True)
                self.assertEqual(
                    a_xpu.is_contiguous(memory_format=torch.channels_last), True
                )
            else:
                self.assertEqual(a_xpu.is_contiguous(), True)
                self.assertEqual(
                    a_xpu.is_contiguous(memory_format=torch.channels_last), False
                )

            a_cpu = torch.randn(N, C, H, W)
            b_cpu = torch.randn(N, C, H, W)
            a_xpu = a_cpu.to("xpu").to(memory_format=torch.channels_last)
            b_xpu = b_cpu.to("xpu").to(memory_format=torch.channels_last)

            a_cpu.div_(b_cpu)
            a_xpu.div_(b_xpu)
            self.assertEqual(a_cpu, a_xpu.cpu())
            if 1 == C or (1 == H and 1 == W):
                self.assertEqual(a_xpu.is_contiguous(), True)
                self.assertEqual(
                    a_xpu.is_contiguous(memory_format=torch.channels_last), True
                )
            else:
                self.assertEqual(a_xpu.is_contiguous(), False)
                self.assertEqual(
                    a_xpu.is_contiguous(memory_format=torch.channels_last), True
                )

            a_cpu = torch.randn(N, C, H, W)
            b_cpu = torch.randn(N, C, H, W)
            a_xpu = a_cpu.to("xpu").to(memory_format=torch.channels_last)
            b_xpu = b_cpu.to("xpu").to(memory_format=torch.contiguous_format)

            a_cpu.div_(b_cpu)
            a_xpu.div_(b_xpu)
            self.assertEqual(a_cpu, a_xpu.cpu())
            if 1 == C or (1 == H and 1 == W):
                self.assertEqual(a_xpu.is_contiguous(), True)
                self.assertEqual(
                    a_xpu.is_contiguous(memory_format=torch.channels_last), True
                )
            else:
                self.assertEqual(a_xpu.is_contiguous(), False)
                self.assertEqual(
                    a_xpu.is_contiguous(memory_format=torch.channels_last), True
                )

            a_cpu = torch.randn(N, C, H, W)
            b_cpu = torch.randn(N, C, H, W)
            a_xpu = a_cpu.to("xpu").to(memory_format=torch.contiguous_format)
            b_xpu = b_cpu.to("xpu").to(memory_format=torch.channels_last)

            a_cpu.div_(b_cpu)
            a_xpu.div_(b_xpu)
            self.assertEqual(a_cpu, a_xpu.cpu())
            if 1 == C or (1 == H and 1 == W):
                self.assertEqual(a_xpu.is_contiguous(), True)
                self.assertEqual(
                    a_xpu.is_contiguous(memory_format=torch.channels_last), True
                )
            else:
                self.assertEqual(a_xpu.is_contiguous(), True)
                self.assertEqual(
                    a_xpu.is_contiguous(memory_format=torch.channels_last), False
                )

            a_cpu = torch.randn(N, C, H, W)
            b_cpu = torch.randn(N, C, H, W)
            y_cpu = a_cpu / b_cpu
            a_xpu = a_cpu.to("xpu").to(memory_format=torch.contiguous_format)
            b_xpu = b_cpu.to("xpu").to(memory_format=torch.contiguous_format)
            y_xpu = a_xpu / b_xpu
            self.assertEqual(y_cpu, y_xpu.cpu())
            if 1 == C or (1 == H and 1 == W):
                self.assertEqual(a_xpu.is_contiguous(), True)
                self.assertEqual(
                    a_xpu.is_contiguous(memory_format=torch.channels_last), True
                )
            else:
                self.assertEqual(a_xpu.is_contiguous(), True)
                self.assertEqual(
                    a_xpu.is_contiguous(memory_format=torch.channels_last), False
                )

            a_cpu = torch.randn(N, C, H, W)
            b_cpu = torch.randn(N, C, H, W)
            y_cpu = a_cpu / b_cpu
            a_xpu = a_cpu.to("xpu").to(memory_format=torch.channels_last)
            b_xpu = b_cpu.to("xpu").to(memory_format=torch.channels_last)
            y_xpu = a_xpu / b_xpu
            self.assertEqual(y_cpu, y_xpu.cpu())
            if 1 == C or (1 == H and 1 == W):
                self.assertEqual(a_xpu.is_contiguous(), True)
                self.assertEqual(
                    a_xpu.is_contiguous(memory_format=torch.channels_last), True
                )
            else:
                self.assertEqual(a_xpu.is_contiguous(), False)
                self.assertEqual(
                    a_xpu.is_contiguous(memory_format=torch.channels_last), True
                )

            a_cpu = torch.randn(N, C, H, W)
            b_cpu = torch.randn(N, C, H, W)
            y_cpu = a_cpu / b_cpu
            a_xpu = a_cpu.to("xpu").to(memory_format=torch.channels_last)
            b_xpu = b_cpu.to("xpu").to(memory_format=torch.contiguous_format)
            y_xpu = a_xpu / b_xpu
            self.assertEqual(y_cpu, y_xpu.cpu())
            if 1 == C or (1 == H and 1 == W):
                self.assertEqual(a_xpu.is_contiguous(), True)
                self.assertEqual(
                    a_xpu.is_contiguous(memory_format=torch.channels_last), True
                )
            else:
                self.assertEqual(a_xpu.is_contiguous(), False)
                self.assertEqual(
                    a_xpu.is_contiguous(memory_format=torch.channels_last), True
                )

            a_cpu = torch.randn(N, C, H, W)
            b_cpu = torch.randn(N, C, H, W)
            y_cpu = a_cpu / b_cpu
            a_xpu = a_cpu.to("xpu").to(memory_format=torch.contiguous_format)
            b_xpu = b_cpu.to("xpu").to(memory_format=torch.channels_last)
            y_xpu = a_xpu / b_xpu
            self.assertEqual(y_cpu, y_xpu.cpu())
            if 1 == C or (1 == H and 1 == W):
                self.assertEqual(a_xpu.is_contiguous(), True)
                self.assertEqual(
                    a_xpu.is_contiguous(memory_format=torch.channels_last), True
                )
            else:
                self.assertEqual(a_xpu.is_contiguous(), True)
                self.assertEqual(
                    a_xpu.is_contiguous(memory_format=torch.channels_last), False
                )

    def test_pow(self, dtype=torch.float):
        x_cpu = torch.tensor(([2.5, 3.1, 1.3]), dtype=torch.float, device=cpu_device)
        x_xpu = torch.tensor(([2.5, 3.1, 1.3]), dtype=torch.float, device=xpu_device)
        y_cpu = torch.tensor(([3.0, 3.0, 3.0]), dtype=torch.float, device=cpu_device)
        y_xpu = torch.tensor(([3.0, 3.0, 3.0]), dtype=torch.float, device=xpu_device)
        self.assertEqual(torch.pow(x_cpu, y_cpu), torch.pow(x_xpu, y_xpu).cpu())
        self.assertEqual(x_cpu.pow(y_cpu), x_xpu.pow(y_xpu).cpu())
        self.assertEqual(x_cpu.pow_(y_cpu), x_xpu.pow_(y_xpu).cpu())

    def test_binary_op(self, dtype=torch.float):
        x_cpu = torch.randn(5)

        x_xpu = x_cpu.to(xpu_device)
        # y_cpu1 = x_cpu.new_ones((2, 3))
        y_cpu1 = torch.randn(5)
        # y_cpu2 = x_cpu.new_ones((2, 3))
        y_cpu2 = torch.randn(5)

        y_cpu1_int = torch.tensor([[3, 1, 2, 3], [2, 3, 4, 1]], dtype=torch.int32)
        # y_cpu2 = x_cpu.new_ones((2, 3))
        y_cpu2_int = torch.tensor([[1, 5, 2, 4], [1, 1, 5, 5]], dtype=torch.int32)

        y_xpu1 = y_cpu1.to(xpu_device)
        y_xpu2 = y_cpu2.to(xpu_device)
        y_xpu1_int = y_cpu1_int.to(xpu_device)
        y_xpu2_int = y_cpu2_int.to(xpu_device)

        x_cpu_b_1 = torch.tensor([True, True])
        x_cpu_b_2 = torch.tensor([False, True])
        x_xpu_b_1 = x_cpu_b_1.to(xpu_device)
        x_xpu_b_2 = x_cpu_b_2.to(xpu_device)

        print("remainder scalar y_cpu", torch.remainder(y_cpu1, 1.5))
        print("remainder scalar y_xpu", torch.remainder(y_xpu1, 1.5).to(cpu_device))
        self.assertEqual(
            torch.remainder(y_cpu1, 1.5), torch.remainder(y_xpu1, 1.5).to(cpu_device)
        )

        print("remainder tensor y_cpu", torch.remainder(y_cpu1, y_cpu2))
        print(
            "remainder tensor y_xpu",
            torch.remainder(y_xpu1, y_xpu2).to(cpu_device),
        )
        self.assertEqual(
            torch.remainder(y_cpu1, y_cpu2),
            torch.remainder(y_xpu1, y_xpu2).to(cpu_device),
        )

        print("fmod scalar y_cpu", torch.fmod(y_cpu1, 1.5))
        print("fmod scalar y_xpu", torch.fmod(y_xpu1, 1.5).to(cpu_device))
        self.assertEqual(
            torch.fmod(y_cpu1, 1.5), torch.fmod(y_xpu1, 1.5).to(cpu_device)
        )

        print("fmod tensor y_cpu", torch.fmod(y_cpu1, y_cpu2))
        print("fmod tensor y_xpu", torch.fmod(y_xpu1, y_xpu2).to(cpu_device))
        self.assertEqual(
            torch.fmod(y_cpu1, y_cpu2), torch.fmod(y_xpu1, y_xpu2).to(cpu_device)
        )
