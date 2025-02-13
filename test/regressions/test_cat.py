# Owner(s): ["module: intel"]
import torch
from torch.testing._internal.common_utils import TestCase


class TestTorchMethod(TestCase):
    def test_cat_8d(self, dtype=torch.float):
        input1 = torch.randn([256, 8, 8, 3, 3, 3, 3], dtype=dtype)
        input2 = torch.randn([256, 8, 8, 3, 3, 3, 3], dtype=dtype)
        input1_xpu = input1.xpu()
        input2_xpu = input2.xpu()
        output1 = torch.stack([input1, input2], dim=0)
        output1_xpu = torch.stack([input1_xpu, input2_xpu], dim=0)
        output2 = output1.reshape([2, 256, 8, 8, 9, 9])
        output2_xpu = output1_xpu.reshape([2, 256, 8, 8, 9, 9])
        output3 = torch.stack([output2, output2], dim=0)
        output3_xpu = torch.stack([output2_xpu, output2_xpu], dim=0)
        self.assertEqual(output3, output3.cpu())

    def test_cat_array(self, dtype=torch.float):
        user_cpu1 = torch.randn([2, 2, 3], dtype=dtype)
        user_cpu2 = torch.randn([2, 2, 3], dtype=dtype)
        user_cpu3 = torch.randn([2, 2, 3], dtype=dtype)
        res_cpu = torch.cat((user_cpu1, user_cpu2, user_cpu3), dim=1)
        res_xpu = torch.cat(
            (
                user_cpu1.xpu(),
                user_cpu2.xpu(),
                user_cpu3.xpu(),
            ),
            dim=1,
        )
        self.assertEqual(res_cpu, res_xpu.cpu())

    def test_cat_array_2(self, dtype=torch.float):
        shapes = [
            (8, 7, 3, 2),
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
            print("\n================== test shape: ", shape, "==================")
            N, C, H, W = shape[0], shape[1], shape[2], shape[3]
            user_cpu1 = torch.randn([N, C, H, W], dtype=dtype)
            user_cpu2 = torch.randn([N, C, H, W], dtype=dtype)
            user_cpu3 = torch.randn([N, C, H, W], dtype=dtype)

            user_cpu1 = user_cpu1.to(memory_format=torch.channels_last)
            user_cpu2 = user_cpu2.to(memory_format=torch.channels_last)
            user_cpu3 = user_cpu3.to(memory_format=torch.channels_last)

            dim_idx = 1
            res_cpu = torch.cat((user_cpu1, user_cpu2, user_cpu3), dim=dim_idx)
            print("\n-------------CPU Result:--------------")
            print(res_cpu.shape)
            print(
                "res_cpu is cl: ",
                res_cpu.is_contiguous(memory_format=torch.channels_last),
            )

            user_xpu1 = user_cpu1.xpu()
            user_xpu2 = user_cpu2.xpu()
            user_xpu3 = user_cpu3.xpu()

            print("\n-------------GPU Result:--------------")
            res_xpu = torch.cat((user_xpu1, user_xpu2, user_xpu3), dim=dim_idx)
            print("SYCL Result:")
            print(res_xpu.cpu().shape)
            print(
                "res_xpu is cl: ",
                res_xpu.is_contiguous(memory_format=torch.channels_last),
            )
            self.assertEqual(res_cpu, res_xpu.cpu())

            if (
                1 == res_xpu.shape[1]
                or (1 == res_xpu.shape[2] and 1 == res_xpu.shape[3])
                or (
                    1 == res_xpu.shape[1]
                    and 1 == res_xpu.shape[2]
                    and 1 == res_xpu.shape[3]
                )
            ):
                self.assertEqual(res_xpu.is_contiguous(), True)
                self.assertEqual(
                    res_xpu.is_contiguous(memory_format=torch.channels_last), True
                )
            else:
                self.assertEqual(res_xpu.is_contiguous(), False)
                self.assertEqual(
                    res_xpu.is_contiguous(memory_format=torch.channels_last), True
                )

            user_cpu1 = torch.randn([N, C, H, W], dtype=dtype)
            user_cpu2 = torch.randn([N, C, H, W], dtype=dtype)
            user_cpu3 = torch.randn([N, C, H, W], dtype=dtype)

            user_cpu1 = user_cpu1.to(memory_format=torch.channels_last)
            user_cpu2 = user_cpu2.to(memory_format=torch.contiguous_format)
            user_cpu3 = user_cpu3.to(memory_format=torch.channels_last)

            dim_idx = 1
            res_cpu = torch.cat((user_cpu1, user_cpu2, user_cpu3), dim=dim_idx)
            print("\n-------------CPU Result:--------------")
            print(res_cpu.shape)
            print(
                "res_cpu is cl: ",
                res_cpu.is_contiguous(memory_format=torch.channels_last),
            )

            user_xpu1 = user_cpu1.xpu()
            user_xpu2 = user_cpu2.xpu()
            user_xpu3 = user_cpu3.xpu()

            print("\n-------------GPU Result:--------------")
            res_xpu = torch.cat((user_xpu1, user_xpu2, user_xpu3), dim=dim_idx)
            print("SYCL Result:")
            print(res_xpu.cpu().shape)
            print(
                "res_xpu is cl: ",
                res_xpu.is_contiguous(memory_format=torch.channels_last),
            )
            self.assertEqual(res_cpu, res_xpu.cpu())

            if (
                1 == res_xpu.shape[1]
                or (1 == res_xpu.shape[2] and 1 == res_xpu.shape[3])
                or (
                    1 == res_xpu.shape[1]
                    and 1 == res_xpu.shape[2]
                    and 1 == res_xpu.shape[3]
                )
            ):
                self.assertEqual(res_xpu.is_contiguous(), True)
                self.assertEqual(
                    res_xpu.is_contiguous(memory_format=torch.channels_last), True
                )
            else:
                self.assertEqual(res_xpu.is_contiguous(), True)
                self.assertEqual(
                    res_xpu.is_contiguous(memory_format=torch.channels_last), False
                )

            user_cpu1 = torch.randn([N, C, H, W], dtype=dtype)
            user_cpu2 = torch.randn([N, C, H, W], dtype=dtype)
            user_cpu3 = torch.randn([N, C, H, W], dtype=dtype)

            user_cpu1 = user_cpu1.to(memory_format=torch.contiguous_format)
            user_cpu2 = user_cpu2.to(memory_format=torch.channels_last)
            user_cpu3 = user_cpu3.to(memory_format=torch.channels_last)

            dim_idx = 1
            res_cpu = torch.cat((user_cpu1, user_cpu2, user_cpu3), dim=dim_idx)
            print("\n-------------CPU Result:--------------")
            print(res_cpu.shape)
            print(
                "res_cpu is cl: ",
                res_cpu.is_contiguous(memory_format=torch.channels_last),
            )

            user_xpu1 = user_cpu1.xpu()
            user_xpu2 = user_cpu2.xpu()
            user_xpu3 = user_cpu3.xpu()

            print("\n-------------GPU Result:--------------")
            res_xpu = torch.cat((user_xpu1, user_xpu2, user_xpu3), dim=dim_idx)
            print("SYCL Result:")
            print(res_xpu.cpu().shape)
            print(
                "res_xpu is cl: ",
                res_xpu.is_contiguous(memory_format=torch.channels_last),
            )
            self.assertEqual(res_cpu, res_xpu.cpu())

            if (
                1 == res_xpu.shape[1]
                or (1 == res_xpu.shape[2] and 1 == res_xpu.shape[3])
                or (
                    1 == res_xpu.shape[1]
                    and 1 == res_xpu.shape[2]
                    and 1 == res_xpu.shape[3]
                )
            ):
                self.assertEqual(res_xpu.is_contiguous(), True)
                self.assertEqual(
                    res_xpu.is_contiguous(memory_format=torch.channels_last), True
                )
            else:
                self.assertEqual(res_xpu.is_contiguous(), True)
                self.assertEqual(
                    res_xpu.is_contiguous(memory_format=torch.channels_last), False
                )
