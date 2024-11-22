# Owner(s): ["module: intel"]

import torch
import pytest

from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)


checking_atol = 1e-2
checking_rtol = 1e-2


class TestSYCLInt4Linear(TestCase):

    @staticmethod
    def unpack_weight(qweight, scales, qzeros, q_config):
        group_size = q_config["group_size"]
        bits = q_config["bits"]
        s32_bits = 32

        assert bits == 4
        # Int32 can store 8 * 4bits data. This is the offset for each data.
        wf = (
            torch.tensor(list(range(0, s32_bits, bits)), dtype=torch.int32)
            .unsqueeze(0)
            .to("xpu")
        )
        zeros = torch.bitwise_right_shift(
            torch.unsqueeze(qzeros, 2).expand(-1, -1, 32 // bits), wf.unsqueeze(0)
        ).to(torch.int16 if bits == 8 else torch.int8)
        torch.bitwise_and(zeros, (2**bits) - 1, out=zeros)

        zeros = zeros + 1
        zeros = zeros.reshape(scales.shape)

        weight = torch.bitwise_right_shift(
            torch.unsqueeze(qweight, 1).expand(-1, 32 // bits, -1), wf.unsqueeze(-1)
        ).to(torch.int16 if bits == 8 else torch.int8)
        torch.bitwise_and(weight, (2**bits) - 1, out=weight)

        return weight, scales, zeros

    @staticmethod
    def dequantize(qweight, scales, qzeros, group_size):
        q_config = {"group_size": group_size, "bits": 4}
        weight, gptq_scales, gptq_zeros = TestSYCLInt4Linear.unpack_weight(
            qweight, scales, qzeros, q_config
        )
        gptq_zeros = (torch.ones_like(gptq_zeros) * 8).to("xpu")  # TODO: hard code zp
        if len(weight.shape) > 2:
            weight = weight.reshape(-1, weight.shape[-1])
        infeatures = weight.shape[0]
        g_idx = torch.tensor(
            [i // q_config["group_size"] for i in range(infeatures)],
            dtype=torch.int32,
        )
        scale_zeros = gptq_zeros * gptq_scales
        weight = gptq_scales[g_idx.long()] * weight - scale_zeros[g_idx.long()]
        return weight

    @staticmethod
    def rand_int4(size, dtype=torch.int32, device="xpu"):
        rand = torch.randint(-128, 128, [size // 2], device=device).to(torch.int8)
        return rand.view(dtype=dtype)

    @parametrize("per_channel", [False], lambda k: "per_channel" * k)
    @parametrize("dtype", [torch.float16])
    @parametrize("m,n,k", [(8, 4096, 4096), (1, 4096, 11008), (32, 4096, 4096)])
    def test_gemm_int4(self, m, n, k, per_channel, dtype):
        input = torch.rand([m, k], device="xpu", dtype=dtype)
        input_torch = input.cpu()
        weight = self.rand_int4(k * n, torch.int32, "xpu").reshape(k // 8, n)

        group_size = min(128, k)
        if per_channel:
            group_size = k
        group_num = int(k / group_size)

        scales = torch.rand([group_num, n], device="xpu", dtype=dtype)
        zero_points = self.rand_int4(group_num * n, torch.int32, "xpu").reshape(
            group_num, n // 8
        )

        weight_fp = self.dequantize(
            weight, scales, zero_points, group_size).cpu()
        # check gemm
        zero_points = torch.Tensor([8]).to(torch.int8).to("xpu")
        weight_ba = weight.transpose(0, 1).contiguous()

        out_onednn =torch._weight_int4pack_mm_with_scales_and_zeros(
            input, weight_ba, scales, zero_points, group_size
        )
        out_torch = torch.matmul(input_torch, weight_fp)
        self.assertEqual(
            out_onednn.cpu().float(),
            out_torch.float(),
            atol=checking_atol,
            rtol=checking_rtol,
        )
        # check gemm + residual
        res0 = torch.rand([m, n], device="xpu", dtype=dtype)
        out_onednn_res = torch._weight_int4pack_mm_with_scales_and_zeros(
            input, weight_ba, scales, zero_points, group_size, res0)
        out_torch_res = out_torch + res0.cpu().float()
        self.assertEqual(
            out_onednn_res.cpu().float(),
            out_torch_res.float(),
            atol=checking_atol,
            rtol=checking_rtol,
        )

        # check gemm + bias
        bias = torch.rand([1, n], device="xpu", dtype=dtype)
        out_onednn_bias = torch._weight_int4pack_mm_with_scales_and_zeros(
            input, weight_ba, bias, scales, zero_points, group_size)
        out_torch_bias = out_torch + bias.cpu().float()
        self.assertEqual(
            out_onednn_bias.cpu().float(),
            out_torch_bias.float(),
            atol=checking_atol,
            rtol=checking_rtol,
        )

        # check gemm + bias + gelu
        out_onednn_gelu = torch._weight_int4pack_mm_with_scales_and_zeros(
            input,
            weight_ba,
            scales,
            zero_points,
            bias,
            group_size,
            "tanh",
        )
        gelu_out = torch.nn.GELU(approximate="tanh")(out_torch_bias)
        self.assertEqual(
            out_onednn_gelu.cpu().float(),
            gelu_out.float(),
            atol=checking_atol,
            rtol=checking_rtol,
        )

        # check gemm + silu + mul
        res0 = torch.rand([m, n], device="xpu", dtype=dtype)
        out_onednn_silu = torch._weight_int4pack_mm_with_scales_and_zeros(
            input, weight_ba, scales, zero_points, group_size, res0
        )
        silu_mul_out = torch.nn.SiLU()(out_torch) * res0.cpu().float()
        self.assertEqual(
            out_onednn_silu.cpu().float(),
            silu_mul_out.float(),
            atol=checking_atol,
            rtol=checking_rtol,
        )

        # check gemm + bias + residual + residual
        res0 = torch.rand([m, n], device="xpu", dtype=dtype)
        res1 = torch.rand([m, n], device="xpu", dtype=dtype)
        out_onednn_bias_2res = torch._weight_int4pack_mm_with_scales_and_zeros(
            input,
            weight_ba,
            bias,
            res0,
            res1,
            scales,
            zero_points,
            group_size,
        )
        out_torch_bias_2res = out_torch_bias + res0.cpu().float() + res1.cpu().float()
        self.assertEqual(
            out_onednn_bias_2res.cpu().float(),
            out_torch_bias_2res.float(),
            atol=checking_atol,
            rtol=checking_rtol,
        )

        # check gemm + bias + residual
        res0 = torch.rand([m, n], device="xpu", dtype=dtype)
        out_onednn_bias_add = torch._weight_int4pack_mm_with_scales_and_zeros(
            input,
            weight_ba,
            bias,
            scales,
            zero_points,
            group_size,
            res0,
        )
        out_torch_bias_add = out_torch_bias + res0.cpu().float()
        self.assertEqual(
            out_onednn_bias_add.cpu().float(),
            out_torch_bias_add.float(),
            atol=checking_atol,
            rtol=checking_rtol,
        )


instantiate_parametrized_tests(TestSYCLInt4Linear, globals(), only_for="xpu", allow_xpu=True)

if __name__ == "__main__":
    TestCase._default_dtype_check_enabled = True
    run_tests()
