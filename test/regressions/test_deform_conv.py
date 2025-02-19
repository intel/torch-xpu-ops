# Owner(s): ["module: intel"]
import math
import os

import pytest
import torch
import torch.fx
import torch.testing._internal.optests as optests
from torch import nn
from torch.nn.modules.utils import _pair
from torchvision import ops
from torchvision.models.feature_extraction import get_graph_node_names

OPTESTS = [
    "test_schema",
    "test_faketensor",
    "test_aot_dispatch_dynamic",
]


class DeformConvModuleWrapper(nn.Module):
    def __init__(self, obj):
        super().__init__()
        self.layer = obj
        self.n_inputs = 3

    def forward(self, a, b, c):
        self.layer(a, b, c)


def bilinear_interpolate(data, y, x, snap_border=False):
    height, width = data.shape

    if snap_border:
        if -1 < y <= 0:
            y = 0
        elif height - 1 <= y < height:
            y = height - 1

        if -1 < x <= 0:
            x = 0
        elif width - 1 <= x < width:
            x = width - 1

    y_low = int(math.floor(y))
    x_low = int(math.floor(x))
    y_high = y_low + 1
    x_high = x_low + 1

    wy_h = y - y_low
    wx_h = x - x_low
    wy_l = 1 - wy_h
    wx_l = 1 - wx_h

    val = 0
    for wx, xp in zip((wx_l, wx_h), (x_low, x_high)):
        for wy, yp in zip((wy_l, wy_h), (y_low, y_high)):
            if 0 <= yp < height and 0 <= xp < width:
                val += wx * wy * data[yp, xp]
    return val


class TestDeformConv:
    dtype = torch.float32

    def expected_fn(
        self, x, weight, offset, mask, bias, stride=1, padding=0, dilation=1
    ):
        stride_h, stride_w = _pair(stride)
        pad_h, pad_w = _pair(padding)
        dil_h, dil_w = _pair(dilation)
        weight_h, weight_w = weight.shape[-2:]

        n_batches, n_in_channels, in_h, in_w = x.shape
        n_out_channels = weight.shape[0]

        out_h = (in_h + 2 * pad_h - (dil_h * (weight_h - 1) + 1)) // stride_h + 1
        out_w = (in_w + 2 * pad_w - (dil_w * (weight_w - 1) + 1)) // stride_w + 1

        n_offset_grps = offset.shape[1] // (2 * weight_h * weight_w)
        in_c_per_offset_grp = n_in_channels // n_offset_grps

        n_weight_grps = n_in_channels // weight.shape[1]
        in_c_per_weight_grp = weight.shape[1]
        out_c_per_weight_grp = n_out_channels // n_weight_grps

        out = torch.zeros(
            n_batches, n_out_channels, out_h, out_w, device=x.device, dtype=x.dtype
        )
        for b in range(n_batches):
            for c_out in range(n_out_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        for di in range(weight_h):
                            for dj in range(weight_w):
                                for c in range(in_c_per_weight_grp):
                                    weight_grp = c_out // out_c_per_weight_grp
                                    c_in = weight_grp * in_c_per_weight_grp + c

                                    offset_grp = c_in // in_c_per_offset_grp
                                    mask_idx = (
                                        offset_grp * (weight_h * weight_w)
                                        + di * weight_w
                                        + dj
                                    )
                                    offset_idx = 2 * mask_idx

                                    pi = (
                                        stride_h * i
                                        - pad_h
                                        + dil_h * di
                                        + offset[b, offset_idx, i, j]
                                    )
                                    pj = (
                                        stride_w * j
                                        - pad_w
                                        + dil_w * dj
                                        + offset[b, offset_idx + 1, i, j]
                                    )

                                    mask_value = 1.0
                                    if mask is not None:
                                        mask_value = mask[b, mask_idx, i, j]

                                    out[b, c_out, i, j] += (
                                        mask_value
                                        * weight[c_out, c, di, dj]
                                        * bilinear_interpolate(x[b, c_in, :, :], pi, pj)
                                    )
        out += bias.view(1, n_out_channels, 1, 1)
        return out

    def get_fn_args(self, device, contiguous, batch_sz, dtype):
        n_in_channels = 6
        n_out_channels = 2
        n_weight_grps = 2
        n_offset_grps = 3

        stride = (2, 1)
        pad = (1, 0)
        dilation = (2, 1)

        stride_h, stride_w = stride
        pad_h, pad_w = pad
        dil_h, dil_w = dilation
        weight_h, weight_w = (3, 2)
        in_h, in_w = (5, 4)

        out_h = (in_h + 2 * pad_h - (dil_h * (weight_h - 1) + 1)) // stride_h + 1
        out_w = (in_w + 2 * pad_w - (dil_w * (weight_w - 1) + 1)) // stride_w + 1

        x = torch.rand(
            batch_sz, n_in_channels, in_h, in_w, dtype=dtype, requires_grad=True
        ).to(device)

        offset = torch.randn(
            batch_sz,
            n_offset_grps * 2 * weight_h * weight_w,
            out_h,
            out_w,
            dtype=dtype,
            requires_grad=True,
        ).to(device)

        mask = torch.randn(
            batch_sz,
            n_offset_grps * weight_h * weight_w,
            out_h,
            out_w,
            dtype=dtype,
            requires_grad=True,
        ).to(device)

        weight = torch.randn(
            n_out_channels,
            n_in_channels // n_weight_grps,
            weight_h,
            weight_w,
            dtype=dtype,
            requires_grad=True,
        ).to(device)

        bias = torch.randn(n_out_channels, dtype=dtype, requires_grad=True).to(device)

        if not contiguous:
            x = x.permute(0, 1, 3, 2).contiguous().permute(0, 1, 3, 2)
            offset = offset.permute(1, 3, 0, 2).contiguous().permute(2, 0, 3, 1)
            mask = mask.permute(1, 3, 0, 2).contiguous().permute(2, 0, 3, 1)
            weight = weight.permute(3, 2, 0, 1).contiguous().permute(2, 3, 1, 0)

        return x, weight, offset, mask, bias, stride, pad, dilation

    def make_obj(
        self, in_channels=6, out_channels=2, kernel_size=(3, 2), groups=2, wrap=False
    ):
        obj = ops.DeformConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=(2, 1),
            padding=(1, 0),
            dilation=(2, 1),
            groups=groups,
        )
        return DeformConvModuleWrapper(obj) if wrap else obj

    @pytest.mark.parametrize("device", ("xpu",))
    def test_is_leaf_node(self, device):
        op_obj = self.make_obj(wrap=True).to(device=device)
        graph_node_names = get_graph_node_names(op_obj)

        assert len(graph_node_names) == 2
        assert len(graph_node_names[0]) == len(graph_node_names[1])
        assert len(graph_node_names[0]) == 1 + op_obj.n_inputs

    @pytest.mark.parametrize("device", ("xpu",))
    @pytest.mark.parametrize("contiguous", (True, False))
    @pytest.mark.parametrize("batch_sz", (0, 1, 33))
    @pytest.mark.opcheck_only_one
    def test_forward(self, device, contiguous, batch_sz, dtype=None):
        dtype = dtype or self.dtype
        x, _, offset, mask, _, stride, padding, dilation = self.get_fn_args(
            device, contiguous, batch_sz, dtype
        )
        in_channels = 6
        out_channels = 2
        kernel_size = (3, 2)
        groups = 2
        tol = 2e-3 if dtype is torch.half else 1e-5

        layer = self.make_obj(
            in_channels, out_channels, kernel_size, groups, wrap=False
        ).to(device=x.device, dtype=dtype)
        res = layer(x, offset, mask)

        weight = layer.weight.data
        bias = layer.bias.data
        expected = self.expected_fn(
            x,
            weight,
            offset,
            mask,
            bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

        torch.testing.assert_close(
            res.to(expected),
            expected,
            rtol=tol,
            atol=tol,
            msg=f"\nres:\n{res}\nexpected:\n{expected}",
        )

        # no modulation test
        res = layer(x, offset)
        expected = self.expected_fn(
            x,
            weight,
            offset,
            None,
            bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

        torch.testing.assert_close(
            res.to(expected),
            expected,
            rtol=tol,
            atol=tol,
            msg=f"\nres:\n{res}\nexpected:\n{expected}",
        )

    @pytest.mark.parametrize("contiguous", (True, False))
    @pytest.mark.opcheck_only_one
    def test_compare_cpu_xpu_grads(self, contiguous):
        true_cpu_grads = None

        init_weight = torch.randn(9, 9, 3, 3, requires_grad=True)
        img = torch.randn(8, 9, 1000, 110)
        offset = torch.rand(8, 2 * 3 * 3, 1000, 110)
        mask = torch.rand(8, 3 * 3, 1000, 110)

        if not contiguous:
            img = img.permute(0, 1, 3, 2).contiguous().permute(0, 1, 3, 2)
            offset = offset.permute(1, 3, 0, 2).contiguous().permute(2, 0, 3, 1)
            mask = mask.permute(1, 3, 0, 2).contiguous().permute(2, 0, 3, 1)
            weight = init_weight.permute(3, 2, 0, 1).contiguous().permute(2, 3, 1, 0)
        else:
            weight = init_weight

        for d in ["xpu"]:
            out = ops.deform_conv2d(
                img.to(d), offset.to(d), weight.to(d), padding=1, mask=mask.to(d)
            )
            out.mean().backward()
            if true_cpu_grads is None:
                true_cpu_grads = init_weight.grad
                assert true_cpu_grads is not None
            else:
                assert init_weight.grad is not None
                res_grads = init_weight.grad.to("cpu")
                torch.testing.assert_close(true_cpu_grads, res_grads)

    def test_forward_scriptability(self):
        # Non-regression test for https://github.com/pytorch/vision/issues/4078
        torch.jit.script(ops.DeformConv2d(in_channels=8, out_channels=8, kernel_size=3))


optests.generate_opcheck_tests(
    testcase=TestDeformConv,
    namespaces=["torchvision"],
    failures_dict_path=os.path.join(
        os.path.dirname(__file__), "optests_failures_dict.json"
    ),
    additional_decorators=[],
    test_utils=OPTESTS,
)


if __name__ == "__main__":
    pytest.main([__file__])
