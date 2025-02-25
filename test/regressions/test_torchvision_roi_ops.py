# Owner(s): ["module: intel"]
import math
from abc import ABC, abstractmethod

import numpy as np
import pytest
import torch
import torch.fx
from torch import nn
from torch.autograd import gradcheck
from torchvision import ops
from torchvision.models.feature_extraction import get_graph_node_names


# Context manager for setting deterministic flag and automatically
# resetting it to its original value
class DeterministicGuard:
    def __init__(self, deterministic, *, warn_only=False):
        self.deterministic = deterministic
        self.warn_only = warn_only

    def __enter__(self):
        self.deterministic_restore = torch.are_deterministic_algorithms_enabled()
        self.warn_only_restore = torch.is_deterministic_algorithms_warn_only_enabled()
        torch.use_deterministic_algorithms(self.deterministic, warn_only=self.warn_only)

    def __exit__(self, exception_type, exception_value, traceback):
        torch.use_deterministic_algorithms(
            self.deterministic_restore, warn_only=self.warn_only_restore
        )


class RoIOpTesterModuleWrapper(nn.Module):
    def __init__(self, obj):
        super().__init__()
        self.layer = obj
        self.n_inputs = 2

    def forward(self, a, b):
        self.layer(a, b)


class MultiScaleRoIAlignModuleWrapper(nn.Module):
    def __init__(self, obj):
        super().__init__()
        self.layer = obj
        self.n_inputs = 3

    def forward(self, a, b, c):
        self.layer(a, b, c)


class RoIOpTester(ABC):
    dtype = torch.float64
    mps_dtype = torch.float32
    mps_backward_atol = 2e-2

    @pytest.mark.parametrize("device", ("xpu",))
    @pytest.mark.parametrize("contiguous", (True, False))
    @pytest.mark.parametrize(
        "x_dtype",
        (
            torch.float16,
            torch.float32,
            torch.float64,
        ),
        ids=str,
    )
    def test_forward(
        self,
        device,
        contiguous,
        x_dtype,
        rois_dtype=None,
        deterministic=False,
        **kwargs,
    ):
        if device == "mps" and x_dtype is torch.float64:
            pytest.skip("MPS does not support float64")

        rois_dtype = x_dtype if rois_dtype is None else rois_dtype

        tol = 1e-5
        if x_dtype is torch.half:
            if device == "mps":
                tol = 5e-3
            else:
                tol = 4e-3
        elif x_dtype == torch.bfloat16:
            tol = 5e-3

        pool_size = 5
        # n_channels % (pool_size ** 2) == 0 required for PS operations.
        n_channels = 2 * (pool_size**2)
        x = torch.rand(2, n_channels, 10, 10, dtype=x_dtype, device=device)
        if not contiguous:
            x = x.permute(0, 1, 3, 2)
        rois = torch.tensor(
            [
                [0, 0, 0, 9, 9],
                [0, 0, 5, 4, 9],
                [0, 5, 5, 9, 9],
                [1, 0, 0, 9, 9],
            ],  # format is (xyxy)
            dtype=rois_dtype,
            device=device,
        )

        pool_h, pool_w = pool_size, pool_size
        with DeterministicGuard(deterministic):
            y = self.fn(
                x, rois, pool_h, pool_w, spatial_scale=1, sampling_ratio=-1, **kwargs
            )
        # the following should be true whether we're running an autocast test or not.
        assert y.dtype == x.dtype
        gt_y = self.expected_fn(
            x,
            rois,
            pool_h,
            pool_w,
            spatial_scale=1,
            sampling_ratio=-1,
            device=device,
            dtype=x_dtype,
            **kwargs,
        )

        torch.testing.assert_close(gt_y.to(y), y, rtol=tol, atol=tol)

    @pytest.mark.parametrize("device", ("xpu",))
    def test_is_leaf_node(self, device):
        op_obj = self.make_obj(wrap=True).to(device=device)
        graph_node_names = get_graph_node_names(op_obj)

        assert len(graph_node_names) == 2
        assert len(graph_node_names[0]) == len(graph_node_names[1])
        assert len(graph_node_names[0]) == 1 + op_obj.n_inputs

    @pytest.mark.parametrize("device", ("xpu",))
    def test_torch_fx_trace(self, device, x_dtype=torch.float, rois_dtype=torch.float):
        op_obj = self.make_obj().to(device=device)
        graph_module = torch.fx.symbolic_trace(op_obj)
        pool_size = 5
        n_channels = 2 * (pool_size**2)
        x = torch.rand(2, n_channels, 5, 5, dtype=x_dtype, device=device)
        rois = torch.tensor(
            [
                [0, 0, 0, 9, 9],
                [0, 0, 5, 4, 9],
                [0, 5, 5, 9, 9],
                [1, 0, 0, 9, 9],
            ],  # format is (xyxy)
            dtype=rois_dtype,
            device=device,
        )
        output_gt = op_obj(x, rois)
        assert output_gt.dtype == x.dtype
        output_fx = graph_module(x, rois)
        assert output_fx.dtype == x.dtype
        tol = 1e-5
        torch.testing.assert_close(output_gt, output_fx, rtol=tol, atol=tol)

    @pytest.mark.parametrize("seed", range(10))
    @pytest.mark.parametrize("device", ("xpu",))
    @pytest.mark.parametrize("contiguous", (True, False))
    def test_backward(self, seed, device, contiguous, deterministic=False):
        atol = self.mps_backward_atol if device == "mps" else 1e-05
        dtype = self.mps_dtype if device == "mps" else self.dtype

        torch.random.manual_seed(seed)
        pool_size = 2
        x = torch.rand(
            1,
            2 * (pool_size**2),
            5,
            5,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        if not contiguous:
            x = x.permute(0, 1, 3, 2)
        rois = torch.tensor(
            [[0, 0, 0, 4, 4], [0, 0, 2, 3, 4], [0, 2, 2, 4, 4]],
            dtype=dtype,
            device=device,  # format is (xyxy)
        )

        def func(z):
            return self.fn(
                z, rois, pool_size, pool_size, spatial_scale=1, sampling_ratio=1
            )

        script_func = self.get_script_fn(rois, pool_size)

        with DeterministicGuard(deterministic):
            gradcheck(func, (x,), atol=atol)

        gradcheck(script_func, (x,), atol=atol)

    @abstractmethod
    def fn(self, *args, **kwargs):
        pass

    @abstractmethod
    def make_obj(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_script_fn(self, *args, **kwargs):
        pass

    @abstractmethod
    def expected_fn(self, *args, **kwargs):
        pass


class TestRoiPool(RoIOpTester):
    def fn(self, x, rois, pool_h, pool_w, spatial_scale=1, sampling_ratio=-1, **kwargs):
        return ops.RoIPool((pool_h, pool_w), spatial_scale)(x, rois)

    def make_obj(self, pool_h=5, pool_w=5, spatial_scale=1, wrap=False):
        obj = ops.RoIPool((pool_h, pool_w), spatial_scale)
        return RoIOpTesterModuleWrapper(obj) if wrap else obj

    def get_script_fn(self, rois, pool_size):
        scriped = torch.jit.script(ops.roi_pool)
        return lambda x: scriped(x, rois, pool_size)

    def expected_fn(
        self,
        x,
        rois,
        pool_h,
        pool_w,
        spatial_scale=1,
        sampling_ratio=-1,
        device=None,
        dtype=torch.float64,
    ):
        if device is None:
            device = torch.device("cpu")

        n_channels = x.size(1)
        y = torch.zeros(
            rois.size(0), n_channels, pool_h, pool_w, dtype=dtype, device=device
        )

        def get_slice(k, block):
            return slice(int(np.floor(k * block)), int(np.ceil((k + 1) * block)))

        for roi_idx, roi in enumerate(rois):
            batch_idx = int(roi[0])
            j_begin, i_begin, j_end, i_end = (
                int(round(x.item() * spatial_scale)) for x in roi[1:]
            )
            roi_x = x[batch_idx, :, i_begin : i_end + 1, j_begin : j_end + 1]

            roi_h, roi_w = roi_x.shape[-2:]
            bin_h = roi_h / pool_h
            bin_w = roi_w / pool_w

            for i in range(0, pool_h):
                for j in range(0, pool_w):
                    bin_x = roi_x[:, get_slice(i, bin_h), get_slice(j, bin_w)]
                    if bin_x.numel() > 0:
                        y[roi_idx, :, i, j] = bin_x.reshape(n_channels, -1).max(dim=1)[
                            0
                        ]
        return y


class TestPSRoIPool(RoIOpTester):
    mps_backward_atol = 5e-2

    def fn(self, x, rois, pool_h, pool_w, spatial_scale=1, sampling_ratio=-1, **kwargs):
        return ops.PSRoIPool((pool_h, pool_w), 1)(x, rois)

    def make_obj(self, pool_h=5, pool_w=5, spatial_scale=1, wrap=False):
        obj = ops.PSRoIPool((pool_h, pool_w), spatial_scale)
        return RoIOpTesterModuleWrapper(obj) if wrap else obj

    def get_script_fn(self, rois, pool_size):
        scriped = torch.jit.script(ops.ps_roi_pool)
        return lambda x: scriped(x, rois, pool_size)

    def expected_fn(
        self,
        x,
        rois,
        pool_h,
        pool_w,
        spatial_scale=1,
        sampling_ratio=-1,
        device=None,
        dtype=torch.float64,
    ):
        if device is None:
            device = torch.device("cpu")
        n_input_channels = x.size(1)
        assert (
            n_input_channels % (pool_h * pool_w) == 0
        ), "input channels must be divisible by ph * pw"
        n_output_channels = int(n_input_channels / (pool_h * pool_w))
        y = torch.zeros(
            rois.size(0), n_output_channels, pool_h, pool_w, dtype=dtype, device=device
        )

        def get_slice(k, block):
            return slice(int(np.floor(k * block)), int(np.ceil((k + 1) * block)))

        for roi_idx, roi in enumerate(rois):
            batch_idx = int(roi[0])
            j_begin, i_begin, j_end, i_end = (
                int(round(x.item() * spatial_scale)) for x in roi[1:]
            )
            roi_x = x[batch_idx, :, i_begin : i_end + 1, j_begin : j_end + 1]

            roi_height = max(i_end - i_begin, 1)
            roi_width = max(j_end - j_begin, 1)
            bin_h, bin_w = roi_height / float(pool_h), roi_width / float(pool_w)

            for i in range(0, pool_h):
                for j in range(0, pool_w):
                    bin_x = roi_x[:, get_slice(i, bin_h), get_slice(j, bin_w)]
                    if bin_x.numel() > 0:
                        area = bin_x.size(-2) * bin_x.size(-1)
                        for c_out in range(0, n_output_channels):
                            c_in = c_out * (pool_h * pool_w) + pool_w * i + j
                            t = torch.sum(bin_x[c_in, :, :])
                            y[roi_idx, c_out, i, j] = t / area
        return y


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


class TestRoIAlign(RoIOpTester):
    mps_backward_atol = 6e-2

    def fn(
        self,
        x,
        rois,
        pool_h,
        pool_w,
        spatial_scale=1,
        sampling_ratio=-1,
        aligned=False,
        **kwargs,
    ):
        return ops.RoIAlign(
            (pool_h, pool_w),
            spatial_scale=spatial_scale,
            sampling_ratio=sampling_ratio,
            aligned=aligned,
        )(x, rois)

    def make_obj(
        self,
        pool_h=5,
        pool_w=5,
        spatial_scale=1,
        sampling_ratio=-1,
        aligned=False,
        wrap=False,
    ):
        obj = ops.RoIAlign(
            (pool_h, pool_w),
            spatial_scale=spatial_scale,
            sampling_ratio=sampling_ratio,
            aligned=aligned,
        )
        return RoIOpTesterModuleWrapper(obj) if wrap else obj

    def get_script_fn(self, rois, pool_size):
        scriped = torch.jit.script(ops.roi_align)
        return lambda x: scriped(x, rois, pool_size)

    def expected_fn(
        self,
        in_data,
        rois,
        pool_h,
        pool_w,
        spatial_scale=1,
        sampling_ratio=-1,
        aligned=False,
        device=None,
        dtype=torch.float64,
    ):
        if device is None:
            device = torch.device("cpu")
        n_channels = in_data.size(1)
        out_data = torch.zeros(
            rois.size(0), n_channels, pool_h, pool_w, dtype=dtype, device=device
        )

        offset = 0.5 if aligned else 0.0

        for r, roi in enumerate(rois):
            batch_idx = int(roi[0])
            j_begin, i_begin, j_end, i_end = (
                x.item() * spatial_scale - offset for x in roi[1:]
            )

            roi_h = i_end - i_begin
            roi_w = j_end - j_begin
            bin_h = roi_h / pool_h
            bin_w = roi_w / pool_w

            for i in range(0, pool_h):
                start_h = i_begin + i * bin_h
                grid_h = sampling_ratio if sampling_ratio > 0 else int(np.ceil(bin_h))
                for j in range(0, pool_w):
                    start_w = j_begin + j * bin_w
                    grid_w = (
                        sampling_ratio if sampling_ratio > 0 else int(np.ceil(bin_w))
                    )

                    for channel in range(0, n_channels):
                        val = 0
                        for iy in range(0, grid_h):
                            y = start_h + (iy + 0.5) * bin_h / grid_h
                            for ix in range(0, grid_w):
                                x = start_w + (ix + 0.5) * bin_w / grid_w
                                val += bilinear_interpolate(
                                    in_data[batch_idx, channel, :, :],
                                    y,
                                    x,
                                    snap_border=True,
                                )
                        val /= grid_h * grid_w

                        out_data[r, channel, i, j] = val
        return out_data

    @pytest.mark.parametrize("aligned", (True, False))
    @pytest.mark.parametrize("device", ("xpu",))
    @pytest.mark.parametrize(
        "x_dtype", (torch.float16, torch.float32, torch.float64)
    )  # , ids=str)
    @pytest.mark.parametrize("contiguous", (True, False))
    @pytest.mark.parametrize("deterministic", (True, False))
    @pytest.mark.opcheck_only_one
    def test_forward(
        self, device, contiguous, deterministic, aligned, x_dtype, rois_dtype=None
    ):
        if deterministic and device == "cpu":
            pytest.skip("cpu is always deterministic, don't retest")
        super().test_forward(
            device=device,
            contiguous=contiguous,
            deterministic=deterministic,
            x_dtype=x_dtype,
            rois_dtype=rois_dtype,
            aligned=aligned,
        )

    @pytest.mark.parametrize("aligned", (True, False))
    @pytest.mark.parametrize("deterministic", (True, False))
    @pytest.mark.parametrize("x_dtype", (torch.float, torch.half))
    @pytest.mark.parametrize("rois_dtype", (torch.float, torch.half))
    @pytest.mark.opcheck_only_one
    def test_autocast(self, aligned, deterministic, x_dtype, rois_dtype):
        with torch.amp.autocast("xpu"):
            self.test_forward(
                torch.device("xpu"),
                contiguous=False,
                deterministic=deterministic,
                aligned=aligned,
                x_dtype=x_dtype,
                rois_dtype=rois_dtype,
            )


class TestPSRoIAlign(RoIOpTester):
    mps_backward_atol = 5e-2

    def fn(self, x, rois, pool_h, pool_w, spatial_scale=1, sampling_ratio=-1, **kwargs):
        return ops.PSRoIAlign(
            (pool_h, pool_w), spatial_scale=spatial_scale, sampling_ratio=sampling_ratio
        )(x, rois)

    def make_obj(
        self, pool_h=5, pool_w=5, spatial_scale=1, sampling_ratio=-1, wrap=False
    ):
        obj = ops.PSRoIAlign(
            (pool_h, pool_w), spatial_scale=spatial_scale, sampling_ratio=sampling_ratio
        )
        return RoIOpTesterModuleWrapper(obj) if wrap else obj

    def get_script_fn(self, rois, pool_size):
        scriped = torch.jit.script(ops.ps_roi_align)
        return lambda x: scriped(x, rois, pool_size)

    def expected_fn(
        self,
        in_data,
        rois,
        pool_h,
        pool_w,
        device,
        spatial_scale=1,
        sampling_ratio=-1,
        dtype=torch.float64,
    ):
        if device is None:
            device = torch.device("cpu")
        n_input_channels = in_data.size(1)
        assert (
            n_input_channels % (pool_h * pool_w) == 0
        ), "input channels must be divisible by ph * pw"
        n_output_channels = int(n_input_channels / (pool_h * pool_w))
        out_data = torch.zeros(
            rois.size(0), n_output_channels, pool_h, pool_w, dtype=dtype, device=device
        )

        for r, roi in enumerate(rois):
            batch_idx = int(roi[0])
            j_begin, i_begin, j_end, i_end = (
                x.item() * spatial_scale - 0.5 for x in roi[1:]
            )

            roi_h = i_end - i_begin
            roi_w = j_end - j_begin
            bin_h = roi_h / pool_h
            bin_w = roi_w / pool_w

            for i in range(0, pool_h):
                start_h = i_begin + i * bin_h
                grid_h = sampling_ratio if sampling_ratio > 0 else int(np.ceil(bin_h))
                for j in range(0, pool_w):
                    start_w = j_begin + j * bin_w
                    grid_w = (
                        sampling_ratio if sampling_ratio > 0 else int(np.ceil(bin_w))
                    )
                    for c_out in range(0, n_output_channels):
                        c_in = c_out * (pool_h * pool_w) + pool_w * i + j

                        val = 0
                        for iy in range(0, grid_h):
                            y = start_h + (iy + 0.5) * bin_h / grid_h
                            for ix in range(0, grid_w):
                                x = start_w + (ix + 0.5) * bin_w / grid_w
                                val += bilinear_interpolate(
                                    in_data[batch_idx, c_in, :, :],
                                    y,
                                    x,
                                    snap_border=True,
                                )
                        val /= grid_h * grid_w

                        out_data[r, c_out, i, j] = val
        return out_data


class TestMultiScaleRoIAlign:
    def make_obj(
        self, fmap_names=None, output_size=(7, 7), sampling_ratio=2, wrap=False
    ):
        if fmap_names is None:
            fmap_names = ["0"]
        obj = ops.poolers.MultiScaleRoIAlign(fmap_names, output_size, sampling_ratio)
        return MultiScaleRoIAlignModuleWrapper(obj) if wrap else obj

    @pytest.mark.parametrize("device", ("xpu",))
    def test_is_leaf_node(self, device):
        op_obj = self.make_obj(wrap=True).to(device=device)
        graph_node_names = get_graph_node_names(op_obj)

        assert len(graph_node_names) == 2
        assert len(graph_node_names[0]) == len(graph_node_names[1])
        assert len(graph_node_names[0]) == 1 + op_obj.n_inputs


if __name__ == "__main__":
    pytest.main([__file__])
