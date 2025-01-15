import math
import os
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import List, Tuple

import numpy as np
import pytest
import torch
import torch.fx
import torch.nn.functional as F
import torch.testing._internal.optests as optests
from torch import nn, Tensor
from torch._dynamo.utils import is_compile_supported
from torch.autograd import gradcheck
from torch.nn.modules.utils import _pair
from torchvision import ops
from torchvision.models.feature_extraction import get_graph_node_names

OPTESTS = [
    "test_schema",
    "test_autograd_registration",
    "test_faketensor",
    "test_aot_dispatch_dynamic",
]


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
        torch.use_deterministic_algorithms(self.deterministic_restore, warn_only=self.warn_only_restore)


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


class DeformConvModuleWrapper(nn.Module):
    def __init__(self, obj):
        super().__init__()
        self.layer = obj
        self.n_inputs = 3

    def forward(self, a, b, c):
        self.layer(a, b, c)


class StochasticDepthWrapper(nn.Module):
    def __init__(self, obj):
        super().__init__()
        self.layer = obj
        self.n_inputs = 1

    def forward(self, a):
        self.layer(a)


class DropBlockWrapper(nn.Module):
    def __init__(self, obj):
        super().__init__()
        self.layer = obj
        self.n_inputs = 1

    def forward(self, a):
        self.layer(a)


class PoolWrapper(nn.Module):
    def __init__(self, pool: nn.Module):
        super().__init__()
        self.pool = pool

    def forward(self, imgs: Tensor, boxes: List[Tensor]) -> Tensor:
        return self.pool(imgs, boxes)


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
    def test_forward(self, device, contiguous, x_dtype, rois_dtype=None, deterministic=False, **kwargs):
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
            [[0, 0, 0, 9, 9], [0, 0, 5, 4, 9], [0, 5, 5, 9, 9], [1, 0, 0, 9, 9]],  # format is (xyxy)
            dtype=rois_dtype,
            device=device,
        )

        pool_h, pool_w = pool_size, pool_size
        with DeterministicGuard(deterministic):
            y = self.fn(x, rois, pool_h, pool_w, spatial_scale=1, sampling_ratio=-1, **kwargs)
        # the following should be true whether we're running an autocast test or not.
        assert y.dtype == x.dtype
        gt_y = self.expected_fn(
            x, rois, pool_h, pool_w, spatial_scale=1, sampling_ratio=-1, device=device, dtype=x_dtype, **kwargs
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
            [[0, 0, 0, 9, 9], [0, 0, 5, 4, 9], [0, 5, 5, 9, 9], [1, 0, 0, 9, 9]],  # format is (xyxy)
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
        x = torch.rand(1, 2 * (pool_size**2), 5, 5, dtype=dtype, device=device, requires_grad=True)
        if not contiguous:
            x = x.permute(0, 1, 3, 2)
        rois = torch.tensor(
            [[0, 0, 0, 4, 4], [0, 0, 2, 3, 4], [0, 2, 2, 4, 4]], dtype=dtype, device=device  # format is (xyxy)
        )

        def func(z):
            return self.fn(z, rois, pool_size, pool_size, spatial_scale=1, sampling_ratio=1)

        script_func = self.get_script_fn(rois, pool_size)

        with DeterministicGuard(deterministic):
            gradcheck(func, (x,), atol=atol)

        gradcheck(script_func, (x,), atol=atol)

    #@pytest.mark.parametrize("x_dtype", (torch.float, torch.half))
    #@pytest.mark.parametrize("rois_dtype", (torch.float, torch.half))
    #def test_autocast(self, x_dtype, rois_dtype):
    #    with torch.cpu.amp.autocast():
    #        self.test_forward(torch.device("cpu"), contiguous=False, x_dtype=x_dtype, rois_dtype=rois_dtype)

    @abstractmethod
    def fn(*args, **kwargs):
        pass

    @abstractmethod
    def make_obj(*args, **kwargs):
        pass

    @abstractmethod
    def get_script_fn(*args, **kwargs):
        pass

    @abstractmethod
    def expected_fn(*args, **kwargs):
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
        self, x, rois, pool_h, pool_w, spatial_scale=1, sampling_ratio=-1, device=None, dtype=torch.float64
    ):
        if device is None:
            device = torch.device("cpu")

        n_channels = x.size(1)
        y = torch.zeros(rois.size(0), n_channels, pool_h, pool_w, dtype=dtype, device=device)

        def get_slice(k, block):
            return slice(int(np.floor(k * block)), int(np.ceil((k + 1) * block)))

        for roi_idx, roi in enumerate(rois):
            batch_idx = int(roi[0])
            j_begin, i_begin, j_end, i_end = (int(round(x.item() * spatial_scale)) for x in roi[1:])
            roi_x = x[batch_idx, :, i_begin : i_end + 1, j_begin : j_end + 1]

            roi_h, roi_w = roi_x.shape[-2:]
            bin_h = roi_h / pool_h
            bin_w = roi_w / pool_w

            for i in range(0, pool_h):
                for j in range(0, pool_w):
                    bin_x = roi_x[:, get_slice(i, bin_h), get_slice(j, bin_w)]
                    if bin_x.numel() > 0:
                        y[roi_idx, :, i, j] = bin_x.reshape(n_channels, -1).max(dim=1)[0]
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
        self, x, rois, pool_h, pool_w, spatial_scale=1, sampling_ratio=-1, device=None, dtype=torch.float64
    ):
        if device is None:
            device = torch.device("cpu")
        n_input_channels = x.size(1)
        assert n_input_channels % (pool_h * pool_w) == 0, "input channels must be divisible by ph * pw"
        n_output_channels = int(n_input_channels / (pool_h * pool_w))
        y = torch.zeros(rois.size(0), n_output_channels, pool_h, pool_w, dtype=dtype, device=device)

        def get_slice(k, block):
            return slice(int(np.floor(k * block)), int(np.ceil((k + 1) * block)))

        for roi_idx, roi in enumerate(rois):
            batch_idx = int(roi[0])
            j_begin, i_begin, j_end, i_end = (int(round(x.item() * spatial_scale)) for x in roi[1:])
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

    def fn(self, x, rois, pool_h, pool_w, spatial_scale=1, sampling_ratio=-1, aligned=False, **kwargs):
        return ops.RoIAlign(
            (pool_h, pool_w), spatial_scale=spatial_scale, sampling_ratio=sampling_ratio, aligned=aligned
        )(x, rois)

    def make_obj(self, pool_h=5, pool_w=5, spatial_scale=1, sampling_ratio=-1, aligned=False, wrap=False):
        obj = ops.RoIAlign(
            (pool_h, pool_w), spatial_scale=spatial_scale, sampling_ratio=sampling_ratio, aligned=aligned
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
        out_data = torch.zeros(rois.size(0), n_channels, pool_h, pool_w, dtype=dtype, device=device)

        offset = 0.5 if aligned else 0.0

        for r, roi in enumerate(rois):
            batch_idx = int(roi[0])
            j_begin, i_begin, j_end, i_end = (x.item() * spatial_scale - offset for x in roi[1:])

            roi_h = i_end - i_begin
            roi_w = j_end - j_begin
            bin_h = roi_h / pool_h
            bin_w = roi_w / pool_w

            for i in range(0, pool_h):
                start_h = i_begin + i * bin_h
                grid_h = sampling_ratio if sampling_ratio > 0 else int(np.ceil(bin_h))
                for j in range(0, pool_w):
                    start_w = j_begin + j * bin_w
                    grid_w = sampling_ratio if sampling_ratio > 0 else int(np.ceil(bin_w))

                    for channel in range(0, n_channels):
                        val = 0
                        for iy in range(0, grid_h):
                            y = start_h + (iy + 0.5) * bin_h / grid_h
                            for ix in range(0, grid_w):
                                x = start_w + (ix + 0.5) * bin_w / grid_w
                                val += bilinear_interpolate(in_data[batch_idx, channel, :, :], y, x, snap_border=True)
                        val /= grid_h * grid_w

                        out_data[r, channel, i, j] = val
        return out_data

    @pytest.mark.parametrize("aligned", (True, False))
    @pytest.mark.parametrize("device", ("xpu",))
    @pytest.mark.parametrize("x_dtype", (torch.float16, torch.float32, torch.float64))  # , ids=str)
    @pytest.mark.parametrize("contiguous", (True, False))
    @pytest.mark.parametrize("deterministic", (True, False))
    @pytest.mark.opcheck_only_one()
    def test_forward(self, device, contiguous, deterministic, aligned, x_dtype, rois_dtype=None):
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
    @pytest.mark.opcheck_only_one()
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
        return ops.PSRoIAlign((pool_h, pool_w), spatial_scale=spatial_scale, sampling_ratio=sampling_ratio)(x, rois)

    def make_obj(self, pool_h=5, pool_w=5, spatial_scale=1, sampling_ratio=-1, wrap=False):
        obj = ops.PSRoIAlign((pool_h, pool_w), spatial_scale=spatial_scale, sampling_ratio=sampling_ratio)
        return RoIOpTesterModuleWrapper(obj) if wrap else obj

    def get_script_fn(self, rois, pool_size):
        scriped = torch.jit.script(ops.ps_roi_align)
        return lambda x: scriped(x, rois, pool_size)

    def expected_fn(
        self, in_data, rois, pool_h, pool_w, device, spatial_scale=1, sampling_ratio=-1, dtype=torch.float64
    ):
        if device is None:
            device = torch.device("cpu")
        n_input_channels = in_data.size(1)
        assert n_input_channels % (pool_h * pool_w) == 0, "input channels must be divisible by ph * pw"
        n_output_channels = int(n_input_channels / (pool_h * pool_w))
        out_data = torch.zeros(rois.size(0), n_output_channels, pool_h, pool_w, dtype=dtype, device=device)

        for r, roi in enumerate(rois):
            batch_idx = int(roi[0])
            j_begin, i_begin, j_end, i_end = (x.item() * spatial_scale - 0.5 for x in roi[1:])

            roi_h = i_end - i_begin
            roi_w = j_end - j_begin
            bin_h = roi_h / pool_h
            bin_w = roi_w / pool_w

            for i in range(0, pool_h):
                start_h = i_begin + i * bin_h
                grid_h = sampling_ratio if sampling_ratio > 0 else int(np.ceil(bin_h))
                for j in range(0, pool_w):
                    start_w = j_begin + j * bin_w
                    grid_w = sampling_ratio if sampling_ratio > 0 else int(np.ceil(bin_w))
                    for c_out in range(0, n_output_channels):
                        c_in = c_out * (pool_h * pool_w) + pool_w * i + j

                        val = 0
                        for iy in range(0, grid_h):
                            y = start_h + (iy + 0.5) * bin_h / grid_h
                            for ix in range(0, grid_w):
                                x = start_w + (ix + 0.5) * bin_w / grid_w
                                val += bilinear_interpolate(in_data[batch_idx, c_in, :, :], y, x, snap_border=True)
                        val /= grid_h * grid_w

                        out_data[r, c_out, i, j] = val
        return out_data


@pytest.mark.parametrize(
    "op",
    (
        torch.ops.torchvision.roi_pool,
        torch.ops.torchvision.ps_roi_pool,
        torch.ops.torchvision.roi_align,
        torch.ops.torchvision.ps_roi_align,
    ),
)
@pytest.mark.parametrize("dtype", (torch.float16, torch.float32, torch.float64))
@pytest.mark.parametrize("device", ("xpu",))
@pytest.mark.parametrize("requires_grad", (True, False))
def test_roi_opcheck(op, dtype, device, requires_grad):
    # This manually calls opcheck() on the roi ops. We do that instead of
    # relying on opcheck.generate_opcheck_tests() as e.g. done for nms, because
    # pytest and generate_opcheck_tests() don't interact very well when it comes
    # to skipping tests - and these ops need to skip the MPS tests since MPS we
    # don't support dynamic shapes yet for MPS.
    rois = torch.tensor(
        [[0, 0, 0, 9, 9], [0, 0, 5, 4, 9], [0, 5, 5, 9, 9], [1, 0, 0, 9, 9]],
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )
    pool_size = 5
    num_channels = 2 * (pool_size**2)
    x = torch.rand(2, num_channels, 10, 10, dtype=dtype, device=device)

    kwargs = dict(rois=rois, spatial_scale=1, pooled_height=pool_size, pooled_width=pool_size)
    if op in (torch.ops.torchvision.roi_align, torch.ops.torchvision.ps_roi_align):
        kwargs["sampling_ratio"] = -1
    if op is torch.ops.torchvision.roi_align:
        kwargs["aligned"] = True

    optests.opcheck(op, args=(x,), kwargs=kwargs)


class TestMultiScaleRoIAlign:
    def make_obj(self, fmap_names=None, output_size=(7, 7), sampling_ratio=2, wrap=False):
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


class TestNMS:
    def _reference_nms(self, boxes, scores, iou_threshold):
        """
        Args:
            boxes: boxes in corner-form
            scores: probabilities
            iou_threshold: intersection over union threshold
        Returns:
             picked: a list of indexes of the kept boxes
        """
        picked = []
        _, indexes = scores.sort(descending=True)
        while len(indexes) > 0:
            current = indexes[0]
            picked.append(current.item())
            if len(indexes) == 1:
                break
            current_box = boxes[current, :]
            indexes = indexes[1:]
            rest_boxes = boxes[indexes, :]
            iou = ops.box_iou(rest_boxes, current_box.unsqueeze(0)).squeeze(1)
            indexes = indexes[iou <= iou_threshold]

        return torch.as_tensor(picked)

    def _create_tensors_with_iou(self, N, iou_thresh):
        # force last box to have a pre-defined iou with the first box
        # let b0 be [x0, y0, x1, y1], and b1 be [x0, y0, x1 + d, y1],
        # then, in order to satisfy ops.iou(b0, b1) == iou_thresh,
        # we need to have d = (x1 - x0) * (1 - iou_thresh) / iou_thresh
        # Adjust the threshold upward a bit with the intent of creating
        # at least one box that exceeds (barely) the threshold and so
        # should be suppressed.
        boxes = torch.rand(N, 4) * 100
        boxes[:, 2:] += boxes[:, :2]
        boxes[-1, :] = boxes[0, :]
        x0, y0, x1, y1 = boxes[-1].tolist()
        iou_thresh += 1e-5
        boxes[-1, 2] += (x1 - x0) * (1 - iou_thresh) / iou_thresh
        scores = torch.rand(N)
        return boxes, scores

    @pytest.mark.parametrize("device", ("xpu",))
    @pytest.mark.parametrize("iou", (0.2, 0.5, 0.8))
    @pytest.mark.opcheck_only_one()
    def test_nms_gpu(self, iou, device, dtype=torch.float64):
        dtype = torch.float32 if device == "mps" else dtype
        tol = 1e-3 if dtype is torch.half else 1e-5
        err_msg = "NMS incompatible between CPU and XPU for IoU={}"

        boxes, scores = self._create_tensors_with_iou(1000, iou)
        r_cpu = ops.nms(boxes, scores, iou)
        r_gpu = ops.nms(boxes.to(device), scores.to(device), iou)

        is_eq = torch.allclose(r_cpu, r_gpu.cpu())
        if not is_eq:
            # if the indices are not the same, ensure that it's because the scores
            # are duplicate
            is_eq = torch.allclose(scores[r_cpu], scores[r_gpu.cpu()], rtol=tol, atol=tol)
        assert is_eq, err_msg.format(iou)

    @pytest.mark.parametrize("iou", (0.2, 0.5, 0.8))
    @pytest.mark.parametrize("dtype", (torch.float, torch.half))
    @pytest.mark.opcheck_only_one()
    def test_autocast(self, iou, dtype):
        with torch.amp.autocast("xpu"):
            self.test_nms_gpu(iou=iou, dtype=dtype, device="xpu")

    @pytest.mark.parametrize("device", ("xpu",))
    @pytest.mark.opcheck_only_one()
    def test_nms_float16(self, device):
        boxes = torch.tensor(
            [
                [285.3538, 185.5758, 1193.5110, 851.4551],
                [285.1472, 188.7374, 1192.4984, 851.0669],
                [279.2440, 197.9812, 1189.4746, 849.2019],
            ]
        ).to(device)
        scores = torch.tensor([0.6370, 0.7569, 0.3966]).to(device)

        iou_thres = 0.2
        keep32 = ops.nms(boxes, scores, iou_thres)
        keep16 = ops.nms(boxes.to(torch.float16), scores.to(torch.float16), iou_thres)
        torch.testing.assert_close(keep32, keep16)


optests.generate_opcheck_tests(
    testcase=TestNMS,
    namespaces=["torchvision"],
    failures_dict_path=os.path.join(os.path.dirname(__file__), "optests_failures_dict.json"),
    additional_decorators=[],
    test_utils=OPTESTS,
)


def get_boxes(dtype, device):
    box1 = torch.tensor([-1, -1, 1, 1], dtype=dtype, device=device)
    box2 = torch.tensor([0, 0, 1, 1], dtype=dtype, device=device)
    box3 = torch.tensor([0, 1, 1, 2], dtype=dtype, device=device)
    box4 = torch.tensor([1, 1, 2, 2], dtype=dtype, device=device)

    box1s = torch.stack([box2, box2], dim=0)
    box2s = torch.stack([box3, box4], dim=0)

    return box1, box2, box3, box4, box1s, box2s


def assert_iou_loss(iou_fn, box1, box2, expected_loss, device, reduction="none"):
    computed_loss = iou_fn(box1, box2, reduction=reduction)
    expected_loss = torch.tensor(expected_loss, device=device)
    torch.testing.assert_close(computed_loss, expected_loss)


def assert_empty_loss(iou_fn, dtype, device):
    box1 = torch.randn([0, 4], dtype=dtype, device=device).requires_grad_()
    box2 = torch.randn([0, 4], dtype=dtype, device=device).requires_grad_()
    loss = iou_fn(box1, box2, reduction="mean")
    loss.backward()
    torch.testing.assert_close(loss, torch.tensor(0.0, device=device))
    assert box1.grad is not None, "box1.grad should not be None after backward is called"
    assert box2.grad is not None, "box2.grad should not be None after backward is called"
    loss = iou_fn(box1, box2, reduction="none")
    assert loss.numel() == 0, f"{str(iou_fn)} for two empty box should be empty"


class TestGeneralizedBoxIouLoss:
    # We refer to original test: https://github.com/facebookresearch/fvcore/blob/main/tests/test_giou_loss.py
    @pytest.mark.parametrize("device", ("xpu",))
    @pytest.mark.parametrize("dtype", [torch.float32, torch.half])
    def test_giou_loss(self, dtype, device):
        box1, box2, box3, box4, box1s, box2s = get_boxes(dtype, device)

        # Identical boxes should have loss of 0
        assert_iou_loss(ops.generalized_box_iou_loss, box1, box1, 0.0, device=device)

        # quarter size box inside other box = IoU of 0.25
        assert_iou_loss(ops.generalized_box_iou_loss, box1, box2, 0.75, device=device)

        # Two side by side boxes, area=union
        # IoU=0 and GIoU=0 (loss 1.0)
        assert_iou_loss(ops.generalized_box_iou_loss, box2, box3, 1.0, device=device)

        # Two diagonally adjacent boxes, area=2*union
        # IoU=0 and GIoU=-0.5 (loss 1.5)
        assert_iou_loss(ops.generalized_box_iou_loss, box2, box4, 1.5, device=device)

        # Test batched loss and reductions
        assert_iou_loss(ops.generalized_box_iou_loss, box1s, box2s, 2.5, device=device, reduction="sum")
        assert_iou_loss(ops.generalized_box_iou_loss, box1s, box2s, 1.25, device=device, reduction="mean")

        # Test reduction value
        # reduction value other than ["none", "mean", "sum"] should raise a ValueError
        with pytest.raises(ValueError, match="Invalid"):
            ops.generalized_box_iou_loss(box1s, box2s, reduction="xyz")

    @pytest.mark.parametrize("device", ("xpu",))
    @pytest.mark.parametrize("dtype", [torch.float32, torch.half])
    def test_empty_inputs(self, dtype, device):
        assert_empty_loss(ops.generalized_box_iou_loss, dtype, device)


class TestCompleteBoxIouLoss:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.half])
    @pytest.mark.parametrize("device", ("xpu",))
    def test_ciou_loss(self, dtype, device):
        box1, box2, box3, box4, box1s, box2s = get_boxes(dtype, device)

        assert_iou_loss(ops.complete_box_iou_loss, box1, box1, 0.0, device=device)
        assert_iou_loss(ops.complete_box_iou_loss, box1, box2, 0.8125, device=device)
        assert_iou_loss(ops.complete_box_iou_loss, box1, box3, 1.1923, device=device)
        assert_iou_loss(ops.complete_box_iou_loss, box1, box4, 1.2500, device=device)
        assert_iou_loss(ops.complete_box_iou_loss, box1s, box2s, 1.2250, device=device, reduction="mean")
        assert_iou_loss(ops.complete_box_iou_loss, box1s, box2s, 2.4500, device=device, reduction="sum")

        with pytest.raises(ValueError, match="Invalid"):
            ops.complete_box_iou_loss(box1s, box2s, reduction="xyz")

    @pytest.mark.parametrize("device", ("xpu",))
    @pytest.mark.parametrize("dtype", [torch.float32, torch.half])
    def test_empty_inputs(self, dtype, device):
        assert_empty_loss(ops.complete_box_iou_loss, dtype, device)


class TestDistanceBoxIouLoss:
    @pytest.mark.parametrize("device", ("xpu",))
    @pytest.mark.parametrize("dtype", [torch.float32, torch.half])
    def test_distance_iou_loss(self, dtype, device):
        box1, box2, box3, box4, box1s, box2s = get_boxes(dtype, device)

        assert_iou_loss(ops.distance_box_iou_loss, box1, box1, 0.0, device=device)
        assert_iou_loss(ops.distance_box_iou_loss, box1, box2, 0.8125, device=device)
        assert_iou_loss(ops.distance_box_iou_loss, box1, box3, 1.1923, device=device)
        assert_iou_loss(ops.distance_box_iou_loss, box1, box4, 1.2500, device=device)
        assert_iou_loss(ops.distance_box_iou_loss, box1s, box2s, 1.2250, device=device, reduction="mean")
        assert_iou_loss(ops.distance_box_iou_loss, box1s, box2s, 2.4500, device=device, reduction="sum")

        with pytest.raises(ValueError, match="Invalid"):
            ops.distance_box_iou_loss(box1s, box2s, reduction="xyz")

    @pytest.mark.parametrize("device", ("xpu",))
    @pytest.mark.parametrize("dtype", [torch.float32, torch.half])
    def test_empty_distance_iou_inputs(self, dtype, device):
        assert_empty_loss(ops.distance_box_iou_loss, dtype, device)


class TestFocalLoss:
    def _generate_diverse_input_target_pair(self, shape=(5, 2), **kwargs):
        def logit(p):
            return torch.log(p / (1 - p))

        def generate_tensor_with_range_type(shape, range_type, **kwargs):
            if range_type != "random_binary":
                low, high = {
                    "small": (0.0, 0.2),
                    "big": (0.8, 1.0),
                    "zeros": (0.0, 0.0),
                    "ones": (1.0, 1.0),
                    "random": (0.0, 1.0),
                }[range_type]
                return torch.testing.make_tensor(shape, low=low, high=high, **kwargs)
            else:
                return torch.randint(0, 2, shape, **kwargs)

        # This function will return inputs and targets with shape: (shape[0]*9, shape[1])
        inputs = []
        targets = []
        for input_range_type, target_range_type in [
            ("small", "zeros"),
            ("small", "ones"),
            ("small", "random_binary"),
            ("big", "zeros"),
            ("big", "ones"),
            ("big", "random_binary"),
            ("random", "zeros"),
            ("random", "ones"),
            ("random", "random_binary"),
        ]:
            inputs.append(logit(generate_tensor_with_range_type(shape, input_range_type, **kwargs)))
            targets.append(generate_tensor_with_range_type(shape, target_range_type, **kwargs))

        return torch.cat(inputs), torch.cat(targets)

    @pytest.mark.parametrize("alpha", [-1.0, 0.0, 0.58, 1.0])
    @pytest.mark.parametrize("gamma", [0, 2])
    @pytest.mark.parametrize("device", ("xpu",))
    @pytest.mark.parametrize("dtype", [torch.float32, torch.half])
    @pytest.mark.parametrize("seed", [0, 1])
    def test_correct_ratio(self, alpha, gamma, device, dtype, seed):
        if device == "cpu" and dtype is torch.half:
            pytest.skip("Currently torch.half is not fully supported on cpu")
        # For testing the ratio with manual calculation, we require the reduction to be "none"
        reduction = "none"
        torch.random.manual_seed(seed)
        inputs, targets = self._generate_diverse_input_target_pair(dtype=dtype, device=device)
        focal_loss = ops.sigmoid_focal_loss(inputs, targets, gamma=gamma, alpha=alpha, reduction=reduction)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction=reduction)

        assert torch.all(
            focal_loss <= ce_loss
        ), "focal loss must be less or equal to cross entropy loss with same input"

        loss_ratio = (focal_loss / ce_loss).squeeze()
        prob = torch.sigmoid(inputs)
        p_t = prob * targets + (1 - prob) * (1 - targets)
        correct_ratio = (1.0 - p_t) ** gamma
        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            correct_ratio = correct_ratio * alpha_t

        tol = 1e-3 if dtype is torch.half else 1e-5
        torch.testing.assert_close(correct_ratio, loss_ratio, atol=tol, rtol=tol)

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    @pytest.mark.parametrize("device", ("xpu",))
    @pytest.mark.parametrize("dtype", [torch.float32, torch.half])
    @pytest.mark.parametrize("seed", [2, 3])
    def test_equal_ce_loss(self, reduction, device, dtype, seed):
        if device == "cpu" and dtype is torch.half:
            pytest.skip("Currently torch.half is not fully supported on cpu")
        # focal loss should be equal ce_loss if alpha=-1 and gamma=0
        alpha = -1
        gamma = 0
        torch.random.manual_seed(seed)
        inputs, targets = self._generate_diverse_input_target_pair(dtype=dtype, device=device)
        inputs_fl = inputs.clone().requires_grad_()
        targets_fl = targets.clone()
        inputs_ce = inputs.clone().requires_grad_()
        targets_ce = targets.clone()
        focal_loss = ops.sigmoid_focal_loss(inputs_fl, targets_fl, gamma=gamma, alpha=alpha, reduction=reduction)
        ce_loss = F.binary_cross_entropy_with_logits(inputs_ce, targets_ce, reduction=reduction)

        torch.testing.assert_close(focal_loss, ce_loss)

        focal_loss.backward()
        ce_loss.backward()
        torch.testing.assert_close(inputs_fl.grad, inputs_ce.grad)

    @pytest.mark.parametrize("alpha", [-1.0, 0.0, 0.58, 1.0])
    @pytest.mark.parametrize("gamma", [0, 2])
    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    @pytest.mark.parametrize("device", ("xpu",))
    @pytest.mark.parametrize("dtype", [torch.float32, torch.half])
    @pytest.mark.parametrize("seed", [4, 5])
    def test_jit(self, alpha, gamma, reduction, device, dtype, seed):
        if device == "cpu" and dtype is torch.half:
            pytest.skip("Currently torch.half is not fully supported on cpu")
        script_fn = torch.jit.script(ops.sigmoid_focal_loss)
        torch.random.manual_seed(seed)
        inputs, targets = self._generate_diverse_input_target_pair(dtype=dtype, device=device)
        focal_loss = ops.sigmoid_focal_loss(inputs, targets, gamma=gamma, alpha=alpha, reduction=reduction)
        scripted_focal_loss = script_fn(inputs, targets, gamma=gamma, alpha=alpha, reduction=reduction)

        tol = 1e-3 if dtype is torch.half else 1e-5
        torch.testing.assert_close(focal_loss, scripted_focal_loss, rtol=tol, atol=tol)

    # Raise ValueError for anonymous reduction mode
    @pytest.mark.parametrize("device", ("xpu",))
    @pytest.mark.parametrize("dtype", [torch.float32, torch.half])
    def test_reduction_mode(self, device, dtype, reduction="xyz"):
        if device == "cpu" and dtype is torch.half:
            pytest.skip("Currently torch.half is not fully supported on cpu")
        torch.random.manual_seed(0)
        inputs, targets = self._generate_diverse_input_target_pair(device=device, dtype=dtype)
        with pytest.raises(ValueError, match="Invalid"):
            ops.sigmoid_focal_loss(inputs, targets, 0.25, 2, reduction)


if __name__ == "__main__":
    pytest.main([__file__])
