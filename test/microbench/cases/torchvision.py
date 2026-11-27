# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""torchvision ops registered by torch-xpu-ops."""

from ..case import randn, State, TableBench, to_dtype, torch, torchvision, with_dtypes

NMS_CASES = [
    dict(shape=(6000, 4), iou_threshold=0.7),
]


class Nms(TableBench):
    """Box centers uniform in a 1024x1024 image, sides in [16, 256), scores
    uniform: pairwise IoU mostly below 0.5, about half the boxes survive."""

    op = "nms"
    family = "torchvision"
    expected = {"fwd": ("nms",)}
    params = with_dtypes(NMS_CASES)

    def setup(self, p, device):
        n = p["shape"][0]
        center = torch.rand(n, 2) * 1024
        side = torch.rand(n, 2) * 240 + 16
        boxes = torch.cat([center - side / 2, center + side / 2], dim=1)
        boxes = boxes.to(to_dtype(p["dtype"])).to(device)
        scores = torch.rand(n).to(to_dtype(p["dtype"])).to(device)
        return State(boxes=boxes, scores=scores, iou_threshold=p["iou_threshold"])

    def fwd(self, s):
        return torchvision.ops.nms(s.boxes, s.scores, s.iou_threshold)


ROI_ALIGN_CASES = [
    dict(
        shape=(16, 256, 304, 336),
        in_stride=(26148864, 1, 86016, 256),
        rois=9246,
        spatial_scale=0.25,
        pooled_height=7,
        pooled_width=7,
        sampling_ratio=0,
        aligned=True,
    ),
    dict(
        shape=(16, 256, 152, 168),
        in_stride=(6537216, 1, 43008, 256),
        rois=3219,
        spatial_scale=0.125,
        pooled_height=7,
        pooled_width=7,
        sampling_ratio=0,
        aligned=True,
    ),
]


class RoiAlign(TableBench):
    """rois: random batch index, boxes inside the image (feature size /
    spatial_scale), sides in [8, 256)."""

    op = "roi_align"
    family = "torchvision"
    expected = {"fwd": ("roi_align",)}
    params = with_dtypes(ROI_ALIGN_CASES)
    pr = [dict(ROI_ALIGN_CASES[0], dtype="bf16")]

    def setup(self, p, device):
        x = randn(p["shape"], p["dtype"], device, stride=p.get("in_stride"))
        n_img, _, h, w = p["shape"]
        k = p["rois"]
        img_h, img_w = h / p["spatial_scale"], w / p["spatial_scale"]
        side = torch.rand(k, 2) * 248 + 8
        x1 = torch.rand(k) * (img_w - side[:, 0])
        y1 = torch.rand(k) * (img_h - side[:, 1])
        rois = torch.stack(
            [
                torch.randint(0, n_img, (k,)).float(),
                x1,
                y1,
                x1 + side[:, 0],
                y1 + side[:, 1],
            ],
            dim=1,
        )
        return State(
            x=x,
            rois=rois.to(to_dtype(p["dtype"])).to(device),
            spatial_scale=p["spatial_scale"],
            pooled_height=p["pooled_height"],
            pooled_width=p["pooled_width"],
            sampling_ratio=p["sampling_ratio"],
            aligned=p["aligned"],
        )

    def fwd(self, s):
        return torchvision.ops.roi_align(
            s.x,
            s.rois,
            (s.pooled_height, s.pooled_width),
            s.spatial_scale,
            s.sampling_ratio,
            s.aligned,
        )
