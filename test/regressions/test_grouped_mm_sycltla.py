# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
"""Numerical tests of the SYCL-TLA grouped GEMM behind torch._grouped_mm.

PyTorch selects the kernel with XPU_GROUP_GEMM_BACKEND (read on every call).
Every test forces it to ``sycltla`` and turns on per-call dispatch logging
(TORCH_XPU_GROUPED_MM_LOG=2), then checks from the log that each supported
call (forward, grad_input, grad_weight) really ran the SYCL-TLA kernel, with
the forced tile and grid, and compares all three against a float32 per-group
reference.

If the PyTorch build does not route _grouped_mm to SYCL-TLA at all (no
companion dispatcher, built without SYCL-TLA, unsupported device), the class
is skipped. Set TORCH_XPU_REQUIRE_SYCLTLA_GROUPED_MM=1 to make that a failure.
"""

import contextlib
import math
import os
import tempfile
import unittest

import torch
from torch.testing._internal.common_utils import run_tests, TestCase

device = "xpu"

# TORCH_XPU_GROUPED_MM_TILE value -> (TileM, TileN, name in the dispatch log).
TILES = {
    1: (256, 128, "256x128"),
    2: (256, 256, "256x256"),
    3: (128, 128, "128x128"),
    4: (128, 256, "128x256"),
}

KNOBS = (
    "XPU_GROUP_GEMM_BACKEND",
    "TORCH_XPU_GROUPED_MM_LOG",
    "TORCH_XPU_GROUPED_MM_TILE",
    "TORCH_XPU_GROUPED_MM_NUM_WORKGROUPS",
)


@contextlib.contextmanager
def env(**values):
    old = {k: os.environ.get(k) for k in values}
    try:
        for k, v in values.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def capture_dispatch_log(fn):
    """Runs fn() and returns (result, [dispatch log lines]).

    The log is written by C++ to file descriptor 2, so redirect the fd itself.
    """
    with tempfile.TemporaryFile(mode="w+b") as tmp:
        saved = os.dup(2)
        try:
            os.dup2(tmp.fileno(), 2)
            result = fn()
            torch.xpu.synchronize()
        finally:
            os.dup2(saved, 2)
            os.close(saved)
        tmp.seek(0)
        text = tmp.read().decode(errors="replace")
    lines = [line for line in text.splitlines() if line.startswith("[xpu grouped_mm]")]
    return result, lines


def sycltla_lines(lines):
    return [line for line in lines if "SYCL-TLA" in line]


def make_offs(rows):
    return torch.tensor(rows, dtype=torch.int64).cumsum(0).to(torch.int32).to(device)


def make_weight(E, K, N, dtype, b_layout):
    """Returns (leaf weight, mat_b view [E, K, N]) for the requested layout."""
    if b_layout == "N-contig":  # row-major [K, N] per group
        w = torch.randn(E, K, N, device=device, dtype=dtype)
        w.requires_grad_(True)
        return w, w
    # K-contig: transposed [N, K] weight, as nn.Linear-style MoE experts use.
    w = torch.randn(E, N, K, device=device, dtype=dtype)
    w.requires_grad_(True)
    return w, w.transpose(-2, -1)


def reference(x, b, go, rows):
    """float32 per-group out, grad_x, grad_b for out = grouped_mm(x, b)."""
    xf, bf, gof = x.float(), b.float(), go.float()
    out = torch.zeros(x.shape[0], b.shape[-1], device=device)
    gx = torch.zeros_like(xf)
    gb = torch.zeros_like(bf)
    start = 0
    for g, r in enumerate(rows):
        s = slice(start, start + r)
        out[s] = xf[s] @ bf[g]
        gx[s] = gof[s] @ bf[g].t()
        gb[g] = xf[s].t() @ gof[s]
        start += r
    return out, gx, gb


def tolerances(dtype, ref):
    scale = max(ref.abs().max().item(), 1.0) if ref.numel() else 1.0
    if dtype == torch.bfloat16:
        return dict(rtol=2e-2, atol=1e-2 * scale)
    return dict(rtol=4e-3, atol=2e-3 * scale)


class TestGroupedMMSyclTla(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not torch.xpu.is_available():
            raise unittest.SkipTest("XPU not available")
        cls._saved_env = {k: os.environ.get(k) for k in KNOBS}
        os.environ["XPU_GROUP_GEMM_BACKEND"] = "sycltla"
        os.environ["TORCH_XPU_GROUPED_MM_LOG"] = "2"
        os.environ.pop("TORCH_XPU_GROUPED_MM_TILE", None)
        os.environ.pop("TORCH_XPU_GROUPED_MM_NUM_WORKGROUPS", None)
        # Probe: one plainly supported 2D x 3D call.
        x = torch.randn(64, 64, device=device, dtype=torch.bfloat16)
        w = torch.randn(2, 64, 64, device=device, dtype=torch.bfloat16)
        offs = make_offs([32, 32])
        _, lines = capture_dispatch_log(lambda: torch._grouped_mm(x, w, offs=offs))
        if not sycltla_lines(lines):
            cls._restore_env()
            msg = (
                "XPU_GROUP_GEMM_BACKEND=sycltla does not reach the SYCL-TLA "
                "grouped GEMM in this PyTorch build/device"
            )
            if os.environ.get("TORCH_XPU_REQUIRE_SYCLTLA_GROUPED_MM") == "1":
                raise RuntimeError(msg)
            raise unittest.SkipTest(msg)

    @classmethod
    def _restore_env(cls):
        for k, v in cls._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    @classmethod
    def tearDownClass(cls):
        cls._restore_env()
        super().tearDownClass()

    def _check(
        self,
        dtype,
        b_layout,
        rows,
        K,
        N,
        tile=None,
        num_workgroups=None,
    ):
        torch.manual_seed(0)
        E = len(rows)
        x = torch.randn(sum(rows), K, device=device, dtype=dtype, requires_grad=True)
        w, b = make_weight(E, K, N, dtype, b_layout)
        offs = make_offs(rows)
        go = torch.randn(sum(rows), N, device=device, dtype=dtype)

        def run():
            out = torch._grouped_mm(x, b, offs=offs)
            out.backward(go)
            return out

        with env(
            TORCH_XPU_GROUPED_MM_TILE=tile,
            TORCH_XPU_GROUPED_MM_NUM_WORKGROUPS=num_workgroups,
        ):
            out, lines = capture_dispatch_log(run)

        what = (
            f"{dtype} B={b_layout} rows={rows} K={K} N={N} "
            f"tile={tile} wgs={num_workgroups}"
        )
        # Forward and grad_input are 2D x 3D, grad_weight is 2D x 2D; all
        # three must have been served by SYCL-TLA.
        hits = sycltla_lines(lines)
        self.assertEqual(len(hits), 3, f"{what}: dispatch log {lines}")
        self.assertEqual(sum("m-grouped" in h for h in hits), 2, f"{what}: {hits}")
        self.assertEqual(sum("k-grouped" in h for h in hits), 1, f"{what}: {hits}")
        if tile is not None:
            for h in hits:
                self.assertIn(f"tile={TILES[tile][2]}", h, what)
        if num_workgroups is not None:
            for h in hits:
                self.assertIn(f"wgs={num_workgroups}", h, what)

        ref_out, ref_gx, ref_gb = reference(x.detach(), b.detach(), go, rows)
        gb = w.grad if b_layout == "N-contig" else w.grad.transpose(-2, -1)
        for name, got, ref in (
            ("out", out, ref_out),
            ("grad_input", x.grad, ref_gx),
            ("grad_weight", gb, ref_gb),
        ):
            torch.testing.assert_close(
                got.float(),
                ref,
                **tolerances(dtype, ref),
                msg=lambda m, name=name: f"{what} {name}: {m}",
            )

    def test_layouts_and_ragged_groups(self):
        cases = [
            ([300, 0, 517, 207], 256, 384),  # ragged, an empty group, ragged N
            ([40, 8, 1], 64, 128),  # groups smaller than a tile: idle sub-groups
            ([128] * 8, 512, 256),  # uniform
            ([0, 1024, 0], 128, 256),  # one group holds every row
            ([0, 0, 0, 0], 256, 256),  # all groups empty: zero-fill, no launch
        ]
        for dtype in (torch.bfloat16, torch.float16):
            for b_layout in ("N-contig", "K-contig"):
                for rows, K, N in cases:
                    with self.subTest(dtype=dtype, b=b_layout, rows=rows):
                        self._check(dtype, b_layout, rows, K, N)

    def test_forced_tiles_persistent_grid(self):
        # Grids far smaller than the tile count: every work-group walks several
        # tiles across group boundaries (the persistent loop and its prefetch
        # of the next tile), including groups with fewer rows than TileM.
        rows = [300, 0, 517, 40, 1, 207]
        for tile in TILES:
            for num_workgroups in (1, 3, 7):
                for b_layout in ("N-contig", "K-contig"):
                    with self.subTest(tile=tile, wgs=num_workgroups, b=b_layout):
                        self._check(
                            torch.bfloat16,
                            b_layout,
                            rows,
                            256,
                            384,
                            tile=tile,
                            num_workgroups=num_workgroups,
                        )

    def test_forced_tiles_wave_tail_split(self):
        # 2D x 3D wave-tail balancing: with T tiles on G work-groups and
        # r = T % G > 0, the last partial wave is split 4 ways if 4r <= G,
        # else 2 ways if 2r <= G. Build T = 8 and T = 9 tiles of one N-tile
        # (the last group ragged in M) and pick G to hit each split.
        for tile, (tm, tn, _) in TILES.items():
            for groups, num_workgroups, split in ((8, 6, 2), (9, 8, 4), (9, 4, 4)):
                rows = [tm] * (groups - 1) + [tm // 2 + 3]
                total = sum(math.ceil(r / tm) for r in rows)  # N == TileN
                rem = total % num_workgroups
                self.assertTrue(rem > 0)
                self.assertEqual(
                    split, 4 if 4 * rem <= num_workgroups else 2, (tile, groups)
                )
                for dtype in (torch.bfloat16, torch.float16):
                    with self.subTest(tile=tile, groups=groups, wgs=num_workgroups):
                        self._check(
                            dtype,
                            "K-contig",
                            rows,
                            128,
                            tn,
                            tile=tile,
                            num_workgroups=num_workgroups,
                        )

    def _forward_with_row_pitch(self, ld):
        # A is [2, 32] bf16 whose rows are `ld` elements apart.
        torch.manual_seed(0)
        storage = torch.randn(ld + 32, device=device, dtype=torch.bfloat16)
        a = storage.as_strided((2, 32), (ld, 1))
        b = torch.randn(2, 32, 64, device=device, dtype=torch.bfloat16)
        offs = make_offs([1, 1])
        out, lines = capture_dispatch_log(lambda: torch._grouped_mm(a, b, offs=offs))
        ref = torch.cat([a[0:1].float() @ b[0].float(), a[1:2].float() @ b[1].float()])
        torch.testing.assert_close(out.float(), ref, **tolerances(torch.bfloat16, ref))
        return lines

    def test_row_pitch_at_2d_block_limit(self):
        # Largest 64-byte-aligned pitch within the 24-bit byte pitch field.
        ld = ((2**24 - 1) // 2) // 32 * 32
        lines = self._forward_with_row_pitch(ld)
        self.assertEqual(len(sycltla_lines(lines)), 1, lines)

    def test_row_pitch_beyond_2d_block_limit_falls_back(self):
        # A 32 MiB row pitch does not fit an Xe 2D block surface: the call
        # must not reach SYCL-TLA and must still be correct (per-group path).
        lines = self._forward_with_row_pitch(2**24)
        self.assertEqual(sycltla_lines(lines), [], lines)


if __name__ == "__main__":
    run_tests()
