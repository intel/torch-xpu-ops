# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]

from functools import partial
from itertools import chain, combinations, permutations, product

import torch
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_dtype import all_types_and_complex_and
from torch.testing._internal.common_utils import run_tests

try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    from .xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_shape_ops import TestShapeOps

    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def _test_flip(self, device, dtype):
        make_from_data = partial(torch.tensor, device=device, dtype=dtype)
        make_from_size = partial(make_tensor, device=device, dtype=dtype)

        def test_flip_impl(input_t, dims, output_t):
            def all_t():
                yield input_t, output_t

            for in_t, out_t in all_t():
                self.assertEqual(in_t.flip(dims), out_t)
                n = in_t.ndim
                if not isinstance(dims, tuple):
                    # Wrap dim
                    self.assertEqual(in_t.flip(-n + dims), out_t)
                else:
                    # Permute dimensions
                    for p_dims in permutations(dims):
                        self.assertEqual(in_t.flip(p_dims), out_t)
                        if len(p_dims) > 0:
                            # Wrap 1st dim
                            self.assertEqual(
                                in_t.flip((-n + p_dims[0],) + p_dims[1:]), out_t
                            )

        def gen_data():
            # Basic tests
            data = make_from_data([1, 2, 3, 4, 5, 6, 7, 8]).view(2, 2, 2)
            nonctg = make_from_size((2, 2, 2), noncontiguous=True).copy_(data)

            dims_result = (
                (0, make_from_data([5, 6, 7, 8, 1, 2, 3, 4]).view(2, 2, 2)),
                (1, make_from_data([3, 4, 1, 2, 7, 8, 5, 6]).view(2, 2, 2)),
                (2, make_from_data([2, 1, 4, 3, 6, 5, 8, 7]).view(2, 2, 2)),
                ((0, 1), make_from_data([7, 8, 5, 6, 3, 4, 1, 2]).view(2, 2, 2)),
                ((0, 1, 2), make_from_data([8, 7, 6, 5, 4, 3, 2, 1]).view(2, 2, 2)),
            )
            for in_tensor, (dims, out_tensor) in product((data, nonctg), dims_result):
                yield in_tensor, dims, out_tensor

            # Expanded
            in_t = make_from_data([1, 2, 3]).view(3, 1).expand(3, 2)
            dims = 0
            out_t = make_from_data([3, 3, 2, 2, 1, 1]).view(3, 2)
            yield in_t, dims, out_t
            # Noop on expanded dimension
            yield in_t, 1, in_t

            # Transposed
            in_t = (
                make_from_data([1, 2, 3, 4, 5, 6, 7, 8]).view(2, 2, 2).transpose(0, 1)
            )
            dims = (0, 1, 2)
            out_t = make_from_data([8, 7, 4, 3, 6, 5, 2, 1]).view(2, 2, 2)
            yield in_t, dims, out_t

            # Rectangular case
            in_t = make_from_data([1, 2, 3, 4, 5, 6]).view(2, 3)
            dims = 0
            out_t = make_from_data([[4, 5, 6], [1, 2, 3]])
            yield in_t, dims, out_t
            dims = 1
            out_t = make_from_data([[3, 2, 1], [6, 5, 4]])
            yield in_t, dims, out_t

            # vectorized NCHW cases (images)
            if device == "cpu" and dtype != torch.bfloat16:
                for mf in [torch.contiguous_format, torch.channels_last]:
                    for c in [2, 3, 8, 16]:
                        in_t = make_from_size((2, c, 32, 32)).contiguous(
                            memory_format=mf
                        )
                        np_in_t = in_t.numpy()

                        np_out_t = np_in_t[:, :, :, ::-1].copy()
                        out_t = torch.from_numpy(np_out_t)
                        yield in_t, 3, out_t

                        np_out_t = np_in_t[:, :, ::-1, :].copy()
                        out_t = torch.from_numpy(np_out_t)
                        yield in_t, 2, out_t

                        # non-contig cases
                        in_tt = in_t[..., ::2, :]
                        np_in_t = in_tt.numpy()
                        np_out_t = np_in_t[:, :, :, ::-1].copy()
                        out_t = torch.from_numpy(np_out_t)
                        yield in_tt, 3, out_t

                        in_tt = in_t[..., ::2]
                        np_in_t = in_tt.numpy()
                        np_out_t = np_in_t[:, :, :, ::-1].copy()
                        out_t = torch.from_numpy(np_out_t)
                        yield in_tt, 3, out_t

            # Noops (edge cases)

            # Size 0
            in_t = make_from_data(())
            yield in_t, 0, in_t
            yield in_t, (), in_t

            # dims = ()
            in_t = make_from_size((3, 2, 1))
            yield in_t, (), in_t

            # Zero elements, non-zero size
            in_t = make_from_size((3, 0, 2))
            for i in range(in_t.ndim):
                yield in_t, i, in_t

            # Size 1
            in_t = make_from_size(())
            yield in_t, 0, in_t
            in_t = make_from_size((1,))
            yield in_t, 0, in_t

        for in_tensor, dims, out_tensor in gen_data():
            test_flip_impl(in_tensor, dims, out_tensor)

        # test for shape
        size = [2, 3, 4]
        data = make_from_size(size)
        possible_dims = range(len(size))
        test_dims = chain(
            combinations(possible_dims, 1), combinations(possible_dims, 2)
        )

        for dims in test_dims:
            self.assertEqual(size, list(data.flip(dims).size()))

    TestShapeOps.test_flip = _test_flip

instantiate_device_type_tests(TestShapeOps, globals(), only_for="xpu", allow_xpu=True)


if __name__ == "__main__":
    run_tests()
