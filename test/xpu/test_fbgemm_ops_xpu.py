# Owner(s): ["module: intel"]
import itertools
from typing import Tuple, Type, List

import hypothesis.strategies as st
import numpy as np
import numpy.typing as npt
import torch
from hypothesis import assume, given, settings, Verbosity
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import TestCase, run_tests

try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    from .xpu_test_utils import XPUPatchForImport


def generate_jagged_tensor(
    num_jagged_dim: int,
    outer_dense_size: int,
    inner_dense_size: int,
    dtype: torch.dtype,
    device: torch.device,
    fold_inner_dense: bool = False,
    # dynamo to mark the input as dynamic shape to make sure symbolic
    # shape is generated
    mark_dynamic: bool = False,
) -> Tuple[torch.Tensor, List[torch.LongTensor], npt.NDArray]:
    max_lengths = np.random.randint(low=1, high=10, size=(num_jagged_dim,))
    x_offsets: List[torch.LongTensor] = []
    num_lengths = outer_dense_size
    for d in range(num_jagged_dim):
        # Sometimes length[i] exceed max_L meaning jagged->dense will be
        # truncation vs. padding
        lengths = torch.randint(
            # PT2 specialize 0/1 dims as non-symbolic shape. So we need
            # to make it non 0/1 for testing. In real cases it'll likelyl
            # not 0/1 anyway (if so, they'll be recompiled)
            low=0 if not mark_dynamic else 1,
            high=max_lengths[d] * 2,
            # pyre-fixme[6]: For 3rd param expected `Union[List[int], Size,
            #  typing.Tuple[int, ...]]` but got `Tuple[Union[bool, float, int]]`.
            size=(num_lengths,),
            device=device,
        )
        offset = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
        if mark_dynamic:
            torch._dynamo.mark_dynamic(offset, 0)
        x_offsets.append(offset)
        num_lengths = x_offsets[-1][-1].item()

    x_values = torch.rand(
        # pyre-fixme[6]: For 1st param expected `Union[List[int], Size,
        #  typing.Tuple[int, ...]]` but got `Tensor`.
        x_offsets[-1][-1] * inner_dense_size,
        dtype=dtype,
        device=device,
    )
    if inner_dense_size != 1 or not fold_inner_dense:
        # pyre-fixme[6]: For 1st param expected `int` but got `Union[bool, float, int]`.
        x_values = x_values.reshape(x_offsets[-1][-1].item(), inner_dense_size)

    if mark_dynamic:
        for i in range(inner_dense_size):
            torch._dynamo.mark_dynamic(x_values, i)

    return x_values, x_offsets, max_lengths

def to_padded_dense(
    values: torch.Tensor,
    offsets: List[torch.LongTensor],
    max_lengths: npt.NDArray,
    padding_value: float = 0,
) -> torch.Tensor:
    outer_dense_size = len(offsets[0]) - 1
    # canonicalize by unsqueeze the last dim if the inner dense dimension
    # is 1 and folded.
    inner_dense_size = 1 if values.ndim == 1 else values.size(-1)
    dense = torch.empty(
        (outer_dense_size,) + tuple(max_lengths) + (inner_dense_size,),
        dtype=values.dtype,
        device=values.device,
    )
    for i in range(outer_dense_size):
        for jagged_coord in itertools.product(
            *(list(range(max_l)) for max_l in max_lengths)
        ):
            cur_offset = i
            is_zero = False
            for d in range(len(max_lengths)):
                # pyre-fixme[6]: For 1st argument expected `Union[None, _NestedSe...
                begin = offsets[d][cur_offset].item()
                # pyre-fixme[6]: For 1st argument expected `Union[None, _NestedSe...
                end = offsets[d][cur_offset + 1].item()
                # pyre-fixme[6]: For 1st param expected `int` but got
                #  `Union[bool, float, int]`.
                if jagged_coord[d] >= end - begin:
                    is_zero = True
                    break
                cur_offset = begin + jagged_coord[d]
            dense[(i,) + jagged_coord] = (
                padding_value
                if is_zero
                # pyre-fixme[6]: For 1st argument expected `Union[None, _NestedSe...
                else values[cur_offset]
            )
    return dense.squeeze(-1) if values.ndim == 1 else dense


with XPUPatchForImport(False):

    class CumSumTest(TestCase):
        @given(
            n=st.integers(min_value=0, max_value=10),
            index_types=st.sampled_from(
                [
                    (torch.int64, np.int64),
                    (torch.int32, np.int32),
                    (torch.float32, np.float32),
                ]
            ),
            device=st.just(torch.device("xpu")),
        )
        @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
        def test_cumsum(
            self,
            n: int,
            index_types: Tuple[Type[object], Type[object]],
            device: torch.device,
        ) -> None:
            (pt_index_dtype, np_index_dtype) = index_types

            # The CPU variants of asynchronous_*_cumsum support floats, since some
            # downstream tests appear to be relying on this behavior.  As such, the
            # test is disabled for GPU + float test cases.
            if device == torch.device("xpu") and pt_index_dtype is torch.float32:
                return

            # pyre-ignore-errors[16]
            x = torch.randint(low=0, high=100, size=(n,)).type(pt_index_dtype).to(device)
            zc = torch.ops.fbgemm.asynchronous_complete_cumsum(x)

            torch.testing.assert_close(
                torch.from_numpy(
                    (np.cumsum([0] + x.cpu().numpy().tolist())).astype(np_index_dtype)
                ),
                zc.cpu(),
            )


    class DenseToJaggedTest(TestCase):
        def _test_dense_to_jagged(
            self,
            num_jagged_dim: int,
            outer_dense_size: int,
            inner_dense_size: int,
            dtype: torch.dtype,
            device: torch.device,
            precompute_total_L: bool,
        ) -> None:

            # import debugpy
            # debugpy.listen(("0.0.0.0", 5678))
            # debugpy.wait_for_client()

            # Generate multi-dim jagged tensor
            values_2d, offsets, max_lengths = generate_jagged_tensor(
                num_jagged_dim, outer_dense_size, inner_dense_size, dtype, device
            )
            # values_2d = values_2d.clone().detach().requires_grad_(True)

            # jagged -> dense
            dense = torch.ops.fbgemm.jagged_to_padded_dense(values_2d, offsets, max_lengths)

            # dense -> jagged (op which is being tested)
            if precompute_total_L:
                total_L = values_2d.size(0)
                jagged_values, jagged_offsets = torch.ops.fbgemm.dense_to_jagged(
                    dense, offsets, total_L
                )
                jagged_values_f = torch.ops.fbgemm.dense_to_jagged_forward(
                    dense, offsets, total_L
                )
                torch.testing.assert_close(jagged_values, jagged_values_f)
            else:
                jagged_values, jagged_offsets = torch.ops.fbgemm.dense_to_jagged(
                    dense, offsets
                )
                jagged_values_f = torch.ops.fbgemm.dense_to_jagged_forward(
                    dense, offsets
                )
                torch.testing.assert_close(jagged_values, jagged_values_f)

            # jagged -> dense
            dense2 = torch.ops.fbgemm.jagged_to_padded_dense(
                jagged_values, jagged_offsets, max_lengths
            )

            # verify forward
            torch.testing.assert_close(dense, dense2)

            # verify backward

        @given(
            num_jagged_dim=st.integers(1, 5),
            outer_dense_size=st.integers(0, 5),
            inner_dense_size=st.integers(0, 5),
            # num_jagged_dim=st.integers(4, 5),
            # outer_dense_size=st.integers(4, 5),
            # inner_dense_size=st.integers(4, 5),
            dtype=st.sampled_from([torch.float, torch.half, torch.bfloat16]),
            device=st.just(torch.device("xpu")),
            precompute_total_L=st.booleans(),
        )
        @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
        def test_dense_to_jagged(
            self,
            num_jagged_dim: int,
            outer_dense_size: int,
            inner_dense_size: int,
            dtype: torch.dtype,
            device: torch.device,
            precompute_total_L: bool,
        ) -> None:
            self._test_dense_to_jagged(
                num_jagged_dim,
                outer_dense_size,
                inner_dense_size,
                dtype,
                device,
                precompute_total_L,
            )

        @given(
            num_jagged_dim=st.just(1),
            outer_dense_size=st.integers(0, 6000),
            inner_dense_size=st.sampled_from([8, 16, 23, 24, 48, 50, 64, 72, 96, 192]),
            dtype=st.just(torch.half),
            device=st.just(torch.device("xpu")),
            precompute_total_L=st.booleans(),
        )
        @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
        def test_dense_to_jagged_opt(
            self,
            num_jagged_dim: int,
            outer_dense_size: int,
            inner_dense_size: int,
            dtype: torch.dtype,
            device: torch.device,
            precompute_total_L: bool,
        ) -> None:
            self._test_dense_to_jagged(
                num_jagged_dim,
                outer_dense_size,
                inner_dense_size,
                dtype,
                device,
                precompute_total_L,
            )

        # (8000+1) * 8 (size of the element of LongTensor/int64_t offsets)
        # = ~62.5KB > 48KB default shared memory on V100/A100.
        @given(
            num_jagged_dim=st.just(1),
            outer_dense_size=st.just(8000),
            inner_dense_size=st.just(16),
            dtype=st.just(torch.half),
            device=st.just(torch.device("xpu")),
            precompute_total_L=st.booleans(),
        )
        @settings(verbosity=Verbosity.verbose, max_examples=1, deadline=None)
        def test_dense_to_jagged_opt_large_batch(
            self,
            num_jagged_dim: int,
            outer_dense_size: int,
            inner_dense_size: int,
            dtype: torch.dtype,
            device: torch.device,
            precompute_total_L: bool,
        ) -> None:
            self._test_dense_to_jagged(
                num_jagged_dim,
                outer_dense_size,
                inner_dense_size,
                dtype,
                device,
                precompute_total_L,
            )

        @given(
            num_jagged_dim=st.integers(1, 5),
            # TODO: size = 0/1 will be incorrectly specialized
            outer_dense_size=st.integers(2, 5),
            inner_dense_size=st.integers(2, 5),
            dtype=st.sampled_from([torch.float, torch.half, torch.bfloat16]),
            device=st.just(torch.device("xpu")),
        )
        @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
        def test_dense_to_jagged_dynamic_shape(
            self,
            num_jagged_dim: int,
            outer_dense_size: int,
            inner_dense_size: int,
            dtype: torch.dtype,
            device: torch.device,
        ) -> None:
            # Start a fresh compile for each parameter of the test case
            torch._dynamo.reset()

            values_2d, offsets, max_lengths = generate_jagged_tensor(
                num_jagged_dim,
                outer_dense_size,
                inner_dense_size,
                dtype,
                device,
                mark_dynamic=True,
            )

            def jagged_to_dense(
                values: torch.Tensor,
                offsets: List[torch.LongTensor],
                max_lengths: List[int],
            ) -> torch.Tensor:
                return torch.ops.fbgemm.jagged_to_padded_dense(values, offsets, max_lengths)

            # jagged -> dense
            dense = jagged_to_dense(values_2d, offsets, max_lengths.tolist())

            # dense -> jagged, it is required to pre-compute totalL
            total_L = values_2d.size(0)
            dense = dense.clone().detach().to(device)

            torch._dynamo.mark_dynamic(dense, 0)
            torch._dynamo.mark_dynamic(dense, -1)

            def dense_to_jagged_withL(
                dense: torch.Tensor, offsets: List[torch.LongTensor], total_L: List[int]
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                jagged_values, jagged_offsets = torch.ops.fbgemm.dense_to_jagged(dense, offsets, total_L)
                jagged_values_f = torch.ops.fbgemm.dense_to_jagged_forward(dense, offsets, total_L)
                torch.testing.assert_close(jagged_values, jagged_values_f)
                return jagged_values, jagged_offsets

            def dense_to_jagged_noL(
                dense: torch.Tensor, offsets: List[torch.LongTensor]
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                jagged_values, jagged_offsets = torch.ops.fbgemm.dense_to_jagged(dense, offsets)
                jagged_values_f = torch.ops.fbgemm.dense_to_jagged_forward(dense, offsets)
                torch.testing.assert_close(jagged_values, jagged_values_f)
                return jagged_values, jagged_offsets

            jagged_values, jagged_offsets = dense_to_jagged_noL(dense, offsets)
            jagged_values, jagged_offsets = dense_to_jagged_withL(dense, offsets, total_L)

            jagged_values.to(device)
            # jagged -> dense
            dense2 = torch.ops.fbgemm.jagged_to_padded_dense(
                jagged_values, jagged_offsets, max_lengths
            )

            # verify forward
            assert dense.size() == dense2.size()


    class JaggedToPaddedDenseTest(TestCase):
        @given(
            num_jagged_dim=st.integers(1, 5),
            outer_dense_size=st.integers(0, 5),
            inner_dense_size=st.integers(0, 5),
            fold_inner_dense=st.booleans(),
            padding_value=st.sampled_from([0, -1e-8]),
            dtype=st.sampled_from([torch.float, torch.half, torch.bfloat16]),
            device_type=st.just("xpu"),
        )
        @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
        def test_jagged_to_padded_dense(
            self,
            num_jagged_dim: int,
            outer_dense_size: int,
            inner_dense_size: int,
            fold_inner_dense: bool,
            padding_value: float,
            dtype: torch.dtype,
            device_type: str,
        ) -> None:
            # CPU doesn't support bfloat16
            assume(device_type != "cpu" or dtype != torch.bfloat16)
            assume(not fold_inner_dense or inner_dense_size == 1)

            # Testing with a basic crafted example.
            # dense representation is
            # [[[[0, 1], [ 0,  0], [0, 0]],
            #   [[2, 3], [ 4,  5], [6, 7]],
            #   [[0, 0], [ 0,  0], [0, 0]],
            #   [[0, 0], [ 0,  0], [0, 0]]],
            #  [[[0, 0], [ 0,  0], [0, 0]],
            #   [[0, 0], [ 0,  0], [0, 0]],
            #   [[0, 0], [ 0,  0], [0, 0]],
            #   [[0, 0], [ 0,  0], [0, 0]]],
            #  [[[8, 9], [10, 11], [0, 0]],
            #   [[0, 0], [ 0,  0], [0, 0]],
            #   [[0, 0], [ 0,  0], [0, 0]],
            #   [[0, 0], [ 0,  0], [0, 0]]]],
            # inner_dense_size = 2
            # x_offsets = [
            #     torch.LongTensor([0, 2, 2, 3]),  # lengths torch.Tensor([2, 0, 1]),
            #     torch.LongTensor([0, 1, 4, 6]),  # lengths torch.Tensor([1, 3, 2]),
            # ]
            # outer_dense_size = len(x_offsets[0]) - 1
            # max_lengths = [4, 3]

            device = torch.device(device_type)

            x_values, x_offsets, max_lengths = generate_jagged_tensor(
                num_jagged_dim,
                outer_dense_size,
                inner_dense_size,
                torch.float,
                device,
                fold_inner_dense,
            )

            output_ref = to_padded_dense(
                x_values, x_offsets, max_lengths, padding_value=padding_value
            )
            output = torch.ops.fbgemm.jagged_to_padded_dense(
                x_values,
                x_offsets,
                max_lengths,
                padding_value=padding_value,
            )

            output_f = torch.ops.fbgemm.jagged_to_padded_dense_forward(
                x_values,
                x_offsets,
                max_lengths,
                padding_value=padding_value,
            )

            torch.testing.assert_close(output, output_ref)
            torch.testing.assert_close(output_f, output_ref)

instantiate_device_type_tests(CumSumTest, globals(), only_for="xpu", allow_xpu=True)

instantiate_device_type_tests(DenseToJaggedTest, globals(), only_for="xpu", allow_xpu=True)

instantiate_device_type_tests(JaggedToPaddedDenseTest, globals(), only_for="xpu", allow_xpu=True)

if __name__ == "__main__":
    run_tests()
