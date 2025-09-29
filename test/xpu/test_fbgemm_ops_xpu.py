# Owner(s): ["module: intel"]
import itertools
import random
from itertools import accumulate
from typing import List, Optional, Tuple, Type

import hypothesis.strategies as st
import numpy as np
import numpy.typing as npt
import torch
from hypothesis import assume, given, settings, Verbosity
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase

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


def permute_indices_ref_(
    lengths: torch.Tensor,
    indices: torch.Tensor,
    weights: Optional[torch.Tensor],
    permute: torch.LongTensor,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    T = lengths.size(0)
    B = lengths.size(1)
    if T == 0 or B == 0:
        return lengths, indices, weights

    permuted_lengths = torch.index_select(lengths.view(T, -1), 0, permute)
    original_segment_lengths = lengths.view(T, -1).sum(dim=1, dtype=torch.int32)
    original_segment_start = [0] + list(accumulate(original_segment_lengths.view(-1)))

    permuted_indices = []
    permuted_weights = []
    for i in range(permute.size(0)):
        start = original_segment_start[permute[i]]
        end = start + original_segment_lengths[permute[i]]
        permuted_indices.append(indices[start:end])
        if weights is not None:
            permuted_weights.append(weights[start:end])

    permuted_indices = torch.cat(permuted_indices, dim=0).flatten()

    if weights is None:
        permuted_weights = None
    else:
        permuted_weights = torch.cat(permuted_weights, dim=0).flatten()

    return permuted_lengths, permuted_indices, permuted_weights


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
            x = (
                torch.randint(low=0, high=100, size=(n,))
                .type(pt_index_dtype)
                .to(device)
            )
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
            # Generate multi-dim jagged tensor
            values_2d, offsets, max_lengths = generate_jagged_tensor(
                num_jagged_dim, outer_dense_size, inner_dense_size, dtype, device
            )
            # values_2d = values_2d.clone().detach().requires_grad_(True)

            # jagged -> dense
            dense = torch.ops.fbgemm.jagged_to_padded_dense(
                values_2d, offsets, max_lengths
            )

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
                return torch.ops.fbgemm.jagged_to_padded_dense(
                    values, offsets, max_lengths
                )

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
                jagged_values, jagged_offsets = torch.ops.fbgemm.dense_to_jagged(
                    dense, offsets, total_L
                )
                jagged_values_f = torch.ops.fbgemm.dense_to_jagged_forward(
                    dense, offsets, total_L
                )
                torch.testing.assert_close(jagged_values, jagged_values_f)
                return jagged_values, jagged_offsets

            def dense_to_jagged_noL(
                dense: torch.Tensor, offsets: List[torch.LongTensor]
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                jagged_values, jagged_offsets = torch.ops.fbgemm.dense_to_jagged(
                    dense, offsets
                )
                jagged_values_f = torch.ops.fbgemm.dense_to_jagged_forward(
                    dense, offsets
                )
                torch.testing.assert_close(jagged_values, jagged_values_f)
                return jagged_values, jagged_offsets

            jagged_values, jagged_offsets = dense_to_jagged_noL(dense, offsets)
            jagged_values, jagged_offsets = dense_to_jagged_withL(
                dense, offsets, total_L
            )

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

    class ElementwiseBinaryTest(TestCase):
        def _test_jagged_elementwise_binary(
            self,
            num_jagged_dim: int,
            outer_dense_size: int,
            inner_dense_size: int,
            operation: str,
            dtype: torch.dtype,
            device: torch.device,
        ) -> None:
            x_values, x_offsets, max_lengths = generate_jagged_tensor(
                num_jagged_dim, outer_dense_size, inner_dense_size, dtype, device
            )
            y = torch.rand(
                outer_dense_size * np.prod(max_lengths) * inner_dense_size,
                dtype=dtype,
                device=device,
            ).reshape((outer_dense_size,) + tuple(max_lengths) + (inner_dense_size,))

            x_padded = to_padded_dense(x_values, x_offsets, max_lengths)

            assert operation == "add_jagged_output"
            # create a jagged tensor and then densify
            y = to_padded_dense(
                torch.rand(
                    (
                        max(outer_dense_size * np.prod(max_lengths), x_values.size(0)),
                        inner_dense_size,
                    ),
                    dtype=dtype,
                    device=device,
                ),
                x_offsets,
                max_lengths,
            )
            output_ref = x_padded + y
            (
                output,
                output_offsets,
            ) = torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output(
                x_values, x_offsets, y
            )
            output = to_padded_dense(output, output_offsets, max_lengths)

            torch.testing.assert_close(output, output_ref)

        @given(
            num_jagged_dim=st.integers(1, 4),
            outer_dense_size=st.integers(0, 4),
            inner_dense_size=st.integers(0, 4),
            operation=st.just("add_jagged_output"),
            dtype=st.sampled_from([torch.float, torch.half, torch.bfloat16]),
            device=st.just(torch.device("xpu")),
        )
        @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
        def test_jagged_elementwise_binary(
            self,
            num_jagged_dim: int,
            outer_dense_size: int,
            inner_dense_size: int,
            operation: str,
            dtype: torch.dtype,
            device: torch.device,
        ) -> None:
            self._test_jagged_elementwise_binary(
                num_jagged_dim,
                outer_dense_size,
                inner_dense_size,
                operation,
                dtype,
                device,
            )

        @given(
            num_jagged_dim=st.just(1),
            outer_dense_size=st.integers(0, 8),
            inner_dense_size=st.sampled_from([16, 64, 96, 192]),
            operation=st.just("add_jagged_output"),
            dtype=st.just(torch.half),
            device=st.just(torch.device("xpu")),
        )
        @settings(verbosity=Verbosity.verbose, max_examples=4, deadline=None)
        def test_jagged_elementwise_binary_opt(
            self,
            num_jagged_dim: int,
            outer_dense_size: int,
            inner_dense_size: int,
            operation: str,
            dtype: torch.dtype,
            device: torch.device,
        ) -> None:
            self._test_jagged_elementwise_binary(
                num_jagged_dim,
                outer_dense_size,
                inner_dense_size,
                operation,
                dtype,
                device,
            )

        @given(
            num_jagged_dim=st.integers(1, 5),
            outer_dense_size=st.integers(2, 5),
            inner_dense_size=st.integers(2, 5),
            operation=st.just("add_jagged_output"),
            dtype=st.sampled_from([torch.float, torch.half, torch.bfloat16]),
            device=st.just(torch.device("xpu")),
        )
        @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
        def test_jagged_elementwise_binary_dynamic_shape(
            self,
            num_jagged_dim: int,
            outer_dense_size: int,
            inner_dense_size: int,
            operation: str,
            dtype: torch.dtype,
            device: torch.device,
        ) -> None:
            # Start a fresh compile for each parameter of the test case
            torch._dynamo.reset()

            x_values, x_offsets, max_lengths = generate_jagged_tensor(
                num_jagged_dim,
                outer_dense_size,
                inner_dense_size,
                dtype,
                device,
                mark_dynamic=True,
            )
            y = torch.rand(
                outer_dense_size * np.prod(max_lengths) * inner_dense_size,
                dtype=dtype,
                device=device,
            ).reshape((outer_dense_size,) + tuple(max_lengths) + (inner_dense_size,))

            x_padded = to_padded_dense(x_values, x_offsets, max_lengths)

            def jagged_dense_elementwise_add(
                x_values: torch.Tensor,
                x_offsets: List[torch.LongTensor],
                y: torch.Tensor,
            ) -> torch.Tensor:
                return torch.ops.fbgemm.jagged_dense_elementwise_add(
                    x_values, x_offsets, y
                )

            def jagged_dense_elementwise_add_jagged_output(
                x_values: torch.Tensor,
                x_offsets: List[torch.LongTensor],
                y: torch.Tensor,
            ) -> Tuple[torch.Tensor, List[torch.LongTensor]]:
                return torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output(
                    x_values, x_offsets, y
                )

            def jagged_dense_elementwise_mul(
                x_values: torch.Tensor,
                x_offsets: List[torch.LongTensor],
                y: torch.Tensor,
            ) -> Tuple[torch.Tensor, List[torch.LongTensor]]:
                return torch.ops.fbgemm.jagged_dense_elementwise_mul(
                    x_values, x_offsets, y
                )

            assert operation == "add_jagged_output"
            # create a jagged tensor and then densify
            y = to_padded_dense(
                torch.rand(
                    (
                        max(outer_dense_size * np.prod(max_lengths), x_values.size(0)),
                        inner_dense_size,
                    ),
                    dtype=dtype,
                    device=device,
                ),
                x_offsets,
                max_lengths,
            )
            output_ref = x_padded + y
            (
                output,
                output_offsets,
            ) = jagged_dense_elementwise_add_jagged_output(x_values, x_offsets, y)
            output = to_padded_dense(output, output_offsets, max_lengths)

            assert output.size() == output_ref.size()

    class ReorderBatchedTest(TestCase):
        @given(
            B=st.integers(min_value=1, max_value=20),
            T=st.integers(min_value=1, max_value=20),
            L=st.integers(min_value=2, max_value=20),
            A=st.integers(min_value=1, max_value=20),
            Dtype=st.sampled_from([torch.int32, torch.float, torch.int64]),
            broadcast_lengths=st.booleans(),
        )
        @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
        def test_reorder_batched_ad_lengths(
            self,
            B: int,
            T: int,
            L: int,
            A: int,
            Dtype: torch.dtype,
            broadcast_lengths: bool,
        ) -> None:
            if broadcast_lengths:
                cat_ad_lengths = (
                    torch.cat(
                        [torch.tensor([L for _ in range(T)]) for _ in range(B)], 0
                    )
                    .xpu()
                    .to(Dtype)
                )
                cat_ad_lengths_broadcasted = cat_ad_lengths.tile([A])
            else:
                cat_ad_lengths = (
                    torch.cat(
                        [torch.tensor([L for _ in range(T * A)]) for _ in range(B)], 0
                    )
                    .xpu()
                    .to(Dtype)
                )
                cat_ad_lengths_broadcasted = cat_ad_lengths
            batch_offsets = torch.tensor([A * b for b in range(B + 1)]).int().xpu()
            num_ads_in_batch = B * A
            reordered_batched_ad_lengths = torch.ops.fbgemm.reorder_batched_ad_lengths(
                cat_ad_lengths, batch_offsets, num_ads_in_batch, broadcast_lengths
            )
            torch.testing.assert_close(
                cat_ad_lengths_broadcasted, reordered_batched_ad_lengths
            )

        @given(
            B=st.integers(min_value=1, max_value=20),
            T=st.integers(min_value=1, max_value=20),
            L=st.integers(min_value=2, max_value=20),
            A=st.integers(min_value=1, max_value=20),
            Dtype=st.sampled_from(
                [torch.int32, torch.float, torch.int64, torch.bfloat16]
            ),
            Itype=st.sampled_from([torch.int32, torch.int64]),
            broadcast_indices=st.booleans(),
        )
        @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
        def test_reorder_batched_ad_indices(
            self,
            B: int,
            T: int,
            L: int,
            A: int,
            Dtype: torch.dtype,
            Itype: torch.dtype,
            broadcast_indices: bool,
        ) -> None:
            if broadcast_indices:
                cat_ad_indices = (
                    torch.randint(
                        low=0,
                        high=100,
                        size=(B * T * L,),
                    )
                    .int()
                    .xpu()
                    .to(Dtype)
                )
                cat_ad_lengths = (
                    torch.cat(
                        [torch.tensor([L for _ in range(T)]) for _ in range(B)],
                        0,
                    )
                    .int()
                    .xpu()
                )
                cat_ad_lengths_broadcasted = cat_ad_lengths.tile([A])
            else:
                cat_ad_indices = (
                    torch.randint(
                        low=0,
                        high=100,
                        size=(B * T * A * L,),
                    )
                    .int()
                    .xpu()
                    .to(Dtype)
                )
                cat_ad_lengths = (
                    torch.cat(
                        [torch.tensor([L for _ in range(T * A)]) for _ in range(B)],
                        0,
                    )
                    .int()
                    .xpu()
                )
                cat_ad_lengths_broadcasted = cat_ad_lengths
            batch_offsets = torch.tensor([A * b for b in range(B + 1)]).int().xpu()
            num_ads_in_batch = B * A
            reordered_cat_ad_lengths = torch.ops.fbgemm.reorder_batched_ad_lengths(
                cat_ad_lengths, batch_offsets, num_ads_in_batch, broadcast_indices
            )
            torch.testing.assert_close(
                cat_ad_lengths_broadcasted, reordered_cat_ad_lengths
            )

            cat_ad_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
                cat_ad_lengths
            ).to(Itype)
            reordered_cat_ad_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
                reordered_cat_ad_lengths
            ).to(Itype)
            reordered_cat_ad_indices = torch.ops.fbgemm.reorder_batched_ad_indices(
                cat_ad_offsets,
                cat_ad_indices,
                reordered_cat_ad_offsets,
                batch_offsets,
                num_ads_in_batch,
                broadcast_indices,
                B * T * A * L,
            )

            torch.testing.assert_close(
                reordered_cat_ad_indices.view(T, B, A, L).permute(1, 0, 2, 3),
                (
                    cat_ad_indices.view(B, T, 1, L).tile([1, 1, A, 1])
                    if broadcast_indices
                    else cat_ad_indices.view(B, T, A, L)
                ),
            )

    class Permute2DSparseFeaturesTest(TestCase):
        @given(
            B=st.integers(min_value=0, max_value=20),
            T=st.integers(min_value=0, max_value=20),
            L=st.integers(min_value=2, max_value=20),
            long_index=st.booleans(),
            has_weight=st.booleans(),
            W=st.integers(min_value=4, max_value=8),
        )
        def test_permute_indices(
            self,
            B: int,
            T: int,
            L: int,
            long_index: bool,
            has_weight: bool,
            W: int,
        ) -> None:
            index_dtype = torch.int64 if long_index else torch.int32
            length_splits: Optional[List[torch.Tensor]] = None
            lengths = torch.randint(low=1, high=L, size=(T, B)).type(index_dtype)

            # pyre-fixme[6]: For 1st param expected `Union[List[int], Size,
            #  typing.Tuple[int, ...]]` but got `Union[bool, float, int]`.
            weights = torch.rand(lengths.sum().item()).float() if has_weight else None
            indices = torch.randint(
                low=1,
                high=int(1e5),
                # pyre-fixme[6]: Expected `Union[int, typing.Tuple[int, ...]]` for 3rd
                #  param but got `Tuple[typing.Union[float, int]]`.
                size=(lengths.sum().item(),),
            ).type(index_dtype)

            permute_list = list(range(T))
            random.shuffle(permute_list)

            permute = torch.IntTensor(permute_list)
            (
                permuted_lengths_ref,
                permuted_indices_ref,
                permuted_weights_ref,
                # pyre-fixme[6]: For 4th param expected `LongTensor` but got `Tensor`.
            ) = permute_indices_ref_(lengths, indices, weights, permute.long())
            (
                permuted_lengths_xpu,
                permuted_indices_xpu,
                permuted_weights_xpu,
            ) = torch.ops.fbgemm.permute_2D_sparse_data(
                permute.xpu(),
                lengths.xpu(),
                indices.xpu(),
                weights.xpu() if has_weight else None,
                None,
            )
            if has_weight:
                torch.testing.assert_close(
                    permuted_weights_xpu.cpu(), permuted_weights_ref
                )
            else:
                assert permuted_weights_xpu is None and permuted_weights_ref is None

                torch.testing.assert_close(
                    permuted_indices_xpu.cpu(), permuted_indices_ref
                )
                torch.testing.assert_close(
                    permuted_lengths_xpu.cpu(), permuted_lengths_ref
                )
                self.assertIsNone(permuted_weights_xpu)

        @given(
            B=st.integers(min_value=2, max_value=20),
            T=st.integers(min_value=2, max_value=20),
            L=st.integers(min_value=2, max_value=20),
            long_index=st.booleans(),
        )
        def test_permute_indices_non_contiguous(
            self,
            B: int,
            T: int,
            L: int,
            long_index: bool,
        ) -> None:
            index_dtype = torch.int64 if long_index else torch.int32
            lengths = torch.randint(low=1, high=L, size=(T, B)).type(index_dtype)

            indices = torch.randint(
                low=1,
                high=int(1e5),
                # pyre-fixme[6]: Expected `Union[int, typing.Tuple[int, ...]]` for 3rd
                #  param but got `Tuple[typing.Union[float, int]]`.
                size=(lengths.sum().item(),),
            ).type(index_dtype)

            permute_list = list(range(T))
            random.shuffle(permute_list)
            permute = torch.IntTensor(permute_list)

            def create_non_contiguous(x: torch.Tensor) -> torch.Tensor:
                # Create a diluted tensor with 2x elements, and then take every other element
                # with the value from the original tensor. For example, if x = [1, 2, 3, 4],
                # then the diluted tensor is [1, 0, 2, 0, 3, 0, 4, 0].
                diluted = x.new_zeros(x.numel() * 2).flatten()
                diluted[::2] = x.flatten()
                # Returns the sliced tensor, which is non-contiguous.
                return diluted[::2].view(x.shape)

            (
                permuted_lengths_ref,
                permuted_indices_ref,
                permuted_weights_ref,
                # pyre-fixme[6]: For 4th param expected `LongTensor` but got `Tensor`.
            ) = permute_indices_ref_(lengths, indices, None, permute.long())

            permute_xpu = create_non_contiguous(permute.xpu())
            lengths_xpu = create_non_contiguous(lengths.xpu())
            indices_xpu = create_non_contiguous(indices.xpu())
            self.assertFalse(permute_xpu.is_contiguous())
            self.assertFalse(lengths_xpu.is_contiguous())
            self.assertFalse(indices_xpu.is_contiguous())

            (
                permuted_lengths_xpu,
                permuted_indices_xpu,
                permuted_weights_xpu,
            ) = torch.ops.fbgemm.permute_2D_sparse_data(
                permute_xpu,
                lengths_xpu,
                indices_xpu,
                None,
                None,
            )
            torch.testing.assert_close(permuted_indices_xpu.cpu(), permuted_indices_ref)
            torch.testing.assert_close(permuted_lengths_xpu.cpu(), permuted_lengths_ref)
            self.assertIsNone(permuted_weights_xpu)

        def test_permute_indices_scripted_with_none_weights(
            self,
        ) -> None:
            index_dtype = torch.int32
            lengths = torch.randint(low=1, high=2, size=(1, 1)).type(index_dtype)
            weights = None
            indices = torch.randint(
                low=1,
                high=int(1e5),
                # pyre-fixme[6]: Expected `Union[int, typing.Tuple[int, ...]]` for 3rd
                #  param but got `Tuple[typing.Union[float, int]]`.
                size=(lengths.sum().item(),),
            ).type(index_dtype)
            permute_list = list(range(1))
            random.shuffle(permute_list)

            permute = torch.IntTensor(permute_list)

            (
                permuted_lengths_xpu,
                permuted_indices_xpu,
                permuted_weights_xpu,
            ) = torch.ops.fbgemm.permute_2D_sparse_data(
                permute.xpu(), lengths.xpu(), indices.xpu(), None, None
            )
            (
                permuted_lengths_ref,
                permuted_indices_ref,
                permuted_weights_ref,
                # pyre-fixme[6]: For 4th param expected `LongTensor` but got `Tensor`.
            ) = permute_indices_ref_(lengths, indices, weights, permute.long())
            self.assertTrue(
                torch.equal(permuted_indices_xpu.cpu(), permuted_indices_ref)
            )
            self.assertTrue(
                torch.equal(permuted_lengths_xpu.cpu(), permuted_lengths_ref)
            )
            self.assertEqual(permuted_weights_xpu, None)
            self.assertEqual(permuted_weights_ref, None)

        @given(
            B=st.integers(min_value=1, max_value=20),
            T=st.integers(min_value=1, max_value=20),
            L=st.integers(min_value=2, max_value=20),
            long_index=st.booleans(),
            has_weight=st.booleans(),
        )
        def test_permute_indices_with_repeats(
            self, B: int, T: int, L: int, long_index: bool, has_weight: bool
        ) -> None:
            index_dtype = torch.int64 if long_index else torch.int32
            lengths = torch.randint(low=1, high=L, size=(T, B)).type(index_dtype)
            # pyre-fixme[6]: For 1st param expected `Union[List[int], Size,
            #  typing.Tuple[int, ...]]` but got `Union[bool, float, int]`.
            weights = torch.rand(lengths.sum().item()).float() if has_weight else None
            indices = torch.randint(
                low=1,
                high=int(1e5),
                # pyre-fixme[6]: Expected `Union[int, typing.Tuple[int, ...]]` for 3rd
                #  param but got `Tuple[typing.Union[float, int]]`.
                size=(lengths.sum().item(),),
            ).type(index_dtype)
            permute_list = list(range(T))

            num_repeats = random.randint(0, T)
            for _ in range(num_repeats):
                permute_list.append(random.randint(0, T - 1))

            random.shuffle(permute_list)
            permute = torch.IntTensor(permute_list)

            (
                permuted_lengths_ref,
                permuted_indices_ref,
                permuted_weights_ref,
                # pyre-fixme[6]: For 4th param expected `LongTensor` but got `Tensor`.
            ) = permute_indices_ref_(lengths, indices, weights, permute.long())

            (
                permuted_lengths_xpu,
                permuted_indices_xpu,
                permuted_weights_xpu,
            ) = torch.ops.fbgemm.permute_2D_sparse_data(
                permute.xpu(),
                lengths.xpu(),
                indices.xpu(),
                # pyre-fixme[16]: `Optional` has no attribute `cuda`.
                weights.xpu() if has_weight else None,
            )
            torch.testing.assert_close(permuted_indices_xpu.cpu(), permuted_indices_ref)
            torch.testing.assert_close(permuted_lengths_xpu.cpu(), permuted_lengths_ref)
            if has_weight:
                torch.testing.assert_close(
                    permuted_weights_xpu.cpu(), permuted_weights_ref
                )
            else:
                assert permuted_weights_xpu is None

        def test_permute_2D_sparse_data(self) -> None:
            lengths = torch.tensor(
                [[0, 0, 1], [0, 1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 1]],
                dtype=torch.int32,
                device="xpu",
            )
            indices = torch.tensor(
                [500, 1000, 1999],
                dtype=torch.int32,
                device="xpu",
            )
            permute = torch.tensor(
                [0, 3, 1, 4, 2, 5],
                dtype=torch.int32,
                device="xpu",
            )
            weights = torch.rand((3, 64), device="xpu")
            (
                lengths_actual,
                values_actual,
                weights_actual,
            ) = torch.ops.fbgemm.permute_2D_sparse_data(
                permute, lengths, indices, weights, indices.numel()
            )
            self.assertTrue(
                torch.equal(
                    lengths_actual,
                    torch.tensor(
                        [
                            [0, 0, 1],
                            [0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 1],
                        ],
                        dtype=torch.int32,
                        device="xpu",
                    ),
                )
            )
            self.assertTrue(torch.equal(values_actual, indices))
            self.assertTrue(torch.equal(weights_actual, weights))


instantiate_device_type_tests(CumSumTest, globals(), only_for="xpu", allow_xpu=True)

instantiate_device_type_tests(
    DenseToJaggedTest, globals(), only_for="xpu", allow_xpu=True
)

instantiate_device_type_tests(
    JaggedToPaddedDenseTest, globals(), only_for="xpu", allow_xpu=True
)

instantiate_device_type_tests(
    ElementwiseBinaryTest, globals(), only_for="xpu", allow_xpu=True
)

instantiate_device_type_tests(
    ReorderBatchedTest, globals(), only_for="xpu", allow_xpu=True
)

instantiate_device_type_tests(
    Permute2DSparseFeaturesTest, globals(), only_for="xpu", allow_xpu=True
)

if __name__ == "__main__":
    run_tests()
