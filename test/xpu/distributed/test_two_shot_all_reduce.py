"""
Tests for two_shot_all_reduce_ and two_shot_all_reduce_out operations.

Usage:
    # Run with pytest (requires multi-GPU setup)
    pytest test_two_shot_all_reduce.py

    # Run with mpirun
    mpirun -n 2 python test_two_shot_all_reduce.py
"""

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import _SymmetricMemory

from test_c10d_xccl import requires_xccl
from torch.testing._internal.common_distributed import MultiProcContinuousTest
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)


@instantiate_parametrized_tests
class TwoShotAllReduceTest(MultiProcContinuousTest):
    @property
    def device(self) -> torch.device:
        return torch.device("xpu", self.rank)

    def _init_process(self):
        torch.xpu.set_device(self.device)
        torch.manual_seed(42 + self.rank)

    def _create_symm_mem_tensor(self, size, dtype):
        """Create a tensor allocated with symmetric memory."""
        group = dist.group.WORLD
        # Calculate contiguous strides
        strides = torch._prims_common.make_contiguous_strides_for(size)
        return _SymmetricMemory.empty_strided_p2p(
            size,
            strides,
            dtype,
            self.device,
            group.group_name,
        )

    @requires_xccl()
    @parametrize("dtype", [torch.float32, torch.bfloat16])
    @parametrize("size", [64, 256, 1024, 4096])
    def test_two_shot_all_reduce_inplace(self, dtype: torch.dtype, size: int) -> None:
        """Test in-place two_shot_all_reduce_ operation."""
        self._init_process()

        group = dist.group.WORLD
        world_size = self.world_size
        rank = self.rank

        # Each rank creates its own tensor with rank-specific values
        torch.manual_seed(42 + rank)
        local_data = torch.rand(size, device=self.device, dtype=dtype)

        # Gather all data for reference computation
        all_data = [torch.zeros_like(local_data) for _ in range(world_size)]
        dist.all_gather(all_data, local_data)
        expected = sum(all_data)

        # Allocate symmetric memory tensor and copy data
        symm_tensor = self._create_symm_mem_tensor((size,), dtype)
        symm_tensor.copy_(local_data)

        # Call two_shot_all_reduce_ (in-place)
        result = torch.ops.symm_mem.two_shot_all_reduce_(
            symm_tensor, "sum", group.group_name
        )

        torch.xpu.synchronize()

        # Verify result
        self.assertEqual(result.data_ptr(), symm_tensor.data_ptr())  # Same buffer
        torch.testing.assert_close(
            result, expected, rtol=1e-3, atol=1e-3,
            msg=f"two_shot_all_reduce_ failed for dtype={dtype}, size={size}"
        )

    @requires_xccl()
    @parametrize("dtype", [torch.float32, torch.bfloat16])
    @parametrize("size", [64, 256, 1024, 4096])
    def test_two_shot_all_reduce_out(self, dtype: torch.dtype, size: int) -> None:
        """Test two_shot_all_reduce_out operation with separate output buffer."""
        self._init_process()

        group = dist.group.WORLD
        world_size = self.world_size
        rank = self.rank

        # Each rank creates its own tensor with rank-specific values
        torch.manual_seed(42 + rank)
        local_data = torch.rand(size, device=self.device, dtype=dtype)

        # Gather all data for reference computation
        all_data = [torch.zeros_like(local_data) for _ in range(world_size)]
        dist.all_gather(all_data, local_data)
        expected = sum(all_data)

        # Allocate symmetric memory tensor and copy data
        symm_tensor = self._create_symm_mem_tensor((size,), dtype)
        symm_tensor.copy_(local_data)

        # Create output tensor (regular XPU tensor)
        output = torch.empty(size, device=self.device, dtype=dtype)

        # Call two_shot_all_reduce_out
        result = torch.ops.symm_mem.two_shot_all_reduce_out(
            symm_tensor, "sum", group.group_name, output
        )

        torch.xpu.synchronize()

        # Verify result
        self.assertEqual(result.data_ptr(), output.data_ptr())  # Output buffer
        torch.testing.assert_close(
            result, expected, rtol=1e-3, atol=1e-3,
            msg=f"two_shot_all_reduce_out failed for dtype={dtype}, size={size}"
        )


if __name__ == "__main__":
    run_tests()

