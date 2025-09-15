import torch
import torch.distributed as dist
from test_c10d_xccl import init_multigpu_helper, requires_xccl
from torch.distributed._symmetric_memory import (
    _fused_all_gather_matmul_fallback,
    _fused_matmul_reduce_scatter_fallback,
)

from torch.testing._internal.common_distributed import MultiProcContinuousTest
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests
)

@instantiate_parametrized_tests
class AsyncTPTest(MultiProcContinuousTest):
    @property
    def device(self) -> torch.device:
        return torch.device("xpu", self.rank)

    def _init_process(self):
        torch.xpu.set_device(self.device)
        torch.manual_seed(42 + self.rank)
        torch.use_deterministic_algorithms(True)
        torch.set_deterministic_debug_mode("warn")
        torch.utils.deterministic.fill_uninitialized_memory = True

    @requires_xccl()
    @parametrize("gather_dim", [0, 1])
    def test_fused_all_gather_matmul(self, gather_dim: int) -> None:
        self._init_process()
        BATCH = 8
        M = 64
        N = 16
        K = 32
        group = dist.group.WORLD
        rank = self.rank    

        torch.manual_seed(42 + rank)
        A_shard = torch.rand(BATCH, M // self.world_size, K, device="xpu")
        Bs = [torch.rand(K, N, device="xpu") for _ in range(3)]

        ag_output_0, mm_outputs_0 = _fused_all_gather_matmul_fallback(
            A_shard, Bs, gather_dim=gather_dim, group_name=group.group_name
        )
        ag_output_1, mm_outputs_1 = torch.ops.symm_mem.fused_all_gather_matmul(
            A_shard, Bs, gather_dim=gather_dim, group_name=group.group_name
        )

        self.assertEqual(ag_output_0, ag_output_1)
        self.assertEqual(ag_output_0.stride(), ag_output_1.stride())
        for mm_output_0, mm_output_1 in zip(mm_outputs_0, mm_outputs_1):
            self.assertEqual(mm_output_0, mm_output_1)
            self.assertEqual(mm_output_0.stride(), mm_output_1.stride())

    @requires_xccl()
    @parametrize("scatter_dim", [0, 1])
    def test_fused_matmul_reduce_scatter(self, scatter_dim: int) -> None:
        self._init_process()
        
        BATCH = 8
        M = 64
        N = 16
        K = 32
        group = dist.group.WORLD
        rank = self.rank

        torch.manual_seed(42 + rank)
        A = torch.rand(BATCH, M, K, device="xpu")
        B = torch.rand(K, N, device="xpu")

        output_0 = _fused_matmul_reduce_scatter_fallback(
            A, B, "avg", scatter_dim=scatter_dim, group_name=group.group_name
        )
        output_1 = torch.ops.symm_mem.fused_matmul_reduce_scatter(
            A, B, "avg", scatter_dim=scatter_dim, group_name=group.group_name
        )

        self.assertEqual(output_0, output_1)
        self.assertEqual(output_0.stride(), output_1.stride())


if __name__ == "__main__":
    run_tests()
