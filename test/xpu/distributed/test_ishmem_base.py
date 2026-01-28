import torch
import torch.nn as nn
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch._C._distributed_c10d import _SymmetricMemory
from torch.distributed._symmetric_memory import enable_symm_mem_for_group
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    requires_cuda_p2p_access,
    run_tests,
    skip_but_pass_in_sandcastle_if,
    skipIfRocm,
)

device_type = "xpu"
device_module = torch.get_device_module(device_type)
class ISHMEMSymmetricMemoryTest(MultiProcContinuousTest):
    def _init_device(self) -> None:
        device_module.set_device(self.device)
        symm_mem.set_backend("ISHMEM")

    @property
    def device(self) -> torch.device:
        return torch.device(device_type, self.rank)

    def test_alloc(self) -> None:
        self._init_device()

        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)

        dtype = torch.float
        numel = 1024

        def foo():
            inp = symm_mem.empty(numel, dtype=dtype, device=self.device)
            symm_mem.rendezvous(inp, group=group_name)

        foo()

        out = symm_mem.empty(numel, dtype=dtype, device=self.device)
        symm_mem.rendezvous(out, group=group_name)


    def test_get_remote_tensor(self) -> None:
        """
        Get a remote tensor and use regular aten ops to write to it.
        """
        self._init_device()
        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)

        dtype = torch.float
        numel = 1024

        x = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(self.rank)
        y = symm_mem.empty(numel, dtype=dtype, device=self.device)

        hdl_y = symm_mem.rendezvous(y, group=group_name)
        peer = (self.rank + 1) % self.world_size  # Shifting pattern
        y_remote = hdl_y.get_remote_tensor(peer, y.size(), y.dtype)
        y_remote.copy_(x)
        dist.barrier()
        # Expecting data from -1 rank
        expected = torch.empty(numel, dtype=dtype, device=self.device).fill_(
            (self.rank - 1) % self.world_size
        )
        self.assertEqual(y, expected)

if __name__ == "__main__":
    run_tests()
