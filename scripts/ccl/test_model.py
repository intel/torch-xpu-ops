
import copy
import torch
import intel_extension_for_pytorch
import oneccl_bindings_for_pytorch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._composable.fsdp import fully_shard
import argparse
import os
import sys

class MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        device: Optional[torch.device] = None,
        *,
        bias: bool = True,
        with_buffer: bool = False,
        dim_multiplier: int = 4,
    ):
        super().__init__()
        self.in_proj = nn.Linear(dim, dim_multiplier * dim, device=device, bias=bias)
        self.out_proj = nn.Linear(dim_multiplier * dim, dim, device=device, bias=bias)
        if with_buffer:
            self.register_buffer("buffer", torch.randn((dim,), device=device))
        else:
            self.buffer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.in_proj(x)
        z = F.relu(z)
        z = self.out_proj(z)
        z = F.relu(z)
        if self.buffer is not None:
            z = z + self.buffer
        return z

    def reset_parameters(self):
        if self.buffer is not None:
            torch.nn.init.normal_(self.buffer)


os.environ['RANK'] = str(os.environ.get('PMI_RANK', 0))
os.environ['WORLD_SIZE'] = str(os.environ.get('PMI_SIZE', 1))
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29800'

msg_size=2048
dist.init_process_group("ccl")
rank = dist.get_rank()
size = dist.get_world_size()
device = "xpu:{}".format(rank)

torch.manual_seed(42)
mlp_dim = 8
ref_model = MLP(mlp_dim)

dp_size = 2
global_mesh = init_device_mesh("xpu", (dp_size, 4), mesh_dim_names=("dp", "tp"))
ref_dp_mesh, tp_mesh = global_mesh["dp"], global_mesh["tp"]
for module in (ref_model.in_proj, ref_model.out_proj, ref_model):
    fully_shard(module, mesh=ref_dp_mesh)

inp = torch.randn((4, mlp_dim), device="xpu")
ref_loss = ref_model(inp).sum()
ref_loss.backward()