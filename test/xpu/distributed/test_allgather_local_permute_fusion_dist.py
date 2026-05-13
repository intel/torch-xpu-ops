"""
Accuracy check for allgather_local_permute_fusion (distributed style)

Usage:
    mpirun -n 2 python test_allgather_local_permute_fusion_dist.py
"""
import torch
import torch.distributed as dist
import os
from allgather_local_permute_fusion import allgather_local_permute_fusion

def init_distributed():
    os.environ['RANK'] = str(os.environ.get('PMI_RANK', 0))
    os.environ['WORLD_SIZE'] = str(os.environ.get('PMI_SIZE', 1))
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29513'
    if not dist.is_initialized():
        dist.init_process_group(backend="xccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.xpu.set_device(rank)
    return rank, world_size

def check_allgather_local_permute_fusion():
    rank, world_size = init_distributed()
    device = f"xpu:{rank}"
    num_tokens = 8
    topk = 2
    hidden_size = 4
    num_tokens_per_rank = num_tokens // world_size

    # Each rank: unique hidden_shard
    torch.manual_seed(1234 + rank)
    hidden_shard = torch.randn(num_tokens_per_rank, hidden_size, device=device)

    # All ranks: same topk_idx
    torch.manual_seed(42)
    topk_idx = torch.randint(0, world_size, (num_tokens, topk), device=device)

    # Output buffer
    remap_hidden_states = torch.empty((num_tokens * topk, hidden_size), device=device)

    # Run fusion
    allgather_local_permute_fusion(hidden_shard, topk_idx, remap_hidden_states=remap_hidden_states)
    torch.xpu.synchronize()
    dist.barrier()

    # Gather all hidden_shards to rank 0 for reference
    gathered = [torch.empty_like(hidden_shard) for _ in range(world_size)]
    dist.all_gather(gathered, hidden_shard)
    if rank == 0:
        ref_hidden = torch.cat(gathered, dim=0)
        # Build reference remap_hidden_states
        ref_remap = torch.empty((num_tokens * topk, hidden_size), device="cpu")
        for token in range(num_tokens):
            for k in range(topk):
                src_rank = token // num_tokens_per_rank
                local_idx = token % num_tokens_per_rank
                ref_remap[token * topk + k] = ref_hidden[token]
        # Compare device result to reference
        out = remap_hidden_states.cpu()
        assert torch.allclose(out, ref_remap, atol=1e-5), "Remap hidden states mismatch!"
        print("[PASS] allgather_local_permute_fusion distributed accuracy test")

if __name__ == "__main__":
    check_allgather_local_permute_fusion()
