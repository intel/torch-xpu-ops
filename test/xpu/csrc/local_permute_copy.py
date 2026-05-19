"""
Python interface for LocalPermuteCopy XPU kernel.

Usage:
    from local_permute_copy import local_permute_copy

    # src_hidden:          [num_tokens_per_rank, hidden_size], XPU tensor
    # scatter_idx:         [num_tokens, topk], int32 XPU tensor (expert-sorted positions)
    # remote_token_offset: int
    # remap_hidden_states: [total_expert_tokens, hidden_size], XPU tensor (output, modified in-place)
    result = local_permute_copy(src_hidden, scatter_idx, remote_token_offset, remap_hidden_states)
"""

import os
import torch

_LIB_DIR = os.path.dirname(os.path.abspath(__file__))
_LIB_PATH = os.path.join(_LIB_DIR, "liblocal_permute_copy.so")

if not os.path.exists(_LIB_PATH):
    raise RuntimeError(
        f"Shared library not found: {_LIB_PATH}\n"
        "Run 'python build.py' first to build the extension."
    )

torch.ops.load_library(_LIB_PATH)


def local_permute_copy(
    src_hidden: torch.Tensor,
    scatter_idx: torch.Tensor,
    remote_token_offset: int,
    remap_hidden_states: torch.Tensor,
) -> torch.Tensor:
    """
    Fused local permute copy: remap tokens from [tokens_per_rank, hidden]
    into remap_hidden_states at expert-sorted positions given by scatter_idx.

    Args:
        src_hidden:          [num_tokens_per_rank, hidden_size] - contiguous XPU tensor (float32 or bfloat16)
        scatter_idx:         [num_tokens, topk] - int32 tensor with output positions for each (token, k)
        remote_token_offset: starting token index for this rank's shard
        remap_hidden_states: [total_expert_tokens, hidden_size] - output tensor (modified in-place)

    Returns:
        remap_hidden_states (same tensor, modified in-place)
    """
    return torch.ops.symm_mem.local_permute_copy_(
        src_hidden, scatter_idx, remote_token_offset, remap_hidden_states
    )
