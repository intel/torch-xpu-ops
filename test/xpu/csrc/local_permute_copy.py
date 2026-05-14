"""
Python interface for LocalPermuteCopy XPU kernel.

Usage:
    from local_permute_copy import local_permute_copy

    # src_hidden:          [num_tokens_per_rank, hidden_size], XPU tensor
    # topk_idx:            [num_tokens, topk],                 XPU tensor
    # remote_token_offset: int
    # remap_hidden_states: [num_tokens * topk, hidden_size],   XPU tensor (output, modified in-place)
    result = local_permute_copy(src_hidden, topk_idx, remote_token_offset, remap_hidden_states)
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
    topk_idx: torch.Tensor,
    remote_token_offset: int,
    remap_hidden_states: torch.Tensor,
) -> torch.Tensor:
    """
    Fused local permute copy: remap tokens from [tokens_per_rank, hidden]
    into remap_hidden_states at positions determined by topk_idx.

    Args:
        src_hidden:          [num_tokens_per_rank, hidden_size] - contiguous XPU tensor (float32 or bfloat16)
        topk_idx:            [num_tokens, topk] - global token-to-expert mapping
        remote_token_offset: starting token index for this rank's shard
        remap_hidden_states: [num_tokens * topk, hidden_size] - output tensor (modified in-place)

    Returns:
        remap_hidden_states (same tensor, modified in-place)
    """
    return torch.ops.symm_mem.local_permute_copy_(
        src_hidden, topk_idx, remote_token_offset, remap_hidden_states
    )
