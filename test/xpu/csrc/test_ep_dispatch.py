import torch
import pytest

def test_basic_float32():
    # Create dummy tensors for testing
    hidden_shard = torch.rand((4, 8), dtype=torch.float32, device='xpu')
    topk_idx = torch.randint(0, 4, (16, 2), dtype=torch.int64, device='xpu')
    remap_hidden_states = torch.zeros((16 * 2, 8), dtype=torch.float32, device='xpu')

    # Call the ep_dispatch function
    torch.ops.load_library("libep_dispatch.so")
    result = torch.ops.symm_mem.ep_dispatch(hidden_shard, topk_idx, remap_hidden_states, 4, 0, 1)

    # Validate the result (basic validation for now)
    assert result.shape == remap_hidden_states.shape

def test_basic_bfloat16():
    # Create dummy tensors for testing
    hidden_shard = torch.rand((4, 8), dtype=torch.bfloat16, device='xpu')
    topk_idx = torch.randint(0, 4, (16, 2), dtype=torch.int64, device='xpu')
    remap_hidden_states = torch.zeros((16 * 2, 8), dtype=torch.bfloat16, device='xpu')

    # Call the ep_dispatch function
    torch.ops.load_library("libep_dispatch.so")
    result = torch.ops.symm_mem.ep_dispatch(hidden_shard, topk_idx, remap_hidden_states, 4, 0, 1)

    # Validate the result (basic validation for now)
    assert result.shape == remap_hidden_states.shape

def test_large_topk():
    # Create dummy tensors for testing
    hidden_shard = torch.rand((4, 8), dtype=torch.float32, device='xpu')
    topk_idx = torch.randint(0, 4, (16, 8), dtype=torch.int64, device='xpu')
    remap_hidden_states = torch.zeros((16 * 8, 8), dtype=torch.float32, device='xpu')

    # Call the ep_dispatch function
    torch.ops.load_library("libep_dispatch.so")
    result = torch.ops.symm_mem.ep_dispatch(hidden_shard, topk_idx, remap_hidden_states, 4, 0, 1)

    # Validate the result (basic validation for now)
    assert result.shape == remap_hidden_states.shape

def test_zero_tokens():
    # Create dummy tensors for testing
    hidden_shard = torch.rand((0, 8), dtype=torch.float32, device='xpu')
    topk_idx = torch.randint(0, 4, (0, 2), dtype=torch.int64, device='xpu')
    remap_hidden_states = torch.zeros((0, 8), dtype=torch.float32, device='xpu')

    # Call the ep_dispatch function
    torch.ops.load_library("libep_dispatch.so")
    result = torch.ops.symm_mem.ep_dispatch(hidden_shard, topk_idx, remap_hidden_states, 4, 0, 1)

    # Validate the result (basic validation for now)
    assert result.shape == remap_hidden_states.shape

def test_inplace_semantics():
    # Create dummy tensors for testing
    hidden_shard = torch.rand((4, 8), dtype=torch.float32, device='xpu')
    topk_idx = torch.randint(0, 4, (16, 2), dtype=torch.int64, device='xpu')
    remap_hidden_states = torch.zeros((16 * 2, 8), dtype=torch.float32, device='xpu')

    # Call the ep_dispatch function
    torch.ops.load_library("libep_dispatch.so")
    torch.ops.symm_mem.ep_dispatch(hidden_shard, topk_idx, remap_hidden_states, 4, 0, 1)

    # Validate the result (in-place modification)
    assert torch.all(remap_hidden_states != 0)

if __name__ == "__main__":
    pytest.main(["-v", "test_ep_dispatch.py"])