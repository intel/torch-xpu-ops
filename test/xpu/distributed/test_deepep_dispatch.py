import torch

from deepep_dispatch import (
    deepep_owner_dispatch,
    get_expert_owner,
    get_owner_expert_ranges,
)


def _build_hidden_shards(tp_world_size: int, tokens_per_rank: int, hidden_size: int):
    shards = []
    for src in range(tp_world_size):
        # Unique rows for easy debugging and exact-match checks
        base = src * 1000
        vals = torch.arange(base, base + tokens_per_rank * hidden_size, dtype=torch.float32)
        shards.append(vals.view(tokens_per_rank, hidden_size))
    return shards


def test_owner_ranges_with_remainder():
    ranges = get_owner_expert_ranges(num_experts=10, tp_world_size=4)
    assert ranges == [(0, 3), (3, 6), (6, 8), (8, 10)]


def test_deepep_owner_dispatch_correctness():
    tp_world_size = 4
    num_experts = 8
    tokens_per_rank = 3
    hidden_size = 2
    topk = 2

    hidden_shards = _build_hidden_shards(tp_world_size, tokens_per_rank, hidden_size)

    # Route table covers all owners and experts.
    topk_idx_shards = [
        torch.tensor([[0, 5], [1, 6], [2, 7]], dtype=torch.int64),
        torch.tensor([[3, 4], [0, 6], [1, 7]], dtype=torch.int64),
        torch.tensor([[2, 5], [3, 4], [0, 6]], dtype=torch.int64),
        torch.tensor([[1, 7], [2, 5], [3, 4]], dtype=torch.int64),
    ]
    topk_w_shards = [
        torch.tensor([[0.8, 0.2], [0.6, 0.4], [0.7, 0.3]], dtype=torch.float32),
        torch.tensor([[0.5, 0.5], [0.9, 0.1], [0.2, 0.8]], dtype=torch.float32),
        torch.tensor([[0.55, 0.45], [0.65, 0.35], [0.75, 0.25]], dtype=torch.float32),
        torch.tensor([[0.11, 0.89], [0.22, 0.78], [0.33, 0.67]], dtype=torch.float32),
    ]

    outputs = deepep_owner_dispatch(
        hidden_shards=hidden_shards,
        topk_idx_shards=topk_idx_shards,
        topk_w_shards=topk_w_shards,
        num_experts=num_experts,
    )

    # Build reference order that matches implementation:
    # owner -> expert -> src_rank -> token -> k
    owner_ranges = get_owner_expert_ranges(num_experts, tp_world_size)
    for owner in range(tp_world_size):
        ref = []
        e_start, e_end = owner_ranges[owner]
        for expert in range(e_start, e_end):
            for src_rank in range(tp_world_size):
                for token in range(tokens_per_rank):
                    for k in range(topk):
                        if int(topk_idx_shards[src_rank][token, k].item()) != expert:
                            continue
                        global_token_id = src_rank * tokens_per_rank + token
                        ref.append(
                            {
                                "hidden": hidden_shards[src_rank][token],
                                "global_token_id": global_token_id,
                                "k_idx": k,
                                "source_rank": src_rank,
                                "route_weight": topk_w_shards[src_rank][token, k],
                                "expert_id": expert,
                            }
                        )

        out = outputs[owner]
        assert out["remap_hidden_states_owner"].shape == (len(ref), hidden_size)
        assert out["global_token_id"].shape == (len(ref),)
        assert out["k_idx"].shape == (len(ref),)
        assert out["source_rank"].shape == (len(ref),)
        assert out["route_weight"].shape == (len(ref),)
        assert out["expert_id"].shape == (len(ref),)

        for i, item in enumerate(ref):
            assert torch.equal(out["remap_hidden_states_owner"][i], item["hidden"])
            assert int(out["global_token_id"][i].item()) == item["global_token_id"]
            assert int(out["k_idx"][i].item()) == item["k_idx"]
            assert int(out["source_rank"][i].item()) == item["source_rank"]
            assert torch.allclose(out["route_weight"][i], item["route_weight"]) 
            assert int(out["expert_id"][i].item()) == item["expert_id"]



def test_get_expert_owner():
    num_experts = 10
    tp_world_size = 4
    owners = [get_expert_owner(e, num_experts, tp_world_size) for e in range(num_experts)]
    assert owners == [0, 0, 0, 1, 1, 1, 2, 2, 3, 3]
