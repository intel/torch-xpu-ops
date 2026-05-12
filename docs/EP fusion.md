## MOE model parallism
 - Starting from BF16

## Attention TP
- Attn layer: Norm + GEMM (QKV) + Attn + GEMM(output projection) + Allreduce
- Output shape: [seq, hidden_size] (**All devices are the same**)

## MLP permute & unpermute
- TopK: [num_of_tokens, topk]
- Activate: [num_of_tokens, hidden_size]
- permute output (shuffle): [num_of_tokens*topk, hidden_size] --> [num_of_experts, tokens_per_expert, hidden_size]
    -  ordering by expert id
- unpermute output: [num_of_tokens, hidden_size]
    - aggregate by route weight

 ## Attention TP + MLP EP
- Attn layer: Norm + GEMM (QKV) + Attn + GEMM(output projection) + Allreduce
  - Output shape: [seq, hidden_size] (**All devices are the same**)
- MLP layer:
  - Norm: [seq, hidden_size]
  - **Permute (local permute)**:
    - Input: [seq, hidden_size] (same on all devices from Attention TP)
    - Routing: TopK tokens per expert → [num_tokens, topk]
    - Output shape: Each device locally routes tokens → [num_experts, tokens_per_expert, hidden_size]
  - GEMM(gate up projection) + Act(SiluAndMul):
    - Each device processes its local experts: [num_tokens_for_expert, hidden_size] @ [hidden_size, intermediate_size]
    - Output: [num_tokens_for_expert, intermediate_size]
  - GEMM(down projection):
    - Each device processes: [num_tokens_for_expert, intermediate_size] @ [intermediate_size, hidden_size]
    - Output: [num_tokens_for_expert, hidden_size]
  - **Unpermute (DeepEP combine - cross-device aggregation)**:
    - Each token needs to read results from its topk assigned experts
    - These experts are distributed across different devices
    - Read expert outputs from corresponding devices and aggregate locally
    - output[token_i] = Σ(route_weights[token_i][j] × expert_outputs[topk_devices[j]][token_i])
    - Collective: **Selective All-to-All / P2P** (gather topk expert outputs from assigned devices)
  - **Optimized**
    - Allreduce + Norm -> Reducescatter + Norm + allgather
    - Allgather + local permute -> DeepEP dispatch


 ## Attention TP + MLP TP
- Attn layer: Norm + GEMM (QKV) + Attn + GEMM(output projection) + Allreduce
  - Output shape: [seq, hidden_size] (**All devices are the same**)
- MLP layer:
  - Norm: [seq, hidden_size]
  - **Permute (local permute)**:
    - Input: [seq, hidden_size] (same on all devices)
    - Output shape: Local routing → [num_experts, tokens_per_expert, hidden_size]
  - GEMM(gate up projection) + Act(SiluAndMul):
    - Each device with sharded experts: [seq, hidden_size] @ [hidden_size, intermediate_size / tp]
    - Output: [seq, intermediate_size / tp]
  - GEMM(down projection):
    - [seq, intermediate_size / tp] @ [intermediate_size / tp, hidden_size]
    - Output: [seq, hidden_size]
  - **Unpermute (local unpermute)**:
    - Input: [seq, hidden_size] from each device (partial results)
    - Output shape: [seq, hidden_size]
    - Collective: **None** (local operation on each device)
  - AllReduce:
    - Synchronize partial results across devices
    - Collective: **All-Reduce** (sum gradients / weight-sharded outputs)
- **Optimized**
    - Allreduce + Norm(mlp) + Permute -> Reducescatter + Norm(mlp) + Allgather + local Permute
    - Unpermute + Allreduce + Norm(attn) -> Unpermute + ReduceScatter + Allgather + Norm

## Projection
- Model https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json
    - Hidden size = 2048
- PCIe: Gen5 * 16 lanes ~~ 53.975GB/s
- CRI BW: 1536	GB/s
- Allgather, permute projection:

  TP=2, topk=8

  | hidden | sequence length | BS | tokens | Bytes(16bits) | size(GB) | allgather(ms) | local permute(ms) |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | 2048 | 3500 | 100 | 350000 | 1433600000 | 1.335144043 | 12.36817085 | 6.953875224 |
  | 2048 | 3500 | 50 | 175000 | 716800000 | 0.667572021 | 6.184085424 | 3.476937612 |
  | 2048 | 3500 | 20 | 70000 | 286720000 | 0.267028809 | 2.473634169 | 1.390775045 |
  | 2048 | 3500 | 8 | 28000 | 114688000 | 0.106811523 | 0.989453668 | 0.556310018 |
  | 2048 | 3500 | 1 | 3500 | 14336000 | 0.01335144 | 0.123681708 | 0.069538752 |
  | 2048 | 1500 | 100 | 150000 | 614400000 | 0.57220459 | 5.300644649 | 2.980232239 |
  | 2048 | 1500 | 50 | 75000 | 307200000 | 0.286102295 | 2.650322324 | 1.490116119 |
  | 2048 | 1500 | 20 | 30000 | 122880000 | 0.114440918 | 1.06012893 | 0.596046448 |
  | 2048 | 1500 | 8 | 12000 | 49152000 | 0.045776367 | 0.424051572 | 0.238418579 |
  | 2048 | 1500 | 1 | 1500 | 6144000 | 0.005722046 | 0.053006446 | 0.029802322 |
  | 2048 | 8192 | 1 | 8192 | 33554432 | 0.03125 | 0.289485873 | 0.162760417 |

  TP=4, topk=8

  | hidden | sequence length | BS | tokens | Bytes(16bits) | size(GB) | allgather(ms) | local permute(ms) |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | 2048 | 3500 | 100 | 350000 | 1433600000 | 1.335144043 | 18.55225627 | 6.953875224 |
  | 2048 | 3500 | 50 | 175000 | 716800000 | 0.667572021 | 9.276128135 | 3.476937612 |
  | 2048 | 3500 | 20 | 70000 | 286720000 | 0.267028809 | 3.710451254 | 1.390775045 |
  | 2048 | 3500 | 8 | 28000 | 114688000 | 0.106811523 | 1.484180502 | 0.556310018 |
  | 2048 | 3500 | 1 | 3500 | 14336000 | 0.01335144 | 0.185522563 | 0.069538752 |
  | 2048 | 1500 | 100 | 150000 | 614400000 | 0.57220459 | 7.950966973 | 2.980232239 |
  | 2048 | 1500 | 50 | 75000 | 307200000 | 0.286102295 | 3.975483487 | 1.490116119 |
  | 2048 | 1500 | 20 | 30000 | 122880000 | 0.114440918 | 1.590193395 | 0.596046448 |
  | 2048 | 1500 | 8 | 12000 | 49152000 | 0.045776367 | 0.636077358 | 0.238418579 |
  | 2048 | 1500 | 1 | 1500 | 6144000 | 0.005722046 | 0.07950967 | 0.029802322 |
  | 2048 | 8192 | 1 | 8192 | 33554432 | 0.03125 | 0.43422881 | 0.162760417 |

