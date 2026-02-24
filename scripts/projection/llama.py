'''
(model):
(
(embed_tokens, input[BS, sq], output[BS,sq, embedding_dim])): VocabParallelEmbedding(num_embeddings=32064, embedding_dim=4096, org_vocab_size=128256, num_embeddings_padded=128256, tp_size=4)
(layers): ModuleList(
  (0-31): 32 x LlamaDecoderLayer(hidden_states, residual, postions?)(
    (input_layernorm): RMSNorm(hidden_size=4096, eps=1e-05)
    (self_attn): LlamaAttention(
      (qkv_proj, output[BS, sq, num_heads * head_dim]): QKVParallelLinear(in_features=4096, output_features=1536, bias=False, tp_size=4, gather_output=False)  # tp on output dim
      (rotary_emb): Llama3RotaryEmbedding(head_size=128, rotary_dim=128, max_position_embeddings=131072, base=500000.0, is_neox_style=True)
      (attn): Attention(head_size=128, num_heads=8, num_kv_heads=2, scale=0.08838834764831845, backend=IPEXAttentionImpl)
      (o_proj, output[BS, sq, embedding_dim): RowParallelLinear(input_features=1024, output_features=4096, bias=False, tp_size=4, reduce_results=True)  # tp on
    )
    (post_attention_layernorm): RMSNorm(hidden_size=4096, eps=1e-05)
    (mlp): LlamaMLP(
      (gate_up_proj, output[]): MergedColumnParallelLinear(in_features=4096, output_features=7168, bias=False, tp_size=4, gather_output=False)
      (down_proj): RowParallelLinear(input_features=3584, output_features=4096, bias=False, tp_size=4, reduce_results=True)
      (act_fn): SiluAndMul()
    )
  )
)
(norm): RMSNorm(hidden_size=4096, eps=1e-05)
(lm_head): ParallelLMHead(num_embeddings=32064, embedding_dim=4096, org_vocab_size=128256, num_embeddings_padded=128256, tp_size=4)
(logits_processor): LogitsProcessor(vocab_size=128256, org_vocab_size=128256, scale=1.0, logits_as_input=False)
'''

# key points
#1. num_heads * head_dimension = embedding_dim

# parameters
1. hidden_size = 4096
2. head_num = 8
3. head_size = 128
4. vocabulary size = 128256
5. embedding size = 4096 (alias: embedding dim), means each token would be projected to a 4096 vector
6. num_embeddings means the vocab_size on local after TP
7. embedding dim usually is same as hidden dim
8. Q, K, V shape is [BS, sq, num_heads * head_dim]
9. multi head attention: num_head * head_dim = hidden_size = embedding dim
10. Q, K, V original weight shape (no tp) [hidden_size, hidden_size]
11. TP is acutally weight division
12. three allreduce to get full activations
    - first one: embemdding layer: [bs * seq, embedding_dim] -> [bs, seq, embedding_dim]
        - reducescatter: [bs * seq , embedding_dim]  -> [bs * seq / tp, embedding_dim]
            -   RMS norm
        - allgather: [bs * seq / tp, embedding_dim]  -> [bs * seq , embedding_dim]
    - second allreduce:  middle allreduce for QKV: [bs * seq, embedding_dim] -> [bs * seq, embedding_dim]
            - Linear (bs* seq, embedding dim) * (embedding_dim, num_heads * head_dim/tp) -> (bs*seq, num_heads * head_dim/tp) # Query projection with Col parallel
            - Linear (bs* seq, embedding dim) * (embedding_dim, kv_num_heads * head_dim/tp) -> (bs*seq, kv_num_heads * head_dim/tp) # Key
            - Linear (bs* seq, embedding dim) * (embedding_dim, kv_num_heads * head_dim/tp) -> (bs*seq, kv_num_heads * head_dim/tp) #value
            - Linear(bs*seq, num_heads * head_dim/tp) * (bs*seq, num_heads * head_dim/tp) -> (bs*seq, hidden_size/tp)  # attention1
            - Linear (bs*seq, bs*seq) * (bs * seq, num_heads * head_dim / tp) -> (bs * seq, hidden_size / tp)  # attention2
            - Linear(bs*seq, hidden_size / tp) * (hidden_size / tp, hidden_size) -> (bs * seq, hidden_size)  # output projection with Row parallel
        - reducescatter: [bs * seq, hidden_size] -> [bs * seq / tp, hidden_size]
        - allgather: [bs * seq / tp, hidden_size] -> [bs * seq , hidden_size]
    - third allreduce: last allreduce for MLP
            - Linear(bs*seq, hidden_size) * (hidden_size, intermediate_size/tp) -> (bs * seq, intermediate_size/tp) # gate up projection
            - SiluAndMul
            - Linear(bs*seq, intermediate_size/(2*tp)) * (intermediate_size/(2*tp), hidden_size) -> (bs * seq, hidden_size) # down projection with Row parallel
        - reducescatter: [bs * seq, hidden_size] -> [bs * seq / tp, hidden_size]
        - allgather: [bs * seq / tp, hidden_size] -> [bs * seq , hidden_size]
13. pipeline allgather+matmul
        - allgather: [bs * seq / tp,  hidden_size] -> [bs * seq, hidden_size]
        - Linear (bs * seq, embedding dim) * (embedding_dim, num_heads * head_dim/tp) -> (bs*seq, num_heads * head_dim/tp)
    - bs * seq / tp * hidden_size
    - Linear (bs * seq / tp, embedding dim) * (embedding_dim, num_heads * head_dim/tp) * tp
