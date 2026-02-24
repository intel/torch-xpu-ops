1. Original workload
'''
Qwen3ForCausalLM(
  (model): Qwen3Model(
    (embed_tokens): VocabParallelEmbedding(num_embeddings=151936, embedding_dim=5120, org_vocab_size=151936, num_embeddings_padded=151936, tp_size=1)
    (layers): ModuleList(
      (0-63): 64 x Qwen3DecoderLayer(
        (self_attn): Qwen3Attention(
          (qkv_proj): QKVParallelLinear(in_features=5120, output_features=10240, bias=False, tp_size=1, gather_output=False)
          (o_proj): RowParallelLinear(input_features=8192, output_features=5120, bias=False, tp_size=1, reduce_results=True)
          (rotary_emb): RotaryEmbedding(head_size=128, rotary_dim=128, max_position_embeddings=40960, base=1000000, is_neox_style=True)
          (attn): Attention(head_size=128, num_heads=64, num_kv_heads=8, scale=0.08838834764831845, backend=IPEXAttentionImpl)
          (q_norm): RMSNorm(hidden_size=128, eps=1e-06)
          (k_norm): RMSNorm(hidden_size=128, eps=1e-06)
        )
        (mlp): Qwen2MLP(
          (gate_up_proj): MergedColumnParallelLinear(in_features=5120, output_features=51200, bias=False, tp_size=1, gather_output=False)
          (down_proj): RowParallelLinear(`input_features`=25600, output_features=5120, bias=False, tp_size=1, reduce_results=True)
          (act_fn): SiluAndMul()
        )
        (input_layernorm): RMSNorm(hidden_size=5120, eps=1e-06)
        (post_attention_layernorm): RMSNorm(hidden_size=5120, eps=1e-06)
      )
    )
    (norm): RMSNorm(hidden_size=5120, eps=1e-06)
  )
  (lm_head): ParallelLMHead(num_embeddings=151936, embedding_dim=5120, org_vocab_size=151936, num_embeddings_padded=151936, tp_size=1)
  (logits_processor): LogitsProcessor(vocab_size=151936, org_vocab_size=151936, scale=1.0, logits_as_input=False)
)
'''

2. QWen workload with TP=4
'''
Qwen3ForCausalLM(
 (model): Qwen3Model(
   (embed_tokens): VocabParallelEmbedding(num_embeddings=37984, embedding_dim=5120, org_vocab_size=151936, num_embeddings_padded=151936, tp_size=4)
        - reducescatter: [bs * seq , embedding_dim]  -> [bs * seq / tp, embedding_dim]
        - allgather: [bs * seq / tp, embedding_dim]  -> [bs * seq, embedding_dim]
   (layers): ModuleList(
     (0-63): 64 x Qwen3DecoderLayer(
       (input_layernorm): RMSNorm(hidden_size=5120, eps=1e-06)
       (self_attn): Qwen3Attention(
         (qkv_proj): QKVParallelLinear(in_features=5120, output_features=2560, bias=False, tp_size=4, gather_output=False)
         (o_proj): RowParallelLinear(input_features=2048, output_features=5120, bias=False, tp_size=4, reduce_results=True)
            - reducescatter: [bs * seq, hidden_size] -> [bs * seq / tp, hidden_size]
            - allgather: [bs * seq / tp, hidden_size] -> [bs * seq , hidden_size]
         (rotary_emb): RotaryEmbedding(head_size=128, rotary_dim=128, max_position_embeddings=40960, base=1000000, is_neox_style=True)
         (attn): Attention(head_size=128, num_heads=16, num_kv_heads=2, scale=0.08838834764831845, backend=IPEXAttentionImpl)
         (q_norm): RMSNorm(hidden_size=128, eps=1e-06)
         (k_norm): RMSNorm(hidden_size=128, eps=1e-06)
       )
       (mlp): Qwen2MLP(
         (gate_up_proj): MergedColumnParallelLinear(in_features=5120, output_features=12800, bias=False, tp_size=4, gather_output=False)
         (down_proj): RowParallelLinear(input_features=6400, output_features=5120, bias=False, tp_size=4, reduce_results=True)
            - reducescatter: [bs * seq, hidden_size] -> [bs * seq / tp, hidden_size]
            - allgather: [bs * seq / tp, hidden_size] -> [bs * seq , hidden_size]
         (act_fn): SiluAndMul()
       )
       (post_attention_layernorm): RMSNorm(hidden_size=5120, eps=1e-06)
     )
   )
   (norm): RMSNorm(hidden_size=5120, eps=1e-06)
 )
 (lm_head): ParallelLMHead(num_embeddings=37984, embedding_dim=5120, org_vocab_size=151936, num_embeddings_padded=151936, tp_size=4)
 (logits_processor): LogitsProcessor(vocab_size=151936, org_vocab_size=151936, scale=1.0, logits_as_input=False)
'''



