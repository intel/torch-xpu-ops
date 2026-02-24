from global_config import *

class ModelConfig:
    hidden_size = 4096
    vocab = 128256
    intermediate_size = 14336
    head_dim = 128
    num_heads = 32
    num_kv_heads = 8
    head_ratio = 4
    block = 32
    moe = 0
    tp = 2
    batch_size = 1
    sequence_length = 8192
    data_type_size = 2

class ModelConfig_LLAMA70B:
    hidden_size = 8192
    vocab = 128256
    intermediate_size = 28672
    head_dim = 128
    num_heads = 64
    num_kv_heads = 8
    head_ratio = 4
    block = 80
    moe = 0
    tp = 8
    batch_size = 1
    sequence_length = 8192
    data_type_size = 2

# deepseek-r1-distill-Qwen-32B
class ModelConfig_QWEN32B:
    hidden_size = 5120
    vocab = 152064
    intermediate_size = 27648
    head_dim = 128
    num_heads = 40
    num_kv_heads = 8
    head_ratio = 4
    block = 64
    moe = 0
    tp = 4
    batch_size = 1
    data_type_size = 2
    sequence_length = 8196


if Config.model == "llama3-8b": #zl_debug - correct?
    hidden = 4096
    vocab = 128256
    dense_interm = 28672
    head_dim = 128
    num_heads = 32
    num_kv_heads = 8 # what does it mean?
    head_ratio = 8
    block = 32

# Llama 2 70B
if Config.model == "llama2-70b":
    #print(Config.model)
    hidden = 8192
    vocab = 32000
    dense_interm = 28672
    head_dim = 128
    num_kv_heads = 8
    head_ratio = 8
    block = 80
    '''    
    total = hidden * vocab * 2 +\
            (hidden * head_dim * num_kv_heads * head_ratio +\
            hidden * head_dim * num_kv_heads * 2 +\
            hidden * head_dim * num_kv_heads * head_ratio +\
            dense_interm * hidden * 3) * block
    print(total)
    '''

# Qwen 2.5 14B
if Config.model == "qwen2.5-14b":
    #print(Config.model)
    hidden = 5120
    vocab = 152064
    dense_interm = 13824
    head_dim = 128
    num_kv_heads = 8
    head_ratio = 5
    block = 48

# Qwen 2.5 32B
if Config.model == "qwen2.5-32b":
    #print(Config.model)
    hidden = 5120
    vocab = 152064
    dense_interm = 27648
    head_dim = 128
    num_kv_heads = 8
    head_ratio = 5
    block = 64

# Qwen 3 235B
if Config.model == "qwen3-235b":
    hidden = 4096
    vocab = 151936
    #dense_interm = 12288
    head_dim = 128
    num_kv_heads = 4
    head_ratio = 16
    moe = 1
    moe_interm = 1536
    n_routed_experts = 128
    num_experts_per_tok = 8
    shared_experts = 0
    block = 94
    dense_layers = 0
    '''
    activated = hidden * vocab * 2 +\
                (hidden * head_dim * num_kv_heads * head_ratio +\
                 hidden * head_dim * num_kv_heads * 2 +\
                 hidden * head_dim * num_kv_heads * head_ratio) * block +\
                (hidden * n_routed_experts +\
                 hidden * moe_interm * 3 * num_experts_per_tok +\
                 hidden * moe_interm * 3 * shared_experts) * (block - dense_layers) +\
                hidden * dense_interm * 3 * dense_layers
    print(activated)
    total = hidden * vocab * 2 +\
            (hidden * head_dim * num_kv_heads * head_ratio +\
            hidden * head_dim * num_kv_heads * 2 +\
            hidden * head_dim * num_kv_heads * head_ratio) * block +\
            (hidden * n_routed_experts +\
             hidden * moe_interm * 3 * (n_routed_experts + shared_experts)) *\
            (block - dense_layers) +\
            hidden * dense_interm * 3 * dense_layers
    print(total)
    '''

if Config.model == "llama4-400b":
    hidden = 5120
    vocab = 202048
    dense_interm = 16384
    moe = 1
    moe_interm = 8192
    head_dim = 128
    num_kv_heads = 8
    head_ratio = 5
    block = 48
    n_routed_experts = 128
    num_experts_per_tok = 1
    shared_experts = 1
    dense_layers = 24
    '''
    activated = hidden * vocab * 2 +\
                (hidden * head_dim * num_kv_heads * head_ratio +\
                 hidden * head_dim * num_kv_heads * 2 +\
                 hidden * head_dim * num_kv_heads * head_ratio) * block +\
                (hidden * n_routed_experts +\
                 hidden * moe_interm * 3 * num_experts_per_tok +\
                 hidden * moe_interm * 3 * shared_experts) * (block - dense_layers) +\
                hidden * dense_interm * 3 * dense_layers
    print(activated)
    total = hidden * vocab * 2 +\
            (hidden * head_dim * num_kv_heads * head_ratio +\
            hidden * head_dim * num_kv_heads * 2 +\
            hidden * head_dim * num_kv_heads * head_ratio) * block +\
            (hidden * n_routed_experts +\
             hidden * moe_interm * 3 * (n_routed_experts + shared_experts)) *\
            (block - dense_layers) +\
            hidden * dense_interm * 3 * dense_layers
    print(total)
    '''
