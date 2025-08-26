#pragma once

void cutlass_sdpa_backward(
    int batch_size,
    int num_head_q,
    int num_head_kv,
    int seq_len_q,
    int seq_len_kv,
    int head_dim_qk,
    int head_dim_v,
    const void* grad_out,
    const void* query,
    const void* key,
    const void* value,
    const void* ps,
    const void* psd,
    void* grad_query,
    void* grad_key,
    void* grad_value,
    void* dps);