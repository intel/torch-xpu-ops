
from model_config import *

pcie_discount = 0.6957
N_uni = 31.5 * pcie_discount * 1024 * 1024 * 1024
T_fp8 = 98 * 1000 * 1000 * 1000 * 1000 * 2 / 2  # for BMG
T_bf16 = 98 * 1000 * 1000 * 1000 * 1000 / 2
mem_capacity = 24
MemoryBW = 0.450 * 1000 * 1000 * 1000 * 1000
N_lat = 0 #3 / 1000 / 1000
N_p2p_uni = N_uni

# def proj_fp8_gemm(m, k, n):
#     weights = k * n
#     overhead = weights * group_overhead / group_size
#     output = m * n
#     macs = m * n * k
#     compute_time = macs / T_fp8
#     mem_time = (weights * 1 + overhead + output * 2)  / MemoryBW  # Include write-back
#     post_process_time = ((output / 128 - 1 + 7) * 128 + output * 2) / V_bf16
#     return max(compute_time, mem_time) + post_process_time, (weights * 1 + overhead)

def proj_bf16_gemm(m, k, n):
    weights = k * n
    output = m * n
    macs = m * n * k
    compute_time = macs / T_bf16
    mem_time = (weights + output) * 2 / MemoryBW # Include write-back
    return max(compute_time, mem_time)

def proj_allgather(input_msg_size, intra_node_tp):
    allgather_time = 0
    size_p2p = input_msg_size
    print(size_p2p)
    print(N_p2p_uni)
    time_p2p = (size_p2p / N_p2p_uni + N_lat) * (intra_node_tp-1)
    print(time_p2p)
    allgather_time += time_p2p
    return allgather_time

def proj_reduce_scatter(input_msg_size, intra_node_tp):
    # Ring reduce-scatter: (n - 1) steps, each step transfers (1/n) of data
    size_p2p = input_msg_size / intra_node_tp
    time_p2p = (size_p2p / N_p2p_uni + N_lat) * (intra_node_tp - 1)
    reduction_time = input_msg_size / MemoryBW
    reducescatter_time = reduction_time + time_p2p
    return reducescatter_time

def proj_copy_from_remote(msg_size):
    return msg_size / N_uni

def proj_copy_from_local(msg_size):
    return msg_size / MemoryBW

def proj_reduction_time(input_msg_size, reduction_op="avg"):
    return input_msg_size / MemoryBW

def proj_overlap(config: Config):
    intra_node_tp = config.tp
    bs = config.batch_size
    seq = config.sequence_length
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    head_dim = config.head_dim
    num_heads = config.num_heads
    num_kv_heads = config.num_kv_heads

    # allgather + matmul in QKV
    all_gather_time_qkv = proj_copy_from_remote(bs * seq * hidden_size / intra_node_tp)
    copy_local_time = proj_copy_from_local(bs * seq * hidden_size / intra_node_tp)
    gemm_time_qkv = proj_bf16_gemm(bs * seq / intra_node_tp, head_dim * (num_heads + num_kv_heads*2) / intra_node_tp, hidden_size)
    print(f"[allgather_matmul_QKV] allgather = {all_gather_time_qkv * 1000 * 1000} copy_to_local = {copy_local_time} gemm = {gemm_time_qkv * 1000 * 1000}")

    # fallback: allgather + matmul in QKV
    all_gather_time_qkv_ref = proj_allgather(bs * seq * hidden_size / intra_node_tp, intra_node_tp)
    gemm_time_qkv_ref = proj_bf16_gemm(bs * seq, head_dim * (num_heads + num_kv_heads * 2) / intra_node_tp, hidden_size)

    # allgather + matmul in MLP
    all_gather_time_mlp = proj_copy_from_remote(bs * seq * hidden_size / intra_node_tp)
    gemm_time_mlp = proj_bf16_gemm(bs * seq / intra_node_tp, intermediate_size / intra_node_tp, hidden_size)
    print(f"[allgather_matmul_MLP] allgather = {all_gather_time_mlp * 1000 * 1000} gemm = {gemm_time_mlp * 1000 * 1000}")

    # fallback: allgather + matmul in MLP
    all_gather_time_mlp_ref = proj_allgather(bs * seq * hidden_size / intra_node_tp, intra_node_tp)
    gemm_time_mlp_ref = proj_bf16_gemm(bs * seq, intermediate_size / intra_node_tp, hidden_size)

    # matmul + reducescatter in QKV
    gemm2_time_qkv = proj_bf16_gemm(bs * seq / intra_node_tp, hidden_size, hidden_size / intra_node_tp)
    reduce_scatter_time_qkv = proj_copy_from_remote(bs * seq * hidden_size / intra_node_tp)
    print(f"[matmul_reduce_scatter_QKV] reducescatter = {reduce_scatter_time_qkv * 1000 * 1000} gemm = {gemm2_time_qkv * 1000 * 1000}")

    # fallback: matmul + reducescatter in QKV
    gemm2_time_qkv_ref = proj_bf16_gemm(bs * seq, hidden_size, hidden_size / intra_node_tp)
    reduce_scatter_time_qkv_ref = proj_reduce_scatter(bs * seq * hidden_size, intra_node_tp)

    # matmul + reducescatter in MLP
    gemm2_time_mlp = proj_bf16_gemm(bs * seq / intra_node_tp, hidden_size, intermediate_size / (2*intra_node_tp))
    reduce_scatter_time_mlp = proj_copy_from_remote(bs * seq * hidden_size / intra_node_tp)
    reduction_time = proj_reduction_time(bs * seq * hidden_size)
    print(f"[matmul_reduce_scatter_MLP] reducescatter = {reduce_scatter_time_mlp * 1000 * 1000} gemm = {gemm2_time_mlp * 1000 * 1000}")

    # fallback: matmul + reducescatter in MLP
    gemm2_time_mlp_ref = proj_bf16_gemm(bs * seq, hidden_size, intermediate_size / (2* intra_node_tp))
    reduce_scatter_time_mlp_ref = proj_reduce_scatter(bs * seq * hidden_size, intra_node_tp)

    time_all_gather_mamtul_QKV_ref = all_gather_time_qkv_ref + gemm_time_qkv_ref
    time_all_gather_mamtul_MLP_ref = all_gather_time_mlp_ref + gemm_time_mlp_ref

    time_mamtul_reduescatter_QKV_ref = gemm2_time_qkv_ref + reduce_scatter_time_qkv_ref
    time_mamtul_reduescatter_MLP_ref = gemm2_time_mlp_ref + reduce_scatter_time_mlp_ref

    if intra_node_tp == 2:
        # fusion all gather and matmul
        # stream 0 (current stream):            [gemm] [b] [fn]
        # stream 1 (backend stream): [gemm] [b] [cp]
        time_all_gather_mamtul_QKV = all_gather_time_qkv + max(all_gather_time_qkv, gemm_time_qkv) + gemm_time_qkv
        time_all_gather_mamtul_MLP = all_gather_time_mlp + max(all_gather_time_mlp, gemm_time_mlp) + gemm_time_mlp
        # stream 0 (current stream):            [gemm] [b] [fn]
        # stream 1 (backend stream): [gemm] [b] [cp]
        time_mamtul_reduescatter_QKV = gemm2_time_qkv + max(reduce_scatter_time_qkv, gemm2_time_qkv) + reduce_scatter_time_qkv + reduction_time
        time_mamtul_reduescatter_MLP = gemm2_time_mlp + max(reduce_scatter_time_mlp, gemm2_time_mlp) + reduce_scatter_time_mlp + reduction_time
    elif intra_node_tp == 4:
        # fusion all gather and matmul
        # stream 0 (current stream): [cp] [gemm] [cp] [gemm]
        # stream 1 (backend stream):      [cp] [gemm]  [cp] [gemm] [b]
        time_all_gather_mamtul_QKV = all_gather_time_qkv + max(all_gather_time_qkv, gemm_time_qkv) * 3 + gemm_time_qkv
        time_all_gather_mamtul_MLP = all_gather_time_mlp + max(all_gather_time_mlp, gemm_time_mlp) * 3 + gemm_time_mlp
        time_mamtul_reduescatter_QKV = gemm2_time_qkv + max(reduce_scatter_time_qkv, gemm2_time_qkv) * 3 + reduce_scatter_time_qkv + reduction_time
        time_mamtul_reduescatter_MLP = gemm2_time_mlp + max(reduce_scatter_time_mlp, gemm2_time_mlp) * 3 + reduce_scatter_time_mlp + reduction_time
    else:
        time_all_gather_mamtul_QKV = all_gather_time_qkv + max(all_gather_time_qkv, gemm_time_qkv) * (intra_node_tp -1) + gemm_time_qkv
        time_all_gather_mamtul_MLP = all_gather_time_mlp + max(all_gather_time_mlp, gemm_time_mlp) * (intra_node_tp -1) + gemm_time_mlp
        time_mamtul_reduescatter_QKV = gemm2_time_qkv + max(reduce_scatter_time_qkv, gemm2_time_qkv) * (intra_node_tp -1) + reduce_scatter_time_qkv + reduction_time
        time_mamtul_reduescatter_MLP = gemm2_time_mlp + max(reduce_scatter_time_mlp, gemm2_time_mlp) * (intra_node_tp -1) + reduce_scatter_time_mlp + reduction_time

    '''
    if K > 208*TP, then communication full overlapped.
    Fot TP=4,
    - allgather_matmul_MLP: gemm 2.3ms, allgather: 0.7ms
    - time_mamtul_reduescatter_MLP, gemm: 0.69ms, reducescatter: 0.675ms
    '''
    print(f"[Fused_matmul_reduce_scatter with TP={intra_node_tp}] [Model config = {config}], perf (us) compared as below: \n"
          f"[time_all_gather_matmul_QKV][M={bs * seq} N={head_dim * (num_heads + num_kv_heads*2) / intra_node_tp} K={hidden_size}] fusion = {time_all_gather_mamtul_QKV * 1000 * 1000}, fallback = {time_all_gather_mamtul_QKV_ref * 1000 * 1000} "
          f"ratio(fusion/fallback) = {time_all_gather_mamtul_QKV/time_all_gather_mamtul_QKV_ref}  "
          f"overlap ratio = {all_gather_time_qkv/gemm_time_qkv} \n"
          f"[time_all_gather_mamtul_MLP][M={bs * seq} N={ intermediate_size / intra_node_tp} K={hidden_size}] fusion = {time_all_gather_mamtul_MLP * 1000 * 1000}, fallback = {time_all_gather_mamtul_MLP_ref * 1000 * 1000} "
          f"ratio(fusion/fallback) = {time_all_gather_mamtul_MLP/time_all_gather_mamtul_MLP_ref} "
          f"overlap ratio = {all_gather_time_mlp/gemm_time_mlp} \n"
          f"[time_mamtul_reduescatter_QKV[M={bs * seq} N={hidden_size} K={hidden_size / intra_node_tp}]] fusion = {time_mamtul_reduescatter_QKV * 1000 * 1000}, fallback = {time_mamtul_reduescatter_QKV_ref * 1000 * 1000} "
          f"ratio(fusion/fallback) = {time_mamtul_reduescatter_QKV/time_mamtul_reduescatter_QKV_ref} "
          f"overlap ratio = {reduce_scatter_time_qkv/gemm2_time_qkv} \n"
          f"[time_mamtul_reduescatter_MLP][M={bs * seq} N={hidden_size} K={intermediate_size / (2*intra_node_tp)}] fusion = {time_mamtul_reduescatter_MLP * 1000 * 1000}, fallback = {time_mamtul_reduescatter_MLP_ref * 1000 * 1000} "
          f"ratio(fusion/fallback) = {time_mamtul_reduescatter_MLP/time_mamtul_reduescatter_MLP_ref}"
          f"overlap ratio = {reduce_scatter_time_mlp/gemm2_time_mlp} \n"
          f"\n")


# test code
if __name__ == '__main__':
    # config = ModelConfig_QWEN32B() #ModelConfig_LLAMA70B() #ModelConfig()
    # proj_overlap(config)
    m = 2048
    n = 12800
    k = 5120
    tp = 4
    #t1 = proj_allgather(n*k*2/tp, tp)*1000*1000
    t1 = proj_allgather(m*k*2/tp, tp) * 1000 * 1000
    t2 = proj_bf16_gemm(m, n, k)*1000*1000
    total = max(t1,t2)* (tp-1) + t2
    print(f"zl_debug allgather = {t1} gemm = {t2} total {total}")

