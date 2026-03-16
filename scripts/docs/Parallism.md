
batch_size, squence_length, hidden_size, k_head, v_head, head_num 


1. TP -> allreduce
2. PP -> send/recv
3. EP -> all_gatherv / reduce_scatterv
4. DP -> allreduce(no need for inference)
