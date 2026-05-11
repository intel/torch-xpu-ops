
# Model inference

- Attn layer: Norm + GEMM (QKV) + Attn + GEMM(output projection) + Allreduce
- MLP layer: Norm + GEMM(gate up projection) + Act + GEMM(down projection) + AllReduce

## No fusion pattern

- Attn layer: Norm + QKV GEMM + Attn + (Quant) + GEMM + Allreduce(**16bit**) + WaitTensor
- MLP layer: Norm + (Quant) + GEMM + Act + (Quant) + GEMM + AllReduce(**16bit**) + WaitTensor

## A16W16 bit fusion pattern

- allgather(16bit) + gemm(a16w16) 

- gemm(a16w16) + reducescatter(16bit)

## A16W8 bit fusion pattern

- allgather(16bit) + gemm(a16w8) 

- gemm(a16w8) + reducescatter(16bit)

## A8W8 bit fusion pattern

- allgather(8bit) + gemm(a8w8, out: 16bit) 

- gemm(a8w8, out: 16bit) + reducescatter(16bit)


## 4bit?

