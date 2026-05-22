
## MOE model pattern

- input: [local_tokens, hidden_size]
- dispatch (allgather + local permute) + GEMM1(gateup) + activation(SwiGLU) + GEMM2(down) + combine (unpermute + reducescatter)
- output: [local_tokens, hidden_size]

- parallelism: DP + (TP) + EP

## Stream Pipeline (单机版)

for loop local_tokens and then push local tokens to remote devices 
- expert parallelism: ep
- pipeline_depth: depth

 以device0为例，形成的pipeline是：
```
 - p0(stream0)：dispatch + (gemm + act + gemm) + combine
 - p1(stream1)：                    dispatch + (gemm + act + gemm) + combine
 - p2(stream0)：                               dispatch + (gemm + act + gemm) + combine
 - p3(stream1)：                                          dispatch + (gemm + act + gemm) + combine
 - p4(stream0):                                                    dispatch + (gemm + act + gemm) + combine
 ...
 ```

 某条流水存在gemm + act + gemm计算的条件是：
 - **当前device收到的tokens数量 >= tileMMA对M的最小要求。**
 
 -TBD: 这样的话，group gemm的efficiency会受到影响么？

dispatch和combine的实现尽可能用copy engine。

## Workgroup Pipelines

- Warp级别的pipeline在单个kernel内做角色分工，不同warp负责不同阶段并通过barrier衔接。

阶段划分（示例）：
1. 分发阶段：
   - `warp_idx < kNumDispatchWarps` 的warp负责dispatch相关操作（路由、拷贝、计数）。
2. 计算阶段：
   - `warp_idx >= kNumDispatchWarps` 且 `< kNumDispatchWarps + kNumMMANonEpilogueWarps` 的warp负责MMA主计算。
3. 尾部阶段：
   - `warp_idx >= kNumDispatchWarps + kNumMMANonEpilogueWarps` 的warp负责epilogue/combine相关操作。

TBD: 得分配多少workgroups才不会影响到gemm的运算么?

