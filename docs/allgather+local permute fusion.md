## TP+EP 场景下的 allgather + local permute 优化

## 基线方案的问题
- 基线路径：
	- allgather hidden states: [num_tokens/TP, hidden_size] -> [num_tokens, hidden_size]
	- local permute: [num_tokens, hidden_size] -> [num_tokens * topk, hidden_size]
- 主要问题：
	- 所有 TP rank 都在做同一份 local permute，计算重复
	- 如果先拆分 permute 再 allgather remap 结果，通信量会明显增大

## 优化目标
- 保持 TP+EP 的并行拓扑
- 去掉 TP rank 间重复的 local permute
- 避免最终对 remap_hidden_states 做全量 allgather
- 让输出可以被 expert owner rank 直接消费

## 核心思路（owner-based dispatch）
- 按 TP rank 进行 expert owner 划分：
	- TP=4 示例
		- RANK0: experts [0, E/4)
		- RANK1: experts [E/4, 2E/4)
		- RANK2: experts [2E/4, 3E/4)
		- RANK3: experts [3E/4, E)
- 每个 rank 只构建并计算自己拥有的 experts
- 不再对 [num_tokens*topk, hidden_size] 做最终 allgather

## 张量定义
- rank r 的输入分片：
	- hidden_shard_r: [num_tokens/TP, hidden_size]
	- topk_idx_shard_r: [num_tokens/TP, topk]
	- topk_w_shard_r: [num_tokens/TP, topk]
- owner 侧 dispatch 缓冲区：
	- remap_hidden_states_owner: [sum(tokens_for_local_experts), hidden_size]
	- remap_token_meta_owner: [sum(tokens_for_local_experts)]
		- 存储 (global_token_id, k_idx, source_rank, route_weight)

## 优化后的执行流程
1. 元数据同步（小通信）
- 在 TP ranks 间交换 topk 元数据（或压缩后的 send counts）
- 构建 send_count[peer][expert_owner]

2. 前缀和计算确定写入偏移
- owner rank 计算 write_base[peer][expert]
- 写入阶段使用固定偏移，避免全局 atomic 热点

3. 融合 dispatch kernel（通信 + permute 写入）
- 对每个 source rank 的 shard：
	- 读取 hidden_shard + topk_idx/topk_w
	- 遍历每个 token 的每个 topk 路径：
		- 查 expert owner rank
		- owner 在本地：直接写入本地 expert bucket
		- owner 不在本地：打包到对应 owner 的发送缓冲区
- 通信模式：按 owner 定向 selective all-to-all / p2p

4. owner 侧本地 expert 计算
- experts 直接消费 remap_hidden_states_owner
- 不需要全局 remap allgather

5. Combine/unpermute（DeepEP 风格）
- 仅把每个 token 需要的 partial output 返回给对应 rank
- 按 topk_w 做加权聚合
- 在每个 TP rank 恢复输出为 [num_tokens/TP, hidden_size]

## 为什么优于“拆分 permute 后再 allgather remap”
- 避免第二次大规模 allgather [num_tokens*topk, hidden_size]
- 消除所有 TP rank 的重复 permute 计算
- 通信从“全复制”变为“按 expert owner 定向传输”
- 更容易和 bucket 写入按 tile 流水重叠

## SYCL kernel 实现建议
- 使用 tile-based persistent kernel
- hidden 维使用向量化拷贝（例如 64B/128B 对齐）
- 优先使用 prefix-sum 固定偏移；仅在 overflow 路径回退 atomic
- 每个 expert 维护独立 overflow buffer 和 counter

## 最小伪代码
```text
for src_rank in TP_group:
	load hidden_shard[src_rank], topk_idx[src_rank], topk_w[src_rank]
	for token in shard:
		for k in [0, topk):
			e = topk_idx[token][k]
			owner = expert_owner(e)
			if owner == my_rank:
				dst = write_base[src_rank][e] + local_offset[token][k]
				remap_hidden_states_owner[dst] = hidden[token]
				remap_token_meta_owner[dst] = (global_token_id, k, src_rank, topk_w[token][k])
			else:
				pack_send_buffer(owner, token, k)
execute selective all-to-all / p2p sends
run local experts on remap_hidden_states_owner
combine outputs back to token owners
```

## 使用判据
- 如果流程里仍然需要对 remap tensor 做最终全量 allgather，这个优化通常不划算
- 只有当 EP owner 可以直接消费 remap 输出、且 combine 为选择性通信时，收益更稳定


## TP only 场景下的 allgather + local permute 优化

### 算法思路
在只有TP、没有EP的场景下，假设所有TP rank都已知全局token到expert的分配信息（如topk_idx），则每个rank可以预先构建remap_hidden_states（[num_tokens * topk, hidden_size]），并在allgather过程中直接将本地和远端的hidden_shard写入remap_hidden_states的正确位置。

#### 步骤：
1. **全局分配映射已知**：所有rank都已知每个token的topk expert分配（topk_idx），可预先计算remap_hidden_states的写入位置。
2. **本地写入**：本地hidden_shard按topk_idx映射写入remap_hidden_states的对应位置。
3. **allgather通信**：allgather时，远端rank的数据直接写入remap_hidden_states的目标位置，无需额外permute。
4. **后续计算**：remap_hidden_states可直接用于后续expert计算或聚合，无需再做全量permute。

### 优点
- 消除allgather后的重复permute计算
- 通信和写入一步到位，提升带宽利用率
- 逻辑简单，易于实现


# unpermute + allreduce

此时的unpermute实际上就是获取topk weights，然后按照topk weights对每个token的的topk做累加。
