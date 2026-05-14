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

### 伪代码示例
```python
# 假设：
# - hidden_shard: [num_tokens/TP, hidden_size]
# - topk_idx: [num_tokens, topk]  # 全局已知
# - world_size = TP
# - rank: 当前TP rank
# - remap_hidden_states: [num_tokens * topk, hidden_size]

# 1. 本地准备待发送数据
send_buffers = []
for peer in range(world_size):
	# 计算需要发给peer的数据及其目标位置
	peer_indices = ... # 需要发给peer的token在本地shard中的索引
	peer_targets = ... # 这些token在remap_hidden_states中的目标位置
	send_buffers.append((peer, hidden_shard[peer_indices], peer_targets))

# 2. allgather通信并直接写入remap_hidden_states
for peer, data, targets in send_buffers:
	# 发送本地数据，接收远端数据
	recv_data = allgather(data, peer)
	# 直接写入remap_hidden_states
	remap_hidden_states[targets] = recv_data

# 3. remap_hidden_states可直接用于后续expert计算
```

### 注意事项
- remap_hidden_states的写入偏移需全局一致（可用prefix-sum等方式预分配）
- 通信和写入顺序需保证一致性，避免写入冲突
- 稀疏分布或未分配token需处理空洞或padding

### Python 实现可行性（按你的 4 点）
结论：可行，但更准确地说是“P2P 接收 + 本地 scatter 写入 remap”，而不是传统语义下的一次 allgather 直接 remote read。Python 侧可以先做原型验证，再把热点路径下沉到 C++/SYCL kernel。

#### 与你的 4 点一一对应
1. 先放本地数据到 remap：可行。先写本地 shard 对应的 routes，减少后续等待。
2. 每个 expert 最多 `max_num_tokens`：可行，但必须做容量保护；若超限需要 overflow 路径或提前截断策略。
3. remap 由 PyTorch symmetric memory 创建：可行。把 remap 当作本地可写目标缓冲区即可。
4. 两个 stream 交替：可行。使用 ping-pong 临时 buffer（`tmp[0]`/`tmp[1]`），`stream[0]` 与 `stream[1]`交替执行“recv -> scatter”。

#### 关键风险与约束
- 写入冲突：同一个 `(expert, slot)` 只能被唯一 token-route 占用，必须有确定性 slot 规划（prefix-sum 或预计算 idx）。
- 本方案不做 overflow 检查：前提是 `route_slot_global` 已保证落在 `[0, max_num_tokens)`。
- 通信原语选择：这里按 symmetric memory remote-get 来写，不再用 irecv 语义。
- stream 同步：通信完成事件必须显式串到 scatter stream，避免读到未完成 recv 的临时 buffer。

### Python 原型骨架（双 stream 交替，工程化版本）
下面这版刻意把“通信后端”抽象掉，只保留你关心的核心：
- 先写本地 shard 到 remap
- 两个 stream ping-pong 接收远端 shard 到 `tmp`
- 接收完成后在同一 lane 上 scatter 到 remap

```python
import torch


class SymmetricMemoryHandle:
	"""对接 PyTorch symmetric memory 的句柄。

	remote_get 语义：
	- 从 peer rank 的 symmetric memory 源地址读取 rows 行
	- 直接 DMA 到本地 dst
	- 在给定 stream 上排队执行
	"""

	def remote_get(self, peer_rank: int, dst: torch.Tensor, src_row_start: int, rows: int, stream):
		raise NotImplementedError


def _stream_api(device):
	if device.type == "cuda":
		return torch.cuda.Stream, torch.cuda.Event, torch.cuda.stream, torch.cuda.current_stream
	return torch.xpu.Stream, torch.xpu.Event, torch.xpu.stream, torch.xpu.current_stream


def _scatter_routes_to_remap(buf, global_ids, route_expert, route_slot, remap):
	# buf: [n_tokens_from_src, hidden]
	topk = route_expert.shape[1]
	for k in range(topk):
		e = route_expert[global_ids, k]   # [n]
		s = route_slot[global_ids, k]     # [n]
		remap[e, s, :].copy_(buf)


def build_remap_tp_only_pingpong(
	hidden_local,                 # [tokens_local, hidden]
	route_expert_global,          # [num_tokens, topk]
	route_slot_global,            # [num_tokens, topk]，预计算唯一slot
	max_num_tokens,
	num_experts,
	rank,
	world,
	device,
	symm_mem: SymmetricMemoryHandle,
):
	tokens_local, hidden = hidden_local.shape
	topk = route_expert_global.shape[1]

	# symmetric memory remap: [expert, max_num_tokens, hidden]
	remap = torch.empty(
		(num_experts, max_num_tokens, hidden),
		device=device,
		dtype=hidden_local.dtype,
	)
	remap.zero_()

	# 1) 先写本地 token routes（简单 copy，不做额外检查）
	local_gids = torch.arange(rank * tokens_local, (rank + 1) * tokens_local, device=device)
	for k in range(topk):
		e = route_expert_global[local_gids, k]
		s = route_slot_global[local_gids, k]
		remap[e, s, :].copy_(hidden_local)

	# 2) 双 stream ping-pong: remote_get(tmp[lane]) -> scatter(remap)
	Stream, Event, stream_ctx, current_stream = _stream_api(device)
	lanes = [Stream(device=device), Stream(device=device)]
	done = [Event(), Event()]
	tmp = [
		torch.empty((tokens_local, hidden), device=device, dtype=hidden_local.dtype),
		torch.empty((tokens_local, hidden), device=device, dtype=hidden_local.dtype),
	]

	for peer in range(world):
		if peer == rank:
			continue
		lane = peer & 1

		# 复用 tmp[lane] 之前，先等上一轮 lane 完成
		current_stream(device).wait_event(done[lane])

		with stream_ctx(lanes[lane]):
			symm_mem.remote_get(
				peer_rank=peer,
				dst=tmp[lane],
				src_row_start=0,
				rows=tokens_local,
				stream=lanes[lane],
			)

			gids = torch.arange(peer * tokens_local, (peer + 1) * tokens_local, device=device)
			_scatter_routes_to_remap(
				tmp[lane],
				gids,
				route_expert_global,
				route_slot_global,
				remap,
			)
			done[lane].record(lanes[lane])

	# 3) 收尾同步，保证 remap 可被后续 kernel 直接消费
	current_stream(device).wait_event(done[0])
	current_stream(device).wait_event(done[1])
	return remap
```

> 注：这里假设 symmetric memory 的 `remote_get` 在目标 stream 上是有序执行的；若你的实现是异步 work-handle 语义，则在 lane 内补一个 wait 再 scatter。

### 原型到高性能实现的迁移建议
- Python 原型先验证正确性：slot 唯一性、overflow 比例、输出一致性。
- 热点阶段（recv+scatter）建议融合为单一 C++/SYCL kernel，减少 Python 调度开销。
- 若通信库支持，优先使用 all-to-all(v) + 固定偏移批量写入，降低逐 peer 循环开销。
