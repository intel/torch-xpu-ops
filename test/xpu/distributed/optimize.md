# EP Combine 优化分析与方案 (optimize.md)

## 0. 背景 / 目标
- 文件:`test/xpu/distributed/test_deepep_combine_dist.py` + `deepep_dispatch.py` + `test/xpu/csrc/EpCombine.cpp`
- 三个需求:
  1. 测试必须走到 `ep_combine` **融合 kernel**,而不是靠多 stream overlap(原来实际走的是 `ep_combine_local_` pipeline 版本)。
  2. 用后 4 张卡:`ZE_AFFINITY_MASK=4,5,6,7 mpirun -np 4 python test_deepep_combine_dist.py`。
  3. `ep_combine` kernel 太慢,优化到 **7.5ms 以内**(参考 flashinfer `moeAlltoAllKernels.cu` 的 `moeA2ACombineKernel`)。
- 测试当前形状(被其他用户改过,沿用):`TOKENS_PER_RANK=4096, HIDDEN=7168, TOPK=8, NUM_EXPERTS=256, world_size=4`,bf16。

## 1. 待确认的关键定义(阻塞项)
"7.5ms" 指的是:
- **A) `ep_combine` 融合 kernel 本身**(当前 7.87ms,只差 ~5%,可达)。
- **B) 完整一次 combine**(push kernel + barrier + reduce + barrier ≈ 8.9ms,受 PCIe 远程写下限制约,较难)。
> 二者优化手段可能相反(见 §5 "跳过空 token")。**需用户确认 A/B 后再定方案。**

## 2. 已完成的改动
### 2.1 让测试走融合 kernel(需求 1)✅
- `deepep_dispatch.py` 里原有**两个** `deepep_owner_combine` 定义;后定义(基于 `ep_combine_local_` 的多 stream pipeline)覆盖了前定义(基于 `ep_combine` 融合 kernel)。
- 删除了后者(第 377–581 行那份 pipeline 版本),让基于 `ep_combine` 的版本生效。
- 修复其 symm-mem workspace 名字:`group_name + "_combine_recv"`(未注册,rendezvous 失败)→ 改用已注册的 `group_name`。

### 2.2 重写 `ep_combine` kernel(需求 3,核心)✅
- 原实现:flat 一线程处理一个 hidden 元素,`dst[i]=...` **逐元素 2 字节远程写**,且每个 (token,hidden_vec) 都重复做一次 ownership 扫描 → 极慢。
- 新实现(参考 flashinfer `vectorized_combine`):
  - **一个 work-group 处理一个 token**;线程 stride 遍历 hidden。
  - **宽向量 load/store**:`EpVec<scalar_t,VEC_SIZE>` 对齐结构体,单次 16B(bf16×8)事务。
  - ownership **每 token 只判一次**,收集 owned 的 (weight, src_row) 到寄存器数组。
  - float 累加,一次写出。
  - ring ordering:相邻 WG 指向不同 target rank,分散 PCIe 写。
- 结果:**137ms → 7.87ms**(~17×)。

### 2.3 融合 reduce(把 Python 的 zero+add 循环换成单 kernel)✅
- 新增 `ep_combine_reduce` kernel:一个 WG 一个 token,向量化地把 `recv[world_size]` 累加写进 `output`。
- **1.92ms → 0.74ms**。

### 2.4 去掉 pre-zero pass ✅
- kernel 加 `write_empty=1`:对无 owned expert 的 token 也写 0,因此调用方不再需要把 235MB recv buffer 预清零,省掉一次 235MB 本地写 + 一次 barrier。

### 2.5 调试开关(临时,最终要清理)
- 环境变量:`EPCOMBINE_VEC`(8/16)、`EPCOMBINE_THREADS`、`EPCOMBINE_NOCOMPUTE`(只写不算,隔离纯写带宽)。
- **注意:交付前需移除这些 env 调试开关和 `no_compute` 字段。**

## 3. 干净环境下的可靠测量(卡空闲时)
| 阶段 | median | 说明 |
|---|---|---|
| `ep_combine`(push kernel) | **7.87 ms** | 主瓶颈,远程 PCIe 写 176MB/rank |
| `ep_combine_reduce` | **0.74 ms** | 本地 |
| barrier ×2 | **~0.25 ms** | |
| **完整 combine** | **~8.9 ms** | |

线程数 128/256/512 对 push 无影响 → **纯带宽受限**。

## 4. 硬件带宽实测(symm-mem copy 微基准)
- **单 peer 远程写**:28.6 GB/s → 176MB ≈ 6.15ms(uplink 上限)。
- **3 peer 并发写**:median 9.9GB/s(3 个独立 copy 互相争用),但 **min 6.56ms**。
- **单 peer 远程读**:24.3 GB/s(7.6ms)。
- 结论:
  - **远程写 > 远程读** → push(写)模型优于 pull(读)模型,**保持 push**,不改 pull。
  - push kernel 的物理下限 ≈ **6.15–6.5ms**(176MB 经共享 PCIe uplink)。当前 7.87ms,离下限还有 **~1.4ms** 开销(本地读未完全 overlap、写空 token、pipeline 不满等)。
- VEC=16(32B 远程写)在本硬件上 **hang**(疑似 32B 远程 store fault)→ 放弃,保持 VEC=8。

## 5. 候选优化方案(按预估收益)
> 注:方案对"目标 A(kernel)"和"目标 B(完整)"的影响不同,已标注。

### (a) 跳过空 token 的远程写 + 预清零  —— 利好 A,拖累 B
- ~10% token 在某 (rank,target) 上无 owned expert((3/4)^8≈0.10)。
- `write_empty=0` 只写非空 → 远程写 176MB→~158MB,**kernel ~7.87→~7.2ms**(可能进 A 的 7.5)。
- 代价:recv buffer 必须预清零(235MB 本地写 ~0.5ms + 1 barrier),**完整流程 B 反而变慢**。

### (b) self 贡献直接写 output,push 只写远程 3 peer —— 利好 B
- 现在 push 也往自己 slot[rank] 写 59MB(本地)、reduce 再读回。
- 改:push 只写 (world_size-1) 个远程 target;self 贡献用一个本地小 kernel 直接算进 `output`;reduce 只加 3 个远程 slot。
- 省掉 self 的一趟本地往返;push 远程量不变(176MB),预估省 ~0.1–0.3ms。收益有限。

### (c) 更好地 overlap 本地读与远程写(kernel 内软件流水)—— 利好 A/B
- 目标:让 kernel 从 7.87 逼近纯写下限 ~6.5ms(§4 的 3-peer min)。
- 手段:线程内对 vec 做软件流水(先发 owned 行的本地 load,再发远程 store,交错多 vec);或调 WG/线程配比让远程写延迟被更多在途 WG 掩盖。
- 需先用 `EPCOMBINE_NOCOMPUTE` 隔离出"纯写"时间,确认 compute 是否真的没 overlap(该实验因环境争用/OOM 未取到干净值)。

### (d) reduce 与 barrier 的 overlap / 双缓冲 —— 利好 B
- recv workspace 双缓冲:去掉 reduce 后那次 barrier(下一轮写到另一 buffer)。省 ~0.13ms。
- 在真实逐层负载里,最后一个 barrier 可与后续算子 overlap。

### (e) reduce 用更宽向量 —— 利好 B
- reduce 是本地,VEC=16(local)大概率安全(hang 只在远程写)。0.74→~0.5ms,省 ~0.2ms。

### (f) 结构性:pull 单 kernel(flashinfer 完整架构)—— 已评估,**不采用**
- flashinfer combine 是 pull:block/token 远程**读** topk 贡献、寄存器 reduce、一次写 output,单 kernel 无中间 barrier。
- 但本硬件远程读(24.3)< 写(28.6),且需要把源数据放进 symm-mem;预估不优于 push。**放弃**。

## 6. 预估能否达标
- **目标 A(kernel ≤7.5ms)**:(a) 或 (c) 单独基本可达(→~7.2ms)。**可行**。
- **目标 B(完整 ≤7.5ms)**:需 (c)+(d)+(e) 叠加,把 push 压到 ~6.5ms、reduce ~0.5ms、barrier ~0.13ms → ~7.1ms。**紧,取决于 push 能否逼近纯写下限**;若 push 卡在 ~7ms 则 B 不可达(受 PCIe 物理下限限制)。

## 7. 环境风险(影响测时,非代码问题)
- 机器为**共享**,当前很不稳定:
  - `4,5,6,7` 掉了一张卡(现只枚举 3 张,`mpirun -np 4` 起不来),疑似他人 `test_xpu_random_dispatch_combine_stress.py` 跑挂;总卡数 8→7。
  - `0,1,2,3` 也出现 `OUT_OF_DEVICE_MEMORY` 和 296ms 这类失真值;显存底噪从 1.7%→18%(可能含我反复 kill 造成的 symm-mem 泄漏 + 他人 vllm serve Qwen3-32B / stress job)。
- **可靠测时需要一段干净窗口**;7.87/0.74/0.25 这些数是卡空闲时多次一致测得,可信。

## 8. 交付前清理清单
- [ ] 移除 `EPCOMBINE_VEC/THREADS/NOCOMPUTE` 调试 env 与 `no_compute` 字段(定稿后固定最优配置)。
- [ ] 删除临时脚本:`prof_combine.py`、`prof2.py`、`prof_p2p.py`、`prof_read.py`(以及 /tmp 下的)。
- [ ] `deepep_dispatch.py` 里重复的 `build_combine_rank_output_ptrs` 定义清理(有 2 份)。
- [ ] test 里 `[Summary]` 打印仍写 "kernel=pipeline",应改为反映走的是 `ep_combine` 融合 kernel。
- [ ] 确认 `HIDDEN=7168/NUM_EXPERTS=256` 是否为期望的最终测试形状(当前是被他人改动后的值)。

## 9. 现状小结
- 需求 1 ✅(走融合 kernel);需求 2 ✅(卡空闲时 4-7 跑通、精度通过);需求 3 🔧(kernel 137→7.87ms,离 7.5 目标视 A/B 定义还差 0.37 或 1.4ms)。
- **下一步:等用户确认 §1 的 A/B,并等一段干净卡窗口,再落地 §5 对应方案并最终测时。**
