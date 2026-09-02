# Four/eight-rank RDMA + NVL token dispatch

Date: 2026-09-02

## Scope and topology

`DispatchRdmaNvl.cpp` supports two fixed two-switch layouts:

| World size | Switch 0 | Switch 1 | Transport inside a switch |
|---:|---|---|---|
| 4 | ranks 0-1 | ranks 2-3 | ISHMEM GPU IPC direct stores |
| 8 | ranks 0-3 | ranks 4-7 | ISHMEM GPU IPC direct stores |

Traffic crossing the two domains uses ISHMEM IBGDA RDMA.

A token may target multiple GPU ranks. `is_token_in_rank[token][dst]` is a
boolean destination mask, so the token is copied at most once to each selected
GPU even if multiple top-k experts belong to that GPU.

## Single-kernel algorithm

The custom op is registered as:

```text
symm_mem::dispatch_rdma_nvl
```

The op contains exactly one `queue.submit` and launches only
`DispatchRdmaNvlKernel`.

One channel corresponds to one work-group:

```text
channel = item.get_group(0)
number of work-groups = number of channels
```

The default 16-channel configuration therefore launches exactly 16
work-groups. Each work-group:

1. owns one contiguous token range;
2. scans every token in that range once;
3. iterates the token's destination bits;
4. packs one payload for each selected destination GPU;
5. writes same-switch payloads directly through IPC;
6. streams cross-switch packed chunks through RDMA;
7. publishes one count for every destination.

There is no destination-derived work-group dimension and no repeated
four-work-group scan of the same channel.

## Packed payload

The final implementation uses an AoS payload:

```text
hidden | aligned source rank | aligned top-k indices | top-k weights | padding
```

For BF16 hidden size 4096 and top-k 8:

```text
logical bytes per copy: 8292
packed bytes per copy:  8304
```

Aligned rows use cooperative 128-bit loads and stores. Unaligned rows use a
byte-strided cooperative fallback. The source-rank field is explicitly aligned
to `alignof(int32_t)`.

The hot copy path loads each hidden/top-k source element once and multicasts
that value to every selected destination payload. This avoids re-reading the
same 8 KiB hidden row separately for each destination, which is especially
important for eight ranks because the measured routing fanout is about 5.27
destinations per token.

The symmetric allocation contains:

1. per-source/per-channel receive counts;
2. `ranks_per_switch` packed staging regions, one for each possible remote
   rank lane;
3. `num_ranks` packed receive regions, indexed by source rank.

The caller receives
`recv_payload[num_ranks, capacity, bytes_per_token]` and
`recv_channel_counts[num_ranks, channels]`. Channel `c` owns the same output
slot range as its input token range. The op requires
`capacity >= num_tokens`, which guarantees that no channel can overflow or
silently drop a route.

## IPC path

The host resolves the local-switch ranks' receive payload and count pointers
with `ishmem_ptr`.

For ranks in the same configured switch group, the work-group packs directly
into the destination rank's source/channel partition. The host resolves direct
IPC pointers for self plus every other rank in the local group. A system-scope
release fence orders the payload stores before the channel count publication.

The sender's own-rank destination uses the same packed layout with a local
pointer.

## RDMA path and overlap

For a cross-switch destination, the work-group packs into the local staging
region for that remote lane.

The initial implementation packed the complete channel before starting RDMA,
which serialized GPU packing and NIC transfer. The final implementation
streams ready chunks:

1. pack tokens into the destination's contiguous staging range;
2. after 8 new tokens, issue one
   `ishmemx_putmem_nbi_work_group_qp`;
3. continue packing while the nonblocking RDMA operation is outstanding;
4. send the final partial chunk;
5. fence the QP;
6. publish the final count with
   `ishmemx_uint64_atomic_set_nbi_qp`;
7. drain outstanding operations with `ishmemx_quiet_work_group`.

Each channel owns one QP per remote rank lane:

```text
qp = channel * ranks_per_switch + destination_rank % ranks_per_switch
```

The default 16-channel configuration requires 32 QPs/PE for four ranks and
64 QPs/PE for eight ranks.

Packing all payload fields into one AoS range also reduces each RDMA chunk from
four puts (hidden, metadata, top-k indices, top-k weights) to one contiguous
put.

After `ishmem_barrier_all`, the op uses queue memcpy commands to expose the
packed receive region and count matrix to PyTorch. These are memory-copy
commands, not additional XPU kernels.

## Accuracy validation

`test_dispatch_rdma_nvl.py` builds random top-k routing over all experts. A
token can select multiple ranks, while repeated experts on the same rank are
collapsed by the boolean destination mask.

For every destination rank, source rank, and channel, the test compares:

- receive count;
- source-rank metadata;
- hidden payload;
- top-k indices;
- top-k weights;
- token order.

The following four-rank configurations passed:

| Case | Purpose |
|---|---|
| BF16, 4096 tokens, hidden 4096, top-k 8 | Full correctness and performance workload |
| 64 tokens, hidden 256 | Small multi-destination correctness |
| 64 tokens, hidden 1 | Misaligned hidden-row and metadata-alignment coverage |
| 64 tokens, capacity 80, chunk size 2 | Capacity slack and repeated streaming-chunk coverage |
| 8 tokens, 16 channels | Empty-channel count publication and fixed channel-count contract |
| 8 ranks, 32 tokens, hidden 64 | IPC across four-rank groups and four remote RDMA lanes |
| 8 ranks, BF16, 4096 tokens, hidden 4096 | Full eight-rank accuracy and performance |

The C++ op additionally requires every input/output tensor to be contiguous,
on the same XPU device, and collectively layout-compatible across ranks.

The profiler parser requires exactly one XPU kernel event per measured
iteration and rejects any kernel name other than `DispatchRdmaNvlKernel`.

## Performance tuning

Final workload:

```text
dtype:                 bfloat16
tokens per rank:       4096
hidden size:           4096
top-k:                 8
experts per rank:      64
channels:              16
RDMA chunk:            8 tokens
work-group threads:    512
QPs per PE:            32
doorbell batch size:   4
warmup / measured:     10 / 20 iterations
GPU/NIC mapping:       rank 0-3 -> GPU 0-3 / mlx5_0-3
```

The channel/chunk sweep showed:

| Channels | Chunk tokens | Kernel algorithm BW |
|---:|---:|---:|
| 16 | 8 | **49.15-49.39 GB/s** |
| 16 | 16 | 47.95-48.20 GB/s |
| 16 | 32 | 45.88-46.19 GB/s |
| 16 | 64 | 44.01-45.30 GB/s |
| 24 | 8 | 41.85-42.29 GB/s |
| 24 | 32 | 44.17-45.04 GB/s |
| 32 | 16 | 41.33-42.42 GB/s |
| 32 | 32 | 43.22-44.49 GB/s |

Sixteen channels preserve larger per-QP transfers. Eight-token chunks start
RDMA early enough to overlap with later packing without the excessive WQE and
barrier overhead of very small chunks.

## Four-rank final result

Definitions:

```text
algorithm BW =
    logical bytes for every selected (token, destination GPU) copy
    / DispatchRdmaNvlKernel time

transport contribution =
    packed IPC-peer bytes + packed RDMA bytes
    / DispatchRdmaNvlKernel time
```

Algorithm bandwidth includes self-rank dispatch copies because they are part of
the dispatch algorithm. The transport contribution excludes self copies and is
reported separately.

| Rank | Kernel avg (min-max) | Algorithm BW | IPC contribution | RDMA contribution | IPC + RDMA contribution | End-to-end algorithm BW |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 2.495 ms (2.486-2.506) | **49.26 GB/s** | 12.33 GB/s | 24.67 GB/s | 37.00 GB/s | 39.70 GB/s |
| 1 | 2.491 ms (2.485-2.500) | **49.18 GB/s** | 12.37 GB/s | 24.60 GB/s | 36.97 GB/s | 39.89 GB/s |
| 2 | 2.492 ms (2.486-2.507) | **49.33 GB/s** | 12.32 GB/s | 24.71 GB/s | 37.03 GB/s | 39.73 GB/s |
| 3 | 2.488 ms (2.480-2.500) | **49.48 GB/s** | 12.24 GB/s | 24.87 GB/s | 37.11 GB/s | 39.75 GB/s |

The requested 40 GB/s kernel algorithm-bandwidth target is exceeded on all
four ranks.

The complete op is slightly below 40 GB/s on this run because its wall time
also includes two global ISHMEM barriers, host-side collective layout checks,
custom-op/Python launch overhead, and copying the full four-source packed
receive capacity from symmetric memory into the caller's PyTorch tensor.
Those costs are outside `DispatchRdmaNvlKernel` and account for roughly
0.60-0.63 ms per iteration.

Run:

```bash
bash 4rank-rdma.sh
```

## Eight-rank result

Eight-rank configuration:

```text
switch 0:              ranks/GPU 0-3
switch 1:              ranks/GPU 4-7
dtype:                 bfloat16
tokens per rank:       4096
hidden size:           4096
top-k:                 8
experts per rank:      64
channels:              16
RDMA chunk:            8 tokens
work-group threads:    384
QPs per PE:            64
warmup / measured:     10 / 20 iterations
```

All eight ranks passed the full payload and count comparison.

| Rank | Kernel avg (min-max) | Algorithm BW | IPC contribution | RDMA contribution | IPC + RDMA contribution | End-to-end algorithm BW |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 3.908 ms (3.885-3.936) | **45.86 GB/s** | 17.12 GB/s | 23.00 GB/s | 40.12 GB/s | 35.70 GB/s |
| 1 | 3.877 ms (3.849-3.903) | **46.05 GB/s** | 17.28 GB/s | 23.03 GB/s | 40.31 GB/s | 35.61 GB/s |
| 2 | 3.873 ms (3.844-3.900) | **46.33 GB/s** | 17.39 GB/s | 23.13 GB/s | 40.52 GB/s | 35.80 GB/s |
| 3 | 3.910 ms (3.891-3.926) | **45.92 GB/s** | 17.38 GB/s | 22.94 GB/s | 40.32 GB/s | 35.81 GB/s |
| 4 | 3.902 ms (3.870-3.929) | **45.83 GB/s** | 17.27 GB/s | 22.84 GB/s | 40.11 GB/s | 35.67 GB/s |
| 5 | 3.907 ms (3.869-3.946) | **45.93 GB/s** | 17.06 GB/s | 23.08 GB/s | 40.15 GB/s | 35.79 GB/s |
| 6 | 3.899 ms (3.880-3.929) | **45.91 GB/s** | 17.14 GB/s | 23.01 GB/s | 40.16 GB/s | 35.70 GB/s |
| 7 | 3.863 ms (3.841-3.878) | **46.39 GB/s** | 17.39 GB/s | 23.15 GB/s | 40.54 GB/s | 35.75 GB/s |

The shared-load multicast optimization and work-group sweep improved the
eight-rank kernel from 41.11-42.03 GB/s to 45.83-46.39 GB/s. The best stable
configuration is 384 work-items and an 8-token RDMA chunk. Smaller chunks lose
bandwidth to extra WQEs and work-group synchronization; larger chunks delay
RDMA and reduce packing/transfer overlap. Sweeping 256, 320, 352, 384, 416,
448, 512, and 768 work-items did not produce a stable result above this range.

The fine-grained chunk sweep around the optimum produced:

| Chunk tokens | Kernel algorithm BW |
|---:|---:|
| 5 | 42.74-43.26 GB/s |
| 6 | 43.98-44.65 GB/s |
| 7 | 44.93-45.60 GB/s |
| 8 | **45.76-46.27 GB/s** |
| 9 | 44.23-45.24 GB/s |
| 10 | 43.31-44.50 GB/s |
| 12 | 43.72-44.39 GB/s |

All configurations passed the full eight-rank payload/count comparison. The
8-token result is also consistent with the longer formal run above
(45.83-46.39 GB/s), so changing the default chunk size does not improve
performance.

The kernel does not reach 50 GB/s. The remaining 3.61-4.17 GB/s gap is caused
by work that grows with rank count rather than logical byte count alone:

- the random top-k workload selects about 5.27 destinations per token at eight
  ranks, versus about 3.62 at four ranks;
- each channel services three IPC peers and four independent remote RDMA lanes,
  increasing destination stores, QP operations, chunk checks, and count
  publications;
- multicast removes repeated source loads, but every selected destination still
  requires a full 8304-byte payload store;
- the measured RDMA contribution is already about 23 GB/s while the corrected
  three-peer IPC contribution is about 17 GB/s, leaving the kernel limited by
  aggregate payload stores and per-destination control work rather than a
  single slow RDMA sender.

End-to-end bandwidth is lower because eight source regions double the packed
receive tensor copied out of symmetric memory, in addition to the barriers and
host-side checks included in the complete op wall time.

Run:

```bash
bash 8rank-rdma.sh
```
