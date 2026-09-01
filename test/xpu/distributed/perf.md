# BF16 internode RDMA sender optimization

Date: 2026-08-31

## Goal and test configuration

The target was to improve the effective RDMA payload bandwidth for the
33.9 MB BF16 workload from about 23 GB/s to at least 35 GB/s, preferably
40 GB/s.

All retained performance numbers use:

```bash
DTYPE=bfloat16 \
RING_GPU_IDS="0 2 4 6" \
RING_NIC_IDS="0 2 4 6" \
bash 4rank.sh
```

The effective bandwidth is the actual payload sent to the remote RDMA node
divided by `InternodeDispatchRdmaSenderKernel` duration. Each rank sends about
33,980,416 bytes. The configuration uses 4096 tokens, hidden size 4096,
top-k 8, 16 channels, 16 QPs per PE, and doorbell batch size 4.

## Baseline

| Rank | Kernel time | Effective RDMA BW |
|---|---:|---:|
| 0 | 1.462 ms | 23.16 GB/s |
| 1 | 1.463 ms | 23.17 GB/s |
| 2 | 1.461 ms | 23.19 GB/s |
| 3 | 1.455 ms | 23.31 GB/s |

Although 33.9 MB is large enough to saturate a 400 Gb/s link when transferred
as large contiguous operations, the sender kernel was dominated by packing,
per-token readiness synchronization, and small incremental puts. Reducing the
payload from FP32 to BF16 halved the useful bytes without similarly reducing
those fixed costs.

## Investigation and experiments

### Put aggregation

The coordinator previously issued a put whenever `ready_tail` advanced, often
after one BF16 token (about 8.3 KB), despite the configured 32 KiB chunk size.
It was changed to wait for at least one full chunk while copying is active and
to flush the final partial chunk after `copy_done`.

This was correct but only improved bandwidth to about 23.6 GB/s. Sweeping
16, 32, 64, and 128 KiB showed little difference after the later copy-path
optimizations. The final version keeps the 32 KiB default.

### Channel, QP, and doorbell sweeps

| Experiment | Result |
|---|---:|
| 8 channels, 128 KiB chunks | 7.36-12.47 GB/s |
| 16 channels | Best configuration |
| 18 channels | 27.82-28.07 GB/s after copy optimization |
| 24 channels | 18.17-18.27 GB/s |
| Doorbell batch 1 | No meaningful gain over batch 4 |

Fewer channels did not provide enough packing parallelism. More than 16
channels increased QP/work-group overhead and reduced performance.

### Work-group scaling attempts

- A 1024-thread copy/coordinator pair did not complete because the two large
  work-groups could not reliably co-reside.
- A 768-thread pair passed the initial correctness call but hung during the
  repeated profiled calls.
- Two copy work-groups per channel plus one coordinator also failed to make
  progress with 512- and 384-thread work-groups.

The final implementation therefore retains one 512-thread copy work-group and
one 512-thread coordinator per channel.

### Readiness protocol experiments

Publishing only one final ready count removed the per-token lock but caused
remote metadata visibility failures, even with a final device or system
release fence. The existing per-token fence/release handshake is required for
the NIC-visible staging path.

Replacing the sliding window with one acquire-scanned flag per slot passed
correctness but regressed average performance and introduced directional
variation. The sliding-window protocol was retained.

One important exception is the local RDMA node. Its coordinator branch always
waits for `copy_done`, so it does not need per-token `ready_tail` updates.
Reading its final `send_count` after `copy_done` safely removes half of the
ready-window lock operations in the two-node workload.

The coordinator now also acquires `copy_done` before its final `ready_tail`
load. This prevents a narrow race where it could observe completion after
reading a stale tail, stop sending, and then publish a newer count containing
slots that were not transmitted.

### Copy-path optimization

The routing used by this test sends almost every token to both RDMA nodes.
Previously, the copy work-group loaded and copied the same hidden row and
top-k data separately for the local and remote staging buffers.

A two-destination fast path now:

1. Allocates the local and remote slots together.
2. Loads each 128-bit source value once.
3. Stores that value to both staging destinations.
4. Writes destination-specific metadata separately.
5. Uses one subgroup barrier and device fence for both copies.
6. Publishes readiness only for the remote destination.

The original path remains as the fallback when there are not exactly two RDMA
nodes, the token does not target both nodes, or the per-channel capacity can
truncate allocations.

Intermediate results:

| Change | Effective RDMA BW |
|---|---:|
| Baseline BF16 | 23.16-23.31 GB/s |
| Full-chunk put batching | 23.60-23.63 GB/s |
| Remote-first copy order | 23.65-23.79 GB/s |
| Split subgroups by destination | 25.87-26.03 GB/s |
| Shared load with dual destination stores | 29.48-29.62 GB/s |
| Skip local readiness publication | **38.24-38.81 GB/s** |

Explicitly unrolling the dual-copy loop by four reduced performance to
37.72-38.29 GB/s, so the compiler-managed loop was retained.

## Final result

Final validation after rebuilding the shared library:

| Rank | Kernel avg (min-max) | Effective RDMA BW |
|---|---:|---:|
| 0 | 0.876 (0.870-0.887) ms | 38.67 GB/s |
| 1 | 0.873 (0.867-0.877) ms | 38.84 GB/s |
| 2 | 0.877 (0.867-0.888) ms | 38.65 GB/s |
| 3 | 0.883 (0.874-0.890) ms | 38.42 GB/s |

All four ranks passed the full sender correctness check. Compared with the
baseline, average kernel time decreased by about 40% and effective payload
bandwidth improved by about 66% (approximately 1.66x). The requested 35 GB/s
threshold is exceeded on every rank. The result remains just below 40 GB/s;
the remaining gap is dominated by remote ready-window publication, RDMA
submission/fence overhead, and the unavoidable second staging store.
