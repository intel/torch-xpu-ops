# Ring collective tuning notes

Fused-MoE ring collectives in `test/xpu/csrc/*.cpp`, driven from
`ring_collectives.py`. All numbers below were measured on **8x Intel B60
(Battlemage / bmg-g31, PCIe fabric)**, icpx 2025.3, PyTorch XPU + xccl,
`bfloat16`, `TOKENS_PER_RANK=2048`, `HIDDEN_SIZE=2048`, `TOPK=8`,
`NUM_EXPERTS=128`, `VEC_SIZE=8`.

**The defaults are already the best-measured configuration at 4 and 8 ranks.**
You only need the knobs below to A/B or to retune on a different fabric.

---

## Optimization: LSC write-back store on cross-PCIe push writes

The push (local-read -> remote-write) path issues 16-byte stores to a peer
GPU's buffer. Using the Intel GPU LSC intrinsic with cache control
`L1WB_L3WB` lets L3 write-combining coalesce these into larger PCIe bursts.

Direction rule (validated by measurement):
- **PUSH remote write** -> LSC `L1WB_L3WB` **store** helps.
- **PULL remote read** -> LSC `L1UC` **load** *hurts* (~13%, exposes latency);
  only the local store benefits marginally, so the pull path uses a plain
  remote load by default.

Enabled by default in all four kernels (compile-time, opt-out only — see below).

---

## Runtime environment variables

| Env var | Values | Default | Applies to | Effect |
|---|---|---|---|---|
| `RING_AGP_PUSH` | `1` / `0` | push (always, if op available) | `ring_allgather_permute` | Force push (`1`) or pull (`0`) topology. Push wins at every measured scale. |
| `RING_AGP_LSC` | `1` / `0` | `world_size > 4` | `ring_allgather_permute` push kernel | Force the LSC `L1WB_L3WB` store on the remote forward write. Auto-off at `ws<=4` (see regression note below). |
| `SYCL_CACHE_DISABLE` | `1` | unset | benchmarking only | **Set to `1` when A/B testing.** The NEO compiler cache can serve a stale kernel after an in-place `.so` rebuild and silently corrupt comparisons. |

### Compile-time opt-outs (rebuild the `.so` to disable an optimization)

| Macro | Disables | Files |
|---|---|---|
| `-DRING_NO_LSC_STORE` | LSC store (plain store) | `RingReduceScatterUnpermute.cpp`, `RingReduceScatter.cpp`, `RingAllgather.cpp` |
| `-DRING_NO_LSC_COPY` | LSC store/copy (plain) | `RingAllgatherPermute.cpp` |
| `-DRING_LSC_COPY_LSC_LOAD` | (forces LSC load on pull remote read — **not recommended**, ~13% slower) | `RingAllgatherPermute.cpp` |

---

## Measured results (cache disabled, time-sandwiched, accuracy match=True)

Times are `avg_fused` / `avg_ring` in ms. "base" = LSC off, "LSC" = default.

### RingReduceScatterUnpermute  (default: LSC store ON)
| ranks | base | LSC (default) | gain |
|---|---|---|---|
| 8 | 2.870 | **2.380** | −17.1% |
| 4 | 0.828 | **0.770** | −7.0% |

### RingReduceScatter  (default: LSC store ON)
| ranks | base | LSC (default) | gain |
|---|---|---|---|
| 8 | 2.659 | **2.154** | −19.0% |
| 4 | 0.621 | **0.526** | −15.3% |

### RingAllgather  (default: LSC store ON)
| ranks | base | LSC (default) | gain |
|---|---|---|---|
| 8 | 2.639 | **2.141** | −18.9% |
| 4 | 0.598 | **0.513** | −14.2% |

### RingAllgatherPermute  (default: PUSH + world_size-gated LSC store)

Push vs pull topology (push is the default at all scales):
| ranks | PULL | PUSH (default) | gain |
|---|---|---|---|
| 8 | 3.310 | **2.444** | −26.2% |
| 4 | 0.882 | **0.775** | −12.1% |

LSC store on the push remote forward write (gated by `world_size > 4`):
| ranks | LSC off | LSC on | best (default) |
|---|---|---|---|
| 8 | 2.657 | **2.444** | LSC **on** |
| 4 | **0.775** | 0.901 | LSC **off** |

Why LSC is gated off at low rank count: the permute push kernel emits 1 remote
forward write plus `topk`(=8) local scatter writes per element, so at 4 ranks it
is dominated by the local scatter stream. The `L1WB_L3WB` store write-back-caches
the remote line in local L1/L3, polluting cache and stealing bandwidth from that
dominant local stream (~+18% regression at ws=4). The win returns at 8 ranks where
relay/forward volume is high. (A two-pass loop split to restore write-combining
was tested and did **not** recover it, confirming the cause is cache pollution,
not interleaving.)

---

## Reproduce

```bash
cd test/xpu/distributed
# always set this when comparing rebuilt .so variants:
export SYCL_CACHE_DISABLE=1

# default (best) run:
mpirun -np 8 python benchmark_ring_allgather_permute_dist.py

# force a variant for A/B:
mpirun -np 4 -genv RING_AGP_PUSH=0 python benchmark_ring_allgather_permute_dist.py   # pull
mpirun -np 8 -genv RING_AGP_LSC=0  python benchmark_ring_allgather_permute_dist.py   # push, no LSC
```

Benchmarks: `benchmark_ring_reduce_scatter_unpermute_dist.py`,
`benchmark_ring_reduce_scatter_dist.py`, `benchmark_ring_allgather_dist.py`,
`benchmark_ring_allgather_permute_dist.py`.

## Notes / caveats
- Measured only at 4 and 8 ranks on a single PCIe box. On a different fabric
  (e.g. Xe-Link) the push/pull crossover and LSC gating threshold may shift —
  use the env overrides to retune.
- Further headroom in `RingAllgatherPermute` is in the `topk` local scatter
  writes (non-coalesced), not the remote path — out of scope for these changes.
