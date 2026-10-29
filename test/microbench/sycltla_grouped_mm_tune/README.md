# Standalone tuner for the SYCL-TLA grouped GEMM

Tunes the `torch._grouped_mm` XPU kernels in
`src/ATen/native/xpu/grouped_mm/sycltla/` without PyTorch. It compiles the
production kernel header once per candidate tile config, runs the six grouped
GEMMs of one DeepSeek-V3 16B MoE layer per EP rank, and reports the best
config and work-group count per op next to the production default. The
kernels are the production ones, so a winning config can be moved into
torch-xpu-ops unchanged.

A config build takes about a minute, against about 20 minutes to rebuild
torch.

## Build

```bash
source /path/to/oneapi/setvars.sh            # icpx 2025+
SYCL_TLA_DIR=/path/to/sycl-tla ./build.sh    # sycl-tla v0.9, the tag torch-xpu-ops pins
# options: AOT_DEVICE=bmg  CONFIGS="p256x256 p128x256"  JOBS=24  ./build.sh /out/dir
```

Each config becomes `libtune_cfg_<name>.so`, built in parallel. A config that
fails to compile is reported and skipped. Register spills reported by the
device compiler are printed next to each config.

## Run

```bash
./build/tune_grouped_mm --list                          # compiled configs
./build/tune_grouped_mm                                 # all routings, ops, configs
./build/tune_grouped_mm --routings uniform --ops w13_fwd,w13_dW \
    --configs p256x256,c256x256_pf2 --wgs 56,112 --csv out.csv
./build/tune_grouped_mm --routings '' --sweep 64,256,1024,6144   # equal rows/expert
```

| option | meaning |
|---|---|
| `--ep N`, `--tokens N` | expert-parallel degree (64 / N local experts) and tokens per rank (default 8, 8192) |
| `--routings` | `balanced,uniform,skewed,hot,sparse`: per-expert row counts from a top-6 router over EP x tokens global tokens |
| `--sweep` | extra cases with the same rows on every local expert |
| `--ops` | `w13_fwd,w2_fwd,w13_dX,w2_dX,w13_dW,w2_dW` |
| `--configs`, `--wgs` | candidates and persistent grid sizes (default: Xe cores x 1..4) |
| `--no-check` | skip the CPU reference check on sampled output rows |
| `--csv FILE` | every measurement |

The six ops use the layouts autograd produces for torchtitan's `GroupedLinear`
(`torch._grouped_mm(x, weight.transpose(-2, -1), offs)`):

| op | A | B | out | kernel |
|---|---|---|---|---|
| w13_fwd | x `[M, 2048]` | w13 `[E, 2816, 2048]` read K-contiguous | `[M, 2816]` | M-grouped, B 'R' |
| w2_fwd | h `[M, 1408]` | w2 `[E, 2048, 1408]` read K-contiguous | `[M, 2048]` | M-grouped, B 'R' |
| w13_dX | dg `[M, 2816]` | w13 read N-contiguous | `[M, 2048]` | M-grouped, B 'C' |
| w2_dX | dy `[M, 2048]` | w2 read N-contiguous | `[M, 1408]` | M-grouped, B 'C' |
| w13_dW | dg^T `[2816, M]` | x `[M, 2048]` | `[E, 2816, 2048]` | K-grouped |
| w2_dW | dy^T `[2048, M]` | h `[M, 1408]` | `[E, 2048, 1408]` | K-grouped |

## Add a candidate

Add a line to `tune_configs.h`:

```cpp
//        name          tileM tileN tileK sgM sgN prefetch
TUNE_CFG(c256x256_pf1,  256,  256,  32,   8,  4,  1)
```

Then rebuild, or build only that config with `CONFIGS=c256x256_pf1 ./build.sh`.
Sub-group tiles of 32x64 (tile / sg) are what the production configs use; the
MMA atom is `XE_DPAS_TT<8, float, bf16, bf16>`, so tileM / sgM must be a
multiple of 8 and tileN / sgN a multiple of 16.

## Move a winner into torch

1. Add the config struct next to `TileCfg256x128` / `TileCfg256x256` in
   `grouped_mm_kernel_impl.h`.
2. Give it a `Params::tile` value and dispatch it in `run_m_grouped` /
   `run_k_grouped`; change `default_tile()` / `default_num_workgroups()` in
   `grouped_mm_util.cpp` if it should become the default.
3. Check it in torch without a rebuild of the defaults using the runtime knobs
   `TORCH_XPU_GROUPED_MM_TILE` and `TORCH_XPU_GROUPED_MM_NUM_WORKGROUPS`,
   with `TORCH_XPU_GROUPED_MM_BACKEND=sycltla` and `TORCH_XPU_GROUPED_MM_LOG=1`.
4. Re-run `test/microbench/grouped_mm.dsv3_16b.py` and
   `test/test_linalg.py -k TestGroupedMM`.
