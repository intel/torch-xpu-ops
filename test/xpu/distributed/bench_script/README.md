# oneCCL Collective Benchmark (C API)

Latency benchmarks for oneCCL collectives using the C API.

## Benchmarks

| File | Collective | Description |
|------|------------|-------------|
| `bench_ccl_allreduce_latency.cpp` | AllReduce | Sum reduction across all ranks |
| `bench_ccl_allgather_latency.cpp` | AllGather | Gather from all ranks to all ranks |
| `bench_ccl_reduce_scatter_latency.cpp` | ReduceScatter | Reduce then scatter to each rank |

## Requirements

- oneAPI 2026.0+ (Intel DLE oneAPI Base Toolkit)

## Build

```bash
# Source oneAPI environment
source /opt/intel/oneapi/setvars.sh

# Build for BMG (B60) GPU
icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg" -O2 -std=c++17 \
    -I${ONEAPI_ROOT}/ccl/latest/include \
    -I${ONEAPI_ROOT}/mpi/latest/include \
    -L${ONEAPI_ROOT}/ccl/latest/lib \
    -L${ONEAPI_ROOT}/mpi/latest/lib \
    -lccl -lmpi \
    bench_ccl_allreduce_latency.cpp -o bench_ccl_allreduce

icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg" -O2 -std=c++17 \
    -I${ONEAPI_ROOT}/ccl/latest/include \
    -I${ONEAPI_ROOT}/mpi/latest/include \
    -L${ONEAPI_ROOT}/ccl/latest/lib \
    -L${ONEAPI_ROOT}/mpi/latest/lib \
    -lccl -lmpi \
    bench_ccl_allgather_latency.cpp -o bench_ccl_allgather

icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg" -O2 -std=c++17 \
    -I${ONEAPI_ROOT}/ccl/latest/include \
    -I${ONEAPI_ROOT}/mpi/latest/include \
    -L${ONEAPI_ROOT}/ccl/latest/lib \
    -L${ONEAPI_ROOT}/mpi/latest/lib \
    -lccl -lmpi \
    bench_ccl_reduce_scatter_latency.cpp -o bench_ccl_reduce_scatter
```

## Usage

### Basic Usage (Power-of-2 sizes)

```bash
export CCL_ATL_TRANSPORT=mpi
export ZE_AFFINITY_MASK=0,1,2,3  # Select GPUs

# 4-GPU AllReduce, sizes 2^12 to 2^28 (8KB to 512MB)
mpirun -n 4 ./bench_ccl_allreduce --min 12 --max 28 --warmup 20 --loop 100

# 8-GPU AllGather
mpirun -n 8 ./bench_ccl_allgather --min 12 --max 28 --warmup 20 --loop 100

# 4-GPU ReduceScatter
mpirun -n 4 ./bench_ccl_reduce_scatter --min 12 --max 28 --warmup 20 --loop 100
```

### Exact Size Mode (AllGather/ReduceScatter only)

For LLM workloads where message sizes are not power-of-2:

```bash
# Exact sizes in MB
mpirun -n 4 ./bench_ccl_allgather --sizes-mb 64,128,224,448,896 --warmup 10 --loop 50
mpirun -n 4 ./bench_ccl_reduce_scatter --sizes-mb 64,128,224,448,896,1792 --warmup 10 --loop 50

# Exact sizes in KB  
mpirun -n 4 ./bench_ccl_allgather --sizes-kb 112,224,256,512 --warmup 20 --loop 100

# Exact sizes in bf16 element count
mpirun -n 4 ./bench_ccl_allgather --sizes 33554432,67108864 --warmup 10 --loop 50
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--min` | Minimum size as log2 (e.g., 12 = 4KB) | 12 |
| `--max` | Maximum size as log2 (e.g., 28 = 512MB) | 28 |
| `--step` | Step between sizes in log2 | 1 |
| `--warmup` | Number of warmup iterations | 20 |
| `--loop` | Number of timed iterations | 100 |
| `--sizes` | Exact sizes in bf16 elements (comma-separated) | - |
| `--sizes-kb` | Exact sizes in KB (comma-separated) | - |
| `--sizes-mb` | Exact sizes in MB (comma-separated) | - |
| `--prefill-n` | Prefill buffer size in bf16 elements | 64M |
| `--prefill-reps` | Prefill computation reps | 100 |

## Output Format

```
  Size(out)         avg_us     min_us     max_us     var_us   busBW(GB/s)
  ------------  ---------- ---------- ---------- ----------  ------------
  8.00 KB             7.24       7.05       7.40       0.35         0.849
  16.00 KB           10.84      10.58      11.13       0.55         1.134
  ...
```

- **Size**: Message size (OUT for AllGather, IN for ReduceScatter)
- **avg_us**: Average latency in microseconds
- **min_us/max_us**: Min/max latency across ranks
- **var_us**: Variance (max - min)
- **busBW**: Bus bandwidth in GB/s, calculated as `(n-1)/n × size / time`

## Size Definitions

- **AllGather**: Size = OUTPUT buffer size (total gathered data)
- **ReduceScatter**: Size = INPUT buffer size (total data before scatter)
- **AllReduce**: Size = buffer size (same for input and output)

Per-rank data = Size / world_size
