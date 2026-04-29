---
name: xpu-cuda-to-sycl-porting
description: >
  Port a CUDA kernel to XPU/SYCL for the torch-xpu-ops repository.
  Use when translating CUDA code to SYCL, adapting GPU kernels from
  aten/src/ATen/native/cuda/ to the XPU backend, or when the user
  mentions porting, translating, or converting CUDA to SYCL/XPU.
---

# CUDA to SYCL Porting Guide

This Skill provides the mapping rules and patterns for porting PyTorch CUDA kernels to SYCL for the XPU backend.

Read `.github/copilot-instructions.md` for full repo context before starting.

---

## Step 1: Locate the CUDA source

Find the upstream CUDA implementation in a local `pytorch/pytorch` checkout:

```
aten/src/ATen/native/cuda/OpNameKernel.cu
aten/src/ATen/native/cuda/OpName.cu
```

Read the full CUDA implementation before porting. Understand:
- Which CUDA APIs are used
- Thread/block launch configuration
- Shared memory usage
- Atomic operations
- Reduction patterns

---

## Step 2: Apply the execution model mapping

### Thread hierarchy

| CUDA | SYCL | Notes |
|------|------|-------|
| `threadIdx.x` | `item.get_local_id(0)` or `item.get_local_linear_id()` | Work-item within work-group |
| `blockIdx.x` | `item.get_group(0)` or `item.get_group_linear_id()` | Work-group index |
| `blockDim.x` | `item.get_local_range(0)` | Work-group size |
| `gridDim.x` | `item.get_group_range(0)` | Number of work-groups |
| Global thread ID | `item.get_global_id(0)` or `item.get_global_linear_id()` | Global work-item |
| `warpSize` | `sg.get_local_range()[0]` | Subgroup size (typically 16 or 32 on Intel GPUs) |
| Lane ID in warp | `sg.get_local_linear_id()` | Subgroup local ID |

### Synchronization

| CUDA | SYCL |
|------|------|
| `__syncthreads()` | `sycl::group_barrier(item.get_group())` |
| `__syncwarp()` | `sycl::group_barrier(sg)` |
| `__threadfence()` | `sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::device)` |

### Memory

| CUDA | SYCL |
|------|------|
| `__shared__ T smem[N]` | `sycl::local_accessor<T, 1> smem(N, cgh)` |
| `__constant__` | Kernel argument or `sycl::buffer` with read-only access |
| Global memory pointer | Same — raw pointer from tensor `.data_ptr<T>()` |

### Warp-level primitives

| CUDA | SYCL | torch-xpu-ops utility |
|------|------|----------------------|
| `__shfl_down_sync` | `sycl::shift_group_left(sg, val, delta)` | — |
| `__shfl_up_sync` | `sycl::shift_group_right(sg, val, delta)` | — |
| `__shfl_sync` | `sycl::select_from_group(sg, val, src_lane)` | — |
| `__shfl_xor_sync` | `sycl::permute_group_by_xor(sg, val, mask)` | — |
| `__ballot_sync` | `sycl::group_ballot(sg, pred)` | — |
| `__popc(__ballot_sync(...))` | `sycl::popcount(sycl::group_ballot(sg, pred))` | — |
| Warp reduce sum | — | `GroupReduceSumSGSizeEqualstoNumSG()` in `SYCLGroupAlgorithm.h` |
| Block reduce | — | `detail::group_reduce()` in `Reduce.h` |

### Atomics

| CUDA | SYCL | torch-xpu-ops utility |
|------|------|----------------------|
| `atomicAdd(addr, val)` | `sycl::atomic_ref<T,...>(*addr).fetch_add(val)` | `atomicAdd()` in `Atomics.h` |
| `atomicCAS` | `sycl::atomic_ref<T,...>.compare_exchange_strong()` | `atomicCAS()` in `Atomics.h` |
| `atomicMax` / `atomicMin` | `sycl::atomic_ref<T,...>.fetch_max/min(val)` | In `Atomics.h` |

**Prefer the wrappers in `Atomics.h`** — they handle half, bfloat16, and edge cases.

### Random number generation

| CUDA | SYCL / torch-xpu-ops |
|------|---------------------|
| `curand_*` / Philox engine | `Philox4x32.h` — same Philox counter-based RNG |

---

## Step 3: Translate the kernel launch

### CUDA pattern:
```cuda
my_kernel<<<blocks, threads, shared_mem, stream>>>(args...);
```

### SYCL pattern (using struct functor):

```cpp
struct MyKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    // kernel body
  }

  // Constructor with captured args
  MyKernelFunctor(T* data, int64_t n) : data_(data), n_(n) {}

 private:
  T* data_;
  int64_t n_;
};

// Launch
auto& queue = at::xpu::getCurrentSYCLQueue();
sycl::range<1> global(blocks * threads);
sycl::range<1> local(threads);
queue.submit([&](sycl::handler& cgh) {
  // local memory (if needed)
  sycl::local_accessor<T, 1> smem(shared_size, cgh);
  cgh.parallel_for(sycl::nd_range<1>(global, local),
                   MyKernelFunctor(data, n));
});
```

### For TensorIterator-based ops (simpler):

Just use `gpu_kernel()` from `Loops.h` — it handles launch configuration:

```cpp
gpu_kernel(iter, MyFunctor<scalar_t>());
```

This is the preferred pattern for pointwise/elementwise ops.

---

## Step 4: Handle common CUDA patterns

### Grid-stride loop

CUDA:
```cuda
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
```

SYCL:
```cpp
for (int64_t i = item.get_global_id(0); i < n; i += item.get_global_range(0)) {
```

### Shared memory reduction

CUDA:
```cuda
__shared__ T shared[BLOCK_SIZE];
shared[threadIdx.x] = val;
__syncthreads();
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) shared[threadIdx.x] += shared[threadIdx.x + s];
    __syncthreads();
}
```

SYCL:
```cpp
// Prefer subgroup-based reduction from SYCLGroupAlgorithm.h:
T result = GroupReduceSumSGSizeEqualstoNumSG(item, val, shared_ptr);

// Or use detail::group_reduce() from Reduce.h for TensorIterator-based reductions
```

### `cub::` primitives

| `cub::` / Thrust | torch-xpu-ops equivalent |
|-------------------|--------------------------|
| `cub::BlockReduce` | `detail::group_reduce()` in `Reduce.h` |
| `cub::BlockScan` | Manual scan or `pstl/PSTLFunctions.h` |
| `thrust::sort` | Use `pstl/PSTLFunctions.h` or manual radix sort (`SortingRadixSort.h`) |
| `thrust::inclusive_scan` | `pstl/PSTLFunctions.h` |

---

## Step 5: Key differences and pitfalls

### Subgroup size is not fixed

Intel GPUs use subgroup size 16 or 32 (varies by hardware). Do not hardcode warp size 32. Use:
```cpp
auto sg = item.get_sub_group();
int sg_size = sg.get_local_range()[0];
```

### No device-side `printf`

Remove all `printf` from device code. Use host-side debugging.

### No `__device__` / `__host__` qualifiers

SYCL uses normal C++ — functions callable from device are just regular functions or members of kernel functors. No special qualifiers needed.

### DPC++ vs NVCC template behavior

DPC++ (icpx) can be stricter about template instantiation. Common issues:
- Dependent name lookups may differ
- Some SFINAE patterns need adjustment
- `constexpr if` can sometimes help where CUDA uses `#ifdef`

### 64-bit indexing

Intel GPU drivers handle 64-bit indexing well. Prefer `int64_t` for index math over `int` to avoid overflow on large tensors.

### Work-group size

- Intel GPUs typically support max 1024 work-items per work-group
- Prefer 256 as a default work-group size for compute-bound kernels
- Use `syclMaxWorkGroupSize()` from `comm/DeviceProperties.h` to query the max

---

## Step 6: Validate the port

1. **Correctness**: Compare outputs against CPU and CUDA for matching dtypes
2. **Edge cases**: empty tensors, non-contiguous, scalar, channels-last
3. **Precision**: Check BF16/FP16 accumulation matches CUDA behavior
4. **Performance**: Run `test/microbench/` benchmarks if the op is performance-critical

---

## Quick reference: file mapping

| CUDA file location | XPU/SYCL file location |
|---------------------|------------------------|
| `aten/src/ATen/native/cuda/OpKernel.cu` | `src/ATen/native/xpu/sycl/OpKernels.cpp` |
| `aten/src/ATen/native/cuda/OpKernel.cuh` | `src/ATen/native/xpu/sycl/OpKernels.h` |
| `aten/src/ATen/native/cuda/Op.cpp` | `src/ATen/native/xpu/Op.cpp` |

---

## Checklist

- [ ] CUDA source fully read and understood
- [ ] Thread/block model translated to work-item/work-group
- [ ] Shared memory → `sycl::local_accessor`
- [ ] Atomics use `Atomics.h` wrappers
- [ ] Warp primitives → subgroup operations
- [ ] No hardcoded warp size 32
- [ ] `cub::` / Thrust replaced with repo utilities
- [ ] No `__device__` / `__host__` qualifiers
- [ ] No device-side `printf`
- [ ] Index math uses `int64_t` for large tensor safety
- [ ] Named functor struct instead of device lambda (when required)
- [ ] Correctness validated against CPU/CUDA reference
