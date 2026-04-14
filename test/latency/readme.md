# Latency Benchmark: build and run

This folder contains a 2-rank, 2-GPU D2D ping-pong latency benchmark:

- source: `pingpong_latency_d2d.cpp`
- helper headers: `symm.hpp`, `ipc.hpp`

The benchmark requires exactly 2 MPI ranks and at least 2 GPUs.

## Prerequisites

- Intel oneAPI Base Toolkit (SYCL compiler)
- Intel oneAPI MPI
- Level Zero runtime and Intel GPU driver installed

## Build

```bash
source /opt/intel/oneapi/setvars.sh
cd test/latency

icpx -O2 -fsycl -std=c++17 pingpong_latency_d2d.cpp -o pingpong_latency_d2d \
	-I"${I_MPI_ROOT}/include" \
	-L"${I_MPI_ROOT}/lib/release" -lmpi \
	-lze_loader \
	-Wl,-rpath,"${I_MPI_ROOT}/lib/release"
```

This command uses `icpx` directly and links Intel MPI from `I_MPI_ROOT`, plus Level Zero (`-lze_loader`).

## Run

Pin your GPU frequency to the max (e.g. pin B60 GPU frequency to 2400MHz)

```bash
cd test/latency
mpirun -n 2 ./pingpong_latency_d2d
```

## Expected output

You should see logs similar to:
```
[rank 1] Device: Intel(R) Arc(TM) Pro B60 Graphics
Using 2-rank D2D ping-pong via SymmMemory IPC
[rank 0] Device: Intel(R) Arc(TM) Pro B60 Graphics
GPU timer (rank0): 52.0 ns/tick
Warming up (2000 rounds)...
Measuring (100000 rounds)...
D2D round-trip mean latency (100000 rounds):
  GPU-timer mean = 2.636 us
  estimated one-way = 1.318 us
```


```
[rank 1] Device: Intel(R) Arc(TM) Pro B60 Graphics
Using 2-rank D2D ping-pong via SymmMemory IPC
[rank 0] Device: Intel(R) Arc(TM) Pro B60 Graphics
GPU timer (rank0): 52.0 ns/tick
Warming up (2000 rounds)...
Measuring (100000 rounds)...
D2D round-trip mean latency (100000 rounds):
  GPU-timer mean = 2.557 us
  estimated one-way = 1.278 us
```
