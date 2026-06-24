---
name: setup
description: Set up Intel GPU unitrace profiling tool. Use this skill whenever the user mentions unitrace, Intel GPU tracing, pti-gpu tracing tool, GPU profiling with unitrace, or wants to build/install unitrace from source. Also trigger when the user asks about tracing Intel GPU workloads with unitrace, profiling SYCL/Level Zero/OpenCL applications on Intel GPUs using unitrace, or setting up pti-gpu tools. This skill handles checking if unitrace is already available, and if not, cloning and building it from source.
license: MIT
metadata:
  unitrace: Intel PTI-GPU
  oneAPI: Intel oneAPI
  XPU: Intel GPU
  LevelZero: Intel Level Zero
---

# Intel unitrace Setup

Set up `unitrace` from Intel PTI-GPU. Always check PATH first before building.

## Instructions

### Step 1: Check if unitrace is already available

```bash
which unitrace 2>/dev/null && unitrace --help > /dev/null 2>&1 && echo "UNITRACE_AVAILABLE" || echo "UNITRACE_NOT_FOUND"
```

- **If found**: Report the path. Use it directly unless the user explicitly asks to rebuild.
- **If NOT found**: Proceed to Step 2.

### Step 2: Check prerequisites

```bash
which g++ 2>/dev/null || which icpx 2>/dev/null || echo "NO_CXX_COMPILER"
cmake --version 2>/dev/null || echo "NO_CMAKE"
echo "CMPLR_ROOT=${CMPLR_ROOT:-NOT_SET}"
python3 --version 2>/dev/null || echo "NO_PYTHON"
```

Required:
- CMake 3.22+
- C++17 compiler (g++ or icpx)
- Intel oneAPI (use the "source-oneapi" skill if not initialized)
- Python 3.9+

### Step 3: Clone and build

Default location: `$HOME/.local/src/pti-gpu`.

```bash
mkdir -p "$HOME/.local/src"
cd "$HOME/.local/src"
git clone https://github.com/intel/pti-gpu.git
cd pti-gpu/tools/unitrace
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

If MPI is not available, add `-DBUILD_WITH_MPI=0`.

Build options:

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_WITH_MPI` | 1 | MPI profiling |
| `BUILD_WITH_ITT` | 1 | oneCCL/oneDNN profiling |
| `BUILD_WITH_XPTI` | 1 | SYCL/UR profiling |
| `BUILD_WITH_OPENCL` | 1 | OpenCL profiling |

### Step 4: Add to PATH

```bash
export PATH="$HOME/.local/src/pti-gpu/tools/unitrace/build:$PATH"
```

### Step 5: Verify

```bash
unitrace --help
```
