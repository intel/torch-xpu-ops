---
name: unitrace-setup
description: Set up Intel GPU unitrace profiling tool. Use this skill whenever the user mentions unitrace, Intel GPU tracing, pti-gpu tracing tool, GPU profiling with unitrace, or wants to build/install unitrace from source. Also trigger when the user asks about tracing Intel GPU workloads with unitrace, profiling SYCL/Level Zero/OpenCL applications on Intel GPUs using unitrace, or setting up pti-gpu tools. This skill handles checking if unitrace is already available, and if not, cloning and building it from source.
metadata:
  unitrace: Intel PTI-GPU
  oneAPI: Intel oneAPI
  XPU: Intel GPU
  LevelZero: Intel Level Zero
---

# Intel unitrace Setup

This SKILL is used to set up `unitrace`, the Unified Tracing and Profiling Tool from Intel PTI-GPU, before running Intel oneAPI, SYCL, Level Zero, OpenCL, or Intel GPU profiling workloads.

`unitrace` may already be available in `PATH`. Always detect and reuse an existing valid `unitrace` first before searching for source trees or building from source.

## When to Use This SKILL

Use this SKILL when the user needs to:

- Check whether `unitrace` is available.
- Set up `unitrace` before profiling an Intel GPU workload.
- Build `unitrace` from the `intel/pti-gpu` source tree.
- Configure `unitrace` for Level Zero, OpenCL, SYCL, oneCCL, oneDNN, MPI, kernel timing, hardware metrics, or timeline collection.
- Run commands such as `unitrace --device-list`, `unitrace --metric-list`, or `unitrace [options] <application> [args]`.

## Core Rule

Always check whether `unitrace` is already available in `PATH` before searching for source code or building it.

If a valid `unitrace` is found in `PATH`, use it directly.

Do not rebuild `unitrace` unless:

1. `unitrace` is not found in `PATH`; or
2. the user explicitly asks to rebuild it; or
3. the existing `unitrace` is invalid or unusable.

## Setup Workflow

Follow these steps **in order**. Do NOT skip the availability check.

### Step 1: Check if unitrace is already available

Before cloning or building anything, check whether `unitrace` is already in the user's PATH or environment:

```bash
# Check if unitrace binary is directly available
which unitrace 2>/dev/null && unitrace --version 2>/dev/null || echo "UNITRACE_NOT_FOUND"
```

- **If found**: Report the path and version to the user. Ask if they want to use the existing installation or rebuild from source. If they want to use the existing one, skip to Step 5 (verification).
- **If NOT found**: Proceed to Step 2.

### Step 2: Check prerequisites

Before building, verify the environment has the required dependencies:

```bash
# Check for C++ compiler
which g++ 2>/dev/null || which icpx 2>/dev/null || echo "NO_CXX_COMPILER"

# Check CMake version (need 3.22+)
cmake --version 2>/dev/null || echo "NO_CMAKE"

# Check if Intel oneAPI environment is set up
echo "CMPLR_ROOT=${CMPLR_ROOT:-NOT_SET}"
echo "ONEAPI_ROOT=${ONEAPI_ROOT:-NOT_SET}"

# Check for Python 3.9+
python3 --version 2>/dev/null || python --version 2>/dev/null || echo "NO_PYTHON"
```

**Required dependencies:**
- CMake 3.22 or above
- C++ compiler with C++17 support (g++ or icpx)
- Intel oneAPI Base Toolkit (must be sourced, e.g., `source /opt/intel/oneapi/setvars.sh`)
- Python 3.9+

**Optional dependencies:**
- Intel MPI (for MPI profiling support)
- Matplotlib 3.8+ and Pandas 2.2.1+ (for visualization features)

If Intel oneAPI is not set up, use the "source-oneapi" skill to source the oneAPI environment.


### Step 3: Clone and build unitrace

```bash
# Clone the pti-gpu repository
cd /tmp  # or user's preferred directory
git clone https://github.com/intel/pti-gpu.git
cd pti-gpu/tools/unitrace

# Create build directory
mkdir -p build
cd build

# Configure with CMake
# Basic build (auto-detect MPI):
cmake -DCMAKE_BUILD_TYPE=Release ..

# Or with explicit options:
# cmake -DCMAKE_BUILD_TYPE=Release \
#   -DBUILD_WITH_MPI=1 \            # Enable/disable MPI support (default: 1 on Linux)
#   -DBUILD_WITH_ITT=1 \            # Enable/disable oneCCL/oneDNN support (default: 1)
#   -DBUILD_WITH_XPTI=1 \           # Enable/disable SYCL/UR support (default: 1)
#   -DBUILD_WITH_OPENCL=1 \         # Enable/disable OpenCL support (default: 1)
#   ..

# Build
make -j$(nproc)

# Optionally install to a specific path:
# cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=<install_path> ..
# make -j$(nproc)
# make install
```

**Build option reference:**

| Option | Default (Linux) | Description |
|--------|----------------|-------------|
| `BUILD_WITH_MPI` | 1 | MPI profiling support |
| `BUILD_WITH_ITT` | 1 | oneCCL/oneDNN profiling support |
| `BUILD_WITH_XPTI` | 1 | SYCL/Unified Runtime profiling support |
| `BUILD_WITH_OPENCL` | 1 | OpenCL profiling support |

If the user does NOT have MPI or does not need MPI support, use `-DBUILD_WITH_MPI=0` to avoid build errors related to missing MPI.

### Step 4: Add unitrace to PATH

After a successful build, add unitrace to the user's PATH so it can be used directly:

```bash
# If built in-tree (no make install):
export PATH=/tmp/pti-gpu/tools/unitrace/build:$PATH

# If installed to a custom prefix:
# export PATH=<install_path>/bin:$PATH

# Verify
which unitrace
```

Suggest the user add this to their `.bashrc` or shell config for persistence:
```bash
echo 'export PATH=/tmp/pti-gpu/tools/unitrace/build:$PATH' >> ~/.bashrc
```

### Step 5: Verify installation

```bash
# Quick smoke test — show help
unitrace --help

# If a GPU device is available, optionally run the test suite:
# cd /tmp/pti-gpu/tools/unitrace/build
# ctest -V
```
