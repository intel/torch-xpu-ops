---
name: source-oneapi
description: Source the Intel oneAPI environment before running workloads that depend on oneAPI, SYCL, DPC++, or Level Zero. Use this skill when the user needs to set up oneAPI, source setvars.sh, configure the Intel compiler environment, or when icpx/sycl-ls are not found in PATH.
license: MIT
metadata:
  oneAPI: Intel oneAPI
  SYCL: Intel DPC++
  XPU: Intel GPU
---

# Source Intel oneAPI Environment

You MUST source oneAPI components individually. Do NOT use `setvars.sh`.

```bash
source <oneapi_root>/compiler/latest/env/vars.sh
source <oneapi_root>/pti/latest/env/vars.sh
source <oneapi_root>/umf/latest/env/vars.sh
source <oneapi_root>/ccl/latest/env/vars.sh
source <oneapi_root>/mpi/latest/env/vars.sh
```

## Instructions

### Step 1: Check if already sourced

```bash
which icpx 2>/dev/null && echo "${CMPLR_ROOT:-}"
```

If `CMPLR_ROOT` is set and `icpx` is found, the environment is already initialized. Skip to Step 4.

### Step 2: Find the oneAPI installation root

Check candidates in this order. Use the first one where `compiler/latest/env/vars.sh` exists:

1. `$HOME/intel/oneapi`
2. `$HOME/oneapi`
3. Other paths under `$HOME`:
   ```bash
   find "$HOME" -maxdepth 4 -type f -path "*/oneapi/compiler/latest/env/vars.sh" 2>/dev/null | sed 's|/compiler/latest/env/vars.sh||' | sort
   ```
4. `/opt/intel/oneapi`
5. Other paths under `/opt`:
   ```bash
   find /opt -maxdepth 4 -type f -path "*/oneapi/compiler/latest/env/vars.sh" 2>/dev/null | sed 's|/compiler/latest/env/vars.sh||' | sort
   ```

If **no** valid root is found, stop and ask the user for the path.

If **multiple** roots are found at the same priority level, ask the user to choose.

Do NOT fall through to `/opt` if a valid root exists under `$HOME`.

### Step 3: Source components

```bash
source <oneapi_root>/compiler/latest/env/vars.sh
source <oneapi_root>/pti/latest/env/vars.sh
source <oneapi_root>/umf/latest/env/vars.sh
source <oneapi_root>/ccl/latest/env/vars.sh
source <oneapi_root>/mpi/latest/env/vars.sh
```

Skip any component whose `env/vars.sh` does not exist. Report skipped components to the user.

### Step 4: Verify

```bash
which icpx && which sycl-ls && echo "${CMPLR_ROOT:-}"
```

If `sycl-ls` is not found, stop and report: "sycl-ls not found. Check whether the oneAPI DPC++/Compiler component is installed."

### Step 5: Run workload

Run all subsequent commands in the same shell session. Environment variables do not persist across sessions.

If the workload targets a specific GPU, use the "intel-gpu-device-selection" skill to set `ZE_AFFINITY_MASK` before running.
