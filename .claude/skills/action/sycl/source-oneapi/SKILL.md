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

This SKILL is used to source the Intel oneAPI environment before running workloads that depend on Intel oneAPI, SYCL, DPC++, Level Zero, or Intel GPU software components.

The Intel oneAPI installation path may be under the user's home directory or under `/opt`. The home directory must be checked first. If a valid oneAPI installation is found under the home directory, use it before checking `/opt`.

## Instructions

### Step 1: Check whether the oneAPI environment is already available

Before sourcing oneAPI, check whether the environment is already initialized.

Use commands such as:

```bash
which icpx || true
which icx || true
which sycl-ls || true
echo "${ONEAPI_ROOT:-}"
```

If `ONEAPI_ROOT` is already set and key tools such as `icpx`, `icx`, or `sycl-ls` are available, the oneAPI environment may already be initialized.

However, if a workload depends on a specific oneAPI environment, still continue to verify the installation path and source the expected `setvars.sh`.

### Step 2: Find `setvars.sh` with home directory priority

Search for Intel oneAPI `setvars.sh` in the following priority order.

The home directory has higher priority than `/opt`.

#### Priority 1: Common oneAPI paths under home directory

Check these paths first:

```bash
$HOME/intel/oneapi/setvars.sh
$HOME/oneapi/setvars.sh
```

#### Priority 2: Search under home directory

If the common home paths do not exist, search under the home directory:

```bash
find "$HOME" -maxdepth 5 -type f -path "*/oneapi/setvars.sh" 2>/dev/null | sort
```

#### Priority 3: Common oneAPI path under `/opt`

If no valid `setvars.sh` is found under the home directory, check:

```bash
/opt/intel/oneapi/setvars.sh
```

#### Priority 4: Search under `/opt`

If the common `/opt` path does not exist, search under `/opt`:

```bash
find /opt -maxdepth 5 -type f -path "*/oneapi/setvars.sh" 2>/dev/null | sort
```

### Step 3: Select the oneAPI environment path

Use the first valid `setvars.sh` found according to the priority order in Step 2.

The selection rule is:

1. Prefer `$HOME/intel/oneapi/setvars.sh`
2. Then prefer `$HOME/oneapi/setvars.sh`
3. Then prefer other `*/oneapi/setvars.sh` paths under `$HOME`
4. Then use `/opt/intel/oneapi/setvars.sh`
5. Then use other `*/oneapi/setvars.sh` paths under `/opt`

Do not use a `/opt` oneAPI installation if a valid home-directory oneAPI installation is found.

### Step 4: If multiple oneAPI installations are found under the same priority level, ask the user and stop

If multiple valid `setvars.sh` files are found under the same priority level, you MUST ask the user to select one.

Do not continue to run any workload until the user provides a valid selection.

Use the following format:

```text
Multiple Intel oneAPI environments were found:

[oneAPI setvars.sh path list]

Please decide which oneAPI environment should be used.
Input the index of the target oneAPI environment, for example: 0, 1:
```

Example:

```text
Multiple Intel oneAPI environments were found:

[0] /home/user/intel/oneapi/setvars.sh
[1] /home/user/workspace/intel/oneapi/setvars.sh

Please decide which oneAPI environment should be used.
Input the index of the target oneAPI environment, for example: 0, 1:
```

If the user input is not a valid listed index, ask again and do not continue.

### Step 5: Source the selected oneAPI environment

After selecting the target `setvars.sh`, source it before running any workload.

Use:

```bash
source <selected_oneapi_path>/setvars.sh
```

or:

```bash
. <selected_oneapi_path>/setvars.sh
```

Example:

```bash
source "$HOME/intel/oneapi/setvars.sh"
```

For non-interactive shell scripts, use:

```bash
#!/usr/bin/env bash
set -e

source "$HOME/intel/oneapi/setvars.sh"

<other commands>
```

### Step 6: Verify the oneAPI environment after sourcing

After sourcing oneAPI, verify that key tools are available.

Use:

```bash
which icpx || true
which icx || true
which sycl-ls || true
echo "${ONEAPI_ROOT:-}"
```

If the workload depends on Intel GPU or Level Zero, also check:

```bash
sycl-ls | grep -i level_zero || true
```

If `sycl-ls` is not found after sourcing oneAPI, stop the workflow and report the issue.

Use the following message:

```text
The Intel oneAPI environment was sourced, but sycl-ls was not found. Please check whether the oneAPI DPC++/Compiler component is installed.
```

### Step 7: If no oneAPI environment is found, ask the user and stop

If no valid `setvars.sh` is found under either the home directory or `/opt`, stop the workflow and ask the user to provide the oneAPI installation path.

Use the following message:

```text
No Intel oneAPI setvars.sh was found under the home directory or /opt.

Please provide the Intel oneAPI installation path, for example:
$HOME/intel/oneapi
/opt/intel/oneapi
```

Do not continue to run any workload until the user provides a valid path.

### Step 8: Use the sourced environment for all following commands

After oneAPI is sourced successfully, run all following commands in the same shell session.

Example:

```bash
source "$HOME/intel/oneapi/setvars.sh"
export ZE_AFFINITY_MASK=0
python test.py
```

Do not source oneAPI in a different shell session and then run the workload in another shell session, because the environment variables may not be preserved.
