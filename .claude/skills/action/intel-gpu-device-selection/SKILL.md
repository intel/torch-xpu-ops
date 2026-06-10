---
name: intel-gpu-device-selection
description: Select the Intel GPU device to use when a system has multiple Intel GPU devices. Use this skill when the user wants to run a workload on Intel GPU, mentions device selection, ZE_AFFINITY_MASK, or when multiple Level Zero GPU devices are detected and one must be chosen.
license: MIT
metadata:
  XPU: Intel GPU
---

# Intel GPU device selection

This SKILL is used to determine which Intel GPU device should be used to run a workload when the system has multiple Intel GPU devices.

Many Intel client CPU platforms include an integrated Intel GPU by default. If the system is also equipped with an Intel discrete GPU, multiple Intel GPU devices may be visible through Level Zero. In this case, the user must explicitly select the target GPU device before running any workload.

This skill selects a single target device. For multi-GPU workloads that require multiple devices simultaneously, `ZE_AFFINITY_MASK` supports comma-separated indices (e.g., `0,1`), but that configuration is outside the scope of this skill.

## Instructions

### Step 1: Detect Level Zero devices

Use `sycl-ls` to list all available SYCL devices.

```bash
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
sycl-ls
```

If `sycl-ls` is not available, use the "source-oneapi" skill to set up the Intel oneAPI environment first.

Only consider devices that match the following pattern:

```text
[level_zero:gpu][level_zero:<index>]
```

If `sycl-ls` runs successfully but no lines match this pattern, stop and report:

```text
sycl-ls completed but no Level Zero GPU devices matching the expected pattern were found.
The expected pattern is: [level_zero:gpu][level_zero:<index>]

Please verify the oneAPI version and check `sycl-ls` output manually.
```

### Step 2: Branch based on device count

Count the Level Zero GPU devices found in Step 1, then follow exactly ONE of the subsections below.

#### No devices found

Stop and report the issue. Do not continue to Step 3.

```text
No Level Zero GPU device was found. Please check whether the Intel GPU driver, Level Zero runtime, and oneAPI environment are correctly installed and initialized.
```

#### Exactly one device found

Use its index as the target device. Proceed to Step 3.

#### Multiple devices found

You MUST ask the user to select the target device. Do not proceed to Step 3 until the user provides a valid device index.

Use the following format:
```
There are multiple level-zero devices:
[level-zero device list]

Please decide which device is the target device.
Input the Level Zero device index, for example: 0, 1:
```

Example:
```
There are multiple level-zero devices:
[level_zero:gpu][level_zero:0] Intel(R) oneAPI Unified Runtime over Level-Zero V2, Intel(R) Arc(TM) B580 Graphics 20.1.0 [1.14.36942]
[level_zero:gpu][level_zero:1] Intel(R) oneAPI Unified Runtime over Level-Zero V2, Intel(R) UHD Graphics 770 12.2.0 [1.14.36942]

Please decide which device is the target device.
Input the Level Zero device index, for example: 0, 1:
```

The user must provide a valid index from the listed devices (e.g., `0`).
If the input is not a valid listed device index, ask again and do not continue.

### Step 3: Set ZE_AFFINITY_MASK

Before running any workload, set ZE_AFFINITY_MASK to the target device index determined in Step 2.

```bash
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ZE_AFFINITY_MASK=<target device index>
<other commands>
```

For example, if the target device is `[level_zero:gpu][level_zero:0]`:
```bash
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ZE_AFFINITY_MASK=0
```
