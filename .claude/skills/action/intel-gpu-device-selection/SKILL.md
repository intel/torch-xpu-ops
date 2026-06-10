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

## Instructions

### Step 1: Get all Level Zero GPU devices

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

### Step 2: If there are multiple level-zero devices of step 1, you MUST ask the user first and stop.

If more than one Level Zero GPU device is found, you MUST ask the user to select the target device.

Do not continue to run any workload until the user provides a valid device index.

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
The user must provide a valid index from the listed devices.
For example: 0

If the user input is not a valid listed device index, ask again and do not continue.

### Step 3: Set ZE_AFFINITY_MASK based on the selected device
Before running any workload, always set ZE_AFFINITY_MASK to the selected Level Zero device index.

```bash
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ZE_AFFINITY_MASK=<N of the user's input>
<other commands>
```

For example, if the user selects device 0:
```bash
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ZE_AFFINITY_MASK=0
<other commands>
```

### Step 4: If only one Level Zero GPU device is found, use it by default
If only one Level Zero GPU device is found, do not ask the user.
Set ZE_AFFINITY_MASK to the index of the only available Level Zero GPU device.

Example device list:
```text
[level_zero:gpu][level_zero:0] Intel(R) oneAPI Unified Runtime over Level-Zero V2, Intel(R) Arc(TM) B580 Graphics 20.1.0 [1.14.36942]
```
In this case, use:
```bash
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ZE_AFFINITY_MASK=<detected device index>
<other commands>
```

For example, if the only device is `[level_zero:gpu][level_zero:0]`, set ZE_AFFINITY_MASK to 0:
```bash
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ZE_AFFINITY_MASK=0
```

### Step 5: If no Level Zero GPU device is found, stop

If no Level Zero GPU device is found, stop the workflow and report the issue.

Use the following message:
```text
No Level Zero GPU device was found. Please check whether the Intel GPU driver, Level Zero runtime, and oneAPI environment are correctly installed and initialized.
```
Do not continue to run any workload.
