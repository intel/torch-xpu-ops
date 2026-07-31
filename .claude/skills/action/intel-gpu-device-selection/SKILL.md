---
name: intel-gpu-device-selection
description: Select the Intel GPU device to use when a system has multiple Intel GPU devices. Use this skill when the user wants to run a workload on Intel GPU, mentions device selection, ZE_AFFINITY_MASK, or when multiple Level Zero GPU devices are detected and one must be chosen.
license: MIT
metadata:
  XPU: Intel GPU
---

# Intel GPU Device Selection

Select a single Intel GPU device via `ZE_AFFINITY_MASK` when multiple Level Zero GPU devices are present.

This skill does NOT set `ZE_FLAT_DEVICE_HIERARCHY`. On PVC, the default is `FLAT` (each tile is a separate device). On other platforms, the default is `COMPOSITE` (tiles merged into one device). If the user needs to override this, they must set it explicitly before running this skill.

## Instructions

### Step 1: Detect Level Zero GPU devices

```bash
sycl-ls | grep -i level_zero
```

If `sycl-ls` is not available, use the "source-oneapi" skill first.

Only consider lines matching `[level_zero:gpu][level_zero:<index>]`.

### Step 2: Select the target device

- **0 devices found**: Stop. Report: "No Level Zero GPU device was found. Check GPU driver, Level Zero runtime, and oneAPI environment."
- **1 device found**: Use its index. Proceed to Step 3.
- **Multiple devices found**: If the list contains both discrete and integrated GPUs, select the discrete GPU by default. If there are multiple discrete GPUs, ask the user to select. Do NOT proceed until a valid index is determined.

To distinguish dGPU from iGPU, run the bundled script:
```bash
python3 .claude/skills/action/intel-gpu-device-selection/scripts/l0_igpu_check.py
```
Output labels each GPU as `iGPU` or `dGPU` based on the Level Zero `ZE_DEVICE_PROPERTY_FLAG_INTEGRATED` flag.

Prompt format when user selection is needed:
```text
Multiple Level Zero GPU devices found:
<device list from sycl-ls>

Which device should be used? Enter the device index (e.g., 0):
```

### Step 3: Set ZE_AFFINITY_MASK

```bash
export ZE_AFFINITY_MASK=<selected index>
```
