<!-- Copyright 2024-2026 Intel Corporation -->
<!-- Co-authored with GitHub Copilot -->
<!-- Licensed under the Apache License, Version 2.0 -->

# XPU Nightly CI UT Fix

Agent-assisted workflow for analyzing and fixing nightly CI test failures on XPU.

## Directory Layout

```
tools/agentic_xpu/nightly_ci_fix/
├── README.md              # This file
└── run_fix.sh             # Pipeline entry point (invokes OpenCode)

.github/skills/xpu-nightly-ci-fix/
├── SKILL.md               # Full workflow definition (6 steps)
└── PRINCIPLES.md          # LLM behavioral guidelines
```

## Prerequisites

- [OpenCode](https://github.com/opencode-ai/opencode) installed and connected to an LLM
- Local PyTorch source with XPU support buildable
- `build_pytorch.env` with oneAPI paths configured (see below)

## Quick Start

```bash
# 1. Prepare CI failure report
cat > /tmp/report.md << 'EOF'
Commit: abc1234
Failures:
- test/test_torch.py::TestTorchDeviceTypeXPU::test_foo
- test/test_ops.py::TestCommonXPU::test_bar
EOF

# 2. Run the fix pipeline
bash tools/agentic_xpu/nightly_ci_fix/run_fix.sh /tmp/report.md 0603
```

## build_pytorch.env

Create `build_pytorch.env` at the repo root (or set `BUILD_ENV` env var):

```bash
export TORCH_XPU_ARCH_LIST=pvc
ONEAPI_ROOT=${ONEAPI_ROOT:-/opt/intel/oneapi}
ONEAPI_VARS=${ONEAPI_VARS:-${ONEAPI_ROOT}/setvars.sh}
ONEAPI_PTI=${ONEAPI_PTI:-${ONEAPI_ROOT}/pti/latest/env/vars.sh}

export USE_XPU=1
export USE_CUDA=0

source "${ONEAPI_VARS}" --force
source "${ONEAPI_PTI}"
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENCODE_BIN` | Path to `opencode` binary | `opencode` in `$PATH` |
| `PYTORCH_DIR` | Path to local PyTorch source | `~/pytorch` |
| `BUILD_ENV` | Path to `build_pytorch.env` | `<repo_root>/build_pytorch.env` |
| `ENV_FILE` | Path to tokens.env | `<repo_root>/tokens.env` |

## Workflow (from SKILL.md)

```
report.md (CI failure report)
      │
      ▼
run_fix.sh (pipeline entry point)
      │
      ├─► OpenCode + SKILL.md (6-step workflow)
      │       │
      │       ├─ Step 1: Parse failure report
      │       ├─ Step 2: Reproduce locally
      │       ├─ Step 3: Analyze & categorize root cause
      │       ├─ Step 4: Fix
      │       ├─ Step 5: Verify fix + lint
      │       └─ Step 6: Generate summary report
      │
      ▼
summary_<date>.md (structured fix report)
```