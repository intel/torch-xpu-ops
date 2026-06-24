# Skills Categories

## Category A — Orchestration / Workflow

| Skill | Description | Calls | References |
|-------|-------------|-------|------------|
| `issue-handler` | End-to-end pipeline for fixing a single GitHub issue | `issue-format`, `test-verification`, `xpu-issues-triaging`, `issue-fix`, `xpu-ops-pr-creation` | `execution-modes.md` |
| `xpu-nightly-ci-fix` | Batch nightly CI failure triage and repair | `xpu-build-pytorch` | `AGENTS.md`, `reference.md` |

Orchestrators own the scheduling logic for their scenario. Leaf skills under
Category B/C are shared between both orchestrators.

---

## Category B — Fix / Code Change

| Skill | Description | Calls | References |
|-------|-------------|-------|------------|
| `xpu-issues-triaging` | Analysis-only: root cause, fix strategy, IMPLEMENTING/NEEDS_HUMAN verdict | `issue-fix`, `test-verification` | `execution-modes.md` |
| `issue-fix` | Implement a triaged fix and verify it | `test-verification`, `xpu-issues-triaging`, `xpu-ops-pr-creation` | `execution-modes.md`, `environment-setup.md` |
| `at-dispatch-v2` | Convert legacy `AT_DISPATCH_*` macros to the V2 API | — | — |

---

## Category C — Verification / Testing

| Skill | Description | Calls | References |
|-------|-------------|-------|------------|
| `test-verification` | Run a test, report PASSED / FAILED / CANNOT_VERIFY | — | `execution-modes.md`, `environment-setup.md` |

---

## Category D — PR / Code Review

| Skill | Description | Calls | References |
|-------|-------------|-------|------------|
| `pr-review` | Review XPU operator PRs | `skill-writer` (when reviewing agent files) | `review-checklist.md`, `bc-guidelines.md`, `torch-xpu-ops-review-notes.md`, `pr-submission-guidelines.md` |
| `xpu-ops-pr-creation` | Prepare branch, lint, push, and draft PR for torch-xpu-ops | `xpu-issues-triaging` | `.github/copilot-instructions.md` |

---

## Category E — Environment Setup

| Skill | Description | Calls | References |
|-------|-------------|-------|------------|
| `source-oneapi` | Find and source Intel oneAPI components individually | `intel-gpu-device-selection` | — |
| `intel-gpu-device-selection` | Select target Intel GPU via `ZE_AFFINITY_MASK` | `source-oneapi` | `scripts/l0_igpu_check.py` |
| `unitrace/setup` | Build and install Intel unitrace profiler from source | `source-oneapi` | — |
| `xpu-build-pytorch` | Configure and build PyTorch with XPU support | `source-oneapi` | `reference.md` |
| `tmux-long-tasks` | Launch and manage long-running jobs in tmux | — | — |

### Environment setup dependency chain

```
source-oneapi  ←──────────────────────────────┐
     ↑                                         │
     │ called by                               │ called by
     │                                         │
intel-gpu-device-selection   unitrace/setup   xpu-build-pytorch
                                               │
                                               ↓
                                    environment-setup.md (reference doc)
                                    delegates oneAPI step to source-oneapi,
                                    adds: venv activation, pip install -e .,
                                    xpu.txt pin workflow, rebuild pitfalls
```

`environment-setup.md` is a reference doc (not a skill) consumed by
`test-verification` and `issue-fix`. Its oneAPI sourcing step delegates to
the `source-oneapi` skill rather than duplicating the path-finding logic.

---

## Category F — Issue / CI Management

| Skill | Description | Calls | References |
|-------|-------------|-------|------------|
| `issue-format` | Classify GitHub issue as bug/nonbug, extract metadata | `xpu-issues-triaging` (next stage) | `execution-modes.md` |
| `auto-label` | Determine `disable_*` CI labels from PR file changes | — | — |
| `release-branching` | Create release branch, tracker issue, and workflow PR | — | — |

---

## Category G — Analysis / Reporting

| Skill | Description | Calls | References |
|-------|-------------|-------|------------|
| `xpu-alignment` | Scan pytorch/pytorch for upstream bugs that may affect XPU | — | `xpu-alignment-environment-setup.md`, `xpu-alignment-buckets-and-routing.md`, `xpu-alignment-report-and-issue-format.md` |
| `oob-perf-analysis` | T1/T2/R roofline reports comparing XPU vs CUDA OOB workloads | — | `methodology.md`, `inputs.md`, `per-model-report.md`, `fleet-summary.md`, `graph-consistency.md`, `insights.md`, `troubleshooting.md` |

---

## Category H — Meta

| Skill | Description | Calls | References |
|-------|-------------|-------|------------|
| `skill-writer` | Guide creation of new Agent Skills (SKILL.md files) | — | — |

---

## Caller Map (who calls whom)

| Skill | Called by |
|-------|-----------|
| `test-verification` | `issue-handler`, `issue-fix`, `xpu-issues-triaging` |
| `issue-fix` | `issue-handler` |
| `xpu-issues-triaging` | `issue-handler`, `issue-format`, `issue-fix`, `xpu-ops-pr-creation` |
| `issue-format` | `issue-handler` |
| `xpu-ops-pr-creation` | `issue-handler`, `issue-fix` |
| `source-oneapi` | `intel-gpu-device-selection`, `unitrace/setup`, `xpu-build-pytorch` |
| `intel-gpu-device-selection` | `source-oneapi` (mutual: each can trigger the other) |
| `xpu-build-pytorch` | `xpu-nightly-ci-fix` |
| `skill-writer` | `pr-review` |

---

## Layer Model

Skills fall into three reusability layers:

```
general/          skill-writer, tmux-long-tasks
                  (no project or hardware dependency)

intel-gpu/        source-oneapi, intel-gpu-device-selection, unitrace/setup
                  (Intel oneAPI / Level Zero / SYCL — no PyTorch dependency)

torch-xpu-ops/    everything else
                  (PyTorch + XPU-specific paths, CI labels, issue templates)
```

The fix scenario (Categories B + C) uses leaf skills shared across both
orchestrators (`issue-handler` and `xpu-nightly-ci-fix`). The orchestrators
differ in input format, scheduling, and output:

| | `issue-handler` | `xpu-nightly-ci-fix` |
|---|---|---|
| Input | Single GitHub issue URL/number | Batch CI failure report |
| Scheduling | Single sequential pipeline | Per-failure independent loop |
| State tracking | GitHub issue body markers | `agent_space_xpu/summary_<date>.md` |
| Output | Fix + PR draft | Multiple commits + summary report |
| Mode | Interactive (default) or pipeline | Interactive only |
