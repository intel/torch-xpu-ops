# Skills Categories

## Category A — Orchestration / Workflow

| Skill | Description | Calls | References |
|-------|-------------|-------|------------|
| `issue-handler` | End-to-end pipeline for fixing a single GitHub issue | `issue-format`, `fix/reproduce`, `fix/triage`, `fix/implement`, `fix/verify`, `xpu-ops-pr-creation` | `execution-modes.md` |
| `nightly-ci-fix` | Batch nightly CI failure triage and repair | `fix/reproduce`, `fix/triage`, `fix/implement`, `fix/verify` | `AGENTS.md` |

Orchestrators own scheduling, mode handling, and all side effects (GitHub writes,
commits, PRs). Leaf skills under Category B own the core logic and are shared
between both orchestrators.

---

## Category B — Fix / Code Change (shared leaf skills)

| Skill | Description | Calls | References |
|-------|-------------|-------|------------|
| `fix/reproduce` | Verify bug exists via three-stage approach: nightly wheel → source build → CI env | — | `fix/references/run-test.md`, `fix/references/environment-setup.md` |
| `fix/triage` | Analysis-only: root cause, fix strategy, IMPLEMENTING/NEEDS_HUMAN verdict | — | `fix/references/failure-categories.md` |
| `fix/implement` | Implement a triaged fix; `allow_skip` parameter controls skip strategy | — | `fix/references/environment-setup.md`, `fix/references/failure-categories.md` |
| `fix/verify` | Verify fix via source build; optional before/after diff and lint | — | `fix/references/run-test.md`, `fix/references/environment-setup.md` |
| `at-dispatch-v2` | Convert legacy `AT_DISPATCH_*` macros to the V2 API | — | — |

### fix/ shared references

| Reference | Used by | Content |
|-----------|---------|---------|
| `fix/references/run-test.md` | `fix/reproduce`, `fix/verify` | Path resolution, test commands, PASSED/FAILED/CANNOT_VERIFY |
| `fix/references/environment-setup.md` | `fix/reproduce`, `fix/implement`, `fix/verify` | Source build, xpu.txt pin, rebuild pitfalls |
| `fix/references/failure-categories.md` | `fix/triage`, `fix/implement`, `nightly-ci-fix` | Root cause taxonomy |

### Stubs (replaced by fix/ skills)

| Stub | Replaced by |
|------|-------------|
| `stubs/xpu-issues-triaging` | `fix/triage` |
| `stubs/issue-fix` | `fix/implement` |
| `stubs/test-verification` | `fix/reproduce` + `fix/verify` |

---

## Category C — PR / Code Review

| Skill | Description | Calls | References |
|-------|-------------|-------|------------|
| `pr-review` | Review XPU operator PRs | `skill-writer` (when reviewing agent files) | `review-checklist.md`, `bc-guidelines.md`, `torch-xpu-ops-review-notes.md`, `pr-submission-guidelines.md` |
| `xpu-ops-pr-creation` | Prepare branch, lint, push, and draft PR for torch-xpu-ops | — | `.github/copilot-instructions.md` |

---

## Category D — Environment Setup

| Skill | Description | Calls | References |
|-------|-------------|-------|------------|
| `source-oneapi` | Find and source Intel oneAPI components individually | `intel-gpu-device-selection` | — |
| `intel-gpu-device-selection` | Select target Intel GPU via `ZE_AFFINITY_MASK` | `source-oneapi` | `scripts/l0_igpu_check.py` |
| `unitrace/setup` | Build and install Intel unitrace profiler from source | `source-oneapi` | — |
| `xpu-build-pytorch` | Configure and build PyTorch with XPU support | `source-oneapi` | `reference.md` |
| `tmux-long-tasks` | Launch and manage long-running jobs in tmux | — | — |

### Environment setup dependency chain

```
source-oneapi  ←─────────────────────────────────┐
     ↑                                            │
     │ called by                                  │ called by
     │                                            │
intel-gpu-device-selection   unitrace/setup   xpu-build-pytorch
                                                  │
                                                  ↓
                                      fix/references/environment-setup.md
                                      delegates oneAPI step to source-oneapi,
                                      adds: venv activation, pip install -e .,
                                      xpu.txt pin workflow, rebuild pitfalls
```

`fix/references/environment-setup.md` is consumed by `fix/reproduce`,
`fix/implement`, and `fix/verify`.

---

## Category E — Issue / CI Management

| Skill | Description | Calls | References |
|-------|-------------|-------|------------|
| `issue-format` | Classify GitHub issue as bug/nonbug, extract metadata | — | `execution-modes.md` |
| `auto-label` | Determine `disable_*` CI labels from PR file changes | — | — |
| `release-branching` | Create release branch, tracker issue, and workflow PR | — | — |

---

## Category F — Analysis / Reporting

| Skill | Description | Calls | References |
|-------|-------------|-------|------------|
| `xpu-alignment` | Scan pytorch/pytorch for upstream bugs that may affect XPU | — | `xpu-alignment-environment-setup.md`, `xpu-alignment-buckets-and-routing.md`, `xpu-alignment-report-and-issue-format.md` |
| `oob-perf-analysis` | T1/T2/R roofline reports comparing XPU vs CUDA OOB workloads | — | `methodology.md`, `inputs.md`, `per-model-report.md`, `fleet-summary.md`, `graph-consistency.md`, `insights.md`, `troubleshooting.md` |

---

## Category G — Meta

| Skill | Description | Calls | References |
|-------|-------------|-------|------------|
| `skill-writer` | Guide creation of new Agent Skills (SKILL.md files) | — | — |

---

## Caller Map

| Skill | Called by |
|-------|-----------|
| `fix/reproduce` | `issue-handler`, `nightly-ci-fix` |
| `fix/triage` | `issue-handler`, `nightly-ci-fix` |
| `fix/implement` | `issue-handler`, `nightly-ci-fix` |
| `fix/verify` | `issue-handler`, `nightly-ci-fix` |
| `issue-format` | `issue-handler` |
| `xpu-ops-pr-creation` | `issue-handler` |
| `source-oneapi` | `intel-gpu-device-selection`, `unitrace/setup`, `xpu-build-pytorch` |
| `intel-gpu-device-selection` | `source-oneapi` (mutual) |
| `skill-writer` | `pr-review` |

---

## Orchestrator Comparison

| | `issue-handler` | `nightly-ci-fix` |
|---|---|---|
| Input | Single GitHub issue URL/number | Batch CI failure report |
| Scheduling | Sequential pipeline, single failure | Per-failure independent loop |
| reproduce | May have no reproducer (→ triage only) | Always has CI test command |
| implement `allow_skip` | `false` — must real fix, no skips | `true` — may skip with tracking issue |
| verify | No before/after diff, no lint | before/after diff + lint required |
| State tracking | GitHub issue body markers (pipeline mode) | `agent_space_xpu/summary_<date>.md` |
| Output | Fix + PR via xpu-ops-pr-creation | Per-failure commits + summary report |
| Mode | Interactive (default) or pipeline | Interactive only |

---

## Layer Model

```
general/      skill-writer, tmux-long-tasks
              (no project or hardware dependency)

intel-gpu/    source-oneapi, intel-gpu-device-selection, unitrace/setup
              (Intel oneAPI / Level Zero — no PyTorch dependency)

torch-xpu-ops/  everything else
                (PyTorch + XPU-specific paths, CI labels, issue templates)
```

Within `torch-xpu-ops/`, the fix skills have an internal layer structure —
see `skills_layer_model.md` for the full general / pytorch-backend /
torch-xpu-ops breakdown per skill.
