# Fix Skills Refactor — Design & Status

## Context

Refactoring `.claude/skills/` to extract shared fix logic into `fix/` leaf
skills, shared between two orchestrators. Working branch: `skills_refactor/zzq`.
New design lives in `.claude/skills_refactor/` (draft only, original
`.claude/skills/` untouched).

---

## Problem Being Solved

Original skills had three issues:
1. Pipeline-mode GitHub writes scattered across leaf skills — belongs in orchestrator
2. Duplicated logic (failure categories, skip handling, run-test) across multiple skills
3. No shared layer between `issue-handler` and `nightly-ci-fix` despite identical fix logic

---

## Final Directory Structure (skills_refactor/)

```
skills_refactor/
├── skills_design.md              # design decisions
├── skills_layer_model.md         # general/pytorch-backend/torch-xpu-ops layer map
├── skills_categories.md          # category tables + caller map + orchestrator comparison
│
├── fix/                          # shared leaf skills (both orchestrators use these)
│   ├── reproduce/SKILL.md
│   ├── triage/SKILL.md
│   ├── implement/SKILL.md
│   ├── verify/SKILL.md
│   └── references/
│       ├── run-test.md           # path resolution, test commands, 3-state result
│       ├── environment-setup.md  # source build, xpu.txt pin, rebuild pitfalls
│       └── failure-categories.md # root cause taxonomy
│
├── issue-handler/                # Orchestrator A: single GitHub issue
│   ├── SKILL.md
│   ├── issue-format/SKILL.md
│   └── references/
│       └── execution-modes.md   # pipeline mode GitHub writes (issue-handler only)
│
├── nightly-ci-fix/               # Orchestrator B: batch CI failures
│   └── SKILL.md
│
├── stubs/                        # redirect markers for replaced skills
│   ├── xpu-issues-triaging/SKILL.md  → fix/triage
│   ├── issue-fix/SKILL.md            → fix/implement
│   └── test-verification/SKILL.md    → fix/reproduce + fix/verify
│
└── [all other skills copied unchanged]
    action/, pr-review/, auto-label/, release-branching/,
    xpu-build-pytorch/, oob-perf-analysis/, xpu-alignment/,
    xpu-ops-pr-creation/, skill-writer/, at-dispatch-v2/
```

---

## Key Design Decisions

### 1. Orchestrators own side effects; leaf skills own logic

Leaf skills (`fix/reproduce`, `fix/triage`, `fix/implement`, `fix/verify`):
- Accept inputs, return outputs
- Do NOT write to GitHub issue bodies or commit code
- Do NOT know which orchestrator called them

Orchestrators (`issue-handler`, `nightly-ci-fix`):
- Own pipeline sequence and branching
- Interpret leaf skill outputs, decide next steps
- Own all GitHub/git side effects

### 2. allow_skip parameter in fix/implement

Single parameter captures the behavioral difference between the two scenarios:
- `allow_skip=false` (issue-handler): never add skip decorators, must real fix
- `allow_skip=true` (nightly-ci-fix): may add @skipIfXpu + tracking issue for
  out-of-scope kernel work; stale skips must still be removed

### 3. reproduce and verify are separate skills

Different environment requirements:
- `fix/reproduce`: uses nightly wheel as fast path; three-stage degradation
- `fix/verify`: always source build (local code changes must be tested)

### 4. execution-modes.md belongs to issue-handler only

Pipeline mode GitHub writes (agent:status markers, label map, log slots) are
specific to issue-handler. nightly-ci-fix tracks state in summary_<date>.md.

---

## fix/reproduce Three-Stage Logic

```
Stage 1: Nightly wheel (fast path — most failures reproduce here)
  pip3 install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/xpu
  nightly_commit = torch.version.git_version
  compare vs ci_commit using git merge-base

  CANNOT_VERIFY (env broken)          → report, stop
  FAILED                              → REPRODUCED(stage=nightly) ✓ done
  PASSED + nightly older than CI      → stage 2
  PASSED + nightly same/newer than CI → stage 3

Stage 2: Source build at CI commit
  (only when nightly too stale to be conclusive)
  git checkout <ci_commit>, build from source

  FAILED        → REPRODUCED(stage=source_build) ✓ done
  PASSED        → stage 3
  CANNOT_VERIFY → report, stop

Stage 3: CI environment alignment
  (only when local passes but CI fails)
  align Python version, oneAPI version, test flags from CI log

  FAILED        → REPRODUCED(stage=ci_env, env_diff=...) ✓ done
  PASSED        → NOT_REPRODUCED; triage collects reason
  CANNOT_VERIFY → report, stop
```

Output: REPRODUCED / NOT_REPRODUCED / NO_REPRODUCER / CANNOT_VERIFY

---

## Issue Types and Which Stages Run

| Issue type | reproduce | triage | implement | verify |
|---|---|---|---|---|
| Has reproducer, bug confirmed | ✓ | ✓ | ✓ | ✓ |
| No reproducer (static analysis) | — | ✓ | ✓ | ✓ |
| Reproduces → already fixed | ✓ | ✓ (collect reason) | — | — |
| Nonbug (task, feature, question) | — | — | — | — |
| NEEDS_HUMAN | ✓ | ✓ | — | — |

---

## Orchestrator Comparison

| | issue-handler | nightly-ci-fix |
|---|---|---|
| Input | Single GitHub issue URL/number | Batch CI failure report |
| Scheduling | Sequential pipeline, single failure | Per-failure independent loop |
| reproduce | NO_REPRODUCER → triage only | Always has CI test command |
| implement allow_skip | false (must real fix) | true (may skip + tracking issue) |
| verify | no diff, no lint | before/after diff + lint required |
| State tracking | GitHub issue body markers (pipeline mode) | agent_space_xpu/summary_<date>.md |
| Output | Fix + PR via xpu-ops-pr-creation | Per-failure commits + summary report |
| Mode | Interactive (default) or pipeline | Interactive only |

---

## Layer Model Summary

```
                          general   pytorch-backend   torch-xpu-ops
fix/reproduce
  three-stage structure     ✓
  nightly wheel URL                                       ✓
  source build at commit              ✓
  CI env alignment          ✓

fix/triage
  analysis-only framework   ✓
  backend vs core                     ✓
  CUDA UT porting                     ✓
  XPU path rules                                          ✓

fix/implement
  engineering principles    ✓
  UT skip removal pattern             ✓
  UT skip grep patterns                                   ✓
  allow_skip semantics                ✓

fix/verify
  before/after diff         ✓
  rebuild decision                    ✓
  spin fixlint                                            ✓

references/run-test.md
  3-state result            ✓
  instantiate_device_type_tests       ✓
  test/xpu path mapping                                   ✓

references/environment-setup.md
  pip install -e .                    ✓
  xpu.txt pin workflow                                    ✓

orchestrators
  pipeline skeleton         ✓
  batch scheduling          ✓
  GitHub issue body                                       ✓
  xpu-ops-pr-creation                                     ✓
```

---

## Current Completion Status

### Done ✓

- [x] `skills_refactor/` directory structure created
- [x] `fix/reproduce/SKILL.md` — three-stage logic, four-state output
- [x] `fix/triage/SKILL.md` — root cause analysis, IMPLEMENTING/NEEDS_HUMAN
- [x] `fix/implement/SKILL.md` — allow_skip parameter, UT skip removal
- [x] `fix/verify/SKILL.md` — source build, before/after diff, lint
- [x] `fix/references/run-test.md` — path resolution, 3-state result
- [x] `fix/references/environment-setup.md` — source build, xpu.txt pin
- [x] `fix/references/failure-categories.md` — root cause taxonomy
- [x] `issue-handler/SKILL.md` — orchestrator with pipeline stages
- [x] `nightly-ci-fix/SKILL.md` — batch orchestrator with per-failure loop
- [x] `stubs/` — redirect markers for replaced skills
- [x] All other skills copied unchanged into skills_refactor/
- [x] `skills_categories.md` updated to reflect new fix/ structure
- [x] `skills_layer_model.md` — general/pytorch-backend/torch-xpu-ops breakdown
- [x] `skills_design.md` — design rationale
- [x] Modifications to original `skills/`:
  - `xpu-build-pytorch/SKILL.md` — delegates oneAPI to source-oneapi skill
  - `issue-handler/references/environment-setup.md` — same
  - `skills/fix/` — partial draft written during planning (triage + references)
  - `skills/skills_categories.md` — new, documents current state
- [x] All committed and pushed to `skills_refactor/zzq`

### Not Yet Done

- [ ] Validate new skills by actually running issue-handler or nightly-ci-fix
      against a real issue/CI report using the new fix/ skills
- [ ] Decide whether to replace original skills/ with skills_refactor/ content
      (currently both exist in parallel)
- [ ] Clean up `skills/fix/` partial draft (only has triage + references,
      missing reproduce/implement/verify — inconsistent with skills_refactor/)
- [ ] If replacing: write stub SKILL.md files in original locations pointing to
      new fix/ paths
- [ ] Update AGENTS.md or any other docs that reference old skill names

---

## Next Steps (when resuming)

1. Pick a real issue or CI failure report to validate the new design end-to-end
2. Load `skills_refactor/issue-handler/SKILL.md` or
   `skills_refactor/nightly-ci-fix/SKILL.md` and trace through the new pipeline
3. Fix any gaps found during validation
4. Once validated, decide on migration strategy (parallel → replace or keep both)
5. Clean up `skills/fix/` partial draft if keeping skills/ and skills_refactor/ separate

---

## Pending Design: No-UT-Gate Bug Flow

### Problem

Current fix flow assumes a UT gate exists. Some bugs have no UT — only an
issue description, a user script, or a CI stack trace. These need:
1. A reproducer script generated and validated first
2. The reproducer added to UT after the fix is verified

### Proposed Flow

```
Has UT gate (existing):
  reproduce → triage → implement → verify (run existing UT)

No UT gate (new):
  generate_reproducer             ← new step
      ↓ human validates reproducer is correct and minimal
  confirm bug (run reproducer)
      ↓
  triage → implement
      ↓
  verify (run reproducer)
      ↓ human validates fix is correct before adding to UT
  add_reproducer_to_ut            ← new step
      ↓
  verify (run new UT)
```

### Key Decisions Made

- **Human checkpoints required at two points:**
  1. After `generate_reproducer` — human confirms the reproducer correctly
     captures the bug and is minimal enough for a UT
  2. After `verify` (reproducer passes) — human confirms fix is correct
     before the reproducer is promoted to a permanent UT

- **Not a new orchestrator** — extend `issue-handler` with a
  `has_ut_gate: bool` field in triage output. issue-handler routes to
  `add_reproducer_to_ut` step only when `has_ut_gate=false`.

### Open Questions (to design in next session)

1. **Reproducer sources** — three cases, need to confirm if handled the same:
   - Issue body has a user script → validate and use directly
   - Issue has description only → triage generates minimal reproducer
   - CI log has stack trace but no script → construct from stack trace

2. **Where to add the UT** — options:
   - `test/repro/` (already referenced in xpu-ops-pr-creation)
   - Existing related test file (e.g. test_ops_xpu.py)
   - New test file under `test/xpu/`

3. **generate_reproducer skill** — new leaf skill under `fix/`, or part of
   triage output when `has_ut_gate=false`?

4. **add_reproducer_to_ut skill** — new leaf skill, or handled inline in
   issue-handler orchestrator?

---

## Files Changed from Original (skills/ only)

```
modified:  .claude/skills/xpu-build-pytorch/SKILL.md
           (Step 2: now says "use source-oneapi skill" instead of hardcoded path)

modified:  .claude/skills/issue-handler/references/environment-setup.md
           (Activate section: now says "use source-oneapi skill" instead of
            hardcoded setvars.sh; Build section: same)

new file:  .claude/skills/skills_categories.md
new file:  .claude/skills/fix/triage/SKILL.md         (partial draft)
new file:  .claude/skills/fix/references/environment-setup.md
new file:  .claude/skills/fix/references/failure-categories.md
```
