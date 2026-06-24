# Skills Layer Model

## Layer definitions

- **general** — no project or hardware dependency; reusable by any software project
- **pytorch-backend** — any PyTorch backend (ROCm, MPS, etc.); depends on
  PyTorch internals but not XPU-specific
- **torch-xpu-ops** — specific to intel/torch-xpu-ops and XPU hardware

## Layer map

```
                                general   pytorch-backend   torch-xpu-ops
fix/reproduce
  three-stage structure           ✓
  nightly wheel URL                                              ✓
  source build at CI commit                   ✓
  CI env alignment                ✓

fix/triage
  analysis-only framework         ✓
  backend vs core decision                    ✓
  CUDA UT porting (Step 3b)                   ✓
  XPU path rules                                                 ✓

fix/implement
  engineering principles          ✓
  UT skip removal pattern                     ✓
  UT skip grep patterns                                          ✓
  allow_skip semantics                        ✓

fix/verify
  before/after diff               ✓
  rebuild decision                            ✓
  spin fixlint                                                   ✓

fix/references/run-test.md
  three-state result              ✓
  instantiate_device_type_tests               ✓
  test/xpu path mapping                                          ✓
  CI metadata format                                             ✓

fix/references/environment-setup.md
  pip install -e . command                    ✓
  PCH cache                                   ✓
  TORCH_XPU_ARCH_LIST                                            ✓
  xpu.txt pin workflow                                           ✓

fix/references/failure-categories.md
  all categories                                                 ✓

issue-handler
  pipeline skeleton               ✓
  interactive/pipeline mode       ✓
  GitHub issue body markers                                      ✓
  xpu-ops-pr-creation handoff                                    ✓

nightly-ci-fix
  batch scheduling                ✓
  per-failure independent loop    ✓
  fix-<date> branch naming                                       ✓
  [xpu][fix] commit format                                       ✓
  agent_space_xpu/ layout                                        ✓

action/source-oneapi                          (intel-gpu)
action/intel-gpu-device-selection             (intel-gpu)
action/unitrace/setup                         (intel-gpu)
action/tmux/tmux-long-tasks     ✓
skill-writer                    ✓
pr-review                                                        ✓
auto-label                                                       ✓
release-branching                                                ✓
xpu-build-pytorch                                                ✓
xpu-ops-pr-creation                                              ✓
oob-perf-analysis                                                ✓
xpu-alignment                                                    ✓
at-dispatch-v2                              ✓
```

## Reuse guide

### For another Intel GPU project (non-PyTorch)

Keep:
- Three-stage reproduce structure
- Triage analysis-only framework and verdict design
- Implement engineering principles
- Verify before/after diff
- Orchestrator pipeline skeleton and batch scheduling

Replace:
- Nightly wheel URL → project-specific nightly index
- `source-oneapi` dependency → project-specific env setup
- XPU path mappings in `run-test.md`
- `spin fixlint` → project-specific lint tool
- `xpu.txt` pin workflow → project-specific submodule management

### For another PyTorch backend (ROCm, MPS, etc.)

Keep everything in the pytorch-backend column, plus all general items.

Replace:
- `@skipIfXpu` patterns → `@skipIfROCm`, `@skipIfMPS`, etc.
- XPU path rules in triage → backend-specific paths
- Nightly wheel URL → backend nightly index
- `spin fixlint` → repo-specific lint tool
- `xpu.txt` pin → backend submodule pin
- GitHub issue body markers → backend repo issue templates
