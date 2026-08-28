# Label Proposal — Design Logic

This document explains the design behind the `intel/torch-xpu-ops` label
proposal. It is derived from the two source-of-truth JSON files:

- **`proposed_labels.json`** — the *target* taxonomy: every label, its axis, its
  `evidence` criterion, and its `keywords`.
- **`label_actions.json`** — the *migration plan*: how each current repo label is
  created, renamed, removed, reconciled, dropped, or reused to reach that target.

The label-issue skill reads `proposed_labels.json` at decision time; nothing in
the workflow hard-codes a label name, keyword, or rule that the JSON already
carries.

## 1. Core idea: labels are a multi-axis classification

An issue is not described by one label but by a small set, one per independent
**axis**. Each axis answers a different question about the failure:

| Axis | Question it answers | Cardinality |
|---|---|---|
| `type` | Is this a Bug / Feature / Task / Epic? | native field (not a label) |
| `priority` | How urgent (P0–P3)? | native field (not a label) |
| `test` | Which test surface failed — ut / e2e / oob? | single |
| `module` | Which code component owns the defect? | single |
| `os` / `hw` | Is it platform-specific, and to what? | single each, often empty |
| `dependency component` | Which external component is at fault? | single, evidence-only |
| `dtype` | Which data type(s) are in the failure signature? | multi |
| `symptom` | What is the nature of the failure? | multi |
| `triage` | duplicate / wontfix / need_split | multi, conditional |

Keeping axes orthogonal is the central design constraint: a label that mixes two
questions (e.g. the old `module: ut`, which conflates *test surface* with *code
module*) is a design smell and is reconciled onto its correct axis.

## 2. Labels are data, not code

Every fact about a label lives in `proposed_labels.json`:

- **`name`** — the exact GitHub label string (with its `type:` / `module:` / …
  prefix).
- **`evidence`** — the human-readable criterion that must be *met* for the label
  to apply. This is the authoritative decision rule.
- **`keywords`** — heuristic hints only; they never override an explicit value or
  an `evidence` match.
- **`origin`** — provenance: whether the label is reused unchanged, renamed, or
  new, and which old label it supersedes.
- **`code`** / **`exists_in_repo`** — for axes where the emitted label differs
  from the internal value (`dependency`), or where the label must be created
  before use.

Design consequence: adding or changing a label means editing JSON, not the
skill. A new dtype like `mxfp8` is a JSON entry plus a `create` action — no code
change.

## 3. Evidence over keywords

Every axis is decided by matching **evidence**, not by keyword spotting.
Keywords exist only to break ties or seed a search. This prevents false
positives — e.g. a Dynamo frame appears in *every* `torch.compile` failure, so
`module: dynamo` requires evidence of a *tracing/guards* defect with no
backend-codegen signal, not merely the word "dynamo".

Two axes are explicitly evidence-only (no keywords by design):

- **`dependency component`** — a library merely appearing in the call path never
  qualifies; the failure must *originate* there.
- **`module`** — the label list is **priority-ordered**; the first label whose
  evidence is met wins, driven by the traced root cause.

## 4. Ordered axes encode decision priority

Where multiple labels on one axis could match, the JSON array order *is* the
tie-break rule:

- `module: inductor` precedes `module: dynamo` — a compile failure is attributed
  to the backend unless the defect is purely in tracing.
- `module: infra` is last — the catch-all when nothing else matches (there is
  deliberately no generic `module: others`).
- `priority` tiers are evaluated in severity order, defaulting to the tier the
  JSON marks as default (Medium/P2).

## 5. Native fields are not labels

Issue **Type** (Bug/Feature/Task/Epic) and project **Priority** (P0–P3) are
first-class GitHub/project fields. Modeling them as labels too would duplicate
state and let the two drift, so they are **dropped from the label proposal**
(the `drop_from_proposal` bucket) and set on their native fields instead.

## 6. The migration plan: six buckets

`label_actions.json` maps the *current* repo state onto the target taxonomy.
Every existing and proposed label lands in exactly one bucket, chosen by a single
question: **does a correctly-named, correctly-placed label already exist for this
concept?**

| Bucket | When | Why |
|---|---|---|
| **create** | Concept needed, no label carries it | Add fresh (e.g. `module: gemm`, `dtype: mxfp8`, `functionality`). |
| **rename** | Right concept, wrong name/format | Edit in place to **preserve issue history** (e.g. `hw: Arc`→`hw: ARC`). |
| **remove** | Obsolete or out-of-axis, no target | Delete (e.g. `mkl`, `bug_fix_stage*`). |
| **reconcile** | Valid label on the wrong axis / merges elsewhere | Remap the concept (e.g. `module: ut`→`test: ut`, `module: others`→`module: infra`). |
| **drop_from_proposal** | Belongs to a native field, not labels | Keep out of the label set (Type/Priority values). |
| **reuse** | Already an exact match | No action. |

Guiding preferences:

1. **Prefer edit over churn** — rename/reconcile beats remove+create because it
   keeps history and existing assignments (why `os: Linux` and the `test:` set
   list a rename/reconcile alternative).
2. **Fold synonyms into the canonical label** — `module: transformers`→
   `module: sdpa`, `module: fx`/`module: op impl`→`module: ops`.
3. **Fold legacy exclusion tags** — `not_target` is removed only after its
   meaning is absorbed into `wontfix`.

## 7. Newly proposed labels

One row per axis listing the labels introduced by this proposal:

| Axis | New labels |
|---|---|
| test | test: ut, test: e2e, test: oob |
| module | module: dynamo, module: gemm, module: eltwise, module: reduction, module: ops, module: utils |
| os | os: Linux |
| hw | hw: ARL, hw: CRI |
| dependency | dependency component: oneCCL, dependency component: IGC, dependency component: Level_Zero |
| dtype | dtype: float64, dtype: float4, dtype: mxfp8, dtype: mxfp4, dtype: int64, dtype: int32, dtype: int16 |
| symptom | functionality |
| triage | need_split |

## 8. Full label set

Every label defined in proposed_labels.json, one row per category. The two
native fields (Type, Priority) are set on the issue/project, not applied as
labels.

| Category | Labels |
|---|---|
| type (native field) | Bug, Feature, Task, Epic |
| priority (native field) | Urgent (P0), High (P1), Medium (P2), Low (P3) |
| type | bug, enhancement, question, documentation |
| test | test: ut, test: e2e, test: oob |
| module | module: distributed, module: sdpa, module: sparse, module: profiler, module: inductor, module: dynamo, module: ao, module: gemm, module: eltwise, module: reduction, module: ops, module: core, module: build, module: dpclang, module: rfc, module: utils, module: infra |
| os | os: Windows, os: Linux, os: WSL |
| hw | hw: PVC, hw: BMG, hw: ARC, hw: ARL, hw: LNL, hw: MTL, hw: CRI, hw: PTL |
| dependency | dependency component: driver, dependency component: oneDNN, dependency component: oneMKL, dependency component: oneAPI, dependency component: Triton, dependency component: MSVC, dependency component: community, dependency component: third_party, dependency component: oneCCL, dependency component: IGC, dependency component: Level_Zero |
| dtype | dtype: float32, dtype: float64, dtype: float16, dtype: bfloat16, dtype: float8, dtype: float4, dtype: mxfp8, dtype: mxfp4, dtype: complex, dtype: int64, dtype: int32, dtype: int16, dtype: int8, dtype: int4, dtype: amp_bf16, dtype: amp_fp16 |
| triage | duplicate, wontfix, need_split |
| symptom | Accuracy, performance, functionality, regression, random, inference, training |
| workflow | good first issue, help wanted, triaged, Ready for merge, client, roadmap, wait_upstream, ut_upstream, skipped, skipped_windows, skipped_bmg, skipped_dpclang |
| ci_control | disable_all, disable_ut, disable_e2e, disable_distributed, disable_win, disable_windows_ut, disable_accelerate, disable_transformers, disable_build, disable_auto, windows_ci, windows_ut, pytorch-ci-failure |
| agent | agent:active, agent:needs-human, agent:blocked, agent:skipped, agent:close, agent:merged, agent:reproduction-needed, ai_generated |
| misc | lib: helion |

## 9. End state

After the plan is applied, the repo's label set is a 1:1 match with
`proposed_labels.json`: orthogonal axes, evidence-driven decisions, no
axis-mixing labels, and all state that belongs in native fields kept out of the
label system. The skill can then label any issue purely by reading the JSON
evidence for each axis.
