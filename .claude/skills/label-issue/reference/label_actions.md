# Label Actions — intel/torch-xpu-ops

Action plan for `intel/torch-xpu-ops` labels, derived from `label_actions.json`
(itself built from `proposed_labels.json` and the live repo label set). Grouped
by the action to take. `label` is the target/new name; `from` names the existing
label(s) affected when applicable. The `note` explains *why* the change is
needed, based on the axis design in `proposed_labels.json`.

## Create

New labels that have no repo equivalent and must be added.

| label | note |
| --- | --- |
| `module: dynamo` | The `module` axis splits `torch.compile` failures into frontend vs backend; a Dynamo-tracing defect (guards/graph breaks) needs its own label distinct from `module: inductor`. |
| `module: gemm` | Finer-grained op modules replace the coarse `torch-ops-*` grouping; matmul/addmm/bmm/gemm failures get a dedicated bucket. |
| `module: eltwise` | Same op-axis refinement: elementwise/pointwise ops need their own label for triage routing. |
| `module: reduction` | Same op-axis refinement: reduction ops (sum/mean/softmax/norm) need their own label. |
| `module: ops` | Catch-all for other ATen/native ops (autograd, optimizer, fx, export); absorbs old `op impl`/`fx` so no op failure is unclassified. |
| `module: utils` | `torch.utils` failures that resolve to no op or runtime surface had no home on the `module` axis. |
| `dependency component: oneCCL` | The `dependency` axis needs a value for collective-comm failures (ProcessGroupXCCL/c10d); none existed. |
| `dependency component: IGC` | JIT/IGC compilation, ocloc, and illegal-instruction failures need a distinct dependency owner. |
| `dependency component: Level_Zero` | Level Zero runtime / device-enumeration failures need a distinct dependency owner. |
| `dependency component: PTI` | Intel PTI (Profiling Tools Interface / Kineto backend) failures need a distinct dependency owner on the `dependency` axis. |
| `hw: ARL` | Arrow Lake is a supported platform on the `hw` axis with no existing label. |
| `hw: CRI` | Crescent Island is a supported platform on the `hw` axis with no existing label. |
| `dtype: float64` | The `dtype` axis must cover every dtype in a failure signature; fp64/double was missing. |
| `dtype: float4` | fp4 was missing from the `dtype` axis. |
| `dtype: mxfp8` | MX microscaling fp8 is a distinct dtype from plain fp8 and had no label. |
| `dtype: mxfp4` | MX microscaling fp4 is a distinct dtype from plain fp4 and had no label. |
| `dtype: int64` | Integer dtypes are part of failure signatures; int64/long was missing. |
| `dtype: int32` | int32 was missing from the `dtype` axis. |
| `dtype: int16` | int16/short was missing from the `dtype` axis. |
| `functionality` | The `symptom` axis needs a value for functional defects (assert/hang/timeout/NotImplementedError) that are not specifically accuracy or performance. |
| `need_split` | Promoted from a report-only field to an applied label so multi-cause issues can be flagged for splitting. |
| `os: Linux` | The `os` axis uses generic OS enums; `os: Ubuntu` is too specific. Create, OR satisfy by renaming `os: Ubuntu` (see rename group). |
| `test: ut` | `test` is a separate axis from `module`; the unit-test surface must be labeled independently. Create, OR satisfy via reconcile of `module: ut`. |
| `test: e2e` | The E2E/benchmark surface needs one label instead of the scattered `E2E`/`module: torchbench`/`benchmark`. Create, OR satisfy via their reconcile. |
| `test: oob` | The out-of-box surface needs a `test:` axis label. Create, OR satisfy via reconcile of `OOB`. |

## Rename

Edit an existing label's name/format. Preferred over remove+create because it
preserves issue history and existing assignments.

| from | label | note |
| --- | --- | --- |
| `hw: Arc` | `hw: ARC` | Casing must match the `hw: <CODE>` convention used by every other hw label. |
| `hw : LNL` | `hw: LNL` | Stray space breaks the `hw: <CODE>` format; normalize. |
| `hw : MTL` | `hw: MTL` | Stray space breaks the `hw: <CODE>` format; normalize. |
| `os: Ubuntu` | `os: Linux` | The `os` axis is defined with generic OS enums, not distro names; Ubuntu is one Linux among several. Alternative to creating `os: Linux`. |
| `dependency: third_party packages` | `dependency component: third_party` | Align to the `dependency component: ` prefix shared by the whole dependency axis, and drop the redundant `packages`. |

## Remove

Delete labels no longer needed — superseded by a better label, or out of scope
of any classification axis.

| label | note |
| --- | --- |
| `module: cpp extension` | Not a distinct axis value; C++ extension failures fold into `module: ops`. |
| `mkl` | CPU-only label; XPU uses `dependency component: oneMKL`, so the old `mkl` is redundant. |
| `feature_stage1_PT2.9` | Release-stage workflow tag, not a classification axis in the proposal. |
| `port_from_skiplist` | Workflow/process tag outside every proposed axis. |
| `bug_fix_stage3` | Release-stage workflow tag outside every proposed axis. |
| `bug_fix_stage4` | Release-stage workflow tag outside every proposed axis. |
| `bug_fix_stage5` | Release-stage workflow tag outside every proposed axis. |
| `bug_fix_stage6` | Release-stage workflow tag outside every proposed axis. |
| `not_target` | Its meaning is absorbed by `wontfix` on the `triage` axis; remove after folding (see reconcile group). |

## Reconcile

Existing label kept in concept but remapped to its correct axis or merged into
another bucket — because the old label mixed axes or duplicated another.

| from | label | note |
| --- | --- | --- |
| `module: ut` | `test: ut` | Test surface is not a code module; it belongs on the `test` axis, keeping `module` orthogonal. |
| `module: torchbench` | `test: e2e` | A benchmark suite is a test surface, not a code module; moves to the `test` axis. |
| `E2E` | `test: e2e` | Consolidate the scattered E2E signals into the single `test` axis value. |
| `OOB` | `test: oob` | Out-of-box is a test surface; moves to the `test` axis. |
| `benchmark` | `test: e2e` | Benchmark runs are the E2E surface; consolidate under `test: e2e`. |
| `module: quant` | `module: ao` | Quantization is renamed to the `ao` (torchao/PT2E) module for consistency. |
| `module: op impl` | `module: ops` | Merged into the unified `module: ops` op bucket. |
| `module: fx` | `module: ops` | torch.fx failures resolve to ops; merged into `module: ops`. |
| `module: transformers` | `module: sdpa` | Transformer/attention failures are covered by the `sdpa` module label. |
| `kernel_optimization` | `module: ops + performance` | Splits across axes: the code owner is `module: ops`, the nature is the `performance` symptom. |
| `dependency` | `dependency component: <value>` | The generic `dependency` label carries no owner; resolve to the specific component per the axis. |
| `module: dependency bug` | `dependency component: <confirmed value>` | Dependency ownership belongs on the `dependency` axis, not `module`; map to the confirmed component. |
| `not_target` | `wontfix` | `wontfix` absorbs the legacy out-of-scope signal on the `triage` axis. |
| `long term` | `Epic` | An umbrella/tracking concept is the native issue Type `Epic`, not a label. |

## Drop from proposal

Not modeled as labels — these values live in native GitHub/project fields, and
duplicating them as labels would split state and let the two drift.

| label | note |
| --- | --- |
| `P0` | PyTorchXPU project Priority field value; set on the field, not applied as a label. |
| `P1` | PyTorchXPU project Priority field value; set on the field, not applied as a label. |
| `P2` | PyTorchXPU project Priority field value; set on the field, not applied as a label. |
| `P3` | PyTorchXPU project Priority field value; set on the field, not applied as a label. |
| `Bug` | Native issue Type field value; set on the field, not applied as a label. |
| `Feature` | Native issue Type field value; set on the field, not applied as a label. |
| `Task` | Native issue Type field value; set on the field, not applied as a label. |
| `Epic` | Native issue Type field value; set on the field, not applied as a label. |

## Reuse

Keep as-is; already an exact match to a proposed label, so no action.

| label | note |
| --- | --- |
| `bug` | github-default; matches the `type` axis exactly. |
| `enhancement` | github-default; matches the `type` axis exactly. |
| `question` | github-default; matches the `type` axis exactly. |
| `documentation` | github-default; matches the `type` axis exactly. |
| `duplicate` | github-default; matches the `triage` axis exactly. |
| `wontfix` | github-default; matches the `triage` axis and absorbs `not_target`. |
| `module: others` | Kept as-is per maintainer decision; not folded into `module: infra`. |
| `<all rows marked (exact)>` | All existing module/dtype/hw/os/dependency labels that map exactly to a proposed label need no change. |
