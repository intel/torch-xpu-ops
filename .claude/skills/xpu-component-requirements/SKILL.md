---
name: xpu-component-requirements
description: "Check whether the Python packages, embedded components (SYCL-TLA, oneDNN, Triton XPU, oneMKL, oneCCL), runtime libraries, and Intel GPU drivers (Compute Runtime/NEO, IGC, Level Zero, OpenCL) used by a PyTorch XPU installation meet official version requirements. Use when verifying PyTorch XPU prerequisites, auditing an XPU environment, or checking if IGC, Compute Runtime, Level Zero, or oneAPI versions are compatible or too old. Do not use for unit-test root-cause analysis or code fixes."
---

<!--
Copyright 2020-2025 Intel Corporation
Licensed under the Apache License, Version 2.0
-->

# Check PyTorch XPU Component Versions

Identify the installation, check its package constraints, then follow component dependencies to their runtime and driver requirements. Compare official declarations with local metadata and version/interface queries. Do not compile or run test programs, investigate unit-test failures, or modify the environment. Batch independent reads and use timeouts for external queries.

## Scope

The default scope is the target PyTorch installation's package dependencies, enabled XPU components, and their downstream runtime and driver dependencies. For a targeted request, follow only the relevant chains. Build toolchains, profiling functionality, and management tools enter scope only when requested or required by an in-scope dependency.

GPU, OS, and driver channel select the applicable requirements. OS/GPU support-list audits and independent kernel/firmware matrix checks are outside scope; a kernel or firmware version is checked only when an in-scope dependency explicitly requires it.

## 1. Identify the Installation

Collect in the target server/container and Python interpreter:

- `sys.executable`, Python version, OS/architecture, and GPU models with PCI IDs and BDFs/UUIDs. Unless devices are specified, cover the distinct models visible to this installation. Retain system enumeration if a device query fails and identify the unconfirmed scope.
- Full torch version, installation path, `git_version`, and installation source, read statically via `importlib.metadata`, `direct_url.json`, `torch/version.py`, and build records without importing `torch`.

Example read-only collection commands (adapt the interpreter path as needed):

```bash
lspci -nn -D | grep -i -E 'vga|display|3d'
timeout 10s sycl-ls
python - <<'PY'
import importlib.metadata as m, importlib.util as u, pathlib, sys
d = m.distribution("torch")
print(sys.executable, d.version, d.read_text("direct_url.json") or "")
spec = u.find_spec("torch")
for root in (spec.submodule_search_locations or []) if spec else []:
    vp = pathlib.Path(root) / "version.py"
    if vp.is_file():
        print("\n".join(l for l in vp.read_text().splitlines() if "git_version" in l or "__version__" in l))
PY
```

Use the matching official artifact for release-package declarations and the actual source/build records for local builds. Record provenance limitations without making whole-package binary verification a prerequisite for version checks.

## 2. Stage One: Check Package Constraints

Use [wheel.md](references/wheel.md) to check torch and its applicable package dependencies: version constraints, environment markers/extras, Python compatibility, wheel ABI/platform tags, and declared glibc baselines. Inventory XPU native libraries and their owning packages for stage two.

For each dependency edge, preserve the parent, original constraint, and installed value. Continue downstream only when the applicable parent-package constraints pass. A missing or out-of-range package is Fail; an unestablished constraint is Unresolved. Stop the affected branch and annotate that item with “downstream requirements not checked.” Continue independent branches, including direct PyTorch driver requirements. Never use an out-of-range package's requirements as the target installation's downstream baseline.

## 3. Stage Two: Follow Component Requirements

| Reference | Purpose |
|---|---|
| [torch-build.md](references/torch-build.md) | Identify enabled embedded components and compliant runtime packages; extract their version-specific runtime, driver, and interface requirements. |
| [platform.md](references/platform.md) | Extract direct PyTorch driver requirements, resolve actual driver/library providers, and follow driver-package dependencies and version mappings. |

Build an inventory as dependencies are discovered: **component, actual version/revision, provider, applicability, and evidence**. Use “included / not applicable / unconfirmed” for applicability. Source presence or default build options do not establish enablement. Keep an unconfirmed in-scope component visible and identify the evidence needed to resolve it.

For each consumer, read documentation covering its actual release; resolve a source revision only for source-specific requirements or unclear release mappings. Reference URLs are entry points, not fixed version requirements: select the matching GitHub revision or Intel Version/archive entry. Use release-series documentation with its declared scope and patch notes; a separate page for every patch is unnecessary.

If a source is unavailable, try official mirrors/source files, tags, commits, archives, or channel package records. Record the specific gap when those alternatives fail. Do not substitute the latest or an adjacent release's requirements. Cache sources already resolved for the same version.

## 4. Compare Requirements

Keep one record per distinct **consumer/version → dependency → applicable constraint**, with the actual value, official source, local evidence, and result. Retain separate constraints from different consumers even when they share a dependency.

- **Evidence:** package metadata establishes installed package versions; resolved file paths and ownership establish runtime-library providers; target-device queries establish API/features. Ambiguous providers affect only requirements that need that distinction.
- **Version semantics:** preserve bounds, alternatives, exclusions, and conditions. Compare package, upstream component, loader, and API versions within their own schemes; require evidence for cross-scheme mappings.
- **Requirement strength:** build pins identify components; explicit version bounds and mandatory dependency/interface conditions determine compliance. Tested, validated, recommended, and OS/GPU support lists do not by themselves establish version constraints.
- **Document differences:** align consumer revision, model, and purpose first. Check compatible requirements together. If sources still disagree about the same requirement, retain both and mark that requirement Unresolved; differing requirements from separate consumers remain separate checks.

| Result | Meaning |
|---|---|
| Fail | A clear applicable requirement is violated: a version is out of range, a mandatory dependency is missing, or a mandatory interface feature is unsupported. |
| Pass | The observed value meets a clear applicable requirement. An unversioned dependency checks presence only. |
| Unresolved | An in-scope check lacks necessary applicability, requirement, actual-value, or provider evidence, or its official requirement remains ambiguous/conflicting. State the specific gap. |
| No explicit requirement found | The relevant version's requirements sections and dependency declarations were checked, but they declare no constraint for the identified in-scope relationship. List the sources checked; an unknown actual value alone does not add an Unresolved item. |

Do not create check records for out-of-scope or confirmed inapplicable objects. Provenance limitations belong in the target description unless they prevent a particular version comparison.

## 5. Report Exceptions and Coverage

Use the records above to produce a concise report. Summarize passing checks; provide a full detail table or appendix only when requested.

1. **Target and counts:** torch version/commit/source, Python environment, GPU, OS, scope, and counts for Fail F, Pass P, Unresolved U, and No explicit requirement found N.
2. **Failures:** one row per failed check: “consumer/version → component | applicability | official requirement | actual value | official source | local evidence.” If F = 0, write “No confirmed failures found.”
3. **Unresolved items and missing requirements:** one row per U or N record: “consumer/version → component | result | known requirement and actual value | specific evidence gap or sources checked | sources.” If U + N = 0, write “None.” Attach stopped-branch annotations to their triggering items.
4. **Official-document differences:** only when present, show both statements/sources and the adopted requirement or unresolved reason. These explain existing records and add no counts.
5. **Coverage:** summarize examined names in three rows below, noting confirmed inapplicable components and layers not examined. Inclusion here does not imply Pass.

| Layer | Components examined |
|---|---|
| Python packages and bundled libraries | Package and native-library names |
| Embedded and cooperating components | Consumers such as SYCL-TLA, oneDNN, and Triton XPU |
| Driver dependencies and interfaces | Driver packages, IGC, loaders, APIs, and required tools |

Count checks, not component names; F and U + N must match their detail-table row counts. When coverage is complete, no dependency branches remain stopped, P > 0, and F = U = 0: conclude “All requirements within this check's scope are satisfied” if N = 0, or “All explicit requirements within this check's scope are satisfied (N relationships have no declared constraint)” if N > 0.

### Sample Report

```markdown
**Target:** torch 2.15.0+xpu (git `abc1234`, official wheel) · Intel Data Center GPU Max 1550 (`8086:0bd5`) · Ubuntu 24.04 x86_64 · Python 3.11 (`/opt/venv/bin/python`) · Default XPU runtime scope
**Result:** Fail 1, Pass 42, Unresolved 1, No explicit requirement found 0.

### Failures
| Consumer/version → Component | Applicability | Official requirement | Actual value | Official source | Local evidence |
|---|---|---|---|---|---|
| SYCL-TLA v0.9 → IGC | Max Series (PVC) | IGC v2.27.10 or later | 2.10.10 (`intel-igc-core-2 2.10.10-1082`) | [SYCL-TLA v0.9 README](https://github.com/intel/sycl-tla/blob/b4f7216580a48b1d443d0c6e85dc7a72105bbdbb/README.md#compatibility) | `dpkg-query` + IGC package record |

### Unresolved Items and Missing Requirements
| Consumer/version → Component | Result | Known requirement and actual value | Evidence gap / sources checked | Sources |
|---|---|---|---|---|
| GPU oneDNN v3.13 → OpenCL C 2.0 | Unresolved | Requires OpenCL C 2.0; device lists 1.2 & 3.0 with `__opencl_c_subgroups` | Upstream wording leaves OpenCL C 3.0 acceptance unclear | [oneDNN v3.13 README](https://github.com/uxlfoundation/oneDNN/blob/v3.13/README.md); `clinfo` |

### Official-Document Differences
| Component and target | Statement A and source | Statement B and source | Adopted requirement and rationale |
|---|---|---|---|
| SYCL-TLA v0.9 → IGC (PVC) | `IGC v2.27.10 or later` ([v0.9 README](https://github.com/intel/sycl-tla/blob/b4f7216580a48b1d443d0c6e85dc7a72105bbdbb/README.md#compatibility)) | `Compute Runtime 25.13 (with IGC 2.10.10)` ([v0.9 build guide](https://github.com/intel/sycl-tla/blob/b4f7216580a48b1d443d0c6e85dc7a72105bbdbb/media/docs/cpp/build/building_with_sycl_support.md)) | Adopted README model-specific minimum; build guide lists a paired combination without an independent IGC lower bound. |

### Coverage
| Layer | Components examined |
|---|---|
| Python packages and bundled libraries | `torch`, `intel-sycl-rt`, `dpcpp-cpp-rt`, `onemkl-sycl-*`, `oneccl`, `impi-rt`, `triton-xpu`, `intel-pti` |
| Embedded and cooperating components | `torch-xpu-ops`, `SYCL-TLA`, `GPU oneDNN`, `SYCL/UR`, `oneMKL`, `oneCCL/MPI`, `Triton XPU` |
| Driver dependencies and interfaces | `intel-level-zero-gpu`, `intel-opencl-icd`, `IGC`, `GmmLib`, `Level Zero API`, `OpenCL C`, `ocloc` |
```
