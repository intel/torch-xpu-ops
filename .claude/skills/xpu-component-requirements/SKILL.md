---
name: xpu-component-requirements
description: "Check the Python packages, embedded components, runtime libraries, and GPU drivers used by a PyTorch XPU installation against their official version requirements. Use for a full component version check or a targeted dependency check."
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
- Full torch version, installation path, `git_version`, and installation source, using `importlib.metadata`, `direct_url.json`, and build records. If importing fails, read `torch/version.py` and continue static checks.

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

Count checks, not component names; F and U + N must match their detail-table row counts. Conclude “All requirements within this check's scope are satisfied” only when coverage is complete, no dependency branches remain stopped, P > 0, and F = U = N = 0.
