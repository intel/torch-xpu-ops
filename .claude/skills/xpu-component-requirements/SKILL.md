---
name: xpu-component-requirements
description: "Check whether the Python dependencies, embedded components, runtime libraries, GPU drivers, and firmware actually used by the current PyTorch XPU installation meet their official requirements. Use for a full environment version check or a targeted component check."
---

<!--
Copyright 2020-2025 Intel Corporation
Licensed under the Apache License, Version 2.0
-->

# Check PyTorch XPU Component Versions

Follow “identify the environment → trace components → compare requirements → report results.” Compare official requirements with local metadata and version queries. Do not compile or run test programs, investigate unit-test failures, or modify the environment. Batch independent local queries and document reads within each stage where possible.

## 1. Identify the Environment

Collect the following in the target server, container, and Python interpreter:

- `sys.executable`, Python version, OS/architecture, GPU models, PCI IDs, and BDFs/UUIDs. Unless devices are specified, cover the distinct models visible to the target PyTorch installation. If querying fails, retain system enumeration results and mark the device scope as unconfirmed.
- The full torch version, installation path, `git_version`, and installation source. Start with `importlib.metadata`, `direct_url.json`, and installation/build records. If importing fails, read `torch/version.py` directly and continue static checks.
- Match official artifacts by channel, version, and platform, then compare their requirements with local metadata. For local builds, use the actual source, build cache, and artifacts. Record provenance evidence and its limitations; do not audit whole-package binary integrity by default.

By default, check all XPU runtime dependencies of this installation. Expand build toolchains, profiling, and management tools only when requested or explicitly required by a runtime dependency. For targeted checks, trace only the relevant dependency chains.

## 2. Trace Components and Official Requirements

First check Python packages against the version constraints of torch and applicable parent packages. Follow a branch's downstream requirements only after its package constraints pass. Mark missing or out-of-range packages as Fail and stop that branch. If a constraint cannot be established, pause only the affected dependency edge; continue other established requirements, independent branches, and platform checks.

Build a coverage inventory from the three sources below, extending it as dependencies are discovered. For each component, record its **actual version/revision, provider, applicability, and evidence**. Applicability is “included / not applicable / unconfirmed.” Default build options do not prove that a component is enabled. Keep uncertain components as unconfirmed rather than silently skipping them.

| Source | Checks |
|---|---|
| [Python packages and bundled libraries](references/wheel.md) | Expand applicable package dependencies, identify native-library ownership, and check Python/ABI compatibility and glibc baselines. |
| [Embedded and cooperating components](references/torch-build.md) | Identify enabled embedded components and independent components whose package constraints passed. Read their version-specific IGC, runtime, and interface requirements. |
| [Platform and drivers](references/platform.md) | Match GPU, OS, and driver channel; extract platform requirements and driver-package dependency constraints. |

Select official documentation for the actual consumer version. A source SHA is unnecessary when requirements are published for the release version. Branches, tags, and latest pages in the references are entry points: select the matching revision on GitHub and the appropriate Version/archive entry in Intel documentation. Apply release-series documentation according to its declared scope and patch notes. A missing patch-specific page is not by itself Unresolved, and adjacent versions must not be assumed to share requirements.

If a page is unreadable, try a web tool or official source files and search by version, filename, or section. If a GitHub release is absent, check tags, commits, and channel package records. Record the specific unavailable source only after these alternatives fail; do not substitute the latest page for historical requirements.

## 3. Compare and Classify

For each check, record **consumer and version, required component, applicability conditions, original constraint, actual value, official source, local evidence, and result**.

- Select evidence for the requirement: use installed metadata for package constraints, the file and version resolved by the consumer for runtime-library constraints, and the host/target device for kernel, KMD, and firmware. A library appearing in a process does not prove that every consumer uses it. If multiple copies cannot be distinguished, mark only the affected runtime-library requirement Unresolved.
- Preserve lower/upper bounds, alternatives, and conditions; compare within each version scheme. Treat driver package versions, upstream IGC versions, loader package versions, and API versions separately. Do not force a conversion when the mapping is unknown.
- Use build pins to identify components. Minimum versions, mandatory conditions, and explicitly defined official support scopes determine the result. A combination outside the OS/GPU support scope is Fail, labeled “outside the official support scope”; do not infer runtime failure. Combinations listed only as tested, validated, or recommended are not hard constraints. An ambiguous support scope is Unresolved.
- Deduplicate the same requirement; check requirements from different consumers separately. Before interpreting different documented values as conflicts, align revision, model, and purpose. Check all compatible hard requirements. Mark irreconcilable conflicts Unresolved and retain both sources.

| Result | Conditions |
|---|---|
| Fail | An applicable requirement is clear, and the version is out of range, a mandatory component is missing, an interface feature is unsupported, or the target combination falls outside an explicit official support scope. |
| Pass | An applicable requirement is clear, and the observed version and relevant interface conditions meet it. |
| Unresolved | Evidence is missing for a requirement's applicability, actual value, or provider; necessary documentation is unavailable; or the requirement is ambiguous/conflicting. Identify the affected check. |
| No explicit requirement found | The applicable version's official requirements sections and relevant dependency declarations were checked, but no explicit constraint was found for the object. List the locations checked; do not also mark it Unresolved solely because its actual version is unknown. |

## 4. Report Results

Put exceptions first. Derive counts from individual check records and summarize coverage for passing checks. Do not output a complete detail table or create an appendix file by default; expand only when requested.

1. **Target and result**: list torch version/commit/source, GPU, OS, Python environment, and scope. Summarize Fail F, Pass P, Unresolved U, and No explicit requirement found N.
2. **Failures**: one row per failure with “component | consumer and version / applicability | official requirement | actual value | official source | local evidence.” If F is zero, write “No confirmed failures found.”
3. **Unresolved items and missing requirements**: use “consumer → required component | result (Unresolved / No explicit requirement found) | known requirement and actual value | specific evidence gap / locations checked without finding a constraint | sources.” Do not use catch-all entries such as “native components not fully verified” without identifying specific objects. If U + N is zero, write “None.”
4. **Differences in official documentation**: include only when found, with “component and target | statement A and source | statement B and source | adopted requirement and rationale / unresolved reason.” Do not count these twice.
5. **Covered components**: summarize names in the table below, deduplicating within each layer. Inclusion means the component was examined, not that every check passed; refer to the exception tables for results.

| Coverage layer | Components included in this check |
|---|---|
| Python packages and bundled runtime libraries | <package and native-library names> |
| Embedded and cooperating components | <consumers traced, such as SYCL-TLA, oneDNN, and Triton XPU> |
| System, drivers, and device interfaces | <system packages, IGC, loaders, APIs, KMD, firmware, etc.> |

Write “Not examined in this check” for layers not covered. Briefly explain confirmed inapplicable components. Put limitations that affect only provenance confidence, not established version requirements, in the target description rather than separate Unresolved entries. Include affected requirements in the unresolved-items table. For stopped dependency edges, annotate the triggering item with “downstream requirements not checked.”

Verify that F equals the failure-table row count and U + N equals the unresolved-items table row count. Count P by requirements passed, not component names. Conclude “All requirements within this check's scope are satisfied” only when all in-scope components and dependencies have been traced, P > 0, and F = U = N = 0.
