<!--
Copyright 2020-2025 Intel Corporation
Licensed under the Apache License, Version 2.0
-->

# Python Dependencies and Bundled Runtime Libraries

## 1. Read Requirement Sources

In the target interpreter, batch-read `Name`, `Version`, `Requires-Python`, `Requires-Dist`, `direct_url.json`, `WHEEL`, and `RECORD` through `importlib.metadata`. Importing torch is unnecessary. For legacy egg-info installations, also check `distribution.requires`.

- **Official release declarations**: match version, channel, and platform in the [stable index](https://download.pytorch.org/whl/xpu/) or [nightly index](https://download.pytorch.org/whl/nightly/xpu/), retaining the artifact URL/hash. If the index provides `data-core-metadata` (formerly `data-dist-info-metadata`), remove the fragment from the wheel URL, append `.metadata`, retrieve it, and verify the digest. Otherwise read `METADATA` from the archive. Requirements that match the local metadata can be used to compare dependencies. This does not establish binary provenance, and established package constraints do not require file-by-file verification against the official `RECORD`.
- **Local builds**: use this installation's metadata, build records, and actual source, labeling requirements “declared by this build.” A same-version official wheel is reference material only. Retain gaps when provenance is unknown or metadata is incomplete.

When build-recipe differences need tracing, inspect the matching PyTorch commit's [tools/metadata/dependencies.py](https://github.com/pytorch/pytorch/blob/v2.15.0/tools/metadata/dependencies.py) for `BASE_DEPENDENCIES` / `dynamic_metadata` (older versions use `install_requires` in [setup.py](https://github.com/pytorch/pytorch/blob/v2.9.0/setup.py)), `PYTORCH_EXTRA_INSTALL_REQUIREMENTS` in [generate_binary_build_matrix.py](https://github.com/pytorch/pytorch/blob/v2.15.0/.github/scripts/generate_binary_build_matrix.py), and [.ci/pytorch/binary_populate_env.sh](https://github.com/pytorch/pytorch/blob/v2.15.0/.ci/pytorch/binary_populate_env.sh). Recipes do not replace the actual release artifact.

## 2. Expand and Compare

1. Start with torch's `Requires-Dist`, retaining the parent package and original expression. Evaluate PEP 508 markers using the target interpreter's environment and requested extras to identify applicable dependency constraints.
2. Match installed values by normalized package name. Use PEP 440 for ranges, exclusions, prereleases, and local versions; check `direct_url.json` for direct-URL dependencies. Mark missing or out-of-range packages Fail and stop that dependency edge. Do not use an incorrectly installed version's downstream requirements to judge the target environment. Missing necessary metadata or an unknown direct-URL identity is Unresolved. For dependencies without version constraints, check presence only. Ordinary version constraints do not require complete verification of installation provenance.
3. Recursively expand only satisfied dependency edges, then use [torch-build.md](torch-build.md) to trace those components' requirements across layers. Prefer batch-reading local metadata; consult official artifacts when metadata is incomplete or release declarations need confirmation. Propagate extras along edges and deduplicate by package name, version, and extras. Use `python -m pip check` only as a cross-check, limiting its results to this dependency scope.
4. Inventory bundled native libraries from `RECORD` or archive listings. Use file ownership to connect package versions to component versions. Do not substitute a package version for an independently versioned component inside it.

Focus on SYCL/JIT, UR loaders/adapters, oneMKL, oneCCL/MPI, PTI, Triton, and bundled libraries such as OFI/libfabric and hwloc. Actual package names come from dependency metadata; common distribution names are listed below. Package file indexes use `https://download.pytorch.org/whl/xpu/<normalized-package-name>/`, or `/whl/nightly/xpu/<normalized-package-name>/` for nightly; for example, [intel-sycl-rt](https://download.pytorch.org/whl/xpu/intel-sycl-rt/). Select the wheel for the actual version and platform, not the latest index entry.

| Category | Common distribution names |
|---|---|
| SYCL / compiler / UR | `intel-sycl-rt`, `dpcpp-cpp-rt`, `intel-cmplr-lib-rt`, `intel-cmplr-lib-ur`, `intel-opencl-rt`, `intel-cmplr-lic-rt` |
| General runtime libraries | `intel-openmp`, `tbb`, `tcmlib`, `umf` |
| oneMKL | `mkl`, `onemkl-license`, `onemkl-sycl-{blas,dft,lapack,rng,sparse}` |
| Communication, collection, and compilation | `oneccl`, `oneccl-devel`, `impi-rt`, `intel-pti`, `pyzes`, `triton-xpu`, `pytorch-triton-xpu` |

## 3. Python and ABI Baselines

Compare `Requires-Python` where declared; an absent field does not count as a passed requirement. For each wheel, expand all `Tag` values into a set and intersect it with the target interpreter's `packaging.tags.sys_tags()`. A nonempty intersection means compatibility. Count once per package; do not mark unmatched alternative tags as failures.

On Linux, interpret `manylinux_<major>_<minor>_<arch>` and legacy aliases as glibc lower bounds according to [PEP 600](https://peps.python.org/pep-0600/). Query `getconf GNU_LIBC_VERSION` in the container/host containing the target Python. A plain `linux_<arch>` tag declares no glibc baseline; use actual build evidence for local builds.

Specification entry points: [Core Metadata](https://packaging.python.org/en/latest/specifications/core-metadata/), [Dependency Specifiers](https://packaging.python.org/en/latest/specifications/dependency-specifiers/), and [Simple Repository API](https://packaging.python.org/en/latest/specifications/simple-repository-api/).
