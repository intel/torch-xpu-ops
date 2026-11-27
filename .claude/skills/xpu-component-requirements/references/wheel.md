<!--
Copyright 2020-2025 Intel Corporation
Licensed under the Apache License, Version 2.0
-->

# Stage One: Package Requirements and Bundled Libraries

Versioned links in this file are structural entry points; substitute the target PyTorch commit or release tag when inspecting source recipes.

## 1. Establish Package Declarations

In the target interpreter, batch-read `Name`, `Version`, `Requires-Python`, `Requires-Dist`, `direct_url.json`, `WHEEL`, and `RECORD` through `importlib.metadata`. Importing torch is unnecessary. For legacy egg-info installations, also check `distribution.requires`.

- **Official artifacts:** match torch's version, channel, and platform in the [stable index](https://download.pytorch.org/whl/xpu/) or [nightly index](https://download.pytorch.org/whl/nightly/xpu/). Retain the artifact URL/hash and compare its dependency declarations with local metadata. If the index provides `data-core-metadata` (formerly `data-dist-info-metadata`), remove the wheel URL fragment, append `.metadata`, retrieve it, and verify the advertised metadata digest; otherwise read `METADATA` from the archive. For dependent packages, retrieve official metadata when local declarations are incomplete or need confirmation.
- **Local builds:** use installed metadata, actual source, and build records; label their constraints “declared by this build.” A same-version official wheel is reference material, not a substitute for that build's declarations.

Resolve declaration differences using artifact/build identity. If the governing requirement remains unclear, mark only that affected constraint Unresolved.

When build-recipe differences need tracing, inspect the matching PyTorch commit's [tools/metadata/dependencies.py](https://github.com/pytorch/pytorch/blob/v2.15.0/tools/metadata/dependencies.py) for `BASE_DEPENDENCIES` / `dynamic_metadata` (older versions use `install_requires` in [setup.py](https://github.com/pytorch/pytorch/blob/v2.9.0/setup.py)), `PYTORCH_EXTRA_INSTALL_REQUIREMENTS` in [generate_binary_build_matrix.py](https://github.com/pytorch/pytorch/blob/v2.15.0/.github/scripts/generate_binary_build_matrix.py), and [.ci/pytorch/binary_populate_env.sh](https://github.com/pytorch/pytorch/blob/v2.15.0/.ci/pytorch/binary_populate_env.sh). Recipes do not replace the actual release artifact.

## 2. Traverse Dependencies

Start with torch's `Requires-Dist`. Apply the branch continuation rules in [SKILL.md](../SKILL.md#2-stage-one-check-package-constraints), using these package-specific semantics:

1. Evaluate PEP 508 markers in the target interpreter's environment and propagate requested extras along dependency edges.
2. Match installed packages by normalized name. Use PEP 440 for ranges, exclusions, prereleases, and local versions; use `direct_url.json` for direct-URL requirements. An unknown required URL identity is Unresolved; an unversioned dependency checks presence only.
3. Cache package metadata by name/version and accumulate extras. Evaluate every parent's constraint even when the dependency was already visited; expand newly activated extras without repeating settled checks. Use `python -m pip check` only as a cross-check limited to this dependency scope.
4. Inventory XPU native libraries using `RECORD` or archive listings, retaining their owning packages for stage two. Independently versioned bundled components need their own version evidence.

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

Example compatibility and glibc check:

```bash
getconf GNU_LIBC_VERSION
python - <<'PY'
import importlib.metadata as m
from packaging.tags import parse_tag, sys_tags
d = m.distribution("torch")
w_tags = {t for l in (d.read_text("WHEEL") or "").splitlines() if l.startswith("Tag:") for t in parse_tag(l.split(":", 1)[1].strip())}
print("Requires-Python:", d.metadata.get("Requires-Python") or "not declared")
print("Wheel tag compatibility:", ("match" if (w_tags & set(sys_tags())) else "mismatch") if w_tags else "no WHEEL tags")
PY
```

Specification entry points: [Core Metadata](https://packaging.python.org/en/latest/specifications/core-metadata/), [Dependency Specifiers](https://packaging.python.org/en/latest/specifications/dependency-specifiers/), and [Simple Repository API](https://packaging.python.org/en/latest/specifications/simple-repository-api/).
