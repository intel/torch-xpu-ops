<!--
Copyright 2020-2025 Intel Corporation
Licensed under the Apache License, Version 2.0
-->

# Driver Dependencies, Interfaces, and Actual Versions

## 1. Locate Driver Requirements and Package Sources

For stable releases, start from the matching PyTorch `major.minor` [XPU prerequisites](https://docs.pytorch.org/docs/2.15/notes/get_start_xpu.html#software-prerequisite), follow the [Intel version page](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-15.html), and use Version to select matching content. For nightly builds or specific commits, start from that commit's [get_start_xpu.md](https://github.com/pytorch/pytorch/blob/v2.15.0/docs/source/notes/get_start_xpu.md) (`.rst` in older versions). Extract explicit driver version requirements for the target GPU/OS. Use the installed driver channel to locate package records below, not to initiate a platform support audit. A missing nightly-specific page must not block independently established component requirements. Validated combinations do not automatically establish minimum versions.

| Channel | Official entry points and content to read |
|---|---|
| OMIX | [Components](https://dgpu-docs.intel.com/overview/release-notes/OMIX/0.4.0.html#components) in the [Release Notes](https://dgpu-docs.intel.com/overview/release-notes/OMIX.html). Select the matching release, GPU, and runtime/developer variant. |
| PVC LTS | Select the release by Version ID in [LTS Release Notes](https://dgpu-docs.intel.com/legacy/release-notes-lts.html). Obtain package versions by release/OS from [Provided Packages](https://dgpu-docs.intel.com/legacy/release-notes/packages.html). |
| Client | Use the system repository or [Intel PPA](https://dgpu-docs.intel.com/legacy/installation-guides/installing-packages-from-the-intel-ppa.html) specified for the target model/OS in the [installation guide](https://dgpu-docs.intel.com/driver/client/overview.html). |
| Windows / WSL2 | Extract applicable driver version constraints from the matching Intel prerequisites. For WSL2, check the Windows host driver and Linux guest userspace components separately. |
| Other models | Follow the matching hardware table to its installation page/release notes. Record general and model-specific requirements separately. |

## 2. Locate Component Versions and Dependency Requirements

Use the table below to enter official sources, then select the established version/tag. **Release pages establish version identity and component combinations; minimum requirements come from consumer requirements or parent-package dependency fields.** Never treat the latest version on a release page as the minimum.

| Component | Official entry points | Version mapping and content to read |
|---|---|---|
| **NEO / Level Zero GPU driver / OpenCL ICD** | [NEO releases](https://github.com/intel/compute-runtime/releases); channel package tables above | Select the actual driver release and read Packages included, Components revisions included, and Assets. Distinguish Additional components revisions used in build. Look for channel packages `libze-intel-gpu*` / `intel-level-zero-gpu` and `intel-opencl-icd`. |
| **IGC core / FCL** | [IGC releases](https://github.com/intel/intel-graphics-compiler/releases), [NEO package.cmake](https://github.com/intel/compute-runtime/blob/26.31.39395.16/package.cmake) | Follow the actual NEO package's dependencies to `intel-igc-core*` / `libigc*` and `intel-igc-opencl*` / `libigdfcl*`, then read FCL → core constraints. Use `NEO__IGC_VERSION` at the matching NEO tag to interpret the recipe, and package provenance records to establish the upstream IGC tag. Also check [consumer requirements such as TLA](torch-build.md). |
| **GmmLib** | [GmmLib releases](https://github.com/intel/gmmlib/releases), matching NEO release | Extract constraints from NEO package dependencies on `libigdgmm*` / `intel-gmmlib*`. Use the release's `intel/gmmlib@` revision to establish the paired version. |
| **Level Zero / OpenCL loaders and APIs** | [Level Zero releases](https://github.com/oneapi-src/level-zero/releases), [OpenCL ICD Loader releases](https://github.com/KhronosGroup/OpenCL-ICD-Loader/releases), [NEO Level Zero BUILD](https://github.com/intel/compute-runtime/blob/master/level_zero/doc/BUILD.md) | Establish the actual loader implementation/version through file ownership. If the distribution uses another implementation, follow that package's source. Obtain minimum loader/API requirements from version-matched [oneDNN, UR, and PTI](torch-build.md) documentation and parent-package dependencies. A loader package version is not an API version. |
| **ocloc** | Channel Provided Packages, [NEO release Assets](https://github.com/intel/compute-runtime/releases), and [Client PPA guide](https://dgpu-docs.intel.com/legacy/installation-guides/installing-packages-from-the-intel-ppa.html) | Identify the package shipping `ocloc` (such as `intel-ocloc`) and its installed version through file listings. Whether it is mandatory and its minimum version come from the actual consumer's matching documentation, such as Triton's. |
| **unitrace / Metrics (when in scope)** | [PTI releases](https://github.com/intel/pti-gpu/releases), [unitrace Requirements](https://github.com/intel/pti-gpu/blob/master/tools/unitrace/README.md#requirements), [NEO METRICS.md](https://github.com/intel/compute-runtime/blob/master/programmers-guide/METRICS.md#dependencies) | Select the installed unitrace tool's own tag and read Requirements. Follow the NEO revision to [Metrics Discovery](https://github.com/intel/metrics-discovery) and [Metrics Library](https://github.com/intel/metrics-library), checking `intel-metrics-discovery` / `libmd*` and `intel-metrics-library` / `libigdml*` separately. |
| **Sysman / Fabric / XPU-SMI (when in scope)** | [NEO Level Zero BUILD](https://github.com/intel/compute-runtime/blob/master/level_zero/doc/BUILD.md), [XPU Manager releases](https://github.com/intel/xpumanager/releases), [Install guide](https://github.com/intel/xpumanager/blob/master/doc/Install_guide.md#requirements) | Use interface conditions from the matching NEO revision and Requirements from the installed XPU Manager tag. Distinguish CLI, service, and build dependencies. |

**Parent-package control fields**: prefer the local package database/cache for an established channel. If unavailable, use the installation page's repository configuration to locate the official package with the same version and architecture:

| Package system | Locate the official package | Read and compare |
|---|---|---|
| Debian / Ubuntu | Obtain `base/suite/component` from the [PVC LTS repository configuration](https://dgpu-docs.intel.com/legacy/installation-guides/max-and-flex/installation.html), Client/PPA, or OMIX installation entry points above. Read the `<component>/binary-<arch>/Packages` index referenced by `dists/<suite>/InRelease` / `Release`, and select `Filename` by package name/version/architecture. For distribution packages, use [Ubuntu package search](https://packages.ubuntu.com/) for the matching release. | Read `Depends`, `Pre-Depends`, and `Provides` from the index or `.deb` control data. Query installed values with `dpkg-query -W -f='${binary:Package} ${Status} ${Version}\n' <package>` and compare with `dpkg --compare-versions`. |
| RPM | Obtain `baseurl` from the corresponding installation page. Read the primary index referenced by `repodata/repomd.xml` and locate the RPM by name/architecture/EVR. For direct upstream installations, use Assets from the matching NEO release. | Read requires/provides or the RPM header. Query installed values with `rpm -q --qf '%{NAME} %{EPOCHNUM}:%{VERSION}-%{RELEASE} %{ARCH}\n' <package>`. For simple capabilities, use `rpm -q --whatprovides '<capability >= EVR>'`; handle compound dependencies according to RPM semantics. |

Preserve epochs, distribution suffixes, bounds, alternatives, and versioned `Provides`. Do not turn `Recommends` into hard requirements. A difference between channel package versions and upstream tags does not prevent comparison within the same package system. Mark Unresolved only when a requirement crosses version schemes and needs an unconfirmed mapping; do not guess by stripping suffixes.

## 3. Query the Correct Environment

| Object | Collection location and evidence |
|---|---|
| Userspace libraries: SYCL, UR, NEO, IGC, loaders, etc. | Inspect the container/host of the target Python/process. Record library paths, owning packages, and versions. When multiple copies exist, verify actual loaded paths, such as `/proc/<pid>/maps`. Merely being installed, or not yet loaded, does not establish that a copy is in use. |
| Level Zero API | Use `ze_info` or `zeDriverGetApiVersion` on the driver handle owning the target GPU. Record the API version separately from loader/driver package versions. |
| OpenCL / OpenCL C | Use `clinfo`, matching the device by BDF/UUID. Record the device API, default and all supported OpenCL C language versions, 3.0 feature macros, and extensions separately. |
| Windows / WSL2 | Map PnP device instances/hardware IDs to `Win32_PnPSignedDriver.DriverVersion`. If only the overall package version is exposed, note that the internal IGC version is unknown. Query WSL2 guest packages inside the guest; retain the other side's evidence gap if only host or guest evidence is available. |
| External tools | Inspect the tool path resolved by the consumer, its owning package, and version output. |

### Classifying OpenCL C 2.0 Requirements

Under the [OpenCL specification](https://registry.khronos.org/OpenCL/specs/unified/html/OpenCL_API.html#CL_DEVICE_OPENCL_C_ALL_VERSIONS), 3.0 makes several capabilities mandatory in 2.0 optional. OpenCL C 3.0 support therefore does not guarantee full 2.0 support; a language list may contain 1.2 and 3.0 without 2.0. `CL_DEVICE_OPENCL_C_VERSION` reports the highest fully backward-compatible language version and is not, by itself, the complete supported-language list.

First align the consumer's version-specific requirement with the target device's `CL_DEVICE_OPENCL_C_ALL_VERSIONS`, `CL_DEVICE_OPENCL_C_FEATURES` / `__opencl_c_*`, and required extensions:

| Evidence and requirement | Result |
|---|---|
| Only the default language is 1.2, or the list lacks 2.0.0 | Do not immediately mark the entire consumer Fail. Distinguish full 2.0 support from specific feature requirements and check whether upstream accepts 3.0 as an alternative. |
| Upstream explicitly accepts 3.0 plus specified features, and all conditions are established | Those requirements Pass. A confirmed missing mandatory feature is Fail. |
| Upstream explicitly requires full 2.0 support, the device declares it unavailable, and there is no official alternative | Fail. A subset of 3.0 features cannot replace full 2.0 support. |
| Upstream only says OpenCL C 2.0, its policy on a 3.0 alternative is unclear, or official statements conflict | Unresolved. Record the language list, features, and upstream wording; do not decide Pass or Fail from version numbers alone. |

Read feature declarations returned by device queries and compare them with official requirements. If evidence is insufficient, retain Unresolved; do not supplement the decision with self-written compilation or execution tests. Compare API, language, header, and package versions separately. Headers apply only to build checks.
