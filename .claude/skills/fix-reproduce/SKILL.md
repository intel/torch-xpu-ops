---
name: fix-reproduce
description: >
  Use when asked to reproduce a bug, verify a nightly CI failure, or
  confirm a failure still exists on latest source. Verifies whether a
  bug still reproduces before an orchestrator commits time to a fix.
  Runs a three-stage fallback (nightly wheel -> source build -> CI
  environment alignment) and returns REPRODUCED / NOT_REPRODUCED /
  NO_REPRODUCER / CANNOT_VERIFY. Called by both issue-handler and
  xpu-nightly-ci-fix orchestrators.
---

# Reproduce — Verify the Bug Exists

Runs a test and determines whether the bug reproduces. Uses a three-stage
approach: nightly wheel first (fast), source build at CI commit second (precise),
CI environment alignment third (last resort).

The orchestrator decides what to do with the output — this skill only reports
the result.

## Inputs

- `reproducer_command` — the sequence of shell commands that triggers the
  failure. Any of the forms below is valid; Stage 1's "Reproducer forms"
  section routes execution:

  1. A pytest node id or `pytest ...` invocation:
     `pytest -v test/xpu/test_ops.py::TestFooXPU::test_bar_xpu_float32`
  2. A `python -c "..."` snippet or a single `python script.py` line.
  3. A multi-line shell block (as is common in issue bodies): env setup,
     `git clone`, `pip install`, followed by the actual failing command.

  Missing or unrunnable inputs return `NO_REPRODUCER` up-front — see
  "## Preflight" below. Non-`NO_REPRODUCER` verdicts come from the
  Stage 1/2/3 execution flow.

  Providers (set by the orchestrator, not this skill):
  - **Issue body** — extracted by `issue-triage` from the reproducer section
    (via `issue-handler`).
  - **CI failure log** — the failing pytest node id from the nightly CI report
    (via `xpu-nightly-ci-fix`).
- `stage` — which reproduction path to run. Default `auto`.
  - `auto` — run the full three-stage fallback chain (nightly → source_build →
    ci_env). Used by orchestrators that need a definitive verdict.
  - `nightly` — only run Stage 1 (nightly wheel). PASS returns
    `NOT_REPRODUCED(checked_stages=[nightly])` immediately; do NOT fall through
    to source build or ci_env. Cheapest option — suitable for a fast "does this
    still reproduce on latest nightly?" answer.
- `ci_commit` — upstream commit hash from the CI report. Only used as a
  fallback base when `origin/main` fails to build (optional; ignored when
  `stage=nightly`).
- `pytorch_dir` — path to a local PyTorch checkout (optional). Used
  whenever Prepare determines `needs_tree=yes` (any pytest form with a
  repo-relative path) as well as by Stage 2's source build. If absent,
  clone to `<torch-xpu-ops-repo-root>/agent_space_xpu/pytorch/`
  (`agent_space_xpu/` is the gitignored scratch dir at the torch-xpu-ops
  repo root — see the containing repo's `AGENTS.md`).
- `ci_repo` — which CI to align against in Stage 3: `pytorch` or
  `torch-xpu-ops`. Optional; when absent, Stage 3 infers from the
  reproducer path (see "Determine `ci_repo`" in Stage 3).

## Preflight

`NO_REPRODUCER` is a pre-execution verdict — the skill decides that
there is nothing to run before touching any stage. Every other verdict
(REPRODUCED / NOT_REPRODUCED / CANNOT_VERIFY) comes from Stage 1/2/3
execution.

**`reproducer_command` present?** If missing or empty:
`NO_REPRODUCER(reason=no_command)`. Stop.

If the input is a stack trace without a command, the orchestrator should
not call this skill in the first place (that is `issue-triage`'s
`reproduction_missing=yes` case); if it slips through, this check
catches it.

The pytest `collected 0 items` check happens in **Prepare** below, after
the source tree it needs to run against is in place.

## Prepare

Some reproducer forms need a pytorch source tree even at Stage 1
(nightly wheel path) — either because the test file lives inside
`pytorch/test/` or because it lives under `torch-xpu-ops/test/xpu/`
and imports common test utilities via
`sys.path.append("../../../../test/functorch")` relative paths that
only resolve from `<pytorch_dir>/third_party/torch-xpu-ops/test/xpu/`.

### When to prepare

Set `needs_tree` from the reproducer form:

| Reproducer form | `needs_tree` |
|---|---|
| pytest, path is repo-relative (`test/xpu/...`, `test/...`) | yes |
| pytest, bare node id without a file path (`TestFoo::test_bar`) | yes — pytest rootdir discovery needs the tree |
| pytest, path is absolute and exists on disk | no |
| `python -c "..."` / `python /abs/path/script.py` | no |
| shell block (issue body: clone + install + run) | no — the block does its own setup |

If `needs_tree=no`, skip this section and go to Stage 1.

### Get the pytorch tree

If `pytorch_dir` was provided as input: `git -C $pytorch_dir fetch origin`.

If not provided, clone into the torch-xpu-ops repo's gitignored scratch
dir. Resolve the path explicitly rather than relying on cwd:

```bash
XPU_OPS_ROOT=$(git -C <path-to-torch-xpu-ops-checkout> rev-parse --show-toplevel)
pytorch_dir="$XPU_OPS_ROOT/agent_space_xpu/pytorch"
if [[ ! -d "$pytorch_dir/.git" ]]; then
  git clone --filter=blob:none https://github.com/pytorch/pytorch.git "$pytorch_dir"
fi
git -C "$pytorch_dir" fetch origin
git -C "$pytorch_dir" checkout --detach origin/main
git -C "$pytorch_dir" submodule update --init --recursive
```

The tree is **not built** here — Stage 1 uses the nightly wheel for the
runtime; the source tree only supplies test files and support modules.
Stage 2 reuses the same tree and builds it there.

### torch-xpu-ops test path

If the reproducer targets `test/xpu/...`, make the working torch-xpu-ops
tree available at `$pytorch_dir/third_party/torch-xpu-ops`. The build's
dev-override recipe (symlink or replace-clone) applies; see
`xpu-build-pytorch`. From here on, Stage 1's cwd for pytest is
`$pytorch_dir/third_party/torch-xpu-ops/test/xpu/`.

### Collect-only check (pytest form)

Regardless of `needs_tree`, if `reproducer_command` matches the pytest
form (starts with `pytest`, `python -m pytest`, or is a bare pytest
node id), run:

```bash
pytest --collect-only <node_id>
```

Cwd:
- `needs_tree=yes` → from the tree just prepared (for `test/xpu/...`
  targets, that's `$pytorch_dir/third_party/torch-xpu-ops/test/xpu/`)
- `needs_tree=no` → from any non-pytorch directory (the reproducer's
  absolute path resolves on its own)

Output shows `collected 0 items`? → `NO_REPRODUCER(reason=collected_zero)`.
Stop. Do not fall through to Stage 2 — the source tree is the same
across stages, so a collect-miss at Stage 1 will miss at 2 and 3 too.

Non-pytest forms have no equivalent pre-execution check.

## Stage 1: Nightly Wheel (fast path)

Most failures reproduce here. Start here before doing anything heavier.

### Reproducer forms

Three forms; each dispatches differently in "Run test" below:

- **Pytest form** — `reproducer_command` starts with `pytest`,
  `python -m pytest`, or is a bare pytest node id
  (`.../test_foo.py::TestBar::test_baz`). The `collect-only` check
  (see "## Prepare" above) has already run.
- **Python one-liner / single-script form** — `python -c "..."` or
  `python path/to/script.py`.
- **Shell-block form** — a multi-line block copied out of an issue
  body: env setup + `git clone` + `pip install` + the failing command.
  Split it into **setup steps** (everything before the failing
  command) and **the reproduce step** (the last command that
  exercises the failing path). Only the reproduce step's outcome
  determines the verdict; setup-step failures return
  `CANNOT_VERIFY(stage=<current>, blocker=<the failing setup step>)`.

The **Working directory** and **Use the test's own assertion** rules
below apply to all three forms.

### Install

Always reproduce against the **latest** available XPU nightly. Do not reuse a
stale wheel from a previous session — a bug may already be fixed in a newer
nightly, and re-verifying an old wheel produces misleading `REPRODUCED`
results.

```bash
# Query available versions (informational — pip install --upgrade below
# will pick a resolvable one, which may lag the newest entry here by a
# day when the index metadata refreshes before all wheels land).
pip3 index versions torch --pre \
  --index-url https://download.pytorch.org/whl/nightly/xpu
pip3 install --pre --upgrade torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/xpu
```

Post-install, check that `torch`, `torchvision`, `torchaudio` are all
from the same day — pip's resolver can leave a mixed set (either torch
older than the auxiliary wheels, or the reverse). If they diverge,
uninstall all three and reinstall together:

```bash
pip3 uninstall -y torch torchvision torchaudio
pip3 install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/xpu
```

Record the exact wheel version used (`python -c "import torch;
print(torch.__version__)"`) in the reproduce output and in any issue comment,
so downstream stages and re-verifications know which nightly was tested.


### Working directory

Do NOT run the nightly-wheel reproducer with `cwd` inside any pytorch
source checkout. Python resolves `import torch` against the local `torch/`
package before site-packages, so it will load the in-tree `torch/_C.so`
built at whatever revision that tree happens to be — typically stale
relative to the installed wheel — and fail with
`ImportError: undefined symbol: ...`. Either `cd $(mktemp -d)` (or any
non-pytorch dir) before running, or invoke the reproducer with an
absolute path from outside the tree.

### torch-xpu-ops test invocation

**Applies when the reproducer targets a test under `torch-xpu-ops/test/xpu/`.**
Prepare has already ensured the pytorch tree exists at `$pytorch_dir`
with the working torch-xpu-ops tree at
`$pytorch_dir/third_party/torch-xpu-ops`. Invoke the reproducer from
`$pytorch_dir/third_party/torch-xpu-ops/test/xpu/` — the relative
`sys.path.append("../../../../test/functorch")` in those tests only
resolves from that cwd.

The pytorch tree does NOT need to be built for the nightly-wheel path —
the wheel provides the runtime, the source tree only supplies test
files and support modules.

### Use the test's own assertion

When writing a standalone reproducer for a `TestCase.assertEqual`
failure, use the test's own assertion. **Do NOT substitute**
`torch.allclose`, `torch.equal`, or bare `==` — they have different
(usually stricter) default tolerances and will manufacture false
positives.

If the failure log says `AssertionError: Tensor-likes are not close`,
the assertion is `torch.testing._comparison.assert_close`, which has
dtype-specific defaults (bf16: `rtol=0.016, atol=1e-5`). Reproduce
through `assert_close` or via `TestCase.assertEqual`:

```python
import sys; sys.path.insert(0, "<pytorch>/test")
from torch._dynamo.test_case import TestCase   # or the base class the failing test uses

class T(TestCase):
    def test_x(self, device):
        ...
        self.assertEqual(out_ref, out)

T().test_x(device='xpu')
```

### Run test

Run according to the reproducer form matched in "Reproducer forms":

- **Pytest form:** run the pytest invocation. Result interpretation:
  - `FAILED` → REPRODUCED.
  - `all skipped` by `@skipIfXpu` → the marker is likely hiding the
    actual failure. Temporarily remove it in place, re-run once to
    check what happens without the skip, then revert the file so no
    change escapes this skill:

    ```bash
    # Remove @skipIfXpu from the target test file(s), then:
    pytest <node_id>
    # Regardless of outcome, revert:
    git checkout <test_file>
    ```

    If the re-run FAILs → REPRODUCED (the skip was hiding it). If it
    PASSes → treat as PASSED per the Decision table. Only return
    `CANNOT_VERIFY` when the skip is environmental (not an XPU marker,
    e.g. `@skipIf(not has_cuda)` shielding an unavailable dep).
  - `xfailed` → treat as `FAILED` (REPRODUCED).
  - `PASSED` → per the Decision table below.

- **Python one-liner / shell block:** run the command (or the reproduce
  step extracted from the shell block, per "Reproducer forms"). Result
  interpretation:
  - Exit code non-zero **and** output matches the failure pattern
    named in the issue (traceback, error message, or specific
    assertion) → REPRODUCED.
  - Exit code zero → PASSED (per the Decision table below).
  - Exit code non-zero **but** cause is unrelated (missing dependency
    surfacing inside the reproduce step, permission error, missing
    device) → `CANNOT_VERIFY(stage=nightly, blocker=<...>)`. Do NOT
    report REPRODUCED on a setup or infra failure.

  `all skipped` / `xfailed` do not apply to these forms — they are
  pytest-specific concepts.

### Decision

| Result | Condition | Action |
|--------|-----------|--------|
| `CANNOT_VERIFY` | env problem (wheel install failed, runtime missing) | Report to orchestrator, stop |
| `REPRODUCED` | FAILED | Return `REPRODUCED(stage=nightly, refined_command=...)` |
| → stage 2 | PASSED **and** `stage=auto` | Proceed to source build at `origin/main` to confirm |
| `NOT_REPRODUCED` | PASSED **and** `stage=nightly` | Return `NOT_REPRODUCED(checked_stages=[nightly])` — do NOT fall through |

## Stage 2: Source Build at origin/main

Nightly passing is not conclusive — it may lag behind CI. Build from
`origin/main` to verify. Even when the failure came from a specific CI
commit, we only consider fixes on top of `origin/main` — downstream
stages branch off it.

### Prepare pytorch checkout

If Prepare (above) already ran (`needs_tree=yes`), `$pytorch_dir` is
detached at `origin/main` with submodules initialized — skip to
"Build and run".

Otherwise (Prepare was skipped because `needs_tree=no`), run Prepare's
"Get the pytorch tree" recipe now: resolve `$pytorch_dir` (from input
or `$XPU_OPS_ROOT/agent_space_xpu/pytorch`), clone if missing, fetch,
`checkout --detach origin/main`, `submodule update --init --recursive`.

Leave HEAD detached at `origin/main` at exit (the `ci_commit` fallback
below re-detaches to a different sha; downstream stages branch off
whatever this stage settled on).

### Build and run

Load the `xpu-build-pytorch` skill and follow it for the build. Do not
hand-roll the build here.

If the `origin/main` build fails for a reason unrelated to the bug
(broken trunk, upstream infra issue, etc.) and `ci_commit` is
available, fall back once:

```bash
git -C $pytorch_dir checkout --detach $ci_commit
git -C $pytorch_dir submodule update --init --recursive
```

Rebuild via `xpu-build-pytorch`. If this succeeds, proceed with the
test on `ci_commit` and record `base=<ci_commit_sha>` in the output so
the orchestrator branches its fix off the same base. If the fallback
build also fails, escalate as
`CANNOT_VERIFY(blocker=trunk and ci_commit both fail to build)`
rather than silently reproducing on some other base.

Then run the reproducer following the form-specific rules in Stage 1
"Run test" (pytest interpretation vs python/shell interpretation).

### Decision

Applies only when `stage=auto` — `stage=nightly` returns at Stage 1.

| Result | Action |
|--------|--------|
| `CANNOT_VERIFY` | Report to orchestrator, stop |
| `REPRODUCED` | Return `REPRODUCED(stage=source_build, base=origin/main\|<ci_commit_sha>, refined_command=...)` |
| `PASSED` | Proceed to stage 3 |

## Stage 3: CI Environment Alignment

Only reached when nightly wheel and source build at `origin/main` both
pass. The failure may be specific to the CI environment: wheels built
under CI toolchain, XPU/oneAPI stack pinned to a specific version,
environment variables set by CI.

### Assumption: the agent already runs inside the CI test container

Both pytorch/pytorch and torch-xpu-ops run their XPU tests **inside**
a container (declared as `container: image:` in the workflow yaml).
When this skill is invoked from within CI (e.g. via `@torchxpubot
fix`), the agent is already inside that container — kernel modules,
`/dev/dri`, oneAPI stack, and python are already the CI ones. This
stage does **not** `docker pull` or `docker run`. It aligns the
installed wheels + pytorch source checkout to the CI wheel, then runs
the reproducer directly.

The skill logs the CI image reference it identified (for context in the
report), but does not exec into it.

### Clean up Stage 2 artifacts

Stage 2 may have left behind `build/`, `torch/lib/*.so`, or a modified
`third_party/xpu.txt` from the dev-override. Left in place, Python
will pick the host-built (stale-relative-to-CI-wheel) `torch/_C.so`
off `sys.path` and error with `undefined symbol` before the
reproducer runs.

```bash
# Restore xpu.txt to origin's pinned commit (in case Stage 2 rewrote it).
git -C "$pytorch_dir" checkout -- third_party/xpu.txt
# Discard stage-2 build outputs. `git clean` does not recurse into
# nested repositories by default, so third_party/torch-xpu-ops (a
# separate git repo) is preserved without needing `-e`.
git -C "$pytorch_dir" clean -fdx
```

Alternatively, run the reproducer with `cwd` outside `$pytorch_dir`
(e.g. `cd /tmp`) so `import torch` resolves against site-packages,
matching Stage 1's "Working directory" rule. Do at least one.

### Determine `ci_repo`

Pick which CI to align against based on the reproducer:

| Reproducer clue | `ci_repo` |
|---|---|
| Path contains `test/xpu/` or `torch-xpu-ops` | `torch-xpu-ops` |
| Path is `pytorch/test/...` or absolute path inside a pytorch tree | `pytorch` |
| Ambiguous / `python -c` snippet with no path | try `torch-xpu-ops` first, fall back to `pytorch` |

The orchestrator (`issue-handler` for issues from either repo,
`xpu-nightly-ci-fix` for torch-xpu-ops nightly failures) may also
pass `ci_repo` explicitly; when set, use it and skip the heuristic.

### Path A — `ci_repo=torch-xpu-ops`

Wheels come from intel/torch-xpu-ops's own build workflow and stay on
GitHub Actions artifact storage. Fetch via `gh run download`, not S3.

#### A1. Find the latest successful wheel-producing run

The build job lives in `_linux_build.yml` (a reusable workflow called
by `pull.yml` and `nightly_ondemand.yml`). It uploads the artifact
`Torch-XPU-Wheel-<pr|sha>-<runid>-<attempt>[-category]`.

```bash
# Nightly is the primary source (fresh main-branch build every night).
# Fall back to pull.yml when nightly has been failing for a stretch —
# pull.yml runs on PRs against main and its wheels are close enough for
# CI-env alignment.
RUN=$(gh run list --repo intel/torch-xpu-ops \
  --workflow nightly_ondemand.yml \
  --status success --limit 1 \
  --json databaseId,headSha,createdAt)
if [[ "$RUN" == "[]" ]]; then
  RUN=$(gh run list --repo intel/torch-xpu-ops \
    --workflow pull.yml \
    --status success --limit 1 \
    --json databaseId,headSha,createdAt)
fi
# Empty here → report CANNOT_VERIFY per the paragraph below (do not
# `exit`; the skill returns a verdict, it does not terminate the shell).
RUN_ID=$(jq -r '.[0].databaseId' <<<"$RUN")
```

If both queries return empty: `CANNOT_VERIFY(stage=ci_env,
blocker=no_recent_successful_torch-xpu-ops_run)`.

#### A2. Download the wheel artifact

```bash
CI_ENV_DIR="$XPU_OPS_ROOT/agent_space_xpu/ci_env"
WHEELS_DIR="$CI_ENV_DIR/wheels"
rm -rf "$WHEELS_DIR" && mkdir -p "$WHEELS_DIR"

# Artifact name is Torch-XPU-Wheel-<pr|sha>-<runid>-<attempt>[-category].
# `gh run download -n <name>` requires exact name; use pattern instead.
# If the run uploaded multiple category variants (target/baseline via
# `_linux_build.yml`'s `category` input), --pattern pulls all of them
# and the flatten below will clobber same-named wheels. In that case
# pass --name <specific-artifact> to pick one variant.
gh run download "$RUN_ID" --repo intel/torch-xpu-ops \
  --pattern 'Torch-XPU-Wheel-*' --dir "$WHEELS_DIR"

# gh unpacks each artifact into its own subdir; flatten:
find "$WHEELS_DIR" -mindepth 2 -name '*.whl' -exec mv {} "$WHEELS_DIR" \;
# Empty here → CANNOT_VERIFY(stage=ci_env, blocker=no_wheel_in_artifact).
# Do not `exit`; return the verdict via the skill's Output section.
find "$WHEELS_DIR" -maxdepth 1 -name '*.whl' | grep -q .
```

#### A3. CI image (reference only)

For torch-xpu-ops the test container is
`intelgpu/ubuntu-24.04-lts2:2523.40` (see
`.github/workflows/_linux_ut.yml`). Record the tag for the report;
do not pull. Kept for local-investigation convenience (someone
reproducing outside CI can `docker run` this image manually) — the
skill itself relies on the Assumption above.

### Path B — `ci_repo=pytorch`

Wheels come from pytorch/pytorch's `xpu` workflow and land on
gha-artifacts S3.

#### B1. Find the latest successful `xpu` workflow run

Accept only runs where every `linux-*/ build` job succeeded — partial
runs still upload partial artifacts.

```bash
# Match by display name first; fall back to path in case pytorch/pytorch
# renames the workflow's `name:` field (the file path is more stable).
WF_ID=$(gh api "repos/pytorch/pytorch/actions/workflows?per_page=100" --paginate \
  --jq '.workflows[] | select(.name=="xpu" or .path==".github/workflows/xpu.yml") | .id' \
  | head -1)

RUN_ID=""
for page in 1 2 3 4 5; do
  while IFS=$'\t' read -r rid _ _; do
    conclusions=$(gh api \
      "repos/pytorch/pytorch/actions/runs/$rid/jobs?per_page=100" --paginate \
      --jq '.jobs[] | select(.name | test("^linux.*/ build$")) | .conclusion')
    [[ -z "$conclusions" ]] && continue
    grep -qv '^success$' <<<"$conclusions" && continue
    RUN_ID=$rid; break 2
  done < <(gh api \
    "repos/pytorch/pytorch/actions/workflows/$WF_ID/runs?status=completed&per_page=20&page=$page" \
    --jq '.workflow_runs[] | [.id, .head_sha, .created_at] | @tsv')
done
```

If no qualifying run in the last 100: `CANNOT_VERIFY(stage=ci_env,
blocker=no_recent_successful_xpu_workflow_run)`.

#### B2. Pick the right `build_env`

Per the Assumption above, the agent is **already inside** a
compatible CI container — the image column below is a lookup for
the report only, not a pull target. It is kept in case a local
investigation (outside CI) wants to spin up the same image
manually to reproduce; the skill itself does not use it.

pytorch/pytorch's `xpu.yml` currently defines only `py3.10` linux builds:

| build_env                          | Hardware | Image (reference)                                             |
|------------------------------------|----------|---------------------------------------------------------------|
| `linux-noble-xpu-n-py3.10`         | PVC      | `ghcr.io/pytorch/ci-image:pytorch-linux-noble-xpu-n-py3-<docker-tree-hash>`|
| `linux-noble-xpu-n-py3.10-client`  | BMG      | `ghcr.io/pytorch/ci-image:pytorch-linux-noble-xpu-n-py3-client-<docker-tree-hash>`|
| `linux-jammy-xpu-n-1-py3.10`       | PVC      | `ghcr.io/pytorch/ci-image:pytorch-linux-jammy-xpu-n-1-py3-<docker-tree-hash>`|

`-client` suffix = BMG (client GPU); no suffix = PVC (datacenter).
Match the runner's hardware; default to PVC when unknown.

If `xpu.yml` grows a new build_env not covered here:
`CANNOT_VERIFY(stage=ci_env, blocker=unknown_build_env=<name>)`. Do
not guess. When multiple envs match, iterate them in sorted order for
determinism.

`<docker-tree-hash>` is the git tree hash of `.ci/docker/` at the run's
commit (see upstream `_runner-determinator.yml` "Compute .ci/docker
tree hash"). Only needed if the report wants a fully-qualified image
reference; skill does not pull the image.

#### B3. Download wheel artifacts

Artifacts live at:

```
https://gha-artifacts.s3.amazonaws.com/pytorch/pytorch/<run_id>/<build_env>/artifacts.zip
```

Probe availability with `--range 0-0 -L` (zero-byte GET); HEAD may be
rejected by some intermediaries in front of this bucket, byte-range
GET returns 200 or 206:

```bash
url="https://gha-artifacts.s3.amazonaws.com/pytorch/pytorch/$RUN_ID/$BUILD_ENV/artifacts.zip"
http_status=$(curl -s -o /dev/null -w "%{http_code}" --range 0-0 -L "$url")
# 200 or 206 -> ok; anything else -> skip this build_env
```

Download and extract:

```bash
CI_ENV_DIR="$XPU_OPS_ROOT/agent_space_xpu/ci_env"
WHEELS_DIR="$CI_ENV_DIR/wheels"
ARTIFACTS_ZIP="$CI_ENV_DIR/artifacts.zip"
rm -rf "$WHEELS_DIR" && mkdir -p "$WHEELS_DIR"
# --retry survives transient drops. A silent truncation of the 1.2 GB
# zip surfaces later as "cannot find zipfile directory" from unzip.
curl -sL -f --retry 3 --retry-delay 5 "$url" -o "$ARTIFACTS_ZIP"

# Layout varies: some build envs pack wheels under dist/, others at root.
unzip -o -j "$ARTIFACTS_ZIP" 'dist/*.whl' -d "$WHEELS_DIR" \
  || unzip -o -j "$ARTIFACTS_ZIP" '*.whl' -d "$WHEELS_DIR"
# Empty here → CANNOT_VERIFY(stage=ci_env, blocker=no_wheel_extracted).
find "$WHEELS_DIR" -maxdepth 1 -name '*.whl' | grep -q .
```

### Install the CI wheel and align source

Same for both paths. Uninstall any existing torch stack first — the
Stage 1 nightly is still resident:

```bash
pip uninstall -y torch torchvision torchaudio pytorch-triton-xpu triton_xpu 2>/dev/null || true
pip install --force-reinstall "$WHEELS_DIR"/*.whl
```

Align the pytorch source tree to the wheel's commit so tests that
import support modules from `pytorch/test/` see matching code (Prepare
left `$pytorch_dir` detached at `origin/main`, which is not the wheel's
commit):

```bash
TORCH_COMMIT_ID=$(python -c 'import torch; print(torch.version.git_version)')
git -C "$pytorch_dir" fetch origin
git -C "$pytorch_dir" checkout --detach "$TORCH_COMMIT_ID"
git -C "$pytorch_dir" submodule update --init --recursive
```

Do not shallow-fetch (`--depth 1`) here — pytorch's nested submodules
resolve against pins that require the full history to be reachable;
a shallow fetch surfaces later as "cannot find <sha>" in `submodule
update`.

`TORCH_COMMIT_ID` (the wheel's build commit) is a **temporary
alignment only** — it makes `pytorch/test/` support modules match the
installed wheel's binary so the test can run. It is **not** the fix
base and is never returned as `base`. Downstream fixes always branch
off `origin/main` (see Stage 2), so this stage still reports
`base=origin/main`; `TORCH_COMMIT_ID` stays internal to Stage 3.

For torch-xpu-ops test paths, ensure the working torch-xpu-ops tree is
at `$pytorch_dir/third_party/torch-xpu-ops` (Prepare already handled
this if `needs_tree=yes`; if it didn't, do it now via the
`xpu-build-pytorch` dev-override recipe).

### Run the reproducer

Run in the current shell — no `docker run` wrapper. Working-directory
rule from Stage 1 applies: for torch-xpu-ops tests use
`$pytorch_dir/third_party/torch-xpu-ops/test/xpu/`; for a
non-test-file reproducer, `cd /tmp` (or any non-pytorch dir).

Result interpretation is the form-specific rule from Stage 1 "Run
test" — pytest FAILED / all skipped / xfailed, or non-pytest exit code
+ output-vs-failure-pattern.

### What to check if the failure still doesn't reproduce

From the CI job log, extract and align remaining differences:
- Full test command with all flags (`--timeout`, `-x`, specific env vars)
- Any environment variables set in the CI job (`ZE_AFFINITY_MASK`,
  `PYTORCH_TEST_WITH_XPU`, `IS_XPU_CI`, etc.)

When aligning yields REPRODUCED, fold discovered pieces (env vars,
flags, cwd) into `refined_command` per its contract in the Output
section.

### Decision

| Result | Action |
|--------|--------|
| `CANNOT_VERIFY` | Report to orchestrator, stop |
| `REPRODUCED` | Return `REPRODUCED(stage=ci_env, base=origin/main, refined_command=...)` — base is `origin/main`, not the wheel's `TORCH_COMMIT_ID` the tree is currently detached at |
| `PASSED` | Return `NOT_REPRODUCED(checked_stages=[nightly, source_build, ci_env])` — issue no longer exists; orchestrator reports to user or triage collects reason |

## Output

Return one of these to the orchestrator:

```
REPRODUCED
  stage: nightly | source_build | ci_env
  base: origin/main | <ci_commit_sha>    # base for downstream build. Default origin/main (also for stage=ci_env). ci_commit_sha only when stage=source_build fell back to ci_commit. Stage 3's TORCH_COMMIT_ID wheel-alignment checkout is never returned as base.
  refined_command: <single shell-executable string>
```

**`refined_command` contract.** A single shell-executable string that,
run by itself, reliably triggers the failure. A downstream skill (a
fix-verifier, a skip-list per-entry runner, etc.) invokes it directly
(e.g. via `bash -c "$refined_command"`) after applying a candidate fix
to check whether the failure is gone. Consequences:

- **Include everything needed to reproduce.** Env vars go as inline
  prefix (`ZE_AFFINITY_MASK=0 pytest ...`), a required cwd goes as a
  `cd <dir> &&` prefix.
- **Do NOT include setup steps.** `git clone`, `pip install`,
  wheel-download, etc. that appeared in the input shell block are
  excluded. The caller has already paid that cost; refined_command
  should re-trigger the failure, not re-provision the environment.
- **No `docker run` wrapper.** Stage 3 runs the reproducer directly in
  the current shell (the CI job is already inside its container, and
  the caller of refined_command runs from an equivalent env). If the
  caller needs container-level isolation, that is its concern, not
  refined_command's.
- **Not just the input command.** For the pytest form, refined_command
  may add `-sv`, `--timeout <N>`, or `-x` that Stage 1 used to get a
  usable failure signal. For the shell-block form, refined_command is
  the extracted "reproduce step" (usually the last line), not the
  whole block.
- **Quoting: use double quotes for inline python.** When the reproducer
  embeds python code, write it as `python -c "..."` (double quotes),
  not `python -c '...'`. Downstream callers wrap the string in
  `bash -c "$refined_command"`; single-quoted python payloads compose
  poorly through that wrapping.

```
NOT_REPRODUCED
  checked_stages: [nightly] | [nightly, source_build, ci_env]
  reason: <what was checked and confirmed to pass>

NO_REPRODUCER
  reason: no_command | collected_zero
  (returned when either no reproducer_command was provided, or pytest
  reports `collected 0 items` for the provided command)

CANNOT_VERIFY
  stage: nightly | source_build | ci_env
  blocker: <what went wrong>
```

The orchestrator decides the next step based on this output.
