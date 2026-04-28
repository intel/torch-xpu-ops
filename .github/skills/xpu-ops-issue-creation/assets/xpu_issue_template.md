### 🐛 Describe the bug

[sentence 1: what fails on XPU and on which build channel]
[sentence 2: the concrete observed mismatch, crash, or wrong result]

Failure source: [CI UT failure / Manual validation]

Affected op/module: [e.g. `aten::native_batch_norm`]

### Upstream reference

Required for CI UT failure. For manual validation, include these fields when the bug was discovered from an upstream change.

- Upstream commit: [commit SHA or URL that triggered the failure]
- Upstream PR: [PR URL]
- Upstream issue: [issue URL, if applicable]
- What the upstream change did: [one-line summary]

### Environment and build context

- Date: [YYYY-MM-DD]
- Build: [torch build/channel or CI build identifier]
- PyTorch version / commit: [nightly tag, wheel version, or git SHA]
- torch-xpu-ops version / branch: [commit SHA, branch, or N/A]
- Platform: [OS / Python / device family if known]
- Assisted-by: opencode: [actual-model] [GitHub-API] [collect_env or CI-link]

### Failure details

Failure type: [new test added / existing test broken / test expectation changed / build break / wrong result]

Fill only the subsection that matches Failure source.

For CI UT failure:
- CI job link: [URL]
- Failing test: [e.g. `test/test_ops.py::TestCommonXPU::test_foo_xpu_float32`]
- Run command: [e.g. `python test/test_ops.py -k "test_foo_xpu_float32"`]

Relevant failure output:

```text
[traceback, assertion failure, or CI log excerpt]
```

For manual validation:

Reproducer:

```python
[minimal runnable reproducer]
```

Observed output:

```text
[actual traceback, assertion failure, or wrong-value output]
```

### Versions

For manual validation:

<details>
<summary>Collected with python -W ignore::RuntimeWarning -m torch.utils.collect_env</summary>

```text
[collect_env output]
```
</details>

For CI UT failure:

- CI job link supplements the environment reference.
- Include the build and version fields above in plain text even if the CI job link already contains them.
- Add collect_env only if a local rerun was also done and it adds useful context.