---
name: xpu-ops-pr-creation
description: >
  How to create a pull request for the intel/torch-xpu-ops repository.
  Use when an agent has finished implementing a fix or feature and needs
  to prepare a branch and PR description that satisfies CI and review requirements.
placeholders:
  - branch       # e.g. agent/fix-relu-nan-handling
  - title        # PR title
  - description  # what the change does (one paragraph)
  - test_line    # the "Test:" line for the PR body (see rules below)
---

# PR Creation — torch-xpu-ops

Read `.github/copilot-instructions.md` for full repo context before starting.

---

## Step 1: Verify your branch

```bash
git status                        # confirm clean working tree
git log main..HEAD --oneline      # confirm commits are on your branch, not main
git diff main --stat              # sanity-check the changed files
```

Branch must follow: `agent/<slug>` (lowercase, hyphens, max 50 chars).

---

## Step 2: Write a reproducer test (mandatory)

Every PR must include a pytest-compatible test under `test/repro/` if it is newly introduced. If the test is in PyTorch or torch-xpu-ops tests, you should explicitly write the run command in the PR body (see Step 3).:

Note: For now, if the test is in PyTorch, explicitly point it out in PR message with bold font!

- File name: `test_<description>.py`
- Contains `def test_...()` functions or `class Test...` classes
- Runnable: `pytest test/repro/test_<description>.py`
- Imports `torch`, targets `xpu` device (also `cuda` if relevant)

**Exceptions** (CI-only changes, docs, build fixes): no new test file required, but you
must name existing tests that validate the change.

---

## Step 3: Compose the PR body

Use this exact template:

```
<description of what the PR does and why>

Test: <one of the three forms below>
```

**Three valid `Test:` forms:**

1. New reproducer added:
   ```
   Test: test/repro/test_<description>.py
   ```

2. Existing test(s) cover it:
   ```
   Test: test/repro/test_foo.py, test/xpu/test_bar_xpu.py
   ```

3. No test applicable:
   ```
   Test: none (<reason>)
   ```

The `Test:` line is required in every PR body. CI reviewers look for it.

---

## Step 4: Run linting checks

Before pushing, run pre-commit and/or lintrunner to catch style and lint issues:

```bash
# Option A — pre-commit (if configured)
pre-commit run --files $(git diff main --name-only)

# Option B — lintrunner (if configured)
lintrunner --take FLAKE8,MYPY,CLANGFORMAT -a $(git diff main --name-only)
```

Fix any reported issues and amend your commit before proceeding.
If neither tool is configured in the repo, skip this step and note it in your summary.

---

## Step 6: Push your branch

```bash
git push origin agent/<slug>
```

`origin` for this repo is `intel/torch-xpu-ops`. This is not a fork — push directly
to origin. **Do not push to `main`.**

---

## Step 7: Open the PR (or hand off)

Agents do not auto-open PRs for this repo. Output your summary so a human can review
and open the PR.

At the end of your response output EXACTLY this block:

```
### Agent Summary
- **What I found:** <one sentence>
- **What I changed:** <bullet list of files>
- **Test file:** test/repro/<filename>.py | none (<reason>)
- **Branch:** agent/<slug>
- **PR body draft:**

<title>

<description paragraph>

Test: <test line>

- **Open questions / risks:** <concerns or "None">
```

**Critical:** The `**Branch:**` line must contain the exact branch name. It is machine-parsed.

---

## Checklist before handing off

- [ ] pre-commit / lintrunner passed (or not configured — noted in summary)
- [ ] `yaml/`, `src/`, and `test/` are consistent (if any were touched)
- [ ] Test file added to `test/repro/` or `Test: none (reason)` written
- [ ] No unrelated files changed
- [ ] Commit message is imperative, max 72 chars
- [ ] Branch is `agent/<slug>`, not `main`
- [ ] No direct push to `main` or `upstream`
