# Plan: Add Local Verification Loop to code_fix.py

## Problem
The pipeline pushes agent fixes without verifying they actually work. The "✅ Fix verified locally" checkbox is never checked because no verification step exists.

## Design

### Flow (replaces current agent-only loop)

```
for attempt in 1..MAX_AGENT_ATTEMPTS:
    1. Agent produces fix (opencode)
    2. Commit changes
    3. Build (if C++/SYCL modified):
       - pytorch: `python setup.py develop` (incremental)
       - If build fails: try clean build once (`python setup.py clean && python setup.py develop`)
       - torch-xpu-ops: no build needed (pure source, tested via pytorch)
    4. Run reproducer/test:
       - Extract test command from issue body ("Failed Tests" or "Reproducer" section)
       - For torch-xpu-ops: also check if agent created `test/repro/test_*.py`
       - Run with pytest or the extracted command
    5. If test passes → break (success)
    6. If test fails → soft-reset to base, feed error output to agent as context, continue loop

If all attempts exhausted → push best attempt, note verification failure in issue body
```

### Key decisions
- Use existing `MAX_AGENT_ATTEMPTS` (3) for the entire fix+verify loop — no new config
- Incremental build by default, clean build only on build error or if LLM says must rebuild
- Time budget: increase `IMPLEMENTING` stage timeout to at least 3600s (already 3600, may need more — monitor)
- Update issue description with time budget note when verification is active
- The verify loop wraps around the existing agent call (not a separate step)

### Files to modify

1. **`issue_handler/fixing_steps/code_fix.py`** — Major rework:
   - Extract `_get_test_command(body)` — parse Failed Tests / Reproducer from issue body
   - Add `_incremental_build(workdir, target_repo)` — runs build, returns (success, output)
   - Add `_run_verification(workdir, test_cmd)` — runs test, returns (success, output)
   - Restructure main loop: agent → commit → build → verify → pass/retry
   - On retry: reset branch to base, include verification error in next agent prompt

2. **`issue-fix skill` (`~/.github/skills/issue-fix/SKILL.md`)** — Update:
   - Agent MUST output the test command it used for verification
   - Agent should create `test/repro/test_*.py` for torch-xpu-ops fixes
   - Emphasize: pipeline will independently verify, so agent must ensure fix actually works

3. **`config/agent_config.yml`** — Possibly increase IMPLEMENTING timeout if 3600s isn't enough for build+3 retries. Consider 7200s (2h).

4. **Issue body update** — When verification starts, append note to fix log with time estimate

### Verification command extraction logic

Priority order:
1. Issue body "Reproducer" section — use verbatim if present
2. Issue body "Failed Tests" section — construct `pytest <test_path>` from test names
3. For torch-xpu-ops: look for `test/repro/test_*.py` files created by agent
4. Fallback: skip verification, log warning, push as-is

### Build logic (pytorch only)

```python
def _incremental_build(workdir):
    # Check if any C++/SYCL files were modified
    diff_files = git_out("diff", "--name-only", f"{remote}/{base_ref}..HEAD", workdir=workdir)
    cpp_extensions = {'.cpp', '.h', '.cu', '.cuh', '.hpp'}
    needs_build = any(Path(f).suffix in cpp_extensions for f in diff_files.splitlines())
    
    if not needs_build:
        return True, ""  # Python-only change, no build needed
    
    # Incremental build
    rc, output = run_cmd("python setup.py develop", workdir=workdir, timeout=1800)
    if rc != 0:
        # Clean build fallback
        rc, output = run_cmd("python setup.py clean && python setup.py develop", workdir=workdir, timeout=2400)
    return rc == 0, output
```

### Error feedback to agent on retry

When verification fails, the next agent prompt includes:
```
Your previous fix attempt FAILED verification.

Test command: <cmd>
Test output (last 200 lines):
<output>

Please analyze the failure and produce a corrected fix. Do NOT repeat the same approach.
```

### Checklist before implementation
- [ ] Confirm pytorch incremental build works in ~/pytorch (`python setup.py develop`)
- [ ] Confirm test extraction from existing triaged issues works
- [ ] Write tests for `_get_test_command()` and `_incremental_build()`
- [ ] Implement and run on one issue as validation
