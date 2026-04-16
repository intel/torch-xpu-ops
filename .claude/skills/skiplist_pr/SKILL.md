---
name: skiplist_pr
description: Add test cases to torch-xpu-ops skip_list_common.py when issues are marked wontfix or not_target, then create a PR. Use when user wants to add a test to XPU skiplist from a GitHub issue.
---

# Skiplist PR

Add test cases to torch-xpu-ops skip list when issues are marked 'wontfix' or 'not_target', then create a PR.

## Quick Start

1. User provides issue URL(s) or number(s) from intel/torch-xpu-ops (can be multiple)
2. Fetch each issue (use webfetch if gh unavailable or unauthenticated)
3. Verify issue has 'wontfix' or 'not_target' label
4. Parse test cases from issue body (op_ut format)
5. Update skip_list_common.py with deduplication
6. Create PR (or user handles PR manually if gh not authenticated)

## Detailed Instructions

### Step 1: Get Issue Information

The user will provide one of:
- Single issue URL: `https://github.com/intel/torch-xpu-ops/issues/1234`
- Single issue number: `1234`
- Multiple issues: `3127, 3133, 3129, 3128, 3126`

Extract issue number(s) if URL(s) provided.

### Step 2: Fetch Issue Details

**Primary: Try gh CLI first:**
```bash
gh issue view <issue-number> --repo intel/torch-xpu-ops
```

**Fallback: Use webfetch:**
```bash
webfetch https://github.com/intel/torch-xpu-ops/issues/<issue-number> markdown
```

Parse from issue:
- Issue body (contains test info in "Cases:" section)
- Labels (check for 'wontfix' or 'not_target')
- Issue title

### Step 3: Validate Labels

Check if issue has either 'wontfix' or 'not_target' label in the labels section.

If labels are not present:
- Notify the user: "The issue does not have 'wontfix' or 'not_target' label"
- Ask: "Do you want to continue anyway?"

### Step 4: Parse Test Info

The issue body contains a "Cases:" section with test info in **op_ut format**:

```
Cases:  
op_ut,third_party.torch-xpu-ops.test.xpu.test_transformers_xpu.TestTransformersXPU,test_math_backend_high_precision_xpu
```

For multiple tests, there are multiple op_ut lines:
```
Cases:  
op_ut,third_party.torch-xpu-ops.test.xpu.test_transformers_xpu.TestSDPACudaOnlyXPU,test_fused_kernels_nested_broadcasting_kernel1_...
op_ut,third_party.torch-xpu-ops.test.xpu.test_transformers_xpu.TestSDPACudaOnlyXPU,test_fused_kernels_nested_broadcasting_kernel1_...
```

**Parse rules:**
1. Extract all text after "Cases:" (may span multiple lines)
2. For each line starting with `op_ut,`:
   - Split by comma: `op_ut, third_party.torch-xpu-ops.test.xpu.test_transformers_xpu.TestClassName , test_method_name_xpu`
   - Test file: extract filename from path → `test_transformers_xpu.py`
   - Test method: last part after last comma (strip `_xpu` suffix if present)
3. Build list of unique test method names

**Alternative formats to support:**

Format 2: pytest command
```
pytest_command:  
cd /third_party/torch-xpu-ops/test/xpu && pytest -v test/xpu/test_transformers_xpu.py -k test_math_backend_high_precision_xpu
```
- Extract filename: `test_transformers_xpu.py` (from path after `test/xpu/`)
- Extract test: after `-k ` (strip `_xpu` suffix)

Format 3: pytest class path
```
python test/xpu/test_transformers_xpu.py TestTransformersXPU.test_math_backend_high_precision_xpu
```
- Extract filename and test method name

### Step 5: Update skip_list_common.py

File location: `third_party/torch-xpu-ops/test/xpu/skip_list_common.py`

Read the current file to check:
1. If the test file already exists in skip_dict
2. If the specific test cases already exist

**If file is set to `None` (entire file skipped):**
Replace `None` with individual test entries (see issue #3127 example):
```python
"test_transformers_xpu.py": (
    # https://github.com/intel/torch-xpu-ops/issues/3127
    "test_math_backend_high_precision_xpu",
    # More tests...
),
```

**If file NOT in skip_dict:**
Add new entry:
```python
"test_transformers_xpu.py": (
    # https://github.com/intel/torch-xpu-ops/issues/<issue-number>
    "test_case_name_xpu",
),
```

**If file EXISTS with tuple:**
Append new test cases (check for duplicates first):
```python
"test_transformers_xpu.py": (
    # existing tests...
    # https://github.com/intel/torch-xpu-ops/issues/<issue-number>
    "test_case_name_xpu",
),
```

**Deduplication:**
- Before adding, check if test name already in tuple
- Skip if already exists, notify user

**Issue link comment:**
Add comment before each test or group of tests with issue URL:
```python
# https://github.com/intel/torch-xpu-ops/issues/<issue-number>
"test_name_xpu",
```

### Step 6: Create PR

1. **Check gh authentication:**
```bash
gh auth status
```

2. **If authenticated:**

   a. Check if user has torch-xpu-ops fork:
   ```bash
   gh repo list <username> --source=false | grep torch-xpu-ops
   ```

   b. If no fork, create one:
   ```bash
   gh repo fork intel/torch-xpu-ops
   ```

   c. Add remote (if not already configured):
   ```bash
   git remote add fork git@github.com:<username>/torch-xpu-ops.git
   ```

   d. Create and switch to new branch:
   ```bash
   git checkout -b skiplist/issues-<comma-separated-numbers>
   ```

   e. Commit changes:
   ```bash
   git add third_party/torch-xpu-ops/test/xpu/skip_list_common.py
   git commit -m "Add tests to skiplist: issues #<numbers>

   - Skip <test_name> in <test_file>
   - Refer to https://github.com/intel/torch-xpu-ops/issues/<number>"
   ```

   f. Push and create PR:
   ```bash
   git push -u fork skiplist/issues-<numbers>
   gh pr create --repo intel/torch-xpu-ops --title "Add tests to skiplist: issues #<numbers>" --body "Refer to https://github.com/intel/torch-xpu-ops/issues/<number>"
   ```

3. **If NOT authenticated:**
   - Inform user the file is ready
   - Provide manual PR creation instructions:
   ```bash
   cd third_party/torch-xpu-ops
   git add test/xpu/skip_list_common.py
   git commit -m "Add tests to skiplist: issues #3127, #3128, #3129, #3133, #3126"
   git push origin main
   ```
   - PR URL: https://github.com/intel/torch-xpu-ops/compare

### Step 7: Report Result

- If PR created: Provide PR URL
- If manual: Inform user changes are ready for commit

## Batch Processing (Multiple Issues)

When user provides multiple issues (e.g., "3127, 3133, 3129, 3128, 3126"):

1. **Fetch all issues** (can do in parallel with webfetch)
2. **Validate labels** for each issue
3. **Collect all test cases** into a map by test file
4. **Update skip_list_common.py** once with all test cases
5. **Add issue link comments** before each group
6. **Single PR** for all issues

## Error Handling

| Error | Handling |
|-------|----------|
| gh CLI not available | Use webfetch instead |
| gh not authenticated | Prompt user to run `gh auth login --with-token`, or proceed for manual PR |
| Token missing scopes | Inform user to create new token with `repo` and `read:org` scopes |
| Parse fails | Ask user to provide test info directly in conversation |
| File not found | Verify path: `third_party/torch-xpu-ops/test/xpu/skip_list_common.py` |
| Test already exists | Skip and notify user |
| File already None | Replace with specific test list |

## File Format Reference

```python
skip_dict = {
    "test_transformers_xpu.py": (
        # https://github.com/intel/torch-xpu-ops/issues/3127
        "test_math_backend_high_precision_xpu",
        # https://github.com/intel/torch-xpu-ops/issues/3133
        "test_fused_kernels_nested_broadcasting_kernel1_..._xpu",
        "test_scaled_dot_product_attention_..._xpu",
        # https://github.com/intel/torch-xpu-ops/issues/3129
        "test_fused_backwards_throws_determinism_warning_..._xpu",
        # https://github.com/intel/torch-xpu-ops/issues/3128
        "test_invalid_fused_inputs_invalid_dtype_kernel1_xpu",
        # https://github.com/intel/torch-xpu-ops/issues/3126
        "test_nested_fails_on_padding_head_dim_xpu",
    ),
    # ... other entries
}
```

## Requirements

- GitHub CLI (`gh`) installed and authenticated with `repo` and `read:org` scopes (for PR creation)
- If gh not authenticated: user handles PR manually
- Write access to push to torch-xpu-ops fork (for PR creation)

## Tools Used

| Tool | Purpose |
|------|---------|
| websearch | Search for issue details on GitHub |
| webfetch | Get issue content from GitHub URL |
| grep | Search for existing test entries in skip_list |
| read | Read skip_list_common.py to find test entries |
| edit | Add skip entries to skip_list_common.py |
| question | Ask user for clarification if needed |
| bash | Run git commands for PR creation |
| gh | Create PR if authenticated |

## Constraints

- Working directory: `.` (torch-xpu-ops repo root)
- Skip list file: `test/xpu/skip_list_common.py` (relative to torch-xpu-ops)
- Tests in `test/xpu/` use skip_list_common.py
- Tests in `pytorch/test/` use @skipIfXpu decorator (use check_pytorch_skip skill)
- gh CLI may not be authenticated - use webfetch/websearch as fallback
- This skill is for tests in torch-xpu-ops repo, NOT pytorch/test/

## Examples

**Single issue:**
```
Add to skiplist: https://github.com/intel/torch-xpu-ops/issues/3127
```
→ Adds test_math_backend_high_precision_xpu

**Multiple issues:**
```
Add to skiplist: 3127, 3133, 3129, 3128, 3126
```
→ Creates single PR with all test cases

**Manual PR message:**
```
Add tests to skiplist: issues #3127, #3128, #3129, #3133, #3126

- Skip test_math_backend_high_precision_xpu in test_transformers_xpu.py
- Skip 68 SDPA nested broadcasting tests
- Skip 4 SDPA failure mode tests
- Refer to respective issues
```

## Integration with create_skill

This skill was created using the **create_skill** workflow. When modifying this skill:
1. Track all new tools used
2. Track all new constraints encountered
3. Update this SKILL.md file with the create_skill pattern

To create a similar skill, load the create_skill skill first:
```bash
skill load create_skill
```