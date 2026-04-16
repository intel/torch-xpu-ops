---
name: check_pytorch_skip
description: Check if torch-xpu-ops issues (wontfix/not_target) are already skipped in pytorch test files by adding @skipIfXpu decorators
---

# Check Pytorch Skip

Check if intel/torch-xpu-ops issues (labeled 'wontfix' or 'not_target') are already handled in PyTorch test files by searching for skip decorators.

## Quick Start

1. User provides issue number(s) from intel/torch-xpu-ops (can be multiple)
2. Search for the issue in PyTorch test files using grep
3. If found with skipIfXpu decorator → already handled
4. If not found → need to add skip decorator to appropriate test file in pytorch/test/

## Detailed Instructions

### Step 1: Get Issue Information

User provides issue URL or number:
- Issue URL: `https://github.com/intel/torch-xpu-ops/issues/3004`
- Issue number: `3004`

Extract the issue number from the URL.

### Step 2: Fetch Issue Details

**Primary: Try gh CLI first:**
```bash
gh issue view <issue-number> --repo intel/torch-xpu-ops
```

**Fallback: Use webfetch:**
```bash
webfetch https://github.com/intel/torch-xpu-ops/issues/<issue-number> markdown
```

**Fallback: Use websearch:**
```bash
websearch --numResults 5 "intel torch-xpu-ops issue <issue-number>"
```

Parse from issue:
- Test cases in "Cases:" section (op_ut format)
- Labels (check for 'wontfix' or 'not_target')
- Issue title

### Step 3: Search PyTorch Test Files

Search for the issue in pytorch repository test files:

**Method 1: Search by issue number in pytorch repo:**
```bash
grep -r "torch-xpu-ops.*<issue-number>" <path-to-pytorch>/test/
```

**Method 2: Search by test name (if available from issue):**
```bash
grep -r "test_<test_name>" <path-to-pytorch>/test/
```

**Using grep tool:**
```python
grep(path="<path-to-pytorch>", pattern="torch-xpu-ops.*<issue-number>")
```

### Step 4: Analyze Results

If found with `@skipIfXpu` decorator:
```
Found: test/inductor/test_cuda_repro.py:2872
@skipIfXpu(msg="Decimal object comparison failed - torch-xpu-ops: 2810")
```
→ Already handled, no action needed

If found WITHOUT skip decorator:
→ Need to add skip decorator

If NOT found:
→ Test is in torch-xpu-ops repo (not pytorch/test/), use skiplist_pr skill instead

### Step 5: Read Test File Context

When skip is needed, read the test file to find exact location:
```python
read(filePath="<path-to-pytorch>/test/inductor/test_cuda_repro.py", limit=20, offset=2865)
```

### Step 6: Add Skip Decorator (If Needed)

If the test is in pytorch/test/ and needs skip:

1. Find the test method in the file
2. Add `@skipIfXpu(msg="Description - torch-xpu-ops: <issue-number>")` decorator

Using edit tool:
```python
edit(filePath="<path-to-pytorch>/test/inductor/test_cuda_repro.py", 
     oldString="    def test_not_disabling_ftz_yields_zero(self):",
     newString="    @skipIfXpu(msg=\"Decimal object comparison failed - torch-xpu-ops: 2810\")\n    def test_not_disabling_ftz_yields_zero(self):")
```

## Tools Used

| Tool | Purpose |
|------|---------|
| websearch | Search for issue details on GitHub |
| webfetch | Get issue content from GitHub URL |
| grep | Search for issue number in pytorch test files |
| read | Read test file to find test method and context |
| edit | Add skipIfXpu decorator to test method |
| question | Ask user for clarification if needed |
| skill | Load other skills (skiplist_pr) |

## Constraints

- Working directory: `.` (torch-xpu-ops repo root)
- PyTorch repo path: relative from working directory (e.g., `../../pytorch` or as specified by user)
- Tests in `pytorch/test/` use `@skipIfXpu` decorator from `torch.testing._internal.common_utils`
- Tests in `third_party/torch-xpu-ops/test/xpu/` use skip_list_common.py (use skiplist_pr skill)
- gh CLI may not be authenticated - use webfetch/websearch as fallback

## Example Workflows

### Example 1: Issue Already Handled
```
User: Check issue #3004
→ Search: grep "torch-xpu-ops.*3004" pytorch/test/
→ Found: test/inductor/test_cuda_repro.py:907
   @skipIfXpu(msg="TypeError, torch-xpu-ops: 3004")
→ Result: Already handled, no action needed
```

### Example 2: Issue Needs Skip Added
```
User: Check issue #1234
→ Search: grep "torch-xpu-ops.*1234" pytorch/test/
→ Not found
→ Check if test exists in pytorch/test/:
   Search for test method name from issue
→ Found test at test/inductor/test_example.py:123
→ Add @skipIfXpu decorator
```

### Example 3: Test in torch-xpu-ops (not pytorch)
```
User: Check issue #5678
→ Search: grep "torch-xpu-ops.*5678" pytorch/test/
→ Not found
→ Issue is for test in third_party/torch-xpu-ops/
→ Use skiplist_pr skill instead
```

## Real Examples Checked

| Issue | Test | Status | File |
|-------|------|--------|------|
| #3004 | test_memory_history_inductor | Already skipped | test/inductor/test_cuda_repro.py:907 |
| #3004 | test_additive_rnumel | Already skipped | test/inductor/test_mix_order_reduction.py:775 |
| #2810 | test_not_disabling_ftz_yields_zero | Already skipped | test/inductor/test_cuda_repro.py:2872 |
| #2999 | test_bitwise_adam_* (helper) | Already skipped | test/inductor/test_compiled_optimizers.py:1089 |

## Common Test File Locations

In pytorch/test/:
- `test/inductor/` - Inductor tests (test_cuda_repro.py, test_mix_order_reduction.py, test_compiled_optimizers.py, etc.)
- `test/dynamo/` - Dynamo tests
- `test/test_*.py` - General tests

## Skip Decorator Patterns

Common skip patterns used in PyTorch:
```python
from torch.testing._internal.common_utils import skipIfXpu

@skipIfXpu(msg="Description - torch-xpu-ops: #<issue-number>")
@skipIfXpu(msg="TypeError, torch-xpu-ops: <issue-number>")
@skipIfXpu(msg="AttributeError, torch-xpu-ops: #<issue-number>")
```

## Error Handling

| Error | Handling |
|-------|----------|
| gh CLI not available | Use webfetch or websearch instead |
| Test not in pytorch/test/ | Use skiplist_pr skill for torch-xpu-ops tests |
| Duplicate skip exists | Notify user, no action needed |
| Test already has skip | Check if it references correct issue |
| 401 Unauthorized (webfetch) | Use websearch to get issue details |
| File path not found | Verify pytorch repo path is correct |

## Related Skills

- **skiplist_pr**: For adding tests to torch-xpu-ops skip list when test is in `third_party/torch-xpu-ops/test/xpu/`
- **check_pytorch_skip**: For checking if tests in `pytorch/test/` already have skip decorators

## Requirements

- Access to pytorch repository (path may vary, use relative or absolute as provided)
- GitHub CLI for fetching issue details (optional)
- Tools: websearch, webfetch, grep, read, edit, question

## Integration with create_skill

This skill was created using the **create_skill** workflow. When modifying this skill:
1. Track all new tools used
2. Track all new constraints encountered
3. Update this SKILL.md file with the create_skill pattern

To create a similar skill, load the create_skill skill first:
```bash
skill load create_skill
```

## Workflow Summary

```
1. Get issue number from user
2. Fetch issue details (websearch/webfetch)
3. Search pytorch/test/ for issue (grep)
4. If found with skip → report "already handled"
5. If found without skip → add @skipIfXpu decorator (edit)
6. If not found → check if test in torch-xpu-ops repo
   - Yes → use skiplist_pr skill
   - No → report "test not found"
```

## Important Notes

- Not all issues labeled 'wontfix' or 'not_target' need pytorch changes - some are handled in torch-xpu-ops
- Check both pytorch/test/ and third_party/torch-xpu-ops/test/xpu/ to determine which skill to use
- PR #174058 is the main reference for how these skips are added in pytorch
