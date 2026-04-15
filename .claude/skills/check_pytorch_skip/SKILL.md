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

Parse from issue:
- Test cases in "Cases:" section (op_ut format)
- Labels (check for 'wontfix' or 'not_target')
- Issue title

### Step 3: Search PyTorch Test Files

Search for the issue in pytorch repository test files:

**Method 1: Search by issue number in pytorch repo:**
```bash
grep -r "torch-xpu-ops.*<issue-number>" /home/daisydeng/case_porting/pytorch/test/
```

**Method 2: Search by test name (if available from issue):**
```bash
grep -r "test_<test_name>" /home/daisydeng/case_porting/pytorch/test/
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

### Step 5: Add Skip Decorator (If Needed)

If the test is in pytorch/test/ and needs skip:

1. Find the test method in the file
2. Add `@skipIfXpu(msg="Description - torch-xpu-ops: <issue-number>")` decorator

Example:
```python
# Before:
def test_memory_history_inductor(self):
    ...

# After:
@skipIfXpu(msg="TypeError, torch-xpu-ops: 3004")
def test_memory_history_inductor(self):
    ...
```

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

## Common Test File Locations

In pytorch/test/:
- `test/inductor/` - Inductor tests
- `test/dynamo/` - Dynamo tests
- `test/common.py` - Common test utilities

## Skip Decorator Patterns

Common skip patterns used in PyTorch:
```python
@skipIfXpu(msg="Description - torch-xpu-ops: #<issue-number>")
@skipIfXpu(msg="TypeError, torch-xpu-ops: <issue-number>")
@skipIfXpu(msg="AttributeError, torch-xpu-ops: #<issue-number>")
```

## Error Handling

| Error | Handling |
|-------|----------|
| gh CLI not available | Use webfetch instead |
| Test not in pytorch/test/ | Use skiplist_pr skill for torch-xpu-ops tests |
| Duplicate skip exists | Notify user, no action needed |
| Test already has skip | Check if it references correct issue |

## Related Skills

- **skiplist_pr**: For adding tests to torch-xpu-ops skip list when test is in third_party/torch-xpu-ops/test/xpu/
- **check_pytorch_skip**: For checking if tests in pytorch/test/ already have skip decorators

## Requirements

- Access to pytorch repository at /home/daisydeng/case_porting/pytorch
- GitHub CLI for fetching issue details (optional)
