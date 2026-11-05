---
name: xpu-release-notes
description: Generate PyTorch release notes worksheet for XPU by searching git log between release branches. Trigger on "xpu-release-notes", "generate XPU release notes".
---

# Generate Release Notes from Git History

Generate and complete the release notes worksheet for xpu.

## Usage

```
/xpu-release-notes <version> <torch-release-notes-clone-path>
```
Where `<version>` is the PyTorch release version (e.g., `2.11.0`) and `<torch-release-notes-clone-path>` is the path to the cloned https://github.com/meta-pytorch/torch-release-notes repository.


## Instructions

### Step 0: Validate inputs

1. Check that the version and torch-release-notes-clone-path arguments were provided. If missing, ask the user for the missing arguments.
2. Check that the torch-release-notes-clone-path exists. If it doesn't exist, tell the user to clone it.
3. Confirm the version directory exists (e.g., `torch-release-notes-clone-path/2.12.0/`). If not, tell the user to update to latest main branch of torch-release-notes.
4. Confirm the worksheet file exists at `<torch-release-notes-clone-path>/<version>/todo/result_xpu.md`. If it's already in `done/`, tell the user it's already completed and ask if they want to re-process it.


### Step 1: Collect all the commits between the previous pyTorch release and the current release

Fetch the previous and current release branches from the PyTorch repo:
```bash
git fetch origin release/<prev_version_minor>:release/<prev_version_minor> release/<current_version_minor>:release/<current_version_minor>
```

For example, for version 2.12.0:
- Previous branch: `release/2.11`
- Current branch: `release/2.12`

### Step 2: Find relevant commits

Search git log between the two release branches for commits related to xpu, and save the results to a <version>.txt for review. Use the following command:

```bash
git log --oneline origin/release/<prev>..origin/release/<current> | grep -Ei "xpu|sycl|oneapi|intel"
```


### Step 3: Read <todo>/result_xpu.md to understand task

- Read `<torch-release-notes-clone-path>/<version>/todo/result_xpu.md` worksheet file with the categorized commits. Follow the instructions in the worksheet to ensure proper categorization, clear descriptions, and formatting.

- Reference previous version's completed notes to understand more about the categories and style.

### Step 4: Finish the <todo>/result_xpu.md worksheet

- Remove all the items categorized.
- for each commit, not only read the title, but also the PR descriptions to understand the context and impact of
- put the commit in the existed categories.

Format each entry as:
```markdown
- Clear description of the change ([#NNNNN](https://github.com/pytorch/pytorch/pull/NNNNN))
```

**Grouping:** Merge related PRs into a single entry when they implement the same logical change:
```markdown
- Improve Inductor UT coverage for XPU ([#174053](https://github.com/pytorch/pytorch/pull/174053), [#174054](https://github.com/pytorch/pytorch/pull/174054), ...)
```

**Style rules:**
- Start with a verb or "Support/Enable/Add/Fix"
- Use backticks for code identifiers: `torch.compile`, `conv2d`, `addmm`
- Be specific about what changed and on which device/backend
- Don't start entries with `[tag]` prefixes — integrate the context into the description


### Step 5: Write the worksheet file

Write the complete worksheet to `<version>/done/result_xpu.md` (or `<version>/todo/result_xpu.md` if not yet finalized).


### Step 6: Review with user

Present the categorized notes for user review. Common feedback patterns:
- Items in wrong categories
- Items that should be merged into one entry
- Items that should be dropped
- Items missing that the user knows about

Iterate until the user is satisfied, then ensure the file is in `done/`.

## Example: XPU for 2.12.0

For XPU, the search terms were `xpu`, `sycl`, `oneapi`, `Intel GPU`. The process found ~80 commits, which were filtered down to ~25 entries across categories. Key decisions:
- Pin updates and CI infra → dropped entirely
- C++20 enforcement → devs (affects source builders)
- Header includes, warning fixes → not user facing
- Test coverage expansion → one grouped entry in improvements
- `torch.accelerator.Graph` support → new feature (new API surface)
