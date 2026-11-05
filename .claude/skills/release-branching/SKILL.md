---
name: release-branching
description: >
  Use when setting up a new torch-xpu-ops release branch corresponding to a
  PyTorch release. Covers branch creation, release tracker issue, workflow
  default changes PR, and tracker comment.
---

# XPU Release Branching

Set up torch-xpu-ops for a new PyTorch release cycle: create the release
branch, open the tracker issue, submit the workflow-defaults PR, and post
the initial tracker comment.

**Invocation:** `/xpu-release-branching release/X.Y`

## Terminology

Given `release/2.12`:
- `BRANCH` = `release/2.12`
- `MAJOR_MINOR` = `2.12`
- `VERSION` = `v.2.12.0`

## Step 1 — Verify PyTorch Release Branch

Confirm the pytorch release branch exists and locate the release tracker
issue:

```bash
# Branch must exist
gh api repos/pytorch/pytorch/branches/release/MAJOR_MINOR --jq '.name'

# Find the pytorch release tracker issue
gh issue list -R pytorch/pytorch -s all \
  --search "[v.MAJOR_MINOR.0] Release Tracker in:title" \
  --json number,title,url -L 1
```

Read the pytorch release tracker issue body to extract:
- **Phase 1 deadline** (e.g. "until 27/4/26")
- **Phase 2 start** (e.g. "after 27/4/26")

These dates are needed for the torch-xpu-ops tracker issue.

## Step 2 — Create torch-xpu-ops Release Branch

Get the pinned commit from pytorch's release branch and create the
torch-xpu-ops branch from it. If the branch already exists (e.g. retrying
after a partial failure), skip this step and verify the branch directly.

```bash
# Get the pinned torch-xpu-ops commit on the pytorch release branch
PINNED=$(gh api repos/pytorch/pytorch/contents/third_party/xpu.txt \
  --jq '.content' -H 'Accept: application/vnd.github.v3+json' \
  -f ref=release/MAJOR_MINOR | base64 -d | tr -d '[:space:]')

# Create the release branch at that commit
gh api repos/intel/torch-xpu-ops/git/refs -f ref=refs/heads/release/MAJOR_MINOR -f sha=$PINNED
```

Verify the branch was created:
```bash
gh api repos/intel/torch-xpu-ops/branches/release/MAJOR_MINOR --jq '.name'
```

## Step 3 — Create Release Tracker Issue

Create a milestone `PTMAJOR_MINOR` in torch-xpu-ops if it does not exist yet:

```bash
# Check if milestone exists; create if not
gh api repos/intel/torch-xpu-ops/milestones --jq '.[] | select(.title == "PTMAJOR_MINOR") | .title' \
  || gh api repos/intel/torch-xpu-ops/milestones -f title="PTMAJOR_MINOR" -f state=open
```

Open the tracker issue with this exact body template:

```
Title: [VERSION] Release Tracker
Milestone: PTMAJOR_MINOR
```

**Issue body** (replace placeholders):

```markdown
We cut a [release branch](https://github.com/intel/torch-xpu-ops/tree/BRANCH) for the MAJOR_MINOR.0 release.

Our plan from this point is roughly:

- Phase 1 (until <PHASE1_DEADLINE>): work on finalizing the release branch

- Phase 2 (after <PHASE2_START>): perform extended integration/stability/performance testing based on Release Candidate builds.

This issue is for tracking cherry-picks to the release branch.

Refer <PYTORCH_TRACKER_URL>
```

Where:
- `<PHASE1_DEADLINE>` / `<PHASE2_START>` come from the pytorch tracker (Step 1)
- `<PYTORCH_TRACKER_URL>` is `https://github.com/pytorch/pytorch/issues/<NUMBER>`

```bash
gh issue create -R intel/torch-xpu-ops \
  --title "[VERSION] Release Tracker" \
  --milestone "PTMAJOR_MINOR" \
  --body "$(cat <<'EOF'
<body from template above>
EOF
)"
```

Record the created issue number for Step 5.

## Step 4 — Create Release-Only Changes PR

This PR changes the default branch parameters in workflow files from `main`
to `BRANCH`.

### 4a. Create a working branch

```bash
git fetch origin BRANCH
git checkout -b release_MAJOR_MINOR_change origin/BRANCH
```

(Branch naming convention: `release_X.Y_change`, e.g. `release_2.12_change`)

### 4b. Edit workflow files

Two files need updating — `.github/workflows/_linux_build.yml` and
`.github/workflows/_windows_ut.yml`:

For each file, change the `pytorch` and `torch_xpu_ops` input defaults:

```yaml
# Before
      pytorch:
        ...
        default: 'main'
        description: Pytorch main by default, ...
      torch_xpu_ops:
        ...
        default: 'main'
        description: Torch-xpu-ops main by default, ...

# After
      pytorch:
        ...
        default: 'BRANCH'
        description: Pytorch BRANCH by default, ...
      torch_xpu_ops:
        ...
        default: 'BRANCH'
        description: Torch-xpu-ops BRANCH by default, ...
```

Only change the `default` value and update `main` to `BRANCH` in the
`description` string. Do not touch any other fields.

### 4c. Commit and push

```bash
git add .github/workflows/_linux_build.yml .github/workflows/_windows_ut.yml
git commit -m "[RELEASE MAJOR_MINOR] Release only changes"
git push origin release_MAJOR_MINOR_change
```

### 4d. Open the PR

```bash
gh pr create -R intel/torch-xpu-ops \
  --base BRANCH \
  --head release_MAJOR_MINOR_change \
  --title "[RELEASE MAJOR_MINOR] Release only changes"
```

No description body is needed.

## Step 5 — Comment on Release Tracker Issue

Post the cherry-pick tracking comment on the torch-xpu-ops tracker issue
created in Step 3:

```bash
gh issue comment <ISSUE_NUMBER> -R intel/torch-xpu-ops --body "$(cat <<'EOF'
Link to landed trunk PR (if applicable):
* NA

Link to release branch PR:
* <PR_URL>

Criteria Category:
* Release only changes
EOF
)"
```

Where `<PR_URL>` is the URL from Step 4d.

## Checklist

- [ ] PyTorch release branch exists and tracker issue located
- [ ] torch-xpu-ops release branch created from pinned commit
- [ ] Release tracker issue created with correct milestone and dates
- [ ] Workflow defaults PR opened against release branch
- [ ] Tracker comment posted with PR link
