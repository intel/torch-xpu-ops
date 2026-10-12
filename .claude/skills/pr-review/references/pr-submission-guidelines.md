# PR Submission Guidelines

Guidelines for PR authors submitting to torch-xpu-ops or pytorch/pytorch. These are process and etiquette expectations, not code review items.

## Before Submitting

- Be familiar with the overall submission process: https://github.com/pytorch/pytorch/wiki/The-Ultimate-Guide-to-PyTorch-Contributions
- For features: submit an RFC discussing the feature before sending a PR
- For bug fixes: submit an issue ticket first, then link the PR to the issue

## PR Structure

- Each PR should address a single feature or fix to maintain clarity and ease of review
- Keep LOC as small as possible; split large PRs into multiple PRs (no more than 350 lines each)
- If your PR is part of a series of changes, link related PRs to give reviewers comprehensive context
- If the PR is not ready for review, mark it as draft and add [WIP] to the title

## PR Description

- Craft a clear and descriptive PR title summarizing the main purpose
  - Good: "Fix memory leak in convolutional layers by optimizing tensor allocations"
  - Bad: "Bug fixes and improvements"
- PR description should answer "Why?" and "How?", not "What?"
  - Good example: https://github.com/pytorch/pytorch/pull/126905
- For bug fixes, link to the issue and state how the PR resolves it
  - Good example: https://github.com/pytorch/pytorch/pull/97703
- Each PR should provide test cases. If there are no tests, the reason must be explicitly explained in the description
- Use meaningful commit messages that accurately describe the changes

## Code Quality

- Add informative code comments to help reviewers understand implementation details. Your code is the best code comment.
- If the current code is overly verbose or poorly structured, your options are: match the existing style for consistency, or refactor the whole section to be consistently better (in a separate PR)
- If your PR introduces new features or modifies existing ones, update relevant documentation. Never add functions to an allowlist to skip doc checks.

## Review Process

- Address reviewers' feedback promptly to maintain momentum and reduce review cycle time
- If you think reviewers may be confused by some changes, add self-review comments preemptively
  - Examples: https://github.com/pytorch/pytorch/pull/137794#discussion_r1798681285, https://github.com/pytorch/pytorch/pull/140624/files#r1841269021
- If review comments have been addressed, re-request reviewers to review the PR
- Do NOT mark review comments as resolved yourself — that is the reviewer's responsibility
  - If you have questions about a comment, raise them and discuss with the reviewer
