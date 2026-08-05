# Guilty Commit Rules

The guilty commit is the commit that introduced the failure. Search the past
week of local history with `git log`, then confirm the candidate with
`git show`. This is read-only history inspection: never check out, revert,
cherry-pick, or bisect during triage.

## Which history to search

| Traced failure location | Repository whose history is authoritative |
|---|---|
| `pytorch/aten/`, `pytorch/torch/`, or `pytorch/c10/` | `pytorch` |
| `third_party/torch-xpu-ops/`, or `src/` in a standalone checkout | `torch-xpu-ops` |
| The test file itself | The repository that owns that test file |

Run `git log` in the checkout that owns the traced path, using
`pytorch_folder`. In a PyTorch checkout, `third_party/xpu.txt` records the
pinned torch-xpu-ops commit; read it to identify the pin, but never modify it.

## Search rules

- Scope the search to the past week: `--since="7 days ago"`.
- Scope it to the traced files, not the whole tree. Without a file path the
  result is unusable noise.
- Search only after the root cause is traced. A guilty-commit search with no
  traced file path or symbol is not a valid search.

```bash
git log --oneline --since="7 days ago" -- <traced_file> [<traced_file>...]
git show <candidate_commit> -- <traced_file>
```

## Confirming a candidate

A commit is the guilty commit only when all three hold:

1. `git show` proves it changed the traced symbol or call path, not merely a
   neighbouring line in the same file.
2. The change plausibly produces the extracted failure signature.
3. The commit is an ancestor of the tested revision, verified with
   `git merge-base --is-ancestor <commit> HEAD`.

Record the short hash and subject. If several commits touch the traced path and
none is clearly responsible, report `null` rather than guessing. A commit that
merely renames, reformats, or moves the traced code is not guilty.

## Fixing commits are not guilty commits

The same search may surface a commit that already fixes the root cause. Record
that separately as `evidence.upstream_fix`, per
[target_component.md](target_component.md), and never report it as the guilty
commit. A skip or xfail decorator is neither a fix nor a guilty commit; its
presence confirms the failure is still relevant.

An empty result means no guilty commit was found in the past week. It does not
mean the failure is new, is fixed, or has no cause. Long-standing failures
predate the window.

## Minimum decision checks

1. Confirm the search ran in the repository that owns the traced path.
2. Confirm `git log` was scoped by both `--since="7 days ago"` and a traced
   file path.
3. Confirm any reported commit was inspected with `git show` and is an ancestor
   of the tested revision.
4. Confirm a fixing commit was recorded as `upstream_fix`, not as the guilty
   commit.
5. Confirm an inconclusive or empty search reports `null`.
