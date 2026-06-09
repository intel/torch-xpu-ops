---
name: xpu-alignment-report-and-issue-format
description: Exact output formats for the scan report and bug reports. Gives the layout of reports/full_scan.md, the issue-draft template, and the steps for filing a draft on GitHub after the user approves it. Read this for Steps 2e and 2f.
---

# Report & Issue Templates

## Full scan report (`reports/full_scan.md`)

Write this file directly (no external renderer). Include every candidate whose
`local_status == done`; exclude rows rejected at deep-filter (`deep_status == reject`).

Each entry must preserve a numbered format and include at least:

- candidate id, title, and kind
- evidence URL
- reproducer script path and output log path
- an exact ``Local XPU result: `<bucket>` `` line
- route suggestion for `confirmed` and `related-failure`

Requirements:

- Keep the report auditable: keep the numbered entry format and include the exact
  ``Local XPU result: `<bucket>` `` line for every tested candidate.
- For confirmed bugs, include enough local evidence and context for issue filing
  without reopening raw logs.
- Use upstream issue/PR content or commit context to describe the scenario; do not
  reduce entries to title-only summaries.
- Blocked and not-reproduced entries may be shorter, but must still include the
  repro path, output log path, and decisive local outcome.

## Local issue drafts (`reports/issue_drafts.md`)

Write this file directly for all `confirmed` and `related-failure` candidates,
using this exact body structure:

````
## Issue 1

**Suggested title:** [xpu_alignment] <original bug title>
**Suggested labels:** xpu-alignment, <upstream-issue|upstream-pr>, <confirmed|related-failure>

**Upstream source:** <upstream URL> (upstream-issue | upstream-pr)
**Scan date:** <YYYY-MM-DD> to <YYYY-MM-DD>
**Local XPU result:** confirmed on torch <version>, <GPU model>

---

### Describe the bug

<clear description of the bug with root-cause analysis where available>

```python
# minimal self-contained XPU-adapted reproducer (copy-pasteable)
```

```
<actual output / error message>
```

---

### Versions

```
<full contents of artifacts/collect_env.txt>
```

---

## Issue 2
...
````

### Label selection rules

- Always include `xpu-alignment`.
- Add `upstream-issue` if sourced from a GitHub issue, `upstream-pr` if sourced
  from a GitHub PR or commit.
- Add `confirmed` if `local_bucket == "confirmed"`, `related-failure` if
  `local_bucket == "related-failure"`.

If no confirmed or related-failure candidates exist, write `reports/issue_drafts.md`
with a single line: `No confirmed or related-failure candidates in this scan.`

## Filing on GitHub (only after user confirmation)

This skill never files issues automatically. After writing the local drafts:

1. Tell the user the drafts are in `reports/issue_drafts.md` and summarize how many
   `confirmed` / `related-failure` candidates were found.
2. Ask whether they want any of them filed on GitHub, and into which repo (the
   routing rules suggest a default).
3. Before filing each approved draft, search the target repo for an existing issue
   covering the same bug (search the upstream URL and the op/error keywords, e.g.
   `gh issue search --repo <repo> "<keywords>"`). If a likely duplicate exists,
   skip filing, link the existing issue in the report, and tell the user.
4. Only on explicit confirmation and once de-duplicated, file the approved drafts
   via the GitHub MCP server (or `gh issue create`) into the routed repository,
   applying the labels above.
5. Report back the URLs of any issues created or matched.
