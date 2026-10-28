#!/usr/bin/env python3
# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Files the microbench regression issues the microbench-issue-authoring skill
drafted. The only step of the pipeline that writes.

Everything here is mechanical:

* a draft whose case rows are not verbatim rows of evidence.json is rejected
  whole, and so is a draft for an unknown group;
* the LLM's suggested commit is kept only if it is one of the candidates;
* an open issue with the same marker gets a comment with the new cases
  instead of a new issue; cases already listed in any open microbench issue
  are not reported again;
* labels follow the attribution; a body too long for one issue is split into
  follow-up comments rather than shortened.

    python3 .github/scripts/microbench_create_issues.py --evidence-dir ev \\
        --drafts drafts.json --report-dir report [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = os.environ.get("GITHUB_REPOSITORY") or "intel/torch-xpu-ops"
TITLE_PREFIX = "[microbench] "
MAX_TITLE = 160
SAFE_BODY_LIMIT = 60000
MAX_ISSUES_PER_RUN = 10
ROW_RE = re.compile(r"^\|.*`logs/[^`/]+/[a-z]+_([0-9a-f]{12})/`.*\|$")
SUGGESTION_HEAD = "**Most likely commit (LLM suggestion, not verified by A/B):**"


def run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{proc.stderr.strip()}")
    return proc.stdout


def check_draft(draft: dict, group: dict) -> list[str]:
    """Problems that void the draft (empty: usable)."""
    rows = {c["row"]: c["case_id"] for c in group["cases"]}
    problems = []
    body = draft.get("body") or ""
    seen = set()
    for line in body.splitlines():
        line = line.rstrip()
        m = ROW_RE.match(line)
        if not m:
            continue
        if line not in rows:
            problems.append(f"case row not copied verbatim from evidence.json: {line[:160]}")
        else:
            seen.add(rows[line])
    missing = [c for c in group["new_cases"] if c not in seen]
    if missing:
        problems.append(f"draft omits {len(missing)} new case(s): {', '.join(missing[:10])}")
    if group["marker"] in body:
        problems.append("draft must not contain the marker (the script adds it)")
    return problems


def apply_suggestion(draft: dict, group: dict) -> str:
    """Body with the suggestion paragraph removed unless it names a candidate."""
    body = draft.get("body") or ""
    sug = (draft.get("suggested_commit") or "").strip()
    shas = [c["sha"] for c in group.get("candidates") or []]
    valid = bool(sug) and any(s.startswith(sug) for s in shas) and len(sug) >= 7
    if valid or SUGGESTION_HEAD not in body:
        return body
    out, skip = [], False
    for line in body.splitlines():
        if line.startswith(SUGGESTION_HEAD):
            skip = True
            continue
        if skip and not line.strip():
            skip = False
            continue
        if not skip:
            out.append(line)
    return "\n".join(out)


def keep_rows(body: str, keep: set[str]) -> str:
    """Drop case rows of cases that an open issue already lists."""
    out = []
    for line in body.splitlines():
        m = ROW_RE.match(line.rstrip())
        if m and m.group(1) not in keep:
            continue
        out.append(line)
    return "\n".join(out)


def split_body(body: str) -> list[str]:
    """[issue body, follow-up comments...], splitting only between case rows."""
    if len(body) <= SAFE_BODY_LIMIT:
        return [body]
    lines = body.splitlines()
    rows = [i for i, line in enumerate(lines) if ROW_RE.match(line.rstrip())]
    head_end = rows[0] if rows else len(lines)
    header = [line for line in lines[max(0, head_end - 2) : head_end] if line.startswith("|")]
    parts, cur = [], lines[:head_end]
    for i in range(head_end, len(lines)):
        if len("\n".join(cur + [lines[i]])) > SAFE_BODY_LIMIT and ROW_RE.match(lines[i].rstrip()):
            parts.append("\n".join(cur))
            cur = ["### Regressed cases (continued)", "", *header]
        cur.append(lines[i])
    parts.append("\n".join(cur))
    return parts


def default_title(group: dict, evidence: dict) -> str:
    n = len(group["new_cases"])
    cands = group.get("candidates") or []
    when = f"between nightly {evidence.get('prev', {}).get('run_id')} and {evidence.get('run_id')}"
    if group["attribution"] == "pytorch":
        return f"{n} case(s) regressed >= 10% with PyTorch {group['pytorch_range'][0][:10]}..{group['pytorch_range'][1][:10]}"
    if len(cands) == 1 and cands[0].get("pr"):
        pr = cands[0]["pr"]
        return f"{n} case(s) regressed >= 10% after {cands[0]['sha'][:10]} ({pr.get('title')} #{pr['number']})"
    return f"{n} case(s) regressed >= 10% in torch-xpu-ops {when}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evidence-dir", required=True)
    ap.add_argument("--drafts", required=True)
    ap.add_argument("--report-dir", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    evidence = json.loads((Path(args.evidence_dir) / "evidence.json").read_text())
    drafts_path = Path(args.drafts)
    drafts = json.loads(drafts_path.read_text()).get("drafts", []) if drafts_path.exists() else []
    by_key = {d.get("key"): d for d in drafts}
    report = []
    filed = 0
    for group in evidence.get("groups") or []:
        entry = {"key": group["key"], "attribution": group["attribution"]}
        report.append(entry)
        if not group["new_cases"]:
            entry["result"] = "every case is already in an open microbench issue"
            continue
        draft = by_key.get(group["key"])
        if not draft:
            entry["result"] = "no draft; not filed"
            continue
        if draft.get("file") is False:
            entry["result"] = f"draft says do not file: {draft.get('reason', '')}"
            continue
        problems = check_draft(draft, group)
        if problems:
            entry.update(result="draft rejected", problems=problems)
            continue
        body = keep_rows(apply_suggestion(draft, group), set(group["new_cases"]))
        parts = split_body(body)
        parts[0] = group["marker"] + "\n" + parts[0]
        title = (draft.get("title") or default_title(group, evidence)).strip().replace("\n", " ")
        title = TITLE_PREFIX + title.removeprefix(TITLE_PREFIX)
        title = title[:MAX_TITLE]
        entry.update(title=title, labels=group["labels"], cases=group["new_cases"], parts=len(parts))
        if group.get("existing_issue"):
            entry["result"] = f"comment on #{group['existing_issue']}"
            if not args.dry_run:
                for part in parts:
                    run(["gh", "issue", "comment", str(group["existing_issue"]), "--repo", REPO, "--body",
                         "New cases from nightly " + str(evidence.get("run_id")) + ":\n\n" + part])
            continue
        if filed >= MAX_ISSUES_PER_RUN:
            entry["result"] = f"not filed: more than {MAX_ISSUES_PER_RUN} issues in one run"
            continue
        filed += 1
        if args.dry_run:
            entry["result"] = "dry run"
            continue
        cmd = ["gh", "issue", "create", "--repo", REPO, "--title", title, "--body", parts[0]]
        for label in group["labels"]:
            cmd += ["--label", label]
        url = run(cmd).strip()
        entry["result"] = f"created {url}"
        for part in parts[1:]:
            run(["gh", "issue", "comment", url, "--repo", REPO, "--body", part])
    out = Path(args.report_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "filed.json").write_text(json.dumps(report, indent=1))
    for e in report:
        print(f"{e['key']}: {e.get('result')}")
        for p in e.get("problems") or []:
            print(f"  - {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
