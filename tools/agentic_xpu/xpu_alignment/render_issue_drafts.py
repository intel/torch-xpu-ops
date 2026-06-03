#!/usr/bin/env python3
# Copyright 2024-2026 Intel Corporation
# Co-authored with GitHub Copilot
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Generate issue_drafts.md from a completed XPU alignment scan run directory.

Reads:
  <run_dir>/artifacts/candidate_ledger.jsonl   — per-candidate state
  <run_dir>/artifacts/details/<id>.json        — upstream issue/PR/commit body
  <run_dir>/artifacts/output_<id>.log          — local repro output
  <run_dir>/artifacts/collect_env.txt          — torch.utils.collect_env snapshot

Writes:
    <run_dir>/reports/issue_drafts.md

Only candidates with local_bucket in {"confirmed", "related-failure"} are included.

Usage:
    python render_issue_drafts.py <run_dir>
    python render_issue_drafts.py <run_dir> --output /path/to/issue_drafts.md
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

# Buckets that warrant filing a tracking issue
ACTIONABLE_BUCKETS = {"confirmed", "related-failure"}

# Label mapping per readme §7
KIND_LABEL = {"issue": "upstream-issue", "pr": "upstream-pr", "commit": "upstream-pr"}
BUCKET_LABEL = {"confirmed": "confirmed", "related-failure": "related-failure"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(errors="replace")


def output_tail(path: Path, max_lines: int = 30) -> str:
    text = read_text(path)
    lines = [l for l in text.splitlines() if l.strip()]
    return "\n".join(lines[-max_lines:]) if lines else "(no output captured)"


def upstream_description(detail: dict[str, Any] | None, row: dict[str, Any]) -> str:
    """Build a human-readable description from the upstream detail JSON."""
    if not detail:
        return f"No upstream detail captured for {row['id']}."

    # Issue or PR
    if "title" in detail:
        body = (detail.get("body") or "").strip()
        if body:
            # Trim to first ~800 chars to keep the draft readable
            if len(body) > 800:
                body = body[:797].rstrip() + "..."
            return body
        return f"Upstream {row.get('kind', 'item')} has no body text."

    # Commit
    commit = detail.get("commit", {})
    message = (commit.get("message") or "").strip()
    files = detail.get("files", [])
    parts: list[str] = []
    if message:
        parts.append(message[:600].rstrip() + ("..." if len(message) > 600 else ""))
    if files:
        names = [f.get("filename", "") for f in files[:8]]
        if len(files) > 8:
            names.append(f"… and {len(files) - 8} more")
        parts.append("Changed files:\n" + "\n".join(f"  - {n}" for n in names))
    return "\n\n".join(parts) if parts else "No commit detail captured."


def build_reproducer(run_dir: Path, candidate_id: str) -> str:
    script = run_dir / "scripts" / f"repro_{candidate_id}.py"
    if script.exists():
        return script.read_text(errors="replace").strip()
    return "# No reproducer script was generated for this candidate."


def suggested_title(row: dict[str, Any]) -> str:
    return f"[cuda_xpu_alignment] {row['title']}"


def labels_for(row: dict[str, Any]) -> str:
    labels = ["xpu-alignment"]
    kind_label = KIND_LABEL.get(row.get("kind", ""), "upstream-issue")
    labels.append(kind_label)
    bucket_label = BUCKET_LABEL.get(row.get("local_bucket", ""), "")
    if bucket_label:
        labels.append(bucket_label)
    return ",".join(labels)


# ---------------------------------------------------------------------------
# Per-issue draft renderer
# ---------------------------------------------------------------------------

def render_draft(
    index: int,
    row: dict[str, Any],
    detail: dict[str, Any] | None,
    run_dir: Path,
    collect_env: str,
    scan_date: str,
) -> list[str]:
    candidate_id = str(row["id"])
    upstream_url = row.get("url", "")
    bucket = row.get("local_bucket", "unknown")
    kind = row.get("kind", "issue")
    output_log = run_dir / "artifacts" / f"output_{candidate_id}.log"

    description = upstream_description(detail, row)
    reproducer = build_reproducer(run_dir, candidate_id)
    actual_output = output_tail(output_log)

    # Detect torch version from collect_env
    torch_version = "unknown"
    gpu_model = "unknown"
    for line in collect_env.splitlines():
        if line.startswith("PyTorch version:"):
            torch_version = line.split(":", 1)[1].strip()
        if line.startswith("* [0] _XpuDeviceProperties(") and gpu_model == "unknown":
            m = re.search(r"name='([^']+)'", line)
            if m:
                gpu_model = m.group(1)

    lines: list[str] = []
    lines.append(f"## Issue {index}")
    lines.append("")
    lines.append(f"**Suggested title:** {suggested_title(row)}")
    lines.append(f"**Labels:** `{labels_for(row)}`")
    lines.append(f"**Upstream:** {upstream_url} ({kind})")
    lines.append(f"**Local XPU result:** `{bucket}`")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"**Upstream source:** {upstream_url} (upstream-{KIND_LABEL.get(kind, 'issue').replace('upstream-', '')})")
    lines.append(f"**Scan date:** {scan_date}")
    lines.append(f"**Local XPU result:** {bucket} on torch {torch_version}, {gpu_model}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("### 🐛 Describe the bug")
    lines.append("")
    lines.append(description)
    lines.append("")
    lines.append("```python")
    lines.append(reproducer)
    lines.append("```")
    lines.append("")
    lines.append("```")
    lines.append(actual_output)
    lines.append("```")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("### Versions")
    lines.append("")
    lines.append("```")
    lines.append(collect_env.strip() if collect_env.strip() else "(collect_env.txt not found)")
    lines.append("```")
    lines.append("")

    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def render_issue_drafts(run_dir: Path, output_path: Path) -> int:
    artifacts_dir = run_dir / "artifacts"
    ledger_path = artifacts_dir / "candidate_ledger.jsonl"
    details_dir = artifacts_dir / "details"
    collect_env = read_text(artifacts_dir / "collect_env.txt")
    scan_date = run_dir.name  # e.g. "2026-05-26"

    if not ledger_path.exists():
        print(f"ERROR: ledger not found: {ledger_path}")
        return 1

    rows = read_jsonl(ledger_path)
    actionable = [r for r in rows if r.get("local_bucket") in ACTIONABLE_BUCKETS]

    if not actionable:
        print(f"No confirmed/related-failure candidates in {run_dir.name} — nothing to draft.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            f"# Issue Drafts — {scan_date}\n\nNo actionable candidates (confirmed or related-failure) found in this scan.\n"
        )
        return 0

    # Sort: confirmed first, then related-failure
    actionable.sort(key=lambda r: (0 if r.get("local_bucket") == "confirmed" else 1, str(r.get("id", ""))))

    lines: list[str] = []
    lines.append(f"# Issue Drafts — {scan_date}")
    lines.append("")
    lines.append(
        f"Generated from scan run `{run_dir}`. "
        f"{len(actionable)} actionable candidate(s): "
        f"{sum(r.get('local_bucket') == 'confirmed' for r in actionable)} confirmed, "
        f"{sum(r.get('local_bucket') == 'related-failure' for r in actionable)} related-failure."
    )
    lines.append("")
    lines.append("> Before filing: re-run the reproducer against the latest nightly to confirm the bug persists.")
    lines.append("> Update the Versions block with the version you tested.")
    lines.append("")
    lines.append("---")
    lines.append("")

    for index, row in enumerate(actionable, 1):
        detail = load_json(details_dir / f"{row['id']}.json")
        lines.extend(render_draft(index, row, detail, run_dir, collect_env, scan_date))

    lines.append("---")
    lines.append("")
    lines.append("### Filing commands")
    lines.append("")
    lines.append("```bash")
    repo = os.environ.get("XPU_ALIGNMENT_ISSUE_REPO", "intel/torch-xpu-ops")
    lines.append(f"REPO={repo}")
    for index, row in enumerate(actionable, 1):
        labels = labels_for(row)
        title = suggested_title(row).replace('"', '\\"')
        lines.append(f"# Issue {index}: {row['id']}")
        lines.append(
            f'gh issue create --repo "$REPO" \\\n'
            f'  --title "{title}" \\\n'
            f'  --label "{labels}" \\\n'
            f'  --body-file <(sed -n \'/^## Issue {index}$/,/^## Issue {index + 1}$/{{ /^## Issue [0-9]/!p }}\' \\\n'
            f'      "{output_path}")'
        )
        lines.append("")
    lines.append("```")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(actionable)} issue draft(s) to: {output_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Scan run directory, e.g. runs/2026-05-26")
    parser.add_argument(
        "--output",
        help="Output path for issue_drafts.md (default: <run_dir>/reports/issue_drafts.md)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()

    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = run_dir / "reports" / "issue_drafts.md"

    exit(render_issue_drafts(run_dir, output_path))


if __name__ == "__main__":
    main()
