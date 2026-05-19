#!/usr/bin/env python3
"""E2E report generator — collects per-issue results from Action Items
and publishes a summary tracking issue.

Usage:
  python scripts/e2e_report.py --issues 191 327 258 --batch "Batch run — 35 issues"
  python scripts/e2e_report.py --issues 191 327 --dry-run
  python scripts/e2e_report.py  # reads from config/e2e_issues.txt if present

Called by run_pipeline.py at the end of each cycle.
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from issue_handler.utils import git as gh
from issue_handler.utils.config import ISSUE_REPO

# Title convention for the tracking issue
TRACKING_TITLE = "[E2E] Pipeline Test Dashboard"

# Failure categories
FAILURE_CATEGORIES = {
    "skill_gap": "No skill covers this domain",
    "knowledge_gap": "Agent lacks codebase understanding",
    "upstream_dependency": "Requires changes in upstream repo",
    "timeout": "Agent exceeded time/loop limit",
    "parse_error": "LLM output couldn't be parsed",
    "wrong_triage": "Root cause analysis was incorrect",
    "infra_bug": "Pipeline code bug",
    "agent_error": "Agent runtime error",
}

# Action item patterns — order matters (later = further along)
STAGE_PATTERNS = [
    ("formatted", re.compile(r"- \[x\] 🔍 Issue formatted")),
    ("triaged", re.compile(r"- \[x\] 🧠 Root cause identified")),
    ("fixed", re.compile(r"- \[x\] 🔧 Fix implemented")),
    ("pr_proposed", re.compile(r"- \[x\] 📋 PR proposed")),
    ("reviewed", re.compile(r"- \[x\] 👀 Human review")),
    ("merged", re.compile(r"- \[x\] 🎉 PR merged")),
]

# Token pattern in Action Items logs
TOKEN_PATTERN = re.compile(
    r"\*\*Tokens:\*\*\s*(.+?)(?:\n|$)"
)

# Sentinel that marks each run section — used to count existing runs
_RUN_HEADER_RE = re.compile(r"^## \[Run \d+\]", re.MULTILINE)


def parse_issue_status(body: str) -> dict:
    """Parse an issue's Action Items to determine stage, tokens, and failure info."""
    from issue_handler.utils.body_templates import get_status as _get_status, get_metadata

    result = {
        "stage": "unformatted",
        "stages_done": [],
        "tokens": {},
        "failure_reason": None,
        "failure_category": None,
        "pipeline_status": _get_status(body),
        "tracking_pr": None,
        "pr_url": None,
        "target_repo": None,
    }

    # Extract PR number, PR URL, and target repo from metadata
    tracking_pr = get_metadata(body, "tracking_pr")
    if tracking_pr:
        result["tracking_pr"] = tracking_pr.strip().lstrip("#")
    target_repo = get_metadata(body, "target_repo")
    if target_repo:
        result["target_repo"] = target_repo.strip()
    # Also try to extract PR URL from fix log
    pr_url_match = re.search(r"PR:\s*(https://github\.com/\S+/pull/\d+)", body)
    if pr_url_match:
        result["pr_url"] = pr_url_match.group(1)

    for stage_name, pattern in STAGE_PATTERNS:
        if pattern.search(body):
            result["stages_done"].append(stage_name)
            result["stage"] = stage_name

    status = result["pipeline_status"]
    if status in ("IMPLEMENTING", "IN_REVIEW", "NEEDS_HUMAN") and "triaged" not in result["stages_done"]:
        result["stages_done"].append("triaged")
        if result["stage"] in ("unformatted", "formatted"):
            result["stage"] = "triaged"
    if status == "DONE" and "triaged" not in result["stages_done"]:
        # Auto-closed by triage verification — both format and triage completed
        result["stages_done"].append("triaged")
        if result["stage"] in ("unformatted", "formatted"):
            result["stage"] = "triaged"
    if status == "IN_REVIEW" and "fixed" not in result["stages_done"]:
        result["stages_done"].append("fixed")
        result["stages_done"].append("pr_proposed")
        result["stage"] = "pr_proposed"
    if "formatted" not in result["stages_done"] and status:
        result["stages_done"].insert(0, "formatted")
        if result["stage"] == "unformatted":
            result["stage"] = "formatted"

    for marker, label in [("discovery", "format"), ("triage", "triage"), ("fix", "fix")]:
        marker_tag = f"<!-- agent:{marker}-log -->"
        pattern = re.compile(
            rf"<details><summary>{re.escape(marker)} log</summary>\s*(.*?){re.escape(marker_tag)}",
            re.DOTALL | re.IGNORECASE,
        )
        m = pattern.search(body)
        if m:
            section = m.group(1)
            token_matches = TOKEN_PATTERN.findall(section)
            if token_matches:
                result["tokens"][label] = token_matches[-1].strip()

    needs_human = re.search(r"\*\*Verdict:\*\*\s*NEEDS_HUMAN", body)
    if needs_human:
        reason_match = re.search(r"\*\*Reason:\*\*\s*(.+?)(?:\n|$)", body)
        reason = reason_match.group(1).strip() if reason_match else "Unknown"
        result["failure_reason"] = reason
        result["failure_category"] = _categorize_failure(reason)

    return result


def _categorize_failure(reason: str) -> str:
    """Auto-categorize a failure reason."""
    reason_lower = reason.lower()
    if "upstream" in reason_lower:
        return "upstream_dependency"
    if "timeout" in reason_lower or "exceeded" in reason_lower:
        return "timeout"
    if "parse" in reason_lower or "json" in reason_lower:
        return "parse_error"
    if "skill" in reason_lower:
        return "skill_gap"
    return "agent_error"


def collect_results(repo: str, issue_numbers: list[int]) -> list[dict]:
    """Collect status for each issue."""
    results = []
    for num in issue_numbers:
        try:
            detail = gh.get_issue_detail(repo, num)
            body = detail.get("body", "") or ""
            title = detail.get("title", "")
            status = parse_issue_status(body)
            status["number"] = num
            status["title"] = title[:60]
            status["state"] = detail.get("state", "open")
            results.append(status)
        except Exception as e:
            results.append({
                "number": num,
                "title": "ERROR",
                "stage": "error",
                "stages_done": [],
                "tokens": {},
                "failure_reason": str(e),
                "failure_category": "infra_bug",
                "state": "unknown",
            })
    return results


def _stage_emoji(stages_done: list[str], stage: str) -> str:
    """Return emoji for a stage column."""
    if stage in stages_done:
        return "✅"
    all_stages = ["formatted", "triaged", "fixed", "pr_proposed", "reviewed", "merged"]
    if not stages_done:
        return "⬜" if stage != "formatted" else "🔄"
    last_done_idx = max(all_stages.index(s) for s in stages_done if s in all_stages)
    stage_idx = all_stages.index(stage) if stage in all_stages else -1
    if stage_idx == last_done_idx + 1:
        return "🔄"
    if stage_idx > last_done_idx + 1:
        return "⬜"
    return "⬜"


def build_section(results: list[dict], batch_name: str, run_number: int,
                  issue_repo: str | None = None) -> str:
    """Build one ## [Run N] markdown section for a batch of results.

    Returns the section string (no leading/trailing blank lines beyond what's needed).
    """
    report_repo = issue_repo or ISSUE_REPO
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        f"## [Run {run_number}] {batch_name} — {ts[:10]}",
        f"",
        f"| # | PR | Title | torch-xpu-ops/pytorch | Format | Triage | Fix | Review | Model | Tokens (fmt/tri/fix) | Cost | Result | Failure Reason |",
        f"|---|-----|-------|----------------------|--------|--------|-----|--------|-------|---------------------|------|--------|----------------|",
    ]

    total_triaged = 0
    total_needs_human = 0
    total_in_progress = 0
    total_cost = 0.0

    for r in results:
        num = r["number"]
        title = r["title"]
        stages = r["stages_done"]

        fmt = _stage_emoji(stages, "formatted")
        tri = _stage_emoji(stages, "triaged")
        fix = _stage_emoji(stages, "fixed")
        pr  = _stage_emoji(stages, "pr_proposed")
        rev = _stage_emoji(stages, "reviewed")

        tokens_parts = []
        row_cost = 0.0
        model_name = "—"
        for key in ["format", "triage", "fix"]:
            t = r["tokens"].get(key, "—")
            total_match = re.search(r"tokens:\s*(\S+)", t)
            tokens_parts.append(total_match.group(1) if total_match else "—")
            cost_match = re.search(r"cost:\s*\$(\S+)", t)
            if cost_match:
                try:
                    row_cost += float(cost_match.group(1))
                except ValueError:
                    pass
            if model_name == "—":
                model_match = re.search(r"model:\s*(\S+)", t)
                if model_match:
                    model_name = model_match.group(1)
        total_cost += row_cost
        tokens_str = " / ".join(tokens_parts)
        cost_str = f"${row_cost:.4f}" if row_cost > 0 else "—"

        failure = r.get("failure_reason", "")
        category = r.get("failure_category", "")
        pipeline_status = r.get("pipeline_status", "")

        if "merged" in stages or "reviewed" in stages:
            result = "✅ PASSED"
            total_triaged += 1
        elif pipeline_status == "DONE":
            result = "✅ DONE"
            total_triaged += 1
        elif pipeline_status == "NEEDS_HUMAN":
            result = "❌ NEEDS_HUMAN"
            total_needs_human += 1
            if not failure:
                failure = "Triage timeout or too complex"
                category = "timeout"
        elif pipeline_status == "IN_REVIEW":
            result = "🔄 IN REVIEW"
            total_in_progress += 1
        elif failure:
            result = "❌ FAILED"
            total_needs_human += 1
        elif r["stage"] == "unformatted":
            result = "⬜ PENDING"
            total_in_progress += 1
        else:
            result = "🔄 IN PROGRESS"
            total_in_progress += 1

        failure_display = f"{category}: {failure}" if failure else "—"

        # PR link and target repo
        pr_url = r.get("pr_url")
        tracking_pr = r.get("tracking_pr")
        if pr_url and tracking_pr:
            pr_display = f"[#{tracking_pr}]({pr_url})"
        elif tracking_pr:
            pr_display = f"#{tracking_pr}"
        else:
            pr_display = "—"
        target_repo = r.get("target_repo", "—") or "—"

        lines.append(
            f"| [#{num}](https://github.com/{report_repo}/issues/{num}) "
            f"| {pr_display} | {title} | {target_repo} | {fmt} | {tri} | {fix} | {rev} "
            f"| {model_name} | {tokens_str} | {cost_str} | {result} | {failure_display} |"
        )

    total = len(results)
    cost_summary = f"${total_cost:.4f}" if total_cost > 0 else "—"
    lines.extend([
        f"",
        f"**Summary:** {total} issues · {total_triaged} TRIAGED/PASSED · "
        f"{total_needs_human} NEEDS_HUMAN · {total_in_progress} IN PROGRESS · cost {cost_summary}",
    ])

    return "\n".join(lines)


def build_report(results: list[dict], repo: str | None = None,
                 existing_body: str | None = None,
                 issue_repo: str | None = None,
                 batch_name: str = "Run") -> str:
    """Build a full dashboard body with one new section appended.

    Sections from previous days are folded into <details> blocks.
    Sections from today stay in the main body.
    """
    run_number = (len(_RUN_HEADER_RE.findall(existing_body)) + 1) if existing_body else 1
    section = build_section(results, batch_name, run_number, issue_repo=issue_repo or ISSUE_REPO)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if existing_body and existing_body.strip():
        if run_number == 1:
            # First run on a fresh or legacy dashboard — replace entire body
            header = _dashboard_header()
            return f"{header}\n\n---\n\n{section}"
        else:
            # Fold previous-day sections into <details>, keep today's in main
            body = existing_body.rstrip()
            body = _fold_old_sections(body, today)
            return f"{body}\n\n---\n\n{section}"
    else:
        header = _dashboard_header()
        return f"{header}\n\n---\n\n{section}"


# Pattern to match ## [Run N] ... — YYYY-MM-DD  (captures run header + date)
_RUN_SECTION_RE = re.compile(
    r"(## \[Run \d+\].*?— (\d{4}-\d{2}-\d{2}))",
    re.MULTILINE,
)


def _fold_old_sections(body: str, today: str) -> str:
    """Wrap any ## [Run N] sections from days before *today* in <details>.

    Sections already inside a <details> block are left untouched.
    Same-day sections remain in the main body.
    """
    # Split body into parts by --- separators and process each section
    # Find all Run sections with their dates
    parts = re.split(r"\n---\n", body)
    new_parts = []

    for part in parts:
        header_match = _RUN_SECTION_RE.search(part)
        if header_match:
            section_date = header_match.group(2)
            # Already folded?
            if "<details>" in part:
                new_parts.append(part)
            elif section_date < today:
                # Fold old section
                # Extract the run header for the summary line
                run_header = header_match.group(1).replace("## ", "")
                folded = f"<details>\n<summary>{run_header}</summary>\n\n{part.strip()}\n\n</details>"
                new_parts.append(folded)
            else:
                # Same day — keep in main
                new_parts.append(part)
        else:
            # Header or other content — keep as-is
            new_parts.append(part)

    return "\n---\n".join(new_parts)


def _dashboard_header() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return f"# E2E Pipeline Test Dashboard\n\n**Last updated:** {ts}"


def find_tracking_issue(repo: str) -> int | None:
    """Find the tracking issue by title."""
    try:
        import subprocess, json as _json
        token = gh._token_for_repo(repo)
        env = {**__import__("os").environ, "GH_TOKEN": token} if token else None
        raw = subprocess.check_output(
            ["gh", "issue", "list", "--repo", repo, "--search", TRACKING_TITLE,
             "--state", "open", "--json", "number,title", "--limit", "10"],
            env=env, text=True,
        )
        for issue in _json.loads(raw) if raw.strip() else []:
            if issue.get("title") == TRACKING_TITLE:
                return issue["number"]
    except Exception:
        pass
    return None


def update_tracking_issue(repo: str, results: list[dict],
                          batch_name: str = "Run") -> int:
    """Create or update the tracking issue, appending a new [Run N] section."""
    existing_num = find_tracking_issue(repo)
    if existing_num:
        existing_body = gh.get_issue_detail(repo, existing_num).get("body", "") or ""
        new_body = build_report(results, repo=repo, existing_body=existing_body,
                                issue_repo=ISSUE_REPO, batch_name=batch_name)
        # Also update the Last updated timestamp in the header
        new_body = re.sub(
            r"\*\*Last updated:\*\*.*",
            f"**Last updated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            new_body, count=1,
        )
        gh.update_issue_body(repo, existing_num, new_body)
        return existing_num
    else:
        report = build_report(results, repo=repo, issue_repo=ISSUE_REPO, batch_name=batch_name)
        import subprocess, json as _json
        token = gh._token_for_repo(repo)
        env = {**__import__("os").environ, "GH_TOKEN": token} if token else None
        raw = subprocess.check_output(
            ["gh", "issue", "create", "--repo", repo,
             "--title", TRACKING_TITLE, "--body", report,
             "--label", "agent:tracking"],
            env=env, text=True,
        )
        import re as _re
        m = _re.search(r"/issues/(\d+)", raw)
        return int(m.group(1)) if m else 0


def load_issue_list() -> list[int]:
    """Load issue numbers from config/e2e_issues.txt if it exists."""
    config_file = Path(__file__).resolve().parent.parent / "config" / "e2e_issues.txt"
    if config_file.exists():
        numbers = []
        for line in config_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                try:
                    numbers.append(int(line))
                except ValueError:
                    pass
        return numbers
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate E2E pipeline report")
    parser.add_argument("--issues", type=int, nargs="+",
                        help="Issue numbers to report on")
    parser.add_argument("--repo", type=str, default=ISSUE_REPO,
                        help=f"Repository (default: {ISSUE_REPO})")
    parser.add_argument("--batch", type=str, default="Run",
                        help="Batch label for this run section (e.g. 'Batch run — 35 issues')")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print report without updating tracking issue")
    args = parser.parse_args()

    issue_numbers = args.issues or load_issue_list()
    if not issue_numbers:
        print("No issues specified. Use --issues or create config/e2e_issues.txt",
              file=sys.stderr)
        sys.exit(1)

    results = collect_results(args.repo, issue_numbers)

    if args.dry_run:
        # Simulate what the dashboard would look like
        section = build_section(results, args.batch, run_number=1, issue_repo=args.repo)
        print(section)
    else:
        tracking_num = update_tracking_issue(args.repo, results, batch_name=args.batch)
        print(f"Updated tracking issue #{tracking_num}")


if __name__ == "__main__":
    main()
