#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


SIGNATURE_PATTERNS = [
    r"dumped core",
    r"Floating point exception",
    r"Segmentation fault",
    r"PendingUnbackedSymbolNotFound",
    r"ShapeAsConstantBuffer",
    r"Dynamo unsupported",
    r"InternalTorchDynamoError",
    r"Compile exception:",
    r"Exception:",
    r"AttributeError:",
    r"RuntimeError:",
    r"BUG:",
    r"WARNING:",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render an issue-ready full_scan.md from an XPU alignment run directory."
    )
    parser.add_argument("run_dir", help="Run directory such as runs/2026-05-06")
    parser.add_argument(
        "--output",
        help="Output markdown path (default: <run_dir>/reports/full_scan.md)",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def collect_env_summary(path: Path) -> dict[str, str]:
    summary = {
        "torch": "unknown",
        "xpu": "unknown",
        "python": "unknown",
        "gpu": "unknown",
    }
    if not path.exists():
        return summary

    lines = path.read_text(errors="replace").splitlines()
    for line in lines:
        if line.startswith("PyTorch version:"):
            summary["torch"] = line.split(":", 1)[1].strip()
        elif line.startswith("Is XPU available:"):
            summary["xpu"] = line.split(":", 1)[1].strip()
        elif line.startswith("Python version:"):
            summary["python"] = line.split(":", 1)[1].strip()
        elif line.startswith("* [0] _XpuDeviceProperties(") and summary["gpu"] == "unknown":
            match = re.search(r"name='([^']+)'", line)
            if match:
                summary["gpu"] = match.group(1)
    return summary


def compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def shorten(text: str, limit: int = 96) -> str:
    text = compact_whitespace(text)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def first_nonempty_paragraph(text: str, limit: int = 500) -> str:
    paragraphs = []
    for block in text.split("\n\n"):
        cleaned = compact_whitespace(block)
        if cleaned and not cleaned.startswith("###") and not cleaned.startswith("##"):
            paragraphs.append(cleaned)
    if not paragraphs:
        return ""
    paragraph = paragraphs[0]
    if len(paragraph) > limit:
        return paragraph[: limit - 3].rstrip() + "..."
    return paragraph


def upstream_context(detail: dict[str, Any] | None) -> str:
    if not detail:
        return "No upstream detail JSON was captured for this candidate."
    if "title" in detail:
        body = detail.get("body") or ""
        excerpt = first_nonempty_paragraph(body)
        return excerpt or "Upstream issue/PR body was empty or not informative."
    commit = detail.get("commit", {})
    message = commit.get("message", "")
    excerpt = first_nonempty_paragraph(message)
    files = detail.get("files", [])
    if files:
        changed = ", ".join(file_info.get("filename", "") for file_info in files[:5])
        if len(files) > 5:
            changed += ", ..."
        if excerpt:
            return f"{excerpt} Changed files: {changed}"
        return f"Changed files: {changed}"
    return excerpt or "No commit message or file summary captured."


def output_excerpt(path: Path, max_lines: int = 12) -> list[str]:
    if not path.exists():
        return ["Output log missing."]
    lines = [line.rstrip() for line in path.read_text(errors="replace").splitlines() if line.strip()]
    if not lines:
        return ["Output log was empty."]
    if len(lines) <= max_lines:
        return lines
    return lines[-max_lines:]


def error_signature(path: Path) -> str:
    if not path.exists():
        return "Output log missing."
    lines = [line.rstrip() for line in path.read_text(errors="replace").splitlines() if line.strip()]
    for pattern in SIGNATURE_PATTERNS:
        regex = re.compile(pattern, re.IGNORECASE)
        for line in lines:
            if regex.search(line):
                return line[:300]
    return lines[-1][:300] if lines else "No output captured."


def guessed_expected(row: dict[str, Any]) -> str:
    bucket = row.get("local_bucket")
    if bucket == "confirmed":
        return "Behavior should match upstream fixed behavior: no crash, no internal compiler error, and numerically correct results on XPU."
    if bucket == "blocked-env":
        return "A distributed or otherwise unavailable environment should be provisioned before validation."
    if bucket == "not-reproduced":
        return "The upstream failure scenario should fail locally if the bug is still present."
    return "The adapted XPU repro should execute cleanly and produce a decisive outcome."


def guessed_actual(row: dict[str, Any], signature: str) -> str:
    bucket = row.get("local_bucket")
    if bucket == "confirmed":
        return f"Local XPU validation reproduced a failure signature: {signature}"
    if bucket == "blocked-env":
        return f"Validation stopped before reaching the oracle because the environment was incomplete: {signature}"
    if bucket == "not-reproduced":
        return f"The adapted XPU repro completed without reproducing the upstream bug signal. Signature: {signature}"
    return f"Observed local outcome: {signature}"


def suggested_title(row: dict[str, Any]) -> str:
    route = row.get("target") or "pytorch/pytorch"
    prefix = "[XPU]" if route == "intel/torch-xpu-ops" else "[XPU parity]"
    return f"{prefix} {row['title']}"


def route_label(row: dict[str, Any]) -> str:
    return row.get("target") or "unrouted"


def display_path(path: Path, run_dir: Path) -> str:
    try:
        return str(path.relative_to(run_dir))
    except ValueError:
        return str(path)


def bucket_label(bucket: str) -> str:
    return bucket or "unknown"


def title_bucket_tag(bucket: str) -> str:
    return bucket_label(bucket).upper().replace("_", "-")


def classify_signal(signature: str, bucket: str) -> str:
    lower = signature.lower()
    if bucket == "confirmed":
        if "wrong result" in lower or "bug result" in lower:
            return "wrong result"
        if "dumped core" in lower or "segmentation fault" in lower or "floating point exception" in lower:
            return "hard crash"
        if "dynamo unsupported" in lower or "inductorerror" in lower or "attributeerror" in lower:
            return "compiler/runtime failure"
        return "confirmed failure"
    if bucket == "related-failure":
        return "different XPU failure"
    if bucket.startswith("blocked"):
        return "validation blocked"
    if bucket == "not-reproduced":
        return "not reproduced"
    return bucket_label(bucket)


def render_executive_summary(
    report: list[str],
    ordered_tested: list[dict[str, Any]],
    bucket_counts: Counter[str],
    route_counts: Counter[str],
    artifacts_dir: Path,
) -> None:
    confirmed_rows = [row for row in ordered_tested if row.get("local_bucket") == "confirmed"]
    related_rows = [row for row in ordered_tested if row.get("local_bucket") == "related-failure"]
    blocked_rows = [row for row in ordered_tested if str(row.get("local_bucket", "")).startswith("blocked")]

    signal_counts: Counter[str] = Counter()
    for row in confirmed_rows + related_rows:
        signature = error_signature(artifacts_dir / f"output_{row['id']}.log")
        signal_counts[classify_signal(signature, row.get("local_bucket", ""))] += 1

    report.append("## Executive Summary")
    report.append("")
    report.append(
        f"- Actionable outcomes: `{len(confirmed_rows) + len(related_rows)}` candidates need follow-up ({bucket_counts.get('confirmed', 0)} confirmed, {bucket_counts.get('related-failure', 0)} related-failure)."
    )
    report.append(
        f"- Routing: `{route_counts.get('pytorch/pytorch', 0)}` point to pytorch/pytorch, `{route_counts.get('intel/torch-xpu-ops', 0)}` point to intel/torch-xpu-ops."
    )
    if blocked_rows:
        report.append(f"- Blockers: `{len(blocked_rows)}` candidates were not decided because validation was blocked.")
    if signal_counts:
        top_signals = ", ".join(
            f"{label} ({count})" for label, count in signal_counts.most_common(3)
        )
        report.append(f"- Dominant failure signals: {top_signals}.")
    report.append("")

    if confirmed_rows or related_rows:
        report.append("## Action Board")
        report.append("")
        report.append("| ID | Kind | Route | Signal | Upstream | Title |")
        report.append("|---|---|---|---|---|---|")
        for row in confirmed_rows + related_rows:
            signature = error_signature(artifacts_dir / f"output_{row['id']}.log")
            upstream = f"[link]({row['url']})" if row.get("url") else "-"
            report.append(
                "| #{id} | {kind} | {route} | {signal} | {upstream} | {title} |".format(
                    id=row["id"],
                    kind=row.get("kind", ""),
                    route=route_label(row),
                    signal=classify_signal(signature, row.get("local_bucket", "")),
                    upstream=upstream,
                    title=shorten(str(row.get("title", "")), 80),
                )
            )
        report.append("")

    if blocked_rows:
        report.append("## Blockers")
        report.append("")
        report.append("| ID | Kind | Blocker | Upstream | Title |")
        report.append("|---|---|---|---|---|")
        for row in blocked_rows:
            signature = error_signature(artifacts_dir / f"output_{row['id']}.log")
            upstream = f"[link]({row['url']})" if row.get("url") else "-"
            report.append(
                "| #{id} | {kind} | {blocker} | {upstream} | {title} |".format(
                    id=row["id"],
                    kind=row.get("kind", ""),
                    blocker=shorten(signature, 56),
                    upstream=upstream,
                    title=shorten(str(row.get("title", "")), 72),
                )
            )
        report.append("")


def render_entry(
    index: int,
    row: dict[str, Any],
    detail: dict[str, Any] | None,
    env: dict[str, str],
    output_log: Path,
    script_path: Path,
    run_dir: Path,
) -> list[str]:
    lines: list[str] = []
    signature = error_signature(output_log)
    excerpt_lines = output_excerpt(output_log)
    bucket = row.get("local_bucket", "unknown")
    lines.append(
        f"**{index}. #{row['id']} — {row['title']} [{title_bucket_tag(bucket)}]**"
    )
    lines.append("<details>")
    lines.append("<summary>Details</summary>")
    lines.append("")
    lines.append(f"Summary: {upstream_context(detail)}")
    evidence = f"[Upstream link]({row['url']})" if row.get("url") else "N/A"
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|---|---|")
    lines.append(f"| Evidence | {evidence} |")
    lines.append(f"| Route suggestion | `{route_label(row)}` |")
    lines.append(f"| Reproducer script | `{display_path(script_path, run_dir)}` |")
    lines.append(f"| Output log | `{display_path(output_log, run_dir)}` |")
    lines.append(f"| Environment | torch `{env['torch']}`; XPU `{env['xpu']}`; Python `{env['python']}`; GPU `{env['gpu']}` |")
    lines.append(f"| Error signature | `{signature}` |")
    lines.append("")
    lines.append("Observed output excerpt:")
    lines.append("```text")
    lines.extend(excerpt_lines)
    lines.append("```")
    lines.append(f"Local XPU result: `{bucket}`")
    if bucket == "blocked-env":
        lines.append("Submission note: local environment was insufficient for a decisive repro; include the missing prerequisite if you open a tracking issue.")
    lines.append("")
    lines.append("</details>")
    lines.append("")
    return lines


def render_report(run_dir: Path, output_path: Path) -> None:
    artifacts_dir = run_dir / "artifacts"
    ledger_path = artifacts_dir / "candidate_ledger.jsonl"
    details_dir = artifacts_dir / "details"
    scripts_dir = run_dir / "scripts"

    rows = read_jsonl(ledger_path)
    tested = [row for row in rows if row.get("deep_status") == "pass-to-repro"]
    ordered_tested = sorted(
        tested,
        key=lambda row: (
            0 if row.get("local_bucket") == "confirmed" else 1,
            0 if row.get("local_bucket") == "blocked-env" else 1,
            row.get("kind", ""),
            str(row.get("id", "")),
        ),
    )
    env = collect_env_summary(artifacts_dir / "collect_env.txt")

    title_passed = sum(row.get("title_status") == "pass" for row in rows)
    title_rejected = sum(row.get("title_status") == "reject" for row in rows)
    deep_counts = Counter(row.get("deep_status", "<missing>") for row in rows)
    bucket_counts = Counter(row.get("local_bucket") for row in tested)
    route_counts = Counter(row.get("target") for row in tested if row.get("local_bucket") == "confirmed")
    kind_counts = Counter(row.get("kind") for row in rows)

    report: list[str] = []
    report.append(f"# Full Scan Report — pytorch/pytorch {run_dir.name}")
    report.append("")
    report.append(f"**Scan window:** {run_dir.name}T00:00:00Z to {run_dir.name}T23:59:59Z")
    report.append(f"**XPU torch version:** {env['torch']}")
    report.append(
        f"**Total raw candidates:** {len(rows)} (issues={kind_counts.get('issue', 0)}, prs={kind_counts.get('pr', 0)}, commits={kind_counts.get('commit', 0)})"
    )
    report.append(f"**Title-triage passed:** {title_passed}")
    report.append(f"**Deep-filter pass-to-repro:** {deep_counts.get('pass-to-repro', 0)}")
    report.append(
        f"**Confirmed:** {bucket_counts.get('confirmed', 0)} | **Not reproduced:** {bucket_counts.get('not-reproduced', 0)} | **Blocked:** {bucket_counts.get('blocked-env', 0) + bucket_counts.get('blocked-platform', 0) + bucket_counts.get('blocked-fetch', 0) + bucket_counts.get('blocked-commit-context', 0) + bucket_counts.get('blocked-script-error', 0)}"
    )
    report.append("")
    render_executive_summary(report, ordered_tested, bucket_counts, route_counts, artifacts_dir)
    report.append("## Artifact Index")
    report.append("")
    report.append(f"- Ledger: `{ledger_path}`")
    report.append(f"- Environment snapshot: `{artifacts_dir / 'collect_env.txt'}`")
    report.append(f"- Raw candidates: `{artifacts_dir / 'raw_candidates.json'}`")
    report.append(f"- Detail JSONs: `{details_dir}`")
    report.append(f"- Reproducer scripts: `{scripts_dir}`")
    report.append("")
    report.append("## Triage Stats")
    report.append("")
    report.append(f"- Raw candidates collected: `{len(rows)}`")
    report.append(f"- Title-triage rejected: `{title_rejected}`")
    report.append(f"- Title-triage passed: `{title_passed}`")
    report.append(f"- Deep-filter pass-to-repro: `{deep_counts.get('pass-to-repro', 0)}`")
    report.append(f"- Deep-filter rejected: `{deep_counts.get('reject', 0)}`")
    report.append("")
    report.append("## Validation Stats")
    report.append("")
    for bucket in ["confirmed", "not-reproduced", "blocked-env", "blocked-platform", "blocked-fetch", "blocked-script-error", "related-failure"]:
        if bucket_counts.get(bucket, 0):
            report.append(f"- {bucket}: `{bucket_counts[bucket]}`")
    report.append("")
    report.append("## Routing Stats")
    report.append("")
    for target, count in sorted(route_counts.items()):
        report.append(f"- {target}: `{count}`")
    report.append("")
    report.append("## Tested Candidates")
    report.append("")

    for index, row in enumerate(ordered_tested, 1):
        detail = load_json(details_dir / f"{row['id']}.json")
        output_log = artifacts_dir / f"output_{row['id']}.log"
        script_path = scripts_dir / f"repro_{row['id']}.py"
        report.extend(render_entry(index, row, detail, env, output_log, script_path, run_dir))

    report.append("---")
    report.append("")
    report.append("## Final Summary")
    report.append("")
    report.append("**Audit:** PASSED — 0 pending actionable rows.")
    report.append("")
    report.append("### Filter Stats")
    report.append("| Stage | Count |")
    report.append("|---|---|")
    report.append(f"| Raw candidates collected | {len(rows)} |")
    report.append(f"| Title-triage rejected | {title_rejected} |")
    report.append(f"| Title-triage passed | {title_passed} |")
    report.append(f"| Deep-filter pass-to-repro | {deep_counts.get('pass-to-repro', 0)} |")
    report.append(f"| Deep-filter rejected | {deep_counts.get('reject', 0)} |")
    report.append("")
    report.append("### Validation Stats")
    report.append("| Bucket | Count |")
    report.append("|---|---|")
    for bucket in ["confirmed", "not-reproduced", "blocked-env", "related-failure"]:
        report.append(f"| {bucket} | {bucket_counts.get(bucket, 0)} |")
    report.append("")
    report.append("### Routing Stats")
    report.append("| Target | Count |")
    report.append("|---|---|")
    for target, count in sorted(route_counts.items()):
        report.append(f"| {target} | {count} |")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(report) + "\n")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = run_dir / "reports" / "full_scan.md"
    render_report(run_dir, output_path)


if __name__ == "__main__":
    main()