#!/usr/bin/env python3
"""Files the issues the ut-issue-authoring skill drafted.

The skill reads the evidence, decides which failures share a root cause, and
writes one draft per group. It does not talk to GitHub. This script does, and
it is the only part of the pipeline that writes anything.

Everything here is mechanical, which is the point of the split: a case line is
checked against the evidence before it can mute anything, a case an open issue
already mutes is dropped, labels and titles follow from the classification, and
a group too large for one body is split rather than shortened. None of that is
a judgement, and none of it should depend on a model having followed an
instruction.

    python .github/scripts/ut_create_issues.py --evidence-dir ./evidence \\
        --drafts ./drafts.json --dry-run
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = os.environ.get("GITHUB_REPOSITORY") or "intel/torch-xpu-ops"
SERVER = os.environ.get("GITHUB_SERVER_URL") or "https://github.com"
PYTORCH_REPO = "pytorch/pytorch"
# The one definition of a skip issue: its sections, their order, and the labels
# it carries. Read rather than mirrored, so a bot issue cannot drift from a
# hand-written one.
SKIP_FORM = Path(".github/ISSUE_TEMPLATE/dynamic-skip.yml")

# Stamped into every body so a later night, and a reader, can tell a
# machine-filed issue from a hand-written one.
MARKER_VERSION = "v1"
MARKER = "<!-- ut-auto-issue:{version}:run={run_id}:part={part}/{parts} -->"
CASES_BEGIN = "<!-- cases:begin -->"
CASES_END = "<!-- cases:end -->"

TITLE_PREFIX = "[Bug Skip]: "
MAX_TITLE = 140
CLS_PREFIX = {"regression": "[Regression] ", "new_case_failure": "[New Case] "}
# An issue may carry any of these, so dedup has to ask for each separately:
# repeated --label on one `gh issue list` means every label at once.
DEDUP_LABELS = ("skipped", "skipped_bmg", "regression", "new_case_failure")
MAX_CASES_PER_ISSUE = 400
# Headroom below GitHub's 65536, so appending to an issue later has room.
SAFE_BODY_LIMIT = 60000
# A burst guard, not a quota: past this the night is a question about the
# machine rather than a set of bugs.
MAX_ISSUES_PER_RUN = 15

NO_TRACEBACK = "No traceback was captured in the JUnit XML for this failure."


def run(cmd: list[str], check: bool = True) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{proc.stderr.strip()}")
    return proc.stdout


def gh_tsv(path: str, jq: str) -> list[list[str]]:
    out = run(["gh", "api", "--paginate", path, "-q", jq])
    return [line.split("\t") for line in out.splitlines() if line.strip()]


def warn(msg: str) -> None:
    print(f"::warning::{msg}")


# --------------------------------------------------------------------------- #
# What the open issues already mute
# --------------------------------------------------------------------------- #


def parse_cases_block(body: str) -> set[str]:
    """The lines of one body that actually mute.

    `ut_result_check.sh:mark_passed_issue` rewrites a line to `~~<line>~~` once
    the case passes, and the subtraction is `grep -vFxf` - whole line, fixed
    string - so a struck line matches nothing and mutes nothing. It is history:
    that issue claimed the case and has released it.
    """
    start, end = body.find(CASES_BEGIN), body.find(CASES_END)
    if start == -1 or end == -1 or end < start:
        return set()
    live = set()
    for raw in body[start + len(CASES_BEGIN):end].splitlines():
        line = raw.strip()
        if line and line != "Cases:" and not (
                line.startswith("~~") and line.endswith("~~")):
            live.add(line)
    return live


def already_muted() -> dict[str, int]:
    """Every case line an open issue still mutes, and which issue mutes it.

    Raises rather than returning what it managed to read: this is the only
    thing standing between a case an issue already covers and a second issue
    covering it again, so an empty answer has to mean "nothing is muted", never
    "the query failed".
    """
    muted: dict[str, int] = {}
    for label in DEDUP_LABELS:
        rows = gh_tsv(
            f"repos/{REPO}/issues?state=open&labels={label}&per_page=100",
            ".[] | select(.pull_request == null) "
            '| [(.number|tostring), (.body // "" | @base64)] | @tsv',
        )
        for row in rows:
            if len(row) < 2:
                continue
            number = int(row[0])
            body = base64.b64decode(row[1]).decode("utf-8", errors="replace")
            for line in parse_cases_block(body):
                muted.setdefault(line, number)
    return muted


# --------------------------------------------------------------------------- #
# The form, read
# --------------------------------------------------------------------------- #


def form_text() -> str:
    return SKIP_FORM.read_text(encoding="utf-8")


def form_fields(text: str) -> list[tuple[str, str]]:
    """`(id, label)` per textarea, in the order GitHub renders them.

    Scanned rather than parsed as YAML so this stays stdlib-only. The two keys
    are unambiguous by indentation: `id` is a sibling of `type` at two spaces,
    `label` sits under `attributes` at four, and any prose that might contain
    either word is indented deeper inside a `description`.
    """
    fields: list[tuple[str, str]] = []
    current: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("- type:"):
            current = {"type": line.split(":", 1)[1].strip()}
        elif line.startswith("  id: "):
            current["id"] = line[6:].strip()
        elif line.startswith("    label: "):
            current["label"] = line[11:].strip()
            if current.get("type") == "textarea" and "id" in current:
                fields.append((current["id"], current["label"]))
    if not fields:
        raise SystemExit(f"::error::no identified textarea in {SKIP_FORM}")
    return fields


def form_labels(text: str) -> list[str]:
    """The labels the form gives every skip issue, plus the ones it cannot.

    `labels:` applies only to the web flow - `gh issue create` does not read a
    template - so it is read here and applied by hand. The rest cannot be
    expressed by a form at all, since a form cannot set a label from what a
    field contains, so the form declares them for this script instead.
    """
    match = re.search(r"^labels:\s*\[(.*?)\]\s*$", text, re.MULTILINE)
    if not match:
        raise SystemExit(f"::error::no labels: line in {SKIP_FORM}")
    return [name.strip().strip("\"'") for name in match.group(1).split(",")
            if name.strip()]


def bot_labels(text: str) -> dict[str, str]:
    found = dict(re.findall(r"^#\s*bot-label\s+(\S+):\s*(\S+)\s*$", text,
                            re.MULTILINE))
    missing = {"bmg-runner", "cls-regression", "cls-new_case_failure"} - set(found)
    if missing:
        raise SystemExit(
            f"::error::{SKIP_FORM} declares no bot-label for {sorted(missing)}")
    return found


# --------------------------------------------------------------------------- #
# Labels and title
# --------------------------------------------------------------------------- #


def is_bmg(runner: str) -> bool:
    # Case-insensitive where fetch_issues.sh is not: the runner label there is
    # `bmg-test` while the hostname recorded in the evidence is `BMG-17691`.
    return "bmg" in runner.lower()


def labels_for(cls: str, runner: str, base: list[str],
               extra: dict[str, str]) -> list[str]:
    """A `persistent` or `unknown` group gets no classification label, because
    neither "it used to pass" nor "it is a new case" is true of it."""
    labels = list(base)
    if is_bmg(runner):
        labels.append(extra["bmg-runner"])
    if f"cls-{cls}" in extra:
        labels.append(extra[f"cls-{cls}"])
    return labels


def title_for(text: str, cls: str, collection_error: bool) -> str:
    prefix = TITLE_PREFIX
    if collection_error:
        prefix += "[Failed to collect] "
    prefix += CLS_PREFIX.get(cls, "")
    room = MAX_TITLE - len(prefix)
    body = " ".join(text.split()).encode("ascii", "replace").decode()
    return prefix + (body if len(body) <= room else body[:room - 3] + "...")


# --------------------------------------------------------------------------- #
# Body
# --------------------------------------------------------------------------- #


def commit_link(repo: str, sha: str) -> str:
    return f"[`{sha[:8]}`]({SERVER}/{repo}/commit/{sha})" if sha else "unknown"


def evidence_block(run: dict, category: str, cls: str, cls_reason: str,
                   contexts: list[dict]) -> str:
    """When the failure started, and what it cost, for one group's category.

    The bisect range is why this is rendered rather than left to a reader: the
    baseline sha and tonight's sha have to come from the same UT job or the
    range spans the wrong commits, and nothing in the rendered link says which
    UT job it came from.
    """
    base = run.get("baselines", {}).get(category)
    ut_job = run["category_ut_job"].get(category, "")
    lines: list[str] = []
    if base is None:
        lines.append(
            f"No nightly in the lookback window completed `{category}` "
            "healthily, so when this failure started could not be determined."
        )
    else:
        lines += [
            "| Category | | Run | Date | torch | torch-xpu-ops |",
            "|---|---|---|---|---|---|",
            f"| {category} | Last good "
            f"| [#{base['run_id']} ({base['ut_job']})]({base['job_url']}) "
            f"| {base['created_at']} "
            f"| {commit_link(PYTORCH_REPO, base['torch'])} "
            f"| {commit_link(REPO, base['torch_xpu_ops'])} |",
            f"| {category} | First seen bad "
            f"| [#{run['run_id']} ({ut_job})]({run['job_urls'].get(ut_job, '')}) "
            f"| {run['created_at']} "
            f"| {commit_link(PYTORCH_REPO, run['torch'].get(ut_job, ''))} "
            f"| {commit_link(REPO, run['torch_xpu_ops'].get(ut_job, ''))} |",
            "",
        ]
        if base["torch"] and run["torch"].get(ut_job):
            link = (f"Changes in range: [pytorch]({SERVER}/{PYTORCH_REPO}/compare/"
                    f"{base['torch']}...{run['torch'][ut_job]})")
            if base["torch_xpu_ops"] and run["torch_xpu_ops"].get(ut_job):
                link += (f" - [torch-xpu-ops]({SERVER}/{REPO}/compare/"
                         f"{base['torch_xpu_ops']}..."
                         f"{run['torch_xpu_ops'][ut_job]})")
            lines += [link, ""]
        # A stale baseline keeps `regression` true but makes the range much
        # weaker evidence, so say so rather than presenting a five-night range
        # in the same shape as a one-night one.
        if base["age_in_runs"] > 1:
            gap = base["age_in_runs"] - 1
            lines += [
                f"Note: the last healthy `{category}` nightly was "
                f"{base['age_in_runs']} runs back ({gap} intervening "
                f"{'nightly' if gap == 1 else 'nightlies'} did not complete this "
                "category), so this range is wider than one night and the "
                "failure may predate the first-seen-bad run.",
                "",
            ]
    if cls == "persistent":
        lines += [f"Classified `persistent`: {cls_reason}. The range above is "
                  "not this failure's onset.", ""]
    if cls == "unknown":
        lines += [f"Unclassified: {cls_reason}.", ""]
    if contexts:
        lines += [
            "These test files did not fail their cases - they erased them. A "
            "module that will not import reports one row and its cases never "
            "run at all.",
            "",
            "| Module | Category | In the baseline | Cases it used to pass |",
            "|---|---|---|---|",
        ]
        lines += [
            f"| `{c['module']}` | {c['category']} | {c['state']} "
            f"| {c['baseline_passed']} |"
            for c in contexts
        ]
        lines.append("")
    return "\n".join(lines).strip()


def reproduce_for(case: dict, reproduce: dict) -> str:
    entry = reproduce.get(case["category"])
    if not entry:
        return f"# No reproduce command was recorded for {case['category']}."
    module, class_name = case["module"], case["class_name"]
    if class_name and module and class_name.startswith(module + "."):
        target = (f"{case['test_file']}::{class_name[len(module) + 1:]}"
                  f"::{case['test_name']}")
    elif class_name:
        target = f"{case['test_file']}::{class_name}::{case['test_name']}"
    else:
        target = case["test_file"]
    command = entry.get("command_template", "").replace("failed_case", target)
    return "\n".join(x for x in (entry.get("file_path", ""), command) if x)


def error_log_for(cases: list[dict], tracebacks: dict, chosen: str = "") -> str:
    """The failing message, then its traceback, for one case of the group.

    `chosen` is the draft's pick, which is a judgement: a group is one root
    cause, not one message, so the clearest traceback in it is not necessarily
    the first. The text itself is copied from the evidence either way - it is
    third-party output, and nothing downstream should be able to reword it.

    The message is a level below the section headings, which are the form's
    fields: a `###` here would read as another section.
    """
    order = sorted(cases, key=lambda c: c["line"] != chosen)
    for case in order:
        text = tracebacks.get(case["line"])
        if text:
            return "\n".join([
                f"#### {case['message'] or 'No message was recorded.'}",
                "",
                "```",
                *text,
                "```",
            ])
    message = next((c["message"] for c in order if c["message"]), "")
    heading = f"#### {message}" if message else (
        "#### No message was recorded for this failure.")
    return "\n".join([heading, "", NO_TRACEBACK])


def cases_section(lines: list[str]) -> str:
    """The muting block, in the shape all three of its parsers expect.

    The blank line before the end marker is what ends the block: the awk filter
    in _linux_ut.yml stops at the first line with no alphanumerics and
    mark_passed_issue at the first empty one, and the marker itself is neither.
    """
    return "\n".join([CASES_BEGIN, "Cases:", *lines, "", CASES_END])


def render_body(fields: list[tuple[str, str]], slots: dict[str, str],
                marker: str) -> str:
    """One `### <label>` section per form field, in the form's order.

    A field the script has nothing for is still rendered, empty, rather than
    dropped: the sections a skip issue has are the form's business, and a body
    missing one would not be the same document as a hand-written issue.
    """
    unknown = set(slots) - {field_id for field_id, _ in fields}
    if unknown:
        raise SystemExit(
            f"::error::{SKIP_FORM} has no field for {sorted(unknown)}")
    out: list[str] = []
    for field_id, label in fields:
        out += [f"### {label}", "", slots.get(field_id, ""), ""]
    return "\n".join(out + [marker, ""])


# --------------------------------------------------------------------------- #
# Drafts, checked before anything is created
# --------------------------------------------------------------------------- #


def check_draft(draft: dict, index: dict[str, dict]) -> tuple[list[dict], str]:
    """The draft's cases, or the reason it cannot be filed.

    A case line is a byte-exact subtraction rule: `grep -vFxf`, whole line,
    fixed string. One that names no real case matches nothing the night it is
    written, looks harmless, and silently swallows a real failure the first
    night a test of that name fails. So a line the evidence does not contain
    is not repaired here; it fails the whole draft.
    """
    lines = draft.get("cases") or []
    if not lines:
        return [], "no cases"
    if not (draft.get("summary") or "").strip():
        return [], "no summary"
    if not (draft.get("title_text") or "").strip():
        return [], "no title"
    unknown = [ln for ln in lines if ln not in index]
    if unknown:
        return [], (f"{len(unknown)} case line(s) name no case in this run's "
                    f"evidence, the first being `{unknown[0]}`")
    cases = [index[ln] for ln in dict.fromkeys(lines)]
    chosen = draft.get("error_case") or ""
    if chosen and chosen not in {c["line"] for c in cases}:
        return [], f"`error_case` names `{chosen}`, which is not in this group"
    classes = {c["cls"] for c in cases}
    if len(classes) > 1:
        return [], f"mixes classifications: {', '.join(sorted(classes))}"
    shapes = {c["is_collection_error"] for c in cases}
    if len(shapes) > 1:
        return [], "mixes whole-module rows with ordinary cases"
    return cases, ""


def quoted_traceback_matches(draft: dict, tracebacks: dict) -> str:
    """Whether the draft's `traceback` really is the evidence it claims to quote.

    The copy is not what the issue renders, so a wrong one mutes nothing. What
    it does is put the summary's argument next to the text it was argued from,
    for a human reviewing the drafts - and a copy that has been tidied up,
    shortened or invented leaves that reader checking the reasoning against
    the reasoning. Reported, not rejected: the harm is a misled reviewer.
    """
    quoted = draft.get("traceback")
    if quoted is None:
        return ""
    chosen = draft.get("error_case") or ""
    if not chosen:
        return "carries a `traceback` without an `error_case` to attribute it to"
    actual = tracebacks.get(chosen)
    if actual is None:
        return f"quotes a traceback for `{chosen}`, which has none in the evidence"
    if list(quoted) != list(actual):
        return (f"quotes {len(quoted)} line(s) for `{chosen}` where the "
                f"evidence has {len(actual)}, and they do not match")
    return ""


def split(cases: list[dict], body_of) -> list[list[dict]]:
    """Into as few parts as each will render inside the body limit.

    Splitting is the only correct response to a group that is too big: the
    `Cases:` block is what mutes, so shortening it to fit leaves cases failing
    every night with an issue that claims to cover them.
    """
    parts: list[list[dict]] = []
    queue = [cases[i:i + MAX_CASES_PER_ISSUE]
             for i in range(0, len(cases), MAX_CASES_PER_ISSUE)]
    while queue:
        chunk = queue.pop(0)
        if len(chunk) > 1 and len(body_of(chunk)) > SAFE_BODY_LIMIT:
            half = len(chunk) // 2
            queue[:0] = [chunk[:half], chunk[half:]]
            continue
        parts.append(chunk)
    return parts


# --------------------------------------------------------------------------- #
# Create
# --------------------------------------------------------------------------- #


def create_issue(title: str, body: str, labels: list[str], work: Path) -> int:
    path = work / "body.md"
    path.write_text(body, encoding="utf-8")
    cmd = ["gh", "issue", "create", "--repo", REPO, "--title", title,
           "--body-file", str(path)]
    for label in labels:
        cmd += ["--label", label]
    url = run(cmd).strip().splitlines()[-1]
    return int(url.rstrip("/").rsplit("/", 1)[-1])


def comment(number: int, text: str) -> None:
    run(["gh", "issue", "comment", str(number), "--repo", REPO, "--body", text],
        check=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--drafts", required=True)
    parser.add_argument("--report-dir", default="ut_auto_issue_report")
    parser.add_argument("--work-dir", default="ut_auto_issue_work")
    parser.add_argument("--dry-run", action="store_true",
                        help="render and check everything, create nothing")
    args = parser.parse_args()

    evidence = json.loads((Path(args.evidence_dir) / "evidence.json").read_text())
    tracebacks = evidence.get("tracebacks", {})
    run_json = evidence["run"]
    index = {c["line"]: c for c in evidence["cases"]}
    contexts = {c["line"]: c for c in evidence.get("collection_context", [])}
    reproduce = evidence.get("reproduce", {})

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    report = {"run_id": run_json["run_id"], "dry_run": args.dry_run,
              "created": [], "skipped": [], "rejected": [], "misquoted": []}
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    drafts_path = Path(args.drafts)
    if not drafts_path.is_file():
        warn(f"no drafts at {drafts_path}; nothing to file")
        return finish(report, report_dir)
    drafts_doc = json.loads(drafts_path.read_text())
    if drafts_doc.get("digest") != run_json["digest"]:
        raise SystemExit(
            "::error::the drafts were written against a different set of "
            "failures than this evidence describes; filing nothing"
        )

    gates = run_json.get("gates", {})
    blocking = [name for name, hit in gates.items() if hit]
    if blocking:
        warn(f"gate(s) {', '.join(blocking)} are set for this run; filing nothing")
        return finish(report, report_dir)

    form = form_text()
    fields = form_fields(form)
    base_labels = form_labels(form)
    extra_labels = bot_labels(form)
    muted = already_muted()
    created: list[tuple[dict, list[int]]] = []

    for draft in drafts_doc.get("drafts", []):
        name = draft.get("id") or draft.get("title_text", "?")
        if not draft.get("file"):
            report["skipped"].append(
                {"draft": name, "reason": draft.get("reason") or "not for filing"})
            continue
        cases, problem = check_draft(draft, index)
        if problem:
            report["rejected"].append({"draft": name, "reason": problem})
            warn(f"draft {name} not filed: {problem}")
            continue
        misquote = quoted_traceback_matches(draft, tracebacks)
        if misquote:
            report["misquoted"].append({"draft": name, "reason": misquote})
            warn(f"draft {name} {misquote}; the issue still renders the "
                 "evidence, but review the summary against it")

        placed = [c for c in cases if c["line"] in muted]
        cases = [c for c in cases if c["line"] not in muted]
        if not cases:
            report["skipped"].append(
                {"draft": name,
                 "reason": f"all {len(placed)} case(s) are already muted by an "
                           f"open issue, e.g. #{muted[placed[0]['line']]}"})
            continue
        if len(report["created"]) >= MAX_ISSUES_PER_RUN:
            report["skipped"].append(
                {"draft": name,
                 "reason": f"past the {MAX_ISSUES_PER_RUN} issues one run may file"})
            continue

        cls = cases[0]["cls"]
        runner = cases[0]["runner_name"]
        labels = labels_for(cls, runner, base_labels, extra_labels)
        category = cases[0]["category"]
        group_contexts = [contexts[c["line"]] for c in cases
                          if c["line"] in contexts]

        def body_of(chunk: list[dict], part: int = 1, parts: int = 1,
                    first: int | None = None) -> str:
            error_log = (f"See #{first} for the failure text." if first
                         else error_log_for(chunk, tracebacks,
                                            draft.get("error_case", "")))
            collect_env = run_json["collect_env"].get(
                cases[0]["ut_job"], "collect_env was not captured.")
            return render_body(fields, {
                "cases": cases_section([c["line"] for c in chunk]),
                "summary": draft["summary"].strip(),
                "error_log": error_log,
                "reproduce": f"```bash\n{reproduce_for(chunk[0], reproduce)}\n```",
                "pytorch_version": evidence_block(
                    run_json, category, cls, cases[0].get("cls_reason", ""),
                    group_contexts if part == 1 else []),
                "versions": ("<details><summary>Detail</summary>\n\n```\n"
                             f"{collect_env}\n```\n\n</details>"),
            }, MARKER.format(version=MARKER_VERSION,
                             run_id=run_json["run_id"],
                             part=part, parts=parts))

        chunks = split(cases, body_of)
        numbers: list[int] = []
        for position, chunk in enumerate(chunks, 1):
            title = title_for(draft["title_text"], cls,
                              cases[0]["is_collection_error"])
            if len(chunks) > 1:
                title = f"{title} (part {position}/{len(chunks)})"
            body = body_of(chunk, position, len(chunks),
                           numbers[0] if numbers else None)
            if args.dry_run:
                preview = report_dir / "drafts"
                preview.mkdir(parents=True, exist_ok=True)
                (preview / f"{name}-{position}.md").write_text(
                    f"# {title}\n# labels: {', '.join(labels)}\n\n{body}",
                    encoding="utf-8")
                number = 0
            else:
                number = create_issue(title, body, labels, work)
            numbers.append(number)
            report["created"].append({
                "draft": name, "issue": number, "title": title,
                "labels": labels, "cases": len(chunk),
                "part": f"{position}/{len(chunks)}",
            })
            for case in chunk:
                muted[case["line"]] = number
        if placed:
            report["created"][-1]["already_muted"] = len(placed)
        created.append((draft, numbers))

    if not args.dry_run:
        cross_link(created)
    return finish(report, report_dir)


def cross_link(created: list[tuple[dict, list[int]]]) -> None:
    """One root cause split across drafts by classification or by shape.

    Nothing else in the pipeline will ever say the two are related, and a
    triager reading them separately sees two unrelated bugs.
    """
    by_id = {d.get("id"): numbers for d, numbers in created if d.get("id")}
    for draft, numbers in created:
        related = [n for rid in draft.get("related") or []
                   for n in by_id.get(rid, []) if n not in numbers]
        if related and numbers:
            links = ", ".join(f"#{n}" for n in related)
            comment(numbers[0],
                    f"Same root cause as {links}, split because the cases do "
                    "not share a classification or a row shape.")


def finish(report: dict, report_dir: Path) -> int:
    (report_dir / "created.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    lines = ["## UT auto-issue - filing", "",
             f"Run `{report['run_id']}`"
             + (" (dry run, nothing created)" if report["dry_run"] else ""), ""]
    if report["created"]:
        lines += ["| Issue | Title | Labels | Cases | Part |",
                  "|---|---|---|---|---|"]
        lines += [
            f"| {'(dry run)' if report['dry_run'] else '#' + str(c['issue'])} "
            f"| {c['title']} | {', '.join(c['labels'])} "
            f"| {c['cases']} | {c['part']} |"
            for c in report["created"]
        ]
        lines.append("")
    for item in report["skipped"]:
        lines.append(f"- Not filed, `{item['draft']}`: {item['reason']}")
    for item in report["rejected"]:
        lines.append(f"- **Rejected**, `{item['draft']}`: {item['reason']}")
    for item in report.get("misquoted", []):
        lines.append(f"- Filed, but `{item['draft']}` {item['reason']}")
    summary = "\n".join(lines) + "\n"
    print(summary)
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as handle:
            handle.write(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
