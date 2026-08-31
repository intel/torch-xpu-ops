#!/usr/bin/env python3
"""File one GitHub issue per root cause for a nightly UT run's new failures.

Fully deterministic, no model calls. Grouping, classification, labels and the
`Cases:` block are all computed from artifacts and set arithmetic. The pipeline
runs as the numbered stages banner-marked below, Stage 0 through Stage 8.

Because the issues carry the `skipped` label, fetch_issues.sh subtracts their
cases from the next nightly. Filing an issue therefore mutes a test, which is
why creating anything takes an explicit --create-issues and why Stage 0 refuses
to file anything from a run that does not look healthy.

Run by hand against a past nightly to inspect what it would do:

    python .github/scripts/ut_auto_issue.py --run-id <run_id>
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

REPO = os.environ.get("GITHUB_REPOSITORY") or "intel/torch-xpu-ops"
SERVER = os.environ.get("GITHUB_SERVER_URL") or "https://github.com"
PYTORCH_REPO = "pytorch/pytorch"
WORKFLOW = "nightly_ondemand.yml"

# Bumped when the normalization rules below change, so that a rules change
# re-files deliberately and visibly instead of silently breaking dedup.
MARKER_VERSION = "v1"
# `sig` identifies the root cause and is deliberately independent of `cls`.
# A filed case is normally subtracted from the next night's new failures and is
# never reclassified. But when it comes back - a human unmutes it, or
# ut_result_check.sh:mark_passed_issue strikes it through after it passes once -
# it is re-evaluated against a newer baseline and can land in a different class.
# Folding that moving value into the identity would strand the original issue.
MARKER_RE = re.compile(
    r"<!--\s*ut-auto-issue:(?P<ver>[\w.]+):sig=(?P<sig>[0-9a-f]+)"
    r":cls=(?P<cls>[a-z_]+):leg=(?P<leg>[^\s:]+)"
    r":part=(?P<part>\d+)/(?P<parts>\d+)\s*-->"
)
CASES_BEGIN = "<!-- cases:begin -->"
CASES_END = "<!-- cases:end -->"

# Four states, from comparing a failing case against its category's baseline:
#   regression        - passed in the baseline, fails now
#   new_case_failure  - absent from the baseline, or present but skipped
#   persistent        - already failing in the baseline; onset predates it
#   unknown           - no usable baseline for that category
CLS_REGRESSION = "regression"
CLS_NEW_CASE = "new_case_failure"
CLS_PERSISTENT = "persistent"
CLS_UNKNOWN = "unknown"
# Only these two are labels; `persistent` and `unknown` are stated in the body
# instead, because neither "it used to pass" nor "it is a new case" is true.
CLS_LABELS = {CLS_REGRESSION, CLS_NEW_CASE}

MAX_BASELINE_LOOKBACK = 5
MAX_CASES_PER_ISSUE = 400
MAX_ISSUES_PER_RUN = 15
ABORT_THRESHOLD = 5000
INFRA_SIGNATURE_RATIO = 0.3
# A share is only evidence once there is something to take a share of. Below
# this many new failures a single infra-looking one clears 30% on its own - it
# does so for any n <= 3 - and the leg would be discarded on one data point.
INFRA_MIN_CASES = 10
# How many distinct test files one infra signature may reach and still be filed
# as the bug it describes. Beyond this it is read as the machine instead. A
# couple of files sharing an OOM is an ordinary way for one memory regression
# to look, since these messages carry no operator to tell them apart; more than
# five unrelated files failing the same way in one night is not something a
# product bug does. Erring high is the safe side: holding a group back wrongly
# costs a night of red, filing wrongly mutes a test that still fails.
INFRA_MAX_FILES_TO_FILE = 5
HEALTH_RATIO = 0.95
GITHUB_BODY_LIMIT = 65536
# Headroom below the hard cap for appending to an issue filed on an earlier night.
SAFE_BODY_LIMIT = 60000

# Covered legs. xpu_distributed is deliberately excluded: it reports through
# run_distributed_tests in ut_result_check.sh, which produces neither the
# per-category passed/failed logs nor a case count, so neither the Stage 0
# health gate nor the Stage 4 baseline comparison has anything to read.
LEG_CATEGORIES = {
    "basic": ["op_regression", "op_regression_dev1", "op_extended"],
    "op_ut": ["op_ut"],
}
CATEGORY_LEG = {c: leg for leg, cats in LEG_CATEGORIES.items() for c in cats}

# nightly_ondemand.yml:166 runs both phase-1 legs on `bmg-test`, so both carry
# the BMG-only known-failure label honoured by fetch_issues.sh:25.
BMG_LEGS = {"basic", "op_ut"}

# Mirrors EXPECTED_CASES in ut_result_check.sh (linux column). Only a fallback:
# runs predating run_health.jsonl carry no recorded verdict of their own.
EXPECTED_CASES = {
    "op_extended": 5349,
    "op_regression": 268,
    "op_regression_dev1": 1,
    "op_ut": 178548,
}

# A run can pass the count check and still be poisoned - a runner losing its GPU
# near the end of the suite barely moves the count. Matching one of these is not
# on its own evidence of that: "infra" is a claim about the machine, and a
# message cannot make a claim about the machine. An OOM or a device-lost in one
# test file is far likelier to be that test allocating too much or hanging the
# GPU, which is a product bug and belongs in an issue. The same signature
# appearing across unrelated files in one night is what the machine looks like,
# so breadth decides - see INFRA_MAX_FILES_TO_FILE and build_groups.
INFRA_PATTERNS = [
    "device lost",
    "ze_result_error",
    "ur_result_error",
    "ur error",
    "out of memory",
    "outofmemoryerror",
    "no space left on device",
    "worker crashed",
    "connection reset",
    "connection refused",
    "bus error",
    "cannot allocate memory",
    "dmesg",
    "gpu hang",
]

DTYPES = (
    "float8_e4m3fn|float8_e5m2|bfloat16|complex128|complex64|float16|float32"
    "|float64|int16|int32|int64|int8|uint8|bool"
)
RE_PATH = re.compile(r"/(?:[^\s/]+/)+([^\s/]+)")
RE_HEX = re.compile(r"0x[0-9a-fA-F]+")
RE_LINE_NO = re.compile(r"\bline \d+")
RE_DTYPE_SUFFIX = re.compile(rf"_(?:xpu|cpu|cuda|meta)(?::\d+)?_(?:{DTYPES})\b")
RE_SAMPLE_INPUT = re.compile(r"SampleInput\(.*", re.DOTALL)
RE_TENSOR_REPR = re.compile(r"Tensor\[size=\([^)]*\)[^\]]*\]")
RE_NUMBER = re.compile(r"\b\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\b")

TEMPLATE_PATH = (
    Path(__file__).resolve().parents[1]
    / "ISSUE_TEMPLATE"
    / "agent"
    / "ut-auto-issue-body.md"
)


# --------------------------------------------------------------------------- #
# gh plumbing
# --------------------------------------------------------------------------- #


def run(cmd: list[str], check: bool = True) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{proc.stderr.strip()}")
    return proc.stdout


def gh_tsv(path: str, jq: str) -> list[list[str]]:
    """Paginated `gh api` returning TSV rows, so no field can contain a newline."""
    out = run(["gh", "api", "--paginate", path, "-q", jq], check=False)
    return [line.split("\t") for line in out.splitlines() if line.strip()]


def gh_json(path: str) -> dict:
    return json.loads(run(["gh", "api", path]))


def warn(msg: str) -> None:
    print(f"::warning::{msg}")


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Case:
    category: str
    class_name: str
    test_name: str
    message: str

    @property
    def line(self) -> str:
        return f"{self.category},{self.class_name},{self.test_name}"

    @property
    def leg(self) -> str:
        return CATEGORY_LEG.get(self.category, "unknown")

    @property
    def is_collection_error(self) -> bool:
        """Whether this row is a whole-module failure rather than a test case.

        pytest reports a module that failed to import with an empty classname
        and the dotted module path as the name, because
        _pytest/junitxml.py:mangle_test_address has nothing before the first
        `::` to put in classname. Every real case carries a class.
        """
        return not self.class_name

    @property
    def module(self) -> str:
        """Dotted path of the test file this row belongs to.

        For a collection error the module is the name; for a test case it is
        the classname with its trailing class segments removed.
        """
        source = self.test_name if self.is_collection_error else self.class_name
        parts = [p for p in source.split(".") if p]
        while parts and parts[-1][:1].isupper():
            parts.pop()
        return ".".join(parts)

    @property
    def test_file(self) -> str:
        module = self.module
        return f"{module.rsplit('.', 1)[-1]}.py" if module else "unknown"


@dataclass
class Group:
    normalized_error: str
    test_file: str
    cases: list[Case] = field(default_factory=list)
    headline: str = ""
    # How many test files this group's signature reached across the whole run.
    # A property of the set, so build_groups fills it in; the infra decision
    # and the issue body both need it and neither can derive it from one group.
    signature_files: int = 1
    # Non-empty means: report it, never file it, never mute it.
    quarantine: str = ""

    @property
    def collection_error(self) -> bool:
        return any(c.is_collection_error for c in self.cases)

    @property
    def sig(self) -> str:
        raw = f"{MARKER_VERSION}\n{self.normalized_error}\n{self.test_file}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @property
    def categories(self) -> list[str]:
        return sorted({c.category for c in self.cases})

    @property
    def legs(self) -> list[str]:
        return sorted({c.leg for c in self.cases})


@dataclass
class SubGroup:
    """One root cause restricted to a single classification == one issue series.

    A group is split by classification so that each issue carries an
    unambiguous label, but the split lives here and never in `Group.sig`.
    """
    group: Group
    cls: str
    cases: list[Case] = field(default_factory=list)
    siblings: list[int] = field(default_factory=list)

    @property
    def categories(self) -> list[str]:
        return sorted({c.category for c in self.cases})

    @property
    def legs(self) -> list[str]:
        return sorted({c.leg for c in self.cases})


@dataclass
class FamilyMember:
    """What one group contributed to its error family, for cross-file linking.

    `created` gates the note: without it, a family whose issues all stand open
    and unchanged would be re-announced every night.
    """
    test_file: str
    cases: int
    urls: list[str]
    created: bool


@dataclass
class IssueRef:
    number: int
    body: str
    cls: str
    part: int
    parts: int
    # Live lines only: these are what actually mute, and ownership is exactly
    # what is muted. A line struck through by ut_result_check.sh:mark_passed_issue
    # or deleted by a human has been released - see Stage 5 below.
    case_lines: set[str]


@dataclass
class Baseline:
    run_id: int
    created_at: str
    age_in_runs: int
    leg: str
    job_url: str
    torch: str
    torch_xpu_ops: str
    passed: set[str]
    failed: set[str]
    all_cases: set[str]
    # Per test module, so a whole-module failure can be measured against the
    # baseline: its own row exists in neither of the sets above.
    passed_by_module: dict[str, int]
    all_by_module: dict[str, int]


@dataclass
class RunInfo:
    run_id: int
    created_at: str
    job_urls: dict[str, str]  # leg -> job url
    torch: dict[str, str]  # leg -> sha
    torch_xpu_ops: dict[str, str]
    collect_env: dict[str, str]


# --------------------------------------------------------------------------- #
# Artifact access
# --------------------------------------------------------------------------- #


def list_artifacts(run_id: int) -> list[tuple[str, bool]]:
    rows = gh_tsv(
        f"repos/{REPO}/actions/runs/{run_id}/artifacts?per_page=100",
        ".artifacts[] | [.name, (.expired|tostring)] | @tsv",
    )
    return [(r[0], r[1] == "true") for r in rows if len(r) >= 2]


def pick_artifact(names: list[tuple[str, bool]], prefix: str, leg: str, run_id: int):
    """Highest run attempt of `<prefix>-<sha>-<leg>-<run_id>-<attempt>`."""
    pat = re.compile(rf"^{re.escape(prefix)}-.+-{re.escape(leg)}-{run_id}-(\d+)")
    best, best_attempt = None, -1
    for name, expired in names:
        m = pat.match(name)
        if m and not expired and int(m.group(1)) > best_attempt:
            best, best_attempt = name, int(m.group(1))
    return best


def download(run_id: int, artifact: str, dest: Path) -> bool:
    dest.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["gh", "run", "download", str(run_id), "--repo", REPO,
         "--name", artifact, "--dir", str(dest)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        warn(f"download of {artifact} failed: {proc.stderr.strip()}")
        return False
    return True


def find_file(root: Path, name: str) -> Path | None:
    """Shallowest match. The summary job moves the per-category logs into
    ut_log/<leg>/, so their depth differs before and after it runs."""
    hits = sorted(root.rglob(name), key=lambda p: (len(p.parts), str(p)))
    return hits[0] if hits else None


def read_lines(path: Path | None) -> list[str]:
    if path is None or not path.is_file():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    return [line.strip() for line in text.splitlines() if line.strip()]


# --------------------------------------------------------------------------- #
# Stage 0 - health
#
# The gate that runs before anything else. Its checks are cited as H1-H7 in the
# warnings and in the report artifact, so they are enumerated here:
#
#   H1  build job conclusion          not success -> nothing downstream can be
#                                     trusted; abort the whole run
#   H2  leg's test job conclusion     cancelled or skipped
#   H3  UT data artifact              missing, or fails to download -> skip the leg
#   H4  category present at all       no health record and no category log means
#                                     the category never ran: a quiet skip, not
#                                     an error
#   H5  case-count health             actual < HEALTH_RATIO * expected: the run
#                                     is truncated and the machine is suspect
#   H6  new-failure CSV row count     disagrees with new_failure_list.txt, so
#                                     some failures lost their error message
#   H7  infra-signature share         above INFRA_SIGNATURE_RATIO, over at
#                                     least INFRA_MIN_CASES failures: the leg is
#                                     infra breakage, not a set of product bugs
#
# H2 needs no code of its own - a cancelled or skipped leg uploads no artifact,
# so H3 catches it. Evaluation is per category rather than per leg, because the
# `basic` leg carries three and they fail independently.
# --------------------------------------------------------------------------- #


def read_run_health(root: Path) -> dict[str, dict]:
    """Last record wins: a re-run of the summary job appends to the copy it
    downloaded from the previous attempt."""
    path = find_file(root, "run_health.jsonl")
    records: dict[str, dict] = {}
    for line in read_lines(path):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "category" in rec:
            records[rec["category"]] = rec
    return records


def category_state(root: Path, category: str) -> tuple[str, int, int]:
    """(complete|truncated|absent, actual, expected).

    `absent` means the category never ran - a quiet skip, not an error. The
    fallback reads category_<cat>.log rather than counting passed+failures,
    because EXPECTED_CASES counts skipped cases too.
    """
    rec = read_run_health(root).get(category)
    if rec is not None:
        state = "complete" if rec.get("healthy") else "truncated"
        return state, int(rec.get("actual", 0)), int(rec.get("expected", 0))

    log = find_file(root, f"category_{category}.log")
    if log is None:
        return "absent", 0, EXPECTED_CASES.get(category, 0)
    actual = 0
    for line in read_lines(log):
        m = re.match(r"Test cases:\s*(\d+)", line)
        if m:
            actual = int(m.group(1))
    expected = EXPECTED_CASES.get(category)
    if not expected:
        return "complete", actual, 0
    state = "complete" if actual >= HEALTH_RATIO * expected else "truncated"
    return state, actual, expected


# --------------------------------------------------------------------------- #
# Stage 1 - collect this run's new failures
# --------------------------------------------------------------------------- #


def parse_failure_csv(path: Path | None) -> list[Case]:
    """Headerless markdown rows written by check-ut.py:print_md_row:
    `| Category | Class name | Test name | Status | Message | Source |`."""
    cases = []
    for line in read_lines(path):
        parts = line.strip().strip("|").split(" | ")
        if len(parts) < 6:
            continue
        # Only Message can contain a pipe, so bound it by the fixed columns.
        cases.append(
            Case(
                category=parts[0].strip(),
                class_name=parts[1].strip(),
                test_name=parts[2].strip(),
                message=" | ".join(parts[4:-1]).strip(),
            )
        )
    return cases


def resolve_jobs(run_id: int) -> list[tuple[int, str, str]]:
    rows = gh_tsv(
        f"repos/{REPO}/actions/runs/{run_id}/jobs?per_page=100",
        '.jobs[] | [(.id|tostring), .name, (.conclusion // "")] | @tsv',
    )
    return [(int(r[0]), r[1], r[2]) for r in rows if len(r) >= 3]


def job_url(run_id: int, jobs: list[tuple[int, str, str]], leg: str) -> str:
    """Job-level link, so the reader lands on the leg's log rather than a matrix
    summary page. Falls back to the run URL."""
    run_url = f"{SERVER}/{REPO}/actions/runs/{run_id}"
    cands = [j for j in jobs if f"({leg})" in j[1]]
    for jid, name, _ in cands:
        if name.endswith("test-in-container"):
            return f"{run_url}/job/{jid}"
    return f"{run_url}/job/{cands[0][0]}" if cands else run_url


def read_versions(root: Path) -> tuple[str, str]:
    versions = find_file(root, "versions.txt")
    if versions is not None:
        kv = dict(
            line.split("=", 1) for line in read_lines(versions) if "=" in line
        )
        return kv.get("torch", ""), kv.get("torch_xpu_ops", "")
    # Runs predating the linux-testenv change: collect_env carries an
    # abbreviated torch sha and no torch-xpu-ops sha at all.
    for line in read_lines(find_file(root, "collect_env.log")):
        m = re.match(r"PyTorch version:.*\+git([0-9a-f]{7,40})", line)
        if m:
            return m.group(1), ""
    return "", ""


def read_collect_env(root: Path) -> str:
    path = find_file(root, "collect_env.log")
    if path is None:
        return "collect_env output was not captured in this run's artifact."
    return path.read_text(encoding="utf-8", errors="replace").strip()


def extract_tracebacks(root: Path, wanted: set[tuple[str, str]]) -> dict:
    """Full <failure> text for one representative case per group. The Message
    column is only the last exception line; the traceback exists solely in the
    JUnit XML."""
    found: dict[tuple[str, str], str] = {}
    for xml in sorted(root.rglob("*.xml")):
        if not wanted - found.keys():
            break
        try:
            for _, elem in ET.iterparse(str(xml), events=("end",)):
                if elem.tag != "testcase":
                    continue
                key = (elem.get("classname", ""), elem.get("name", ""))
                if key in wanted and key not in found:
                    for child in elem:
                        if child.tag in ("failure", "error"):
                            found[key] = (child.text or child.get("message") or "").strip()
                            break
                elem.clear()
        except ET.ParseError as exc:
            warn(f"could not parse {xml.name}: {exc}")
    return found


# --------------------------------------------------------------------------- #
# Stage 2 - signature and grouping
# --------------------------------------------------------------------------- #


def last_segment(message: str) -> str:
    """check-ut.py joins several `ExceptionType: msg` hits with ' ; '."""
    segments = [s.strip() for s in message.split(" ; ") if s.strip()]
    return segments[-1] if segments else ""


def headline_of(message: str) -> str:
    return " ".join(last_segment(message).split())[:200]


def normalize_error(message: str) -> str:
    """Collapse a failure message to a signature stable across nights.

    Empty messages (segfault, worker crash) share a sentinel, which correctly
    collapses a whole crashed file into one group.
    """
    text = last_segment(message)
    if not text:
        return "CRASH_NO_MESSAGE"
    text = RE_SAMPLE_INPUT.sub("SampleInput(...)", text)
    text = RE_TENSOR_REPR.sub("Tensor[...]", text)
    text = RE_PATH.sub(r"\1", text)
    text = RE_HEX.sub("0xX", text)
    text = RE_LINE_NO.sub("line N", text)
    text = RE_DTYPE_SUFFIX.sub("", text)
    text = RE_NUMBER.sub("N", text)
    return " ".join(text.split())[:200]


def is_infra(normalized: str) -> bool:
    low = normalized.lower()
    return any(p in low for p in INFRA_PATTERNS)


def build_groups(cases: list[Case]) -> list[Group]:
    groups: dict[tuple[str, str], Group] = {}
    for case in cases:
        key = (normalize_error(case.message), case.test_file)
        group = groups.get(key)
        if group is None:
            group = Group(normalized_error=key[0], test_file=key[1])
            groups[key] = group
        group.cases.append(case)
    # Reach of each signature, which the infra decision below needs and no
    # single group can see. Stage 2 keys on (error, file), so one signature
    # spanning several files is several groups.
    files_per_error: dict[str, set[str]] = {}
    for group in groups.values():
        files_per_error.setdefault(group.normalized_error, set()).add(group.test_file)
    for group in groups.values():
        group.cases.sort(key=lambda c: (c.category, c.class_name, c.test_name))
        # Taken from the same representative the ErrorLog traceback comes from
        # (main() reads cases[0]), so the title and the body describe one case.
        # Members share a normalized error but not a literal message, so reading
        # it before the sort would quote whichever case the CSV happened to list
        # first.
        group.headline = headline_of(group.cases[0].message) or group.normalized_error
        group.signature_files = len(files_per_error[group.normalized_error])
        if (is_infra(group.normalized_error)
                and group.signature_files > INFRA_MAX_FILES_TO_FILE):
            group.quarantine = "infra"
    return sorted(groups.values(), key=lambda g: (-len(g.cases), g.sig))


# --------------------------------------------------------------------------- #
# Stage 3 - per-category baseline
# --------------------------------------------------------------------------- #


def baseline_candidates(run_id: int) -> list[dict]:
    out = run([
        "gh", "run", "list", "--repo", REPO, "--workflow", WORKFLOW,
        "--status", "completed", "--limit", "30",
        "--json", "databaseId,displayTitle,createdAt",
    ])
    pat = re.compile(r"^(Nightly|Weekly) / Build-from-source")
    runs = [r for r in json.loads(out) if pat.match(r.get("displayTitle", ""))]
    runs = [r for r in runs if int(r["databaseId"]) < run_id]
    return sorted(runs, key=lambda r: r["createdAt"], reverse=True)


def read_case_sets(root: Path, category: str) -> tuple[set, set, set]:
    """(passed, failed, all). `all` includes skipped cases, so `all - passed -
    failed` is exactly the set that did not run."""
    passed = set(read_lines(find_file(root, f"passed_{category}.log")))
    failed = set(read_lines(find_file(root, f"failures_{category}.log")))
    every = set(read_lines(find_file(root, f"all_cases_{category}.log")))
    return passed, failed, every | passed | failed


def module_counts(lines: set[str]) -> dict[str, int]:
    """`category,class_name,test_name` lines, counted per test module."""
    counts: dict[str, int] = {}
    for line in lines:
        parts = line.split(",")
        if len(parts) < 3:
            continue
        module = Case(parts[0], parts[1], ",".join(parts[2:]), "").module
        if module:
            counts[module] = counts.get(module, 0) + 1
    return counts


def resolve_baselines(run_id: int, categories: set[str], work: Path,
                      report: dict) -> dict[str, Baseline]:
    """One pass over candidates, accumulating per category.

    There is no such thing as "the last good nightly" - only "the last nightly
    in which this category completed healthily". A run truncated in op_ut is
    still a perfectly good op_extended baseline, and one download of a
    candidate's `basic` artifact can resolve up to three categories at once.
    """
    pending = set(categories)
    baselines: dict[str, Baseline] = {}
    for age, cand in enumerate(baseline_candidates(run_id)[:MAX_BASELINE_LOOKBACK], 1):
        if not pending:
            break
        cand_id = int(cand["databaseId"])
        names = list_artifacts(cand_id)
        jobs = resolve_jobs(cand_id)
        dirs: dict[str, Path] = {}
        for leg in {CATEGORY_LEG[c] for c in pending}:
            artifact = pick_artifact(names, "Inductor-XPU-UT-Data", leg, cand_id)
            dest = work / f"baseline-{cand_id}-{leg}"
            if artifact and download(cand_id, artifact, dest):
                dirs[leg] = dest
        for category in sorted(pending):
            root = dirs.get(CATEGORY_LEG[category])
            if root is None:
                continue
            state, actual, expected = category_state(root, category)
            report["baseline_walk"].append({
                "run_id": cand_id, "category": category,
                "state": state, "actual": actual, "expected": expected,
            })
            if state != "complete":
                continue
            leg = CATEGORY_LEG[category]
            passed, failed, every = read_case_sets(root, category)
            torch, tpo = read_versions(root)
            baselines[category] = Baseline(
                run_id=cand_id,
                created_at=cand["createdAt"][:10],
                age_in_runs=age,
                leg=leg,
                job_url=job_url(cand_id, jobs, leg),
                torch=torch,
                torch_xpu_ops=tpo,
                passed=passed,
                failed=failed,
                all_cases=every,
                passed_by_module=module_counts(passed),
                all_by_module=module_counts(every),
            )
            pending.discard(category)
        for path in dirs.values():
            shutil.rmtree(path, ignore_errors=True)
    for category in sorted(pending):
        warn(
            f"no nightly in the last {MAX_BASELINE_LOOKBACK} runs completed "
            f"{category} healthily; its issues will be filed unclassified"
        )
    return baselines


# --------------------------------------------------------------------------- #
# Stage 4 - classify
# --------------------------------------------------------------------------- #


def classify_case(case: Case, baselines: dict[str, Baseline]) -> str:
    """Per case, against its own category's baseline. Exact set membership.

    A whole-module row is compared at module granularity instead, because
    exact membership cannot see it: pytest only emits the row when collection
    fails, so a healthy baseline recorded the module's individual cases and
    never the module itself, and the row would fall through to CLS_NEW_CASE for
    a file of any age. One level up the question is the same one - did this
    used to work - and the answer is exact, because the baseline's per-module
    index is built from the same case sets.

    Module granularity stays confined to these rows. Widening it to real cases
    would be wrong in the other direction: a dtype parametrization added
    upstream is a new case even though its module is years old.
    """
    baseline = baselines.get(case.category)
    if baseline is None:
        return CLS_UNKNOWN
    if case.is_collection_error:
        if baseline.passed_by_module.get(case.module):
            return CLS_REGRESSION
        if case.module in baseline.all_by_module:
            return CLS_PERSISTENT
        return CLS_NEW_CASE
    if case.line in baseline.passed:
        return CLS_REGRESSION
    if case.line in baseline.failed:
        return CLS_PERSISTENT
    # Either absent from the baseline or present but skipped: in both
    # readings the case has never been observed working here.
    return CLS_NEW_CASE


def new_case_reason(case: Case, baseline: Baseline | None) -> str:
    if baseline is None or case.line not in baseline.all_cases:
        return "absent"
    return "skipped"


# --------------------------------------------------------------------------- #
# Stage 4b - what stopped running
#
# A module that fails to import does not fail its cases, it erases them: they
# reach neither passed_<cat>.log nor failures_<cat>.log, so they never enter
# new_ut_failure_list.csv, and a few hundred missing cases sit far below the 5%
# count gate in ut_result_check.sh:check_test_cases. Comparing module coverage
# against the baseline is the only thing here that sees them.
#
# The same per-module index is what classifies a collection error in Stage 4:
# the module row is in neither the baseline's passed nor its failed set, but
# the module is in passed_by_module, so "did this used to work" is answerable
# exactly, one level up from the case.
#
# Nothing in this stage mutes on its own. What it produces is the blast radius
# - how many cases the file used to pass - which Stage 6 renders into the
# issue. The issue itself does mute, like any other: it carries the whole-
# module row, so the row stops being a new failure on the next run and the job
# goes green with the file still dark. That trade is deliberate; leaving it red
# forever ends with nobody reading the nightly at all. The count is what keeps
# the muted state honest, so it belongs in the issue body and not only a log.
# --------------------------------------------------------------------------- #


def collection_error_context(group: Group,
                             baselines: dict[str, Baseline]) -> list[dict]:
    """Per module in a collection-error group, what it used to run."""
    context = []
    for case in group.cases:
        if not case.is_collection_error:
            continue
        base = baselines.get(case.category)
        if base is None:
            state, passed = "no baseline", 0
        elif base.passed_by_module.get(case.module):
            state, passed = "was passing", base.passed_by_module[case.module]
        elif case.module in base.all_by_module:
            state, passed = "known, none passing", 0
        else:
            state, passed = "new test file", 0
        context.append({
            "category": case.category,
            "module": case.module,
            "state": state,
            "baseline_passed": passed,
            "baseline_run": base.run_id if base else None,
        })
    return context


def record_quarantined(groups: list[Group], report: dict) -> None:
    """Record every group, but warn once per signature.

    The finding is that one error reached many files, which is a statement
    about the set. A warning per group states it once per file and buries the
    count that is the whole reason the groups were held back.
    """
    for group in groups:
        report["quarantined"].append({
            "sig": group.sig, "cases": len(group.cases),
            "test_file": group.test_file, "error": group.headline,
            "reason": group.quarantine, "test_files": group.signature_files,
        })
    for family in error_families(groups).values():
        cases = sum(len(g.cases) for g in family)
        warn(
            f"possible infra: {cases} case(s) across "
            f"{family[0].signature_files} test files failed with "
            f"\"{family[0].headline[:80]}\". Past {INFRA_MAX_FILES_TO_FILE} "
            "files this is more likely the runner than the tests, so no issue "
            "was filed and nothing was muted. If the machine was healthy these "
            "are real failures and need filing by hand."
        )


def record_vanished_modules(work: Path, categories: set[str],
                            baselines: dict[str, Baseline],
                            report: dict) -> None:
    """Modules that produced cases in the baseline and none at all in this run.

    Independent of the collection-error rows above, so it also catches a file
    that stops producing cases without reporting an error. The likeliest way
    for that to happen here is the skip list: xpu_test_utils.py:launch_test
    turns it into `pytest -k "not ..."`, and deselected cases are absent from
    the JUnit XML entirely rather than recorded as skipped, so a pattern that
    happens to match a whole file empties it silently.
    """
    for category in sorted(categories):
        base = baselines.get(category)
        if base is None:
            continue
        root = work / f"current-{CATEGORY_LEG[category]}"
        if not root.is_dir():
            continue
        _, _, every = read_case_sets(root, category)
        tonight = module_counts(every)
        gone = [
            (module, count)
            for module, count in base.passed_by_module.items()
            if module not in tonight
        ]
        for module, count in sorted(gone, key=lambda kv: (-kv[1], kv[0])):
            report["vanished_modules"].append({
                "category": category, "module": module,
                "baseline_passed": count, "baseline_run": base.run_id,
            })
    if report["vanished_modules"]:
        total = sum(v["baseline_passed"] for v in report["vanished_modules"])
        warn(
            f"{len(report['vanished_modules'])} module(s) produced no cases in "
            f"this run but had {total} passing case(s) in their baseline; see "
            "the report artifact. Reported only - nothing filed, nothing muted."
        )


# --------------------------------------------------------------------------- #
# Stage 6 - render
# --------------------------------------------------------------------------- #


def load_template() -> str:
    text = TEMPLATE_PATH.read_text(encoding="utf-8")
    if text.startswith("<!--"):
        text = text.split("-->", 1)[1].lstrip("\n")
    return text


def detected_in(sub: SubGroup, current: RunInfo) -> str:
    if len(sub.legs) == 1:
        leg = sub.legs[0]
        link = f"[nightly #{current.run_id} / {leg}]({current.job_urls[leg]})"
    else:
        links = ", ".join(f"[{leg}]({current.job_urls[leg]})" for leg in sub.legs)
        link = f"nightly #{current.run_id} - {links}"
    return f"Detected in : {link} ({current.created_at})"


def version_block(sub: SubGroup, current: RunInfo,
                  baselines: dict[str, Baseline]) -> str:
    lines = [detected_in(sub, current)]
    # "latest good" is only meaningful for a regression; for the other states
    # the baseline run is not a last-known-good for these cases.
    if sub.cls == CLS_REGRESSION:
        resolved = [baselines[c] for c in sub.categories if c in baselines]
        if len(sub.categories) == 1 and resolved:
            lines.append(f"latest good : {resolved[0].torch or 'unknown'}")
        elif resolved:
            # Per-category baselines mean more than one last-good sha; picking
            # one would be arbitrary, so the table below is authoritative.
            lines.append("latest good : see table below")
    lines.append(f"current     : {current.torch.get(sub.legs[0], '') or 'unknown'}")
    return "  \n".join(lines)


def commit_link(repo: str, sha: str) -> str:
    return f"[`{sha[:8]}`]({SERVER}/{repo}/commit/{sha})" if sha else "unknown"


def regression_evidence(sub: SubGroup, current: RunInfo,
                        baselines: dict[str, Baseline]) -> str:
    """Show the work behind the "it used to pass" claim.

    The claim is exact here: every case in this issue is in its baseline's
    passed set, by construction of classify_case. A module row classified as a
    regression passed at module granularity instead and would make this block
    false, which is why evidence_block routes those elsewhere.
    """
    rows = [
        "**Regression evidence**",
        "",
        "These cases passed in the previous healthy nightly for their category "
        "and fail now.",
        "",
        "| Category | | Run | Date | torch | torch-xpu-ops |",
        "|---|---|---|---|---|---|",
    ]
    compares, notes = [], []
    for category in sub.categories:
        base = baselines[category]
        leg = CATEGORY_LEG[category]
        rows.append(
            f"| {category} | Last good | [#{base.run_id} ({base.leg})]({base.job_url}) "
            f"| {base.created_at} | {commit_link(PYTORCH_REPO, base.torch)} "
            f"| {commit_link(REPO, base.torch_xpu_ops)} |"
        )
        rows.append(
            f"| {category} | First seen bad | "
            f"[#{current.run_id} ({leg})]({current.job_urls[leg]}) "
            f"| {current.created_at} "
            f"| {commit_link(PYTORCH_REPO, current.torch.get(leg, ''))} "
            f"| {commit_link(REPO, current.torch_xpu_ops.get(leg, ''))} |"
        )
        if base.torch and current.torch.get(leg):
            compares.append(
                f"Changes in range ({category}): "
                f"[pytorch]({SERVER}/{PYTORCH_REPO}/compare/{base.torch}...{current.torch[leg]})"
                + (
                    f" - [torch-xpu-ops]({SERVER}/{REPO}/compare/"
                    f"{base.torch_xpu_ops}...{current.torch_xpu_ops.get(leg, '')})"
                    if base.torch_xpu_ops and current.torch_xpu_ops.get(leg)
                    else ""
                )
            )
        # A stale baseline keeps "regression" true but makes the range much
        # weaker evidence, so say so rather than presenting a 5-night range in
        # the same shape as a 1-night one.
        if base.age_in_runs > 1:
            gap = base.age_in_runs - 1
            notes.append(
                f"Note: the last healthy {category} nightly was "
                f"{base.age_in_runs} runs back ({gap} intervening "
                f"{'nightly' if gap == 1 else 'nightlies'} did not complete this "
                "category), so this range is wider than one night and the failure "
                "may predate the first-seen-bad run."
            )
    # Emitted after the table: a blank line between rows would terminate it and
    # render the remaining rows as literal text.
    for block in (compares, notes):
        for entry in block:
            rows.extend(("", entry))
    rows.append("")
    rows.append("This is a bisect *range*, not a culprit commit.")
    return "\n".join(rows)


def new_case_evidence(sub: SubGroup, baselines: dict[str, Baseline]) -> str:
    """Distinguish "upstream added it" from "the skip list changed"."""
    lines = ["**Not previously observed passing**", ""]
    for category in sub.categories:
        base = baselines[category]
        cases = [c for c in sub.cases if c.category == category]
        absent = sum(1 for c in cases if new_case_reason(c, base) == "absent")
        skipped = len(cases) - absent
        ref = f"[#{base.run_id}]({base.job_url}) ({base.created_at})"
        if absent:
            lines.append(
                f"- `{category}`: {absent} case(s) did not exist in the last "
                f"healthy nightly {ref}; newly added upstream."
            )
        if skipped:
            lines.append(
                f"- `{category}`: {skipped} case(s) existed but were "
                f"skipped in {ref}; they now run and fail."
            )
    return "\n".join(lines)


def persistent_evidence(sub: SubGroup, baselines: dict[str, Baseline]) -> str:
    """Neither a regression nor a new case: it was already failing."""
    lines = ["**Onset not determined**", ""]
    for category in sub.categories:
        base = baselines[category]
        lines.append(
            f"- `{category}`: these cases also failed in the last healthy "
            f"nightly [#{base.run_id}]({base.job_url}) ({base.created_at}), so "
            "the failure predates that run and its onset was not determined by "
            "this bot. They are neither a regression against that baseline nor "
            "new cases."
        )
    lines.append("")
    lines.append(
        "Reaching this state usually means the cases were never filed - the "
        "group was infra-quarantined, exceeded the per-run issue cap, or predates "
        "this workflow - or that a human reopened them while they were still failing."
    )
    return "\n".join(lines)


def unknown_evidence(sub: SubGroup) -> str:
    cats = ", ".join(f"`{c}`" for c in sub.categories)
    return (
        f"**Baseline unavailable** for {cats} - no nightly in the last "
        f"{MAX_BASELINE_LOOKBACK} runs completed the category healthily, so "
        "regression status could not be determined. Stating the reason matters; "
        "a silently missing block is indistinguishable from a bug."
    )


def collection_error_evidence(sub: SubGroup,
                              baselines: dict[str, Baseline]) -> str:
    """What the muted row stands for, and the work behind its classification.

    Stands in for the per-classification blocks rather than joining them: those
    say "these cases passed in the baseline", which is false of a module row.
    It is the module that used to pass, and the cases it hides that stopped
    running. The count is the only measure of how much, and once the row is
    muted and the job is green this is the only place that says it at all.
    """
    rows, dropped = [], 0
    for entry in collection_error_context(sub.group, baselines):
        rows.append(
            f"| `{entry['module']}` | {entry['category']} | {entry['state']} "
            f"| {entry['baseline_passed']} |"
        )
        dropped += entry["baseline_passed"]
    verdict = {
        CLS_REGRESSION: (
            f"Classified as a **regression**: the module's {dropped} case(s) "
            "passed in the baseline and do not run now. The row itself is in "
            "neither the baseline's passed nor its failed set - a healthy run "
            "records a module's cases, never the module - so the comparison "
            "behind that label is at module granularity."
        ),
        CLS_PERSISTENT: (
            "Classified as **persistent**: the baseline knew this module but "
            "had nothing passing in it, so the breakage predates the baseline."
        ),
        CLS_NEW_CASE: (
            "Classified as a **new test file**: the baseline had never seen "
            "this module, so it has not been observed importing here."
        ),
        CLS_UNKNOWN: (
            "**Baseline unavailable**, so whether this module used to import "
            "could not be determined."
        ),
    }[sub.cls]
    return "\n".join([
        "**Whole-module collection error**",
        "",
        f"`{sub.group.test_file}` failed to import, so none of its cases ran. "
        f"{dropped} case(s) that passed in the baseline are missing from this "
        "run entirely. They are absent rather than failing, so they reach no "
        "failure list and no count that would otherwise notice them.",
        "",
        "| Module | Category | In the baseline | Cases it used to pass |",
        "|---|---|---|---|",
        *rows,
        "",
        verdict,
        "",
        "The skip row above is that module, not a test case. Skipping it stops "
        f"the import error being reported as a new failure; the {dropped} "
        "case(s) stay unrun until the import is fixed.",
    ])


def evidence_block(sub: SubGroup, current: RunInfo,
                   baselines: dict[str, Baseline]) -> str:
    """Everything below `## Pytorch Version`. Exactly one state applies, because
    a sub-group is by construction single-classification."""
    parts = []
    if sub.siblings:
        refs = ", ".join(f"#{n}" for n in sorted(sub.siblings))
        parts.append(
            f"Same root cause as {refs}, split out because those cases have a "
            "different baseline classification."
        )
    if is_infra(sub.group.normalized_error):
        # Reaching here means build_groups declined to call it machine
        # breakage. Say so: an issue filed against a driver-flavoured error
        # otherwise reads as the bot having missed one.
        reach = sub.group.signature_files
        where = (f"only `{sub.group.test_file}`" if reach == 1
                 else f"{reach} test files, within the "
                      f"{INFRA_MAX_FILES_TO_FILE} it may reach and still be "
                      "read as a bug in the tests")
        parts.append(
            "This error matches the infra denylist, but in this run it "
            f"reached {where}. A runner losing its GPU or its disk does not "
            "stop at a handful of files, so it is filed as a bug in the test - "
            "most often allocating too much, or hanging the device - rather "
            "than treated as machine breakage. If the runner was at fault, "
            "closing this returns the case to the next run's new failures."
        )
    if sub.group.collection_error:
        # Routed on the row's shape, not on its classification: a module row
        # can be classified any of the four ways, and none of the four blocks
        # describes one correctly.
        parts.append(collection_error_evidence(sub, baselines))
        return "\n" + "\n\n".join(p.rstrip() for p in parts) + "\n\n"
    renderer = {
        CLS_REGRESSION: lambda: regression_evidence(sub, current, baselines),
        CLS_NEW_CASE: lambda: new_case_evidence(sub, baselines),
        CLS_PERSISTENT: lambda: persistent_evidence(sub, baselines),
        CLS_UNKNOWN: lambda: unknown_evidence(sub),
    }[sub.cls]
    parts.append(renderer())
    return "\n" + "\n\n".join(p.rstrip() for p in parts) + "\n\n"


def render_body(sub: SubGroup, chunk: list[Case], part: int, parts_total: int,
                current: RunInfo, baselines: dict[str, Baseline],
                traceback: str, first_part_url: str | None) -> str:
    marker = (
        f"ut-auto-issue:{MARKER_VERSION}:sig={sub.group.sig}:cls={sub.cls}"
        f":leg={sub.legs[0]}:part={part}/{parts_total}"
    )
    if part == 1:
        error_log = (
            f"## ErrorLog\n\n### {sub.group.headline}\n\n"
            f"```\n{traceback or 'No traceback captured in the JUnit XML.'}\n```\n\n"
        )
    else:
        back = f"[part 1]({first_part_url})" if first_part_url else "part 1"
        error_log = f"Part {part} of {parts_total}. See {back} for the error log.\n\n"
    return load_template().format(
        cases="\n".join(case.line for case in chunk),
        error_log=error_log,
        version_block=version_block(sub, current, baselines),
        evidence_block=evidence_block(sub, current, baselines),
        collect_env=current.collect_env.get(sub.legs[0], ""),
        marker=marker,
    )


TITLE_PREFIX = {CLS_REGRESSION: "[Regression] ", CLS_NEW_CASE: "[New Case] "}
# Says up front that the skip row is a file that would not import rather than a
# failing test. Independent of the classification, which a module row shares
# with ordinary cases, because the two need completely different triage.
COLLECTION_ERROR_PREFIX = "[Failed to collect] "


def render_title(sub: SubGroup, part: int, parts_total: int) -> str:
    suffix = f" (part {part}/{parts_total})" if parts_total > 1 else ""
    prefix = TITLE_PREFIX.get(sub.cls, "")
    if sub.group.collection_error:
        # The subject is the file, not one of its cases, and an import error
        # usually opens with an absolute build path long enough to push the
        # file name past the truncation if it went last.
        return (f"[Bug Skip]: {COLLECTION_ERROR_PREFIX}{prefix}"
                f"{sub.group.test_file}: {sub.group.headline[:100]}{suffix}")
    return (
        f"[Bug Skip]: {prefix}"
        f"{sub.group.headline[:120]} in {sub.group.test_file}{suffix}"
    )


def labels_for(sub: SubGroup) -> list[str]:
    labels = ["skipped"]
    if any(leg in BMG_LEGS for leg in sub.legs):
        labels.append("skipped_bmg")
    if sub.cls in CLS_LABELS:
        labels.append(sub.cls)
    return labels


# --------------------------------------------------------------------------- #
# Stage 7 - size handling
# --------------------------------------------------------------------------- #


def chunk_cases(cases: list[Case]) -> list[list[Case]]:
    """The Cases: block is never truncated - fetch_issues.sh needs it complete
    for the skip to take effect - so an oversized group splits into parts."""
    return [
        cases[i:i + MAX_CASES_PER_ISSUE]
        for i in range(0, len(cases), MAX_CASES_PER_ISSUE)
    ] or [[]]


def error_families(groups: list[Group]) -> dict[str, list[Group]]:
    """Groups sharing a normalized error, keyed by it.

    Stage 2 keys on (normalized_error, test_file) so that a generic message -
    a precision mismatch, or the CRASH_NO_MESSAGE sentinel - cannot collapse
    unrelated files into a single mute. The cost is that one root cause hitting
    several files is several groups, and nothing downstream would say so.
    """
    families: dict[str, list[Group]] = {}
    for group in groups:
        families.setdefault(group.normalized_error, []).append(group)
    return families


def apply_issue_budget(groups: list[Group]) -> tuple[list[Group], list[Group]]:
    """Split into (filed, overflow) under MAX_ISSUES_PER_RUN, family-atomically.

    `groups` is ordered largest-first, so a plain head/tail cut would file the
    big test files of a root cause and defer its small ones - muting half a bug
    and leaving the other half failing every night. Families are therefore
    admitted whole or not at all.

    A family that alone exceeds the cap is deferred entirely rather than
    truncated, and the walk continues: that burst is precisely what the cap
    exists to catch, and skipping it should not also cost the smaller,
    ordinary-looking failures their chance to be filed.
    """
    families = error_families(groups)
    filed: list[Group] = []
    overflow: list[Group] = []
    budget = MAX_ISSUES_PER_RUN
    for group in groups:
        family = families[group.normalized_error]
        if group is not family[0]:
            continue  # already decided with the rest of its family
        if len(family) <= budget:
            filed.extend(family)
            budget -= len(family)
        else:
            overflow.extend(family)
    return filed, overflow


# --------------------------------------------------------------------------- #
# Stage 5 / 8 - dedup, create, comment
# --------------------------------------------------------------------------- #


def open_bot_issues() -> dict[str, list[IssueRef]]:
    """Our own open issues, indexed by root-cause signature.

    Keyed on the stored hash rather than the title, so dedup survives a human
    editing the title and survives a change to the normalization rules (old
    issues keep matching their old hash; a rules change is then a deliberate,
    visible re-file rather than a silent duplicate).
    """
    seen: dict[int, str] = {}
    for label in ("skipped", "skipped_bmg", "new_case_failure", "regression"):
        rows = gh_tsv(
            f"repos/{REPO}/issues?state=open&labels={label}&per_page=100",
            ".[] | select(.pull_request == null) "
            '| [(.number|tostring), (.body // "" | @base64)] | @tsv',
        )
        for row in rows:
            if len(row) >= 2 and int(row[0]) not in seen:
                seen[int(row[0])] = base64.b64decode(row[1]).decode(
                    "utf-8", errors="replace"
                )
    index: dict[str, list[IssueRef]] = {}
    for number, body in seen.items():
        m = MARKER_RE.search(body)
        if not m or m.group("ver") != MARKER_VERSION:
            continue
        index.setdefault(m.group("sig"), []).append(IssueRef(
            number=number,
            body=body,
            cls=m.group("cls"),
            part=int(m.group("part")),
            parts=int(m.group("parts")),
            case_lines=parse_cases_block(body),
        ))
    for refs in index.values():
        refs.sort(key=lambda r: (r.cls, r.part))
    return index


def append_cases(body: str, new_lines: list[str]) -> str:
    """Append inside the cases:begin/end bounds only, so the version block and
    anything a human added stay untouched."""
    start = body.find(CASES_BEGIN)
    end = body.find(CASES_END)
    if start == -1 or end == -1 or end < start:
        return body
    adding = set(new_lines)
    kept, seen = [], set()
    for raw in body[start + len(CASES_BEGIN):end].splitlines():
        line = raw.strip()
        if not line:
            continue
        # Drop the struck-through twin of a case we are re-adding: it no longer
        # mutes anything and keeping both is just noise.
        if line.startswith("~~") and line.endswith("~~") and line[2:-2].strip() in adding:
            continue
        kept.append(line)
        seen.add(line)
    kept += [line for line in new_lines if line not in seen]
    return body[:start + len(CASES_BEGIN)] + "\n" + "\n".join(kept) + "\n" + body[end:]


def parse_cases_block(body: str) -> set[str]:
    """The lines that actually mute. `mark_passed_issue` rewrites a line to
    `~~<line>~~` once the case passes, which stops it muting; such a line is
    kept as history but no longer claims the case."""
    start = body.find(CASES_BEGIN)
    end = body.find(CASES_END)
    if start == -1 or end == -1 or end < start:
        return set()
    live = set()
    for raw in body[start + len(CASES_BEGIN):end].splitlines():
        line = raw.strip()
        if not line or line == "Cases:":
            continue
        if not (line.startswith("~~") and line.endswith("~~")):
            live.add(line)
    return live


def ensure_labels(labels: set[str]) -> None:
    out = run(["gh", "label", "list", "--repo", REPO, "--limit", "200",
               "--json", "name", "-q", ".[].name"], check=False)
    have = set(out.splitlines())
    missing = labels - have
    if missing:
        raise SystemExit(
            f"::error::labels do not exist in {REPO}: {', '.join(sorted(missing))}. "
            "Create them before enabling UT_AUTO_ISSUE_ENABLED."
        )


def create_issue(title: str, body: str, labels: list[str]) -> str:
    cmd = ["gh", "issue", "create", "--repo", REPO, "--title", title,
           "--body-file", "-"]
    for label in labels:
        cmd += ["--label", label]
    proc = subprocess.run(cmd, input=body, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"gh issue create failed: {proc.stderr.strip()}")
    out = proc.stdout.strip().splitlines()
    return out[-1] if out else ""


def still_failing_comment(group: Group, current: RunInfo, total: int,
                          appended: int) -> str:
    leg = group.legs[0]
    note = (
        f" {appended} of them were not in the skip list and have been added."
        if appended else ""
    )
    return (
        f"Still failing in [nightly #{current.run_id} / {leg}]"
        f"({current.job_urls[leg]}). "
        f"torch {current.torch.get(leg, 'unknown')}, "
        f"torch-xpu-ops {current.torch_xpu_ops.get(leg, 'unknown')}. "
        f"{total} cases.{note}"
    )


def cross_file_note(members: list[FamilyMember]) -> str:
    """Link the issues one error signature produced across several test files.

    Posted as a comment rather than rendered into the body for the same reason
    the classification-split note is: most of these issues do not exist yet when
    the earlier ones are rendered.
    """
    lines = [
        f"Same error signature in {len(members)} test files. Grouping is per test "
        "file so that a generic message cannot mute unrelated files, but these are "
        "usually one root cause:",
        "",
    ]
    for member in sorted(members, key=lambda m: (-m.cases, m.test_file)):
        links = ", ".join(member.urls)
        lines.append(f"- `{member.test_file}` ({member.cases} cases): {links}")
    return "\n".join(lines)


def comment_issue(number: int, body: str) -> None:
    subprocess.run(
        ["gh", "issue", "comment", str(number), "--repo", REPO, "--body-file", "-"],
        input=body, capture_output=True, text=True, check=True,
    )


def edit_body(number: int, body: str) -> None:
    subprocess.run(
        ["gh", "issue", "edit", str(number), "--repo", REPO, "--body-file", "-"],
        input=body, capture_output=True, text=True, check=True,
    )


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def collect_leg(run_id: int, leg: str, names: list[tuple[str, bool]], work: Path,
                jobs: list, report: dict, current: RunInfo) -> list[Case]:
    """Stage 0 H3-H7 plus Stage 1 for one leg."""
    data_artifact = pick_artifact(names, "Inductor-XPU-UT-Data", leg, run_id)
    if data_artifact is None:
        report["skipped_legs"].append({"leg": leg, "reason": "no UT data artifact (H3)"})
        warn(f"{leg}: no usable Inductor-XPU-UT-Data artifact; filing nothing")
        return []
    root = work / f"current-{leg}"
    if not download(run_id, data_artifact, root):
        report["skipped_legs"].append({"leg": leg, "reason": "artifact download failed (H3)"})
        return []

    current.job_urls[leg] = job_url(run_id, jobs, leg)
    torch, tpo = read_versions(root)
    current.torch[leg] = torch
    current.torch_xpu_ops[leg] = tpo
    current.collect_env[leg] = read_collect_env(root)

    healthy_categories = set()
    for category in LEG_CATEGORIES[leg]:
        state, actual, expected = category_state(root, category)
        report["categories"].append({
            "category": category, "state": state,
            "actual": actual, "expected": expected,
        })
        if state == "complete":
            healthy_categories.add(category)
        elif state == "truncated":
            warn(
                f"{category}: {actual}/{expected} cases, below the "
                f"{int(HEALTH_RATIO * 100)}% threshold (H5). The failures may be "
                "real but the machine is suspect; filing nothing for it."
            )
        else:
            print(f"note: {category} never ran in this leg; nothing to file")

    failures_artifact = pick_artifact(names, "New-UT-Failures", leg, run_id)
    if failures_artifact is None:
        print(f"note: {leg} produced no new failures")
        return []
    csv_dir = work / f"current-{leg}-newfail"
    if not download(run_id, failures_artifact, csv_dir):
        return []
    cases = parse_failure_csv(find_file(csv_dir, "new_ut_failure_list.csv"))

    # H6: the CSV is built by grepping ut_failure_list.csv per filtered-log line,
    # so a mismatch means some failures lost their error message.
    expected_rows = len(read_lines(find_file(root, "new_failure_list.txt")))
    if expected_rows and expected_rows != len(cases):
        warn(f"{leg}: new failure count mismatch: filtered={expected_rows}, csv={len(cases)} (H6)")

    kept = [c for c in cases if c.category in healthy_categories]
    dropped = len(cases) - len(kept)
    if dropped:
        print(f"note: dropped {dropped} {leg} cases from unhealthy categories")

    # H7: a leg where infra signatures dominate is infra breakage, not a set of
    # product bugs, so none of it is filed. build_groups already quarantines a
    # signature that reached several files; this gate is the wider reading, and
    # distrusts the cases around them too, which takes a sample big enough for
    # a share to mean anything.
    infra = {c for c in kept if is_infra(normalize_error(c.message))}
    if len(kept) >= INFRA_MIN_CASES and len(infra) / len(kept) > INFRA_SIGNATURE_RATIO:
        warn(
            f"{leg}: {len(infra)}/{len(kept)} new failures match the infra "
            f"denylist (> {INFRA_SIGNATURE_RATIO:.0%}); treating the whole leg "
            "as infra breakage and filing nothing (H7)"
        )
        # Recorded case by case: H7 is the one gate that drops failures without
        # rendering them anywhere, so without this the report cannot say what
        # was discarded.
        report["skipped_legs"].append({
            "leg": leg,
            "reason": "infra signature ratio (H7)",
            "dropped": [
                {"case": c.line, "infra": c in infra, "error": c.message[:200]}
                for c in kept
            ],
        })
        return []
    if infra and len(kept) < INFRA_MIN_CASES:
        print(
            f"note: {leg} has {len(infra)}/{len(kept)} infra-looking new "
            f"failures, below the {INFRA_MIN_CASES}-case floor for H7; they are "
            "still quarantined individually"
        )
    return kept


def build_umbrella(overflow: list[Group], current: RunInfo) -> tuple[str, str, str]:
    sig = hashlib.sha256(
        "\n".join(sorted(g.sig for g in overflow)).encode()
    ).hexdigest()[:16]
    distinct = {g.normalized_error for g in overflow}
    rows = [
        f"| {len(g.cases)} | `{g.test_file}` | {g.headline[:120]} |" for g in overflow
    ]
    # Calling these N independent bugs when they share a handful of signatures
    # sends the reader looking for the wrong thing, and is the likeliest reading
    # to be wrong: one root cause across many files is exactly how a group count
    # gets this high without anything being broken at the infra level.
    if len(distinct) < len(overflow):
        diagnosis = (
            f"These {len(overflow)} groups carry only {len(distinct)} distinct error "
            f"signature{'' if len(distinct) == 1 else 's'}. Grouping is per test file, "
            "so one root cause spread across several files arrives here as several "
            "groups - triage the signatures, not the groups."
        )
    else:
        diagnosis = (
            "A burst this wide, with no signature shared between groups, usually "
            f"means infra or a broken build rather than {len(overflow)} independent "
            "bugs. Triage by hand before filing."
        )
    body = "\n".join([
        f"Nightly [#{current.run_id}]({SERVER}/{REPO}/actions/runs/{current.run_id}) "
        f"({current.created_at}) produced more new-failure root causes than the "
        f"per-run cap of {MAX_ISSUES_PER_RUN} allows to be filed. Groups sharing one "
        "error signature are filed or deferred together, never split. These were "
        "deferred, and are **not** muted:",
        "",
        "| Count | Test file | Error |",
        "|---|---|---|",
        *rows,
        "",
        diagnosis,
        "",
        f"<!-- ut-auto-issue:{MARKER_VERSION}:sig={sig}:leg=umbrella:part=1/1 -->",
    ])
    title = (
        f"[Bug Skip]: {len(overflow)} further new-failure groups not filed "
        f"from nightly #{current.run_id}"
    )
    return sig, title, body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, type=int)
    parser.add_argument("--test-type", default="")
    parser.add_argument("--work-dir", default="ut_auto_issue_work")
    parser.add_argument("--report-dir", default="ut_auto_issue_report")
    # Off by default: filing an issue mutes a test, so a bare invocation must
    # only ever report.
    parser.add_argument("--create-issues", action="store_true")
    args = parser.parse_args()

    work = Path(args.work_dir)
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    run_meta = gh_json(f"repos/{REPO}/actions/runs/{args.run_id}")
    current = RunInfo(
        run_id=args.run_id,
        created_at=run_meta.get("created_at", "")[:10],
        job_urls={}, torch={}, torch_xpu_ops={}, collect_env={},
    )
    report = {
        "run_id": args.run_id,
        "test_type": args.test_type,
        "create_issues": args.create_issues,
        "categories": [],
        "skipped_legs": [],
        "quarantined": [],
        "vanished_modules": [],
        "baseline_walk": [],
        "actions": [],
    }

    jobs = resolve_jobs(args.run_id)
    # H1: if the build failed nothing downstream can be trusted.
    build_jobs = [j for j in jobs if j[1].startswith("linux-build")]
    if any(j[2] in ("failure", "cancelled") for j in build_jobs):
        warn("build job did not succeed (H1); filing nothing for this run")
        report["skipped_legs"].append({"leg": "*", "reason": "build not successful (H1)"})
        return finish(report, report_dir)

    names = list_artifacts(args.run_id)
    cases: list[Case] = []
    for leg in LEG_CATEGORIES:
        cases.extend(collect_leg(args.run_id, leg, names, work, jobs, report, current))

    if len(cases) > ABORT_THRESHOLD:
        print(
            f"::error::{len(cases)} new failures exceeds ABORT_THRESHOLD "
            f"({ABORT_THRESHOLD}); assuming infra breakage and creating nothing"
        )
        report["skipped_legs"].append({"leg": "*", "reason": "abort threshold"})
        return finish(report, report_dir)

    groups = build_groups(cases)
    filed, quarantined = [], []
    for group in groups:
        (quarantined if group.quarantine else filed).append(group)

    filed, overflow = apply_issue_budget(filed)
    if overflow:
        warn(
            f"{len(overflow)} groups did not fit MAX_ISSUES_PER_RUN "
            f"({MAX_ISSUES_PER_RUN}) and were not filed; see the umbrella issue"
        )

    # Stage 4b runs before the nothing-to-file exit and needs a baseline for
    # every healthy category, not just the ones with something to file: a
    # category whose only symptom is that a file stopped producing cases
    # reports no failure at all, so a night that is otherwise green is exactly
    # the night this check has to survive to.
    healthy = {c["category"] for c in report["categories"] if c["state"] == "complete"}
    needed = {c.category for g in filed for c in g.cases}
    needed |= {c.category for g in quarantined for c in g.cases} | healthy
    baselines = resolve_baselines(args.run_id, needed, work, report)

    record_quarantined(quarantined, report)
    record_vanished_modules(work, healthy, baselines, report)

    if not filed and not overflow:
        print("Nothing to file.")
        return finish(report, report_dir)

    wanted = {(g.cases[0].class_name, g.cases[0].test_name) for g in filed}
    tracebacks: dict[tuple[str, str], str] = {}
    for leg in {c.leg for g in filed for c in g.cases}:
        root = work / f"current-{leg}"
        if root.is_dir():
            tracebacks.update(extract_tracebacks(root, wanted))

    if args.create_issues:
        ensure_labels({"skipped", "skipped_bmg", "regression", "new_case_failure"})
    # Queried when only reporting too, so the report shows real
    # create/comment decisions.
    index = open_bot_issues()

    families: dict[str, list[FamilyMember]] = {}
    for group in filed:
        key = (group.cases[0].class_name, group.cases[0].test_name)
        traceback = tracebacks.get(key, "")
        siblings = index.get(group.sig, [])

        # Ownership is exactly what is muted. A case an open issue still mutes
        # stays there whatever it would classify as today; re-routing it would
        # strand that issue. A case released - struck through by
        # mark_passed_issue, or deleted by a human - is deliberately
        # re-classified instead: it passed at least once, so the failure that
        # follows may have a different onset than the old issue describes.
        placed = {line for ref in siblings for line in ref.case_lines}
        tonight = {c.line for c in group.cases}

        for ref in siblings:
            overlap = ref.case_lines & tonight
            if not overlap:
                continue
            report["actions"].append({
                "sig": group.sig, "cls": ref.cls, "action": "comment",
                "issue": ref.number, "classification": ref.cls,
                "cases": len(overlap), "appended_cases": 0,
                "title": f"#{ref.number} (part {ref.part}/{ref.parts})",
            })
            if args.create_issues:
                comment_issue(ref.number, still_failing_comment(
                    group, current, len(overlap), 0))

        created: list[tuple[str, str]] = []   # (cls, url) filed for this sig now
        # `created` stays empty when only reporting, so the cross-file gate
        # needs a decision rather than an outcome to report on truthfully.
        will_create = False
        buckets: dict[str, list[Case]] = {}
        for case in group.cases:
            if case.line not in placed:
                buckets.setdefault(classify_case(case, baselines), []).append(case)

        for cls in sorted(buckets):
            sub = SubGroup(group=group, cls=cls, cases=buckets[cls])
            same_cls = [r for r in siblings if r.cls == cls]
            sub.siblings = [r.number for r in siblings if r.cls != cls]

            if same_cls:
                # The series already exists: append to its last part rather
                # than opening a parallel issue for the same classification.
                target = same_cls[-1]
                merged = append_cases(target.body, [c.line for c in sub.cases])
                action = {
                    "sig": group.sig, "cls": cls, "action": "append",
                    "issue": target.number, "classification": cls,
                    "cases": len(sub.cases), "appended_cases": len(sub.cases),
                    "title": f"#{target.number} (part {target.part}/{target.parts})",
                    "body_chars": len(merged),
                }
                if len(merged) > SAFE_BODY_LIMIT:
                    # Refusing to append leaves the case running rather than
                    # silently muted; that is the safe direction to fail.
                    warn(
                        f"#{target.number} would grow to {len(merged)} chars; "
                        f"not appending {len(sub.cases)} cases. They stay unmuted."
                    )
                    action["action"] = "append-refused"
                elif args.create_issues:
                    edit_body(target.number, merged)
                    comment_issue(target.number, still_failing_comment(
                        group, current, len(sub.cases), len(sub.cases)))
                report["actions"].append(action)
                continue

            chunks = chunk_cases(sub.cases)
            first_url = None
            will_create = True
            for part, chunk in enumerate(chunks, 1):
                body = render_body(sub, chunk, part, len(chunks), current,
                                   baselines, traceback, first_url)
                title = render_title(sub, part, len(chunks))
                labels = labels_for(sub)
                action = {
                    "sig": group.sig, "cls": cls, "action": "create",
                    "part": f"{part}/{len(chunks)}", "title": title,
                    "labels": labels, "classification": cls,
                    "cases": len(chunk), "body_chars": len(body),
                }
                if len(body) > GITHUB_BODY_LIMIT:
                    warn(f"body for {group.sig}/{cls} part {part} is {len(body)} chars")
                (report_dir / f"{group.sig}-{cls}-{part}.md").write_text(
                    body, encoding="utf-8")
                if args.create_issues:
                    url = create_issue(title, body, labels)
                    action["url"] = url
                    if part == 1:
                        first_url = url
                        created.append((cls, url))
                report["actions"].append(action)

        # Issues split out of one root cause in the same run cannot reference
        # each other in their bodies, because none of them existed when the
        # earlier ones were rendered. Three near-identical titles with no stated
        # relationship is the worst possible outcome, so link them afterwards.
        # Existing issues are notified too: when a released case forks into a
        # new issue, the one holding its struck-through line has nothing failing
        # tonight and would otherwise never learn the fork happened.
        # Gated on `created` - without it a sig with two standing issues would
        # re-post this every night.
        family = [(r.cls, f"{SERVER}/{REPO}/issues/{r.number}")
                  for r in siblings if r.part == 1] + created
        if created and len(family) > 1 and args.create_issues:
            note = ("One root cause, split by how each case compares with its "
                    "baseline:\n"
                    + "\n".join(f"- `{c}`: {u}" for c, u in sorted(family)))
            for _, url in family:
                comment_issue(int(url.rsplit("/", 1)[-1]), note)

        # The same signature in another test file is a separate group by design
        # (Stage 2). Record what this one produced so the files can be linked
        # once all of them exist.
        if family or will_create:
            families.setdefault(group.normalized_error, []).append(FamilyMember(
                test_file=group.test_file,
                cases=len(group.cases),
                urls=[url for _, url in family],
                created=will_create,
            ))

    # Cross-file linking, for the same reason as the per-classification note
    # above: several issues with near-identical titles and nothing saying they
    # are related is worse than either merging them or filing one.
    for error, member_list in sorted(families.items()):
        if len(member_list) < 2 or not any(m.created for m in member_list):
            continue
        report["actions"].append({
            "action": "cross-file-link", "classification": "",
            "cases": sum(m.cases for m in member_list),
            "title": f"{len(member_list)} test files share `{error[:80]}`",
        })
        if not args.create_issues:
            continue
        note = cross_file_note(member_list)
        for member in member_list:
            for url in member.urls:
                comment_issue(int(url.rsplit("/", 1)[-1]), note)

    if overflow:
        sig, title, body = build_umbrella(overflow, current)
        (report_dir / f"umbrella-{sig}.md").write_text(body, encoding="utf-8")
        # Deliberately unlabelled and not deduped: it mutes nothing, and each
        # night's overflow set is a different statement about a different run.
        action = {"sig": sig, "title": title, "labels": [], "action": "create",
                  "classification": "umbrella",
                  "cases": sum(len(g.cases) for g in overflow)}
        if args.create_issues:
            action["url"] = create_issue(title, body, [])
        report["actions"].append(action)

    return finish(report, report_dir)


def finish(report: dict, report_dir: Path) -> int:
    (report_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    lines = [
        "## UT auto-issue report",
        "",
        f"Run `{report['run_id']}` - "
        f"{'live' if report['create_issues'] else 'report only'}",
        "",
    ]
    if report["categories"]:
        lines += ["| Category | State | Cases | Expected |", "|---|---|---|---|"]
        lines += [
            f"| {c['category']} | {c['state']} | {c['actual']} | {c['expected']} |"
            for c in report["categories"]
        ]
        lines.append("")
    for skipped in report["skipped_legs"]:
        lines.append(f"- Skipped `{skipped['leg']}`: {skipped['reason']}")
        for d in skipped.get("dropped", []):
            mark = "infra" if d["infra"] else "not infra"
            lines.append(f"  - ({mark}) `{d['case']}` - {d['error'][:100]}")
    if report["quarantined"]:
        # One row per signature, for the same reason record_quarantined warns
        # once per signature: the reach is the finding.
        by_error: dict[str, list[dict]] = {}
        for q in report["quarantined"]:
            by_error.setdefault(q["error"], []).append(q)
        lines += [
            "",
            "### Possible infra - reported only, nothing filed, nothing muted",
            "",
            f"One error reaching more than {INFRA_MAX_FILES_TO_FILE} test files "
            "in a single run is read as the runner rather than the tests. If "
            "the machine was healthy these are real failures and need filing "
            "by hand.",
            "",
            "| Cases | Test files | Error |",
            "|---|---|---|",
        ]
        lines += [
            f"| {sum(q['cases'] for q in qs)} | {qs[0].get('test_files', 1)} "
            f"| {error[:100]} |"
            for error, qs in sorted(by_error.items())
        ]
        lines.append("")
    if report["vanished_modules"]:
        lines += [
            "",
            "### Modules that produced no cases in this run",
            "",
            "These passed in their baseline and are absent here, so they did not "
            "fail - they did not run. Nothing below is filed or muted.",
            "",
            "| Category | Module | Passing in baseline | Baseline run |",
            "|---|---|---|---|",
        ]
        lines += [
            f"| {v['category']} | `{v['module']}` | {v['baseline_passed']} "
            f"| {v['baseline_run']} |"
            for v in report["vanished_modules"]
        ]
        lines.append("")
    if report["actions"]:
        lines += ["", "| Action | Classification | Cases | Title |", "|---|---|---|---|"]
        lines += [
            f"| {a['action']} | {a.get('classification', '')} | {a['cases']} "
            f"| {a['title'][:100]} |"
            for a in report["actions"]
        ]
    summary = "\n".join(lines) + "\n"
    print(summary)
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as handle:
            handle.write(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
