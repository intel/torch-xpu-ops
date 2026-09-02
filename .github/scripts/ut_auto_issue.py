#!/usr/bin/env python3
"""Facts and audit for the nightly UT auto-issue pipeline.

Nightly failures are turned into GitHub issues by the ut-issue-authoring skill.
This script does the two parts of that which are not judgement, and neither of
them writes an issue:

  emit-evidence  downloads the run's artifacts and states what happened - which
                 cases failed, how each compares with its category's baseline,
                 which modules stopped producing cases at all, what the
                 tracebacks say, and the markdown for the parts of an issue
                 body that are error-prone to assemble by hand. It groups
                 nothing and decides nothing.

  audit          reads back the `Cases:` block of every open bot issue and
                 reports any line naming a case this run has never heard of.

The audit exists because of how muting works. An issue carrying the `skipped`
label has its case lines subtracted from the next nightly by `grep -vFxf` in
ut_result_check.sh - whole line, fixed string, no tolerance. A line naming a
case that does not exist matches nothing on the day it is written and looks
harmless, stays in the issue indefinitely, and silently swallows a real failure
the first night a test of that name fails. Nothing else in the pipeline would
ever mention it.

Run by hand against a past nightly to see what it collected:

    python .github/scripts/ut_auto_issue.py --run-id <run_id> \
        --evidence-dir ./evidence
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

# Stamped into every filed body so a reader, and the next night's dedup pass,
# can tell a machine-filed issue from a hand-written one.
MARKER_VERSION = "v1"
MARKER_TEMPLATE = "<!-- ut-auto-issue:{version}:run={run_id}:part={part}/{parts} -->"
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

# How far back to look for a category's baseline. Charged per category, and
# only against nightlies that had something to say about it: one whose artifact
# is gone is not evidence that the category was unhealthy, and letting it spend
# the budget is how a category ends up unclassified with no comparison made.
MAX_BASELINE_LOOKBACK = 5
# Hard cap on nightlies walked, so a stretch of artifact-less ones cannot turn
# the walk into an unbounded crawl.
MAX_BASELINE_CANDIDATES = 25
# Pages of 100 workflow runs to scan while collecting those candidates. Most
# runs of this workflow are on-demand, so a page holds far fewer than 100.
MAX_CANDIDATE_PAGES = 5
MAX_CASES_PER_ISSUE = 400
MAX_ISSUES_PER_RUN = 15
ABORT_THRESHOLD = 5000
# Above this many new failures the evidence stops being something a model can
# read in one pass, and a night this red is a question about the machine rather
# than about which bug is which. Grouping falls back to the deterministic rule.
OVERSIZED_THRESHOLD = 1000
# How many distinct (test file, exact message) strata get a traceback captured.
# Sampling, not grouping: two rows with byte-identical messages are one message.
MAX_TRACEBACK_SAMPLES = 300
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
# Headroom below the hard cap, so appending to an issue on a later night has room.
SAFE_BODY_LIMIT = 60000

# Covered legs. xpu_distributed is deliberately excluded: it reports through
# run_distributed_tests in ut_result_check.sh, which produces neither the
# per-category passed/failed logs nor a case count, so neither the health
# gate nor the baseline comparison has anything to read.
LEG_CATEGORIES = {
    "basic": ["op_regression", "op_regression_dev1", "op_extended"],
    "op_ut": ["op_ut"],
}
CATEGORY_LEG = {c: leg for leg, cats in LEG_CATEGORIES.items() for c in cats}

# fetch_issues.sh:25 honours the BMG-only known-failure label only on a runner
# whose name contains `bmg`, so the label has to follow the machine that ran the
# leg rather than the leg itself: nightly_ondemand.yml:166 sends `xpu_distributed`
# to `distributed`, and a leg that lands off BMG must not carry a BMG-only skip.
BMG_LABEL = "skipped_bmg"

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
# so breadth decides - see INFRA_MAX_FILES_TO_FILE.
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
class BaselineMeta:
    """The part of a baseline an issue body quotes.

    Split out from the case sets because rendering never needs those and they
    run to six figures of lines, which is more than is worth carrying between
    the two halves of this script.
    """
    run_id: int
    created_at: str
    age_in_runs: int
    leg: str
    job_url: str
    torch: str
    torch_xpu_ops: str


@dataclass
class Baseline:
    meta: BaselineMeta
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
    runners: dict[str, str] = field(default_factory=dict)  # leg -> runner name


@dataclass
class Evidence:
    """Everything the filing half is allowed to treat as true.

    Self-contained by design: it holds the baseline-derived numbers the issue
    bodies quote rather than the baselines themselves, so the filing half never
    needs to download a past nightly, and a model reading it between the two
    halves sees the same facts the filing half will use.
    """
    run: RunInfo
    cases: list[Case]
    classification: dict[str, str]
    new_case_reason: dict[str, str]
    collection_context: dict[str, dict]
    baselines: dict[str, BaselineMeta]
    tracebacks: dict[str, list[str]]
    reproduce: dict[str, dict]
    leg_health: dict[str, dict]
    gates: dict[str, bool]
    report: dict = field(default_factory=dict)

    @property
    def digest(self) -> str:
        return case_digest(c.line for c in self.cases)


def case_digest(lines) -> str:
    return hashlib.sha256("\n".join(sorted(lines)).encode()).hexdigest()


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
# Health gate
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
#
# H1-H6 are facts about the artifacts and are settled here. H7 is a reading of
# them - a share is only infra breakage if you decide it is - so collection
# records the share and infra_leg_gate in the filing half decides on it. The
# threshold and the outcome are unchanged; only the place moved, so that the
# facts a model sees are not already filtered by one verdict.
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
# This run's new failures
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


def resolve_jobs(run_id: int) -> list[tuple[int, str, str, str]]:
    rows = gh_tsv(
        f"repos/{REPO}/actions/runs/{run_id}/jobs?per_page=100",
        '.jobs[] | [(.id|tostring), .name, (.conclusion // ""), '
        '(.runner_name // "")] | @tsv',
    )
    return [(int(r[0]), r[1], r[2], r[3]) for r in rows if len(r) >= 4]


def leg_jobs(jobs: list[tuple[int, str, str, str]], leg: str) -> list[tuple]:
    """The leg's jobs, the container one first when there is one."""
    cands = [j for j in jobs if f"({leg})" in j[1]]
    return sorted(cands, key=lambda j: (not j[1].endswith("test-in-container"), j[0]))


def job_url(run_id: int, jobs: list[tuple[int, str, str, str]], leg: str) -> str:
    """Job-level link, so the reader lands on the leg's log rather than a matrix
    summary page. Falls back to the run URL."""
    run_url = f"{SERVER}/{REPO}/actions/runs/{run_id}"
    cands = leg_jobs(jobs, leg)
    return f"{run_url}/job/{cands[0][0]}" if cands else run_url


def job_runner(jobs: list[tuple[int, str, str, str]], leg: str) -> str:
    """Which machine ran the leg.

    One error on one box reads differently from the same error on two, and
    nothing else in the artifacts says which box a leg landed on.
    """
    cands = leg_jobs(jobs, leg)
    return cands[0][3] if cands else ""


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


def sample_traceback_targets(cases: list[Case], limit: int) -> list[Case]:
    """One case per (test file, exact message), largest strata first.

    Bounded because the JUnit failure text of a bad night runs to megabytes
    and nothing downstream reads more than a handful of them. Byte-identical
    messages are one message - a statement of fact, not the normalization
    guess that grouping makes, so this samples without deciding anything.
    """
    strata: dict[tuple[str, str], list[Case]] = {}
    for case in cases:
        strata.setdefault((case.test_file, case.message), []).append(case)
    ranked = sorted(strata.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    return [
        sorted(members, key=lambda c: c.line)[0]
        for _, members in ranked[:limit]
    ]


def extract_tracebacks(root: Path, wanted: dict[tuple[str, str], str]) -> dict[str, list[str]]:
    """Full <failure> text, split into lines, keyed by case line.

    The Message column is only the last exception line; the traceback exists
    solely in the JUnit XML. Lines rather than one blob so that a caller can
    point at frames by index instead of restating them.
    """
    found: dict[str, list[str]] = {}
    for xml in sorted(root.rglob("*.xml")):
        if len(found) == len(wanted):
            break
        try:
            for _, elem in ET.iterparse(str(xml), events=("end",)):
                if elem.tag != "testcase":
                    continue
                key = (elem.get("classname", ""), elem.get("name", ""))
                line = wanted.get(key)
                if line is not None and line not in found:
                    for child in elem:
                        if child.tag in ("failure", "error"):
                            text = (child.text or child.get("message") or "").strip()
                            found[line] = text.splitlines()
                            break
                elem.clear()
        except ET.ParseError as exc:
            warn(f"could not parse {xml.name}: {exc}")
    return found


def read_reproduce(root: Path, category: str) -> dict:
    """The `cd` and the pytest invocation linux-uttest/action.yml recorded.

    Written by every UT leg and never read until now, which is why the issues
    have carried no reproduce line: the path differs per category and there is
    nowhere else it is stated.
    """
    entry: dict[str, str] = {}
    for line in read_lines(find_file(root, f"reproduce_{category}.log")):
        if line.startswith("File Path:"):
            entry["file_path"] = line.split(":", 1)[1].strip()
        elif line.startswith("Reproduce Command:"):
            entry["command_template"] = line.split(":", 1)[1].strip()
    return entry


# --------------------------------------------------------------------------- #
# Signature and deterministic grouping
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


# --------------------------------------------------------------------------- #
# Per-category baseline
# --------------------------------------------------------------------------- #


def baseline_candidates(run_id: int) -> list[dict]:
    """Nightlies older than `run_id`, newest first.

    Paged until enough runs older than the target are found, rather than taking
    a fixed window of the most recent ones. That window is anchored to today,
    not to the run being classified, and most runs of this workflow are
    on-demand: it slides past an older target within days and leaves it with no
    candidates at all, so the same run classifies as `regression` one week and
    `unknown` the next.
    """
    pat = re.compile(r"^(Nightly|Weekly) / Build-from-source")
    out: list[dict] = []
    for page in range(1, MAX_CANDIDATE_PAGES + 1):
        rows = gh_json(
            f"repos/{REPO}/actions/workflows/{WORKFLOW}/runs"
            f"?status=completed&per_page=100&page={page}"
        ).get("workflow_runs", [])
        for r in rows:
            if int(r["id"]) < run_id and pat.match(r.get("display_title") or ""):
                out.append({"databaseId": int(r["id"]),
                            "createdAt": r.get("created_at", "")})
        if len(rows) < 100 or len(out) >= MAX_BASELINE_CANDIDATES:
            break
    out.sort(key=lambda r: r["databaseId"], reverse=True)
    return out[:MAX_BASELINE_CANDIDATES]


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

    So the lookback is spent per category rather than per run: a category whose
    leg keeps failing goes on looking after its neighbours have settled, and a
    candidate that produced no readable artifact for it costs it nothing.
    """
    pending = set(categories)
    looked = {c: 0 for c in categories}
    baselines: dict[str, Baseline] = {}
    walked = 0
    for age, cand in enumerate(baseline_candidates(run_id), 1):
        if not pending:
            break
        walked = age
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
                # Recorded but not charged, so that an empty walk is legible:
                # "nothing to read" and "read and found unhealthy" are the two
                # ways to reach `unknown` and they call for different fixes.
                report["baseline_walk"].append({
                    "run_id": cand_id, "category": category,
                    "state": "no artifact", "actual": 0,
                    "expected": EXPECTED_CASES.get(category, 0),
                })
                continue
            state, actual, expected = category_state(root, category)
            report["baseline_walk"].append({
                "run_id": cand_id, "category": category,
                "state": state, "actual": actual, "expected": expected,
            })
            looked[category] += 1
            if state != "complete":
                if looked[category] >= MAX_BASELINE_LOOKBACK:
                    pending.discard(category)
                continue
            leg = CATEGORY_LEG[category]
            passed, failed, every = read_case_sets(root, category)
            torch, tpo = read_versions(root)
            baselines[category] = Baseline(
                meta=BaselineMeta(
                    run_id=cand_id,
                    created_at=cand["createdAt"][:10],
                    age_in_runs=age,
                    leg=leg,
                    job_url=job_url(cand_id, jobs, leg),
                    torch=torch,
                    torch_xpu_ops=tpo,
                ),
                passed=passed,
                failed=failed,
                all_cases=every,
                passed_by_module=module_counts(passed),
                all_by_module=module_counts(every),
            )
            pending.discard(category)
        for path in dirs.values():
            shutil.rmtree(path, ignore_errors=True)
    for category in sorted(set(categories) - set(baselines)):
        warn(
            f"no baseline for {category}: walked {walked} nightly/nightlies "
            f"older than this run, {looked[category]} of which had a readable "
            "artifact for it, and none completed it healthily; its issues will "
            "be filed unclassified"
        )
    return baselines


# --------------------------------------------------------------------------- #
# Classify
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
# What stopped running
#
# A module that fails to import does not fail its cases, it erases them: they
# reach neither passed_<cat>.log nor failures_<cat>.log, so they never enter
# new_ut_failure_list.csv, and a few hundred missing cases sit far below the 5%
# count gate in ut_result_check.sh:check_test_cases. Comparing module coverage
# against the baseline is the only thing here that sees them.
#
# The same per-module index is what classifies a collection error above:
# the module row is in neither the baseline's passed nor its failed set, but
# the module is in passed_by_module, so "did this used to work" is answerable
# exactly, one level up from the case.
#
# Nothing in this stage mutes on its own. What it produces is the blast radius
# - how many cases the file used to pass - which the renderer puts into the
# issue. The issue itself does mute, like any other: it carries the whole-
# module row, so the row stops being a new failure on the next run and the job
# goes green with the file still dark. That trade is deliberate; leaving it red
# forever ends with nobody reading the nightly at all. The count is what keeps
# the muted state honest, so it belongs in the issue body and not only a log.
# --------------------------------------------------------------------------- #


def collection_error_context(case: Case,
                             baselines: dict[str, Baseline]) -> dict:
    """For one whole-module row, what that module used to run."""
    base = baselines.get(case.category)
    if base is None:
        state, passed = "no baseline", 0
    elif base.passed_by_module.get(case.module):
        state, passed = "was passing", base.passed_by_module[case.module]
    elif case.module in base.all_by_module:
        state, passed = "known, none passing", 0
    else:
        state, passed = "new test file", 0
    return {
        "line": case.line,
        "category": case.category,
        "module": case.module,
        "state": state,
        "baseline_passed": passed,
        "baseline_run": base.meta.run_id if base else None,
    }


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
                "baseline_passed": count, "baseline_run": base.meta.run_id,
            })
    if report["vanished_modules"]:
        total = sum(v["baseline_passed"] for v in report["vanished_modules"])
        warn(
            f"{len(report['vanished_modules'])} module(s) produced no cases in "
            f"this run but had {total} passing case(s) in their baseline; see "
            "the report artifact. Reported only - nothing filed, nothing muted."
        )


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


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def collect_leg(run_id: int, leg: str, names: list[tuple[str, bool]], work: Path,
                jobs: list, report: dict, current: RunInfo,
                leg_health: dict) -> list[Case]:
    """Artifact health checks plus this run's new failures, for one leg."""
    data_artifact = pick_artifact(names, "Inductor-XPU-UT-Data", leg, run_id)
    if data_artifact is None:
        report["skipped_legs"].append({"leg": leg, "reason": "no UT data artifact"})
        warn(f"{leg}: no usable Inductor-XPU-UT-Data artifact; filing nothing")
        return []
    root = work / f"current-{leg}"
    if not download(run_id, data_artifact, root):
        report["skipped_legs"].append({"leg": leg, "reason": "artifact download failed"})
        return []

    current.job_urls[leg] = job_url(run_id, jobs, leg)
    current.runners[leg] = job_runner(jobs, leg)
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
                f"{int(HEALTH_RATIO * 100)}% expected for a complete run. The "
                "failures may be real but the machine is suspect; filing "
                "nothing for it."
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
        warn(
            f"{leg}: new failure count mismatch: filtered={expected_rows}, "
            f"csv={len(cases)}, so some failures lost their error message"
        )

    kept = [c for c in cases if c.category in healthy_categories]
    dropped = len(cases) - len(kept)
    if dropped:
        print(f"note: dropped {dropped} {leg} cases from unhealthy categories")

    # H7 is decided later, by infra_leg_gate: what share of a leg's failures
    # carry a denylisted message is a fact, and calling that share infra
    # breakage is a reading of it. Recorded here, acted on there.
    infra = {c.line for c in kept if is_infra(normalize_error(c.message))}
    leg_health[leg] = {
        "runner_name": current.runners.get(leg, ""),
        "new_failures": len(kept),
        "infra_pattern_cases": sorted(infra),
        "infra_pattern_ratio": round(len(infra) / len(kept), 4) if kept else 0.0,
    }
    if infra and len(kept) < INFRA_MIN_CASES:
        print(
            f"note: {leg} has {len(infra)}/{len(kept)} infra-looking new "
            f"failures. That is under the {INFRA_MIN_CASES} it takes for the "
            "share to mean anything, so the leg is kept. Each error is still "
            "judged on how many test files it reached."
        )
    return kept


def new_report(args) -> dict:
    return {
        "run_id": args.run_id,
        "test_type": args.test_type,
        "mode": args.mode,
        "categories": [],
        "skipped_legs": [],
        "vanished_modules": [],
        "baseline_walk": [],
        "unknown_case_lines": [],
    }


# --------------------------------------------------------------------------- #
# Collect - facts only
# --------------------------------------------------------------------------- #


def collect_evidence(args, work: Path, report: dict) -> Evidence:
    """Everything that can be read off the artifacts, and nothing else.

    No grouping, no infra verdict, no GitHub write. What comes out is meant to
    be enough for the filing half to work from alone, which is why the
    baseline-derived numbers are computed here rather than the baselines
    carried across.
    """
    run_meta = gh_json(f"repos/{REPO}/actions/runs/{args.run_id}")
    current = RunInfo(
        run_id=args.run_id,
        created_at=run_meta.get("created_at", "")[:10],
        job_urls={}, torch={}, torch_xpu_ops={}, collect_env={}, runners={},
    )
    gates = {"build_failed": False, "abort": False, "oversized": False}
    leg_health: dict[str, dict] = {}
    cases: list[Case] = []

    jobs = resolve_jobs(args.run_id)
    # H1: if the build failed nothing downstream can be trusted.
    build_jobs = [j for j in jobs if j[1].startswith("linux-build")]
    if any(j[2] in ("failure", "cancelled") for j in build_jobs):
        warn(
            "build job did not succeed, so nothing downstream can be trusted; "
            "filing nothing for this run"
        )
        report["skipped_legs"].append({"leg": "*", "reason": "build not successful"})
        gates["build_failed"] = True
        return Evidence(
            run=current, cases=[], classification={}, new_case_reason={},
            collection_context={}, baselines={}, tracebacks={}, reproduce={},
            leg_health=leg_health, gates=gates, report=carried_report(report),
        )

    names = list_artifacts(args.run_id)
    for leg in LEG_CATEGORIES:
        cases.extend(collect_leg(args.run_id, leg, names, work, jobs, report,
                                 current, leg_health))

    if len(cases) > ABORT_THRESHOLD:
        print(
            f"::error::{len(cases)} new failures exceeds ABORT_THRESHOLD "
            f"({ABORT_THRESHOLD}); assuming infra breakage and creating nothing"
        )
        report["skipped_legs"].append({"leg": "*", "reason": "abort threshold"})
        gates["abort"] = True
        # Nothing downstream will read these, and resolving baselines for them
        # means downloading five past nightlies to answer a question already
        # settled.
        return Evidence(
            run=current, cases=cases, classification={}, new_case_reason={},
            collection_context={}, baselines={}, tracebacks={}, reproduce={},
            leg_health=leg_health, gates=gates, report=carried_report(report),
        )
    gates["oversized"] = len(cases) > OVERSIZED_THRESHOLD

    # Every healthy category, not just the ones with something to file: a
    # category whose only symptom is that a file stopped producing cases
    # reports no failure at all, so a night that is otherwise green is exactly
    # the night the vanished-module check has to survive to.
    healthy = {c["category"] for c in report["categories"] if c["state"] == "complete"}
    baselines = resolve_baselines(
        args.run_id, healthy | {c.category for c in cases}, work, report)
    record_vanished_modules(work, healthy, baselines, report)

    classification = {c.line: classify_case(c, baselines) for c in cases}
    reasons = {
        c.line: new_case_reason(c, baselines.get(c.category))
        for c in cases if not c.is_collection_error
    }
    context = {
        c.line: collection_error_context(c, baselines)
        for c in cases if c.is_collection_error
    }

    tracebacks: dict[str, list[str]] = {}
    samples = sample_traceback_targets(cases, MAX_TRACEBACK_SAMPLES)
    for leg in sorted({c.leg for c in samples}):
        root = work / f"current-{leg}"
        if root.is_dir():
            tracebacks.update(extract_tracebacks(root, {
                (c.class_name, c.test_name): c.line
                for c in samples if c.leg == leg
            }))

    reproduce: dict[str, dict] = {}
    for category in sorted({c.category for c in cases}):
        root = work / f"current-{CATEGORY_LEG[category]}"
        entry = read_reproduce(root, category) if root.is_dir() else {}
        if entry:
            reproduce[category] = entry

    return Evidence(
        run=current, cases=cases, classification=classification,
        new_case_reason=reasons, collection_context=context,
        baselines={cat: b.meta for cat, b in baselines.items()},
        tracebacks=tracebacks, reproduce=reproduce, leg_health=leg_health,
        gates=gates, report=carried_report(report),
    )


CARRIED_SECTIONS = ("categories", "skipped_legs", "vanished_modules", "baseline_walk")


def carried_report(report: dict) -> dict:
    return {key: report[key] for key in CARRIED_SECTIONS}


# --------------------------------------------------------------------------- #
# Evidence on disk
# --------------------------------------------------------------------------- #


def commit_link(repo: str, sha: str) -> str:
    return f"[`{sha[:8]}`]({SERVER}/{repo}/commit/{sha})" if sha else "unknown"


def rendered_blocks(evidence: Evidence) -> dict:
    """Paste-ready markdown that does not depend on how failures are grouped.

    Composed here rather than left to the filing step because a bisect range
    is the part most easily got wrong and most misleading when wrong: the
    baseline sha and tonight's sha have to come from the same leg, and nothing
    in the rendered text says which leg it came from. Everything below is a
    string to be copied, not data to be assembled.
    """
    run = evidence.run
    baseline_rows: dict[str, list[str]] = {}
    compare: dict[str, str] = {}
    staleness: dict[str, str] = {}
    for category, base in sorted(evidence.baselines.items()):
        leg = CATEGORY_LEG[category]
        baseline_rows[category] = [
            f"| {category} | Last good | [#{base.run_id} ({base.leg})]({base.job_url}) "
            f"| {base.created_at} | {commit_link(PYTORCH_REPO, base.torch)} "
            f"| {commit_link(REPO, base.torch_xpu_ops)} |",
            f"| {category} | First seen bad | "
            f"[#{run.run_id} ({leg})]({run.job_urls.get(leg, '')}) "
            f"| {run.created_at} "
            f"| {commit_link(PYTORCH_REPO, run.torch.get(leg, ''))} "
            f"| {commit_link(REPO, run.torch_xpu_ops.get(leg, ''))} |",
        ]
        if base.torch and run.torch.get(leg):
            link = (f"Changes in range ({category}): "
                    f"[pytorch]({SERVER}/{PYTORCH_REPO}/compare/"
                    f"{base.torch}...{run.torch[leg]})")
            if base.torch_xpu_ops and run.torch_xpu_ops.get(leg):
                link += (f" - [torch-xpu-ops]({SERVER}/{REPO}/compare/"
                         f"{base.torch_xpu_ops}...{run.torch_xpu_ops[leg]})")
            compare[category] = link
        # A stale baseline keeps "regression" true but makes the range much
        # weaker evidence, so say so rather than presenting a five-night range
        # in the same shape as a one-night one.
        if base.age_in_runs > 1:
            gap = base.age_in_runs - 1
            staleness[category] = (
                f"Note: the last healthy {category} nightly was "
                f"{base.age_in_runs} runs back ({gap} intervening "
                f"{'nightly' if gap == 1 else 'nightlies'} did not complete this "
                "category), so this range is wider than one night and the failure "
                "may predate the first-seen-bad run."
            )

    collection: dict[str, dict] = {}
    for line, ctx in sorted(evidence.collection_context.items()):
        dropped = ctx["baseline_passed"]
        collection[line] = {
            "table_row": f"| `{ctx['module']}` | {ctx['category']} "
                         f"| {ctx['state']} | {dropped} |",
            "verdict": {
                CLS_REGRESSION: (
                    f"Classified as a **regression**: the module's {dropped} "
                    "case(s) passed in the baseline and do not run now. The row "
                    "itself is in neither the baseline's passed nor its failed "
                    "set - a healthy run records a module's cases, never the "
                    "module - so the comparison behind that label is at module "
                    "granularity."
                ),
                CLS_PERSISTENT: (
                    "Classified as **persistent**: the baseline knew this module "
                    "but had nothing passing in it, so the breakage predates the "
                    "baseline."
                ),
                CLS_NEW_CASE: (
                    "Classified as a **new test file**: the baseline had never "
                    "seen this module, so it has not been observed importing here."
                ),
                CLS_UNKNOWN: (
                    "**Baseline unavailable**, so whether this module used to "
                    "import could not be determined."
                ),
            }[evidence.classification.get(line, CLS_UNKNOWN)],
            "baseline_passed": dropped,
        }
    return {
        "baseline_table_header": [
            "| Category | | Run | Date | torch | torch-xpu-ops |",
            "|---|---|---|---|---|---|",
        ],
        "baseline_table_rows": baseline_rows,
        "compare_links": compare,
        "baseline_staleness": staleness,
        "collection_error": collection,
    }


def is_bmg(runner: str) -> bool:
    # Case-insensitive where fetch_issues.sh is not, because the runner label
    # there is `bmg-test` while the hostname recorded here is `BMG-17691`.
    return "bmg" in runner.lower()


def labels_for(cls: str, runner: str) -> list[str]:
    """The final list, not the rule that produces it.

    Which labels an issue carries is a pure function of its classification and
    the machine that ran it, with no judgement in it anywhere, so it is
    resolved here and copied at filing time. Handing the filing step three
    lookups to perform instead is how a `persistent` group ends up labelled
    `new_case_failure`.
    """
    labels = ["skipped"]
    if is_bmg(runner):
        labels.append(BMG_LABEL)
    if cls in CLS_LABELS:
        labels.append(cls)
    return labels


def emit_evidence(evidence: Evidence, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    run = evidence.run
    write_json(out / "run.json", {
        "run_id": run.run_id,
        "created_at": run.created_at,
        # Per leg throughout, because a bisect range is per leg: the baseline
        # sha and tonight's sha have to come from the same one or the compare
        # link spans the wrong commits.
        "job_urls": run.job_urls,
        "torch": run.torch,
        "torch_xpu_ops": run.torch_xpu_ops,
        "runners": run.runners,
        "collect_env": run.collect_env,
        "category_leg": CATEGORY_LEG,
        "gates": evidence.gates,
        "legs": evidence.leg_health,
        "report": evidence.report,
        # Stated here so that the filing rules have one source of truth and a
        # change to a threshold does not have to be chased into prose.
        "limits": {
            "max_issues_per_run": MAX_ISSUES_PER_RUN,
            "max_cases_per_issue": MAX_CASES_PER_ISSUE,
            "safe_body_chars": SAFE_BODY_LIMIT,
            "hard_body_chars": GITHUB_BODY_LIMIT,
            "infra_max_test_files": INFRA_MAX_FILES_TO_FILE,
            "infra_leg_share": INFRA_SIGNATURE_RATIO,
            "infra_leg_min_cases": INFRA_MIN_CASES,
        },
        # Resolved, keyed `<cls>|<leg>`, because the runner is per leg. Every
        # case also carries its own resolved list; this map is here for a group
        # whose cases have all been placed already and for cross-checking a split.
        "labels": {
            f"{cls}|{leg}": labels_for(cls, runner)
            for cls in (CLS_REGRESSION, CLS_NEW_CASE, CLS_PERSISTENT, CLS_UNKNOWN)
            for leg, runner in sorted(run.runners.items())
        },
        "marker_template": MARKER_TEMPLATE,
        "marker_version": MARKER_VERSION,
    })
    write_json(out / "cases.json", {
        "count": len(evidence.cases),
        "cases": [
            {
                "line": c.line,
                "category": c.category,
                "leg": c.leg,
                "class_name": c.class_name,
                "test_name": c.test_name,
                "test_file": c.test_file,
                "module": c.module,
                "is_collection_error": c.is_collection_error,
                "message": c.message,
                "cls": evidence.classification.get(c.line, CLS_UNKNOWN),
                "labels": labels_for(
                    evidence.classification.get(c.line, CLS_UNKNOWN),
                    run.runners.get(c.leg, "")),
                "runner_name": run.runners.get(c.leg, ""),
                "has_traceback": c.line in evidence.tracebacks,
            }
            for c in evidence.cases
        ],
        "reproduce": evidence.reproduce,
    })
    counts: dict[str, int] = {}
    for cls in evidence.classification.values():
        counts[cls] = counts.get(cls, 0) + 1
    write_json(out / "classifications.json", {
        "by_case": evidence.classification,
        "counts": counts,
        "new_case_reason": evidence.new_case_reason,
        "collection_context": list(evidence.collection_context.values()),
        "baselines": {cat: vars(meta) for cat, meta in evidence.baselines.items()},
    })
    write_json(out / "tracebacks.json", {"by_case": evidence.tracebacks})
    write_json(out / "blocks.json", rendered_blocks(evidence))
    write_json(out / "digest.json", {
        "all_cases": evidence.digest, "count": len(evidence.cases),
    })


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Decisions - opinions, checked before use
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Apply - the only half that writes
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Audit - what the filed issues actually mute
#
# The `Cases:` block of an issue is a byte-exact subtraction rule: every line
# in it is removed from the next run's failures by `grep -vFxf` in
# ut_result_check.sh:92. A line matching nothing looks harmless tonight and
# stays in the issue forever, so the night a test with that exact name really
# does fail, it is subtracted in silence.
#
# This runs after the issues exist, so it prevents nothing. What it does is
# make such a line visible on the night it appears instead of months later.
# --------------------------------------------------------------------------- #


def bot_issue_bodies() -> dict[int, str]:
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
    return seen


def known_case_lines(work: Path) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Every case and every module this run saw, per category.

    Read from all_cases_<category>.log in the artifacts the collection step
    already downloaded, which is the full roster including skipped cases - not
    just the failures. A muting line naming a case in here is legitimate
    whatever it did tonight; one naming nothing at all cannot have come from
    the artifacts at all.
    """
    cases: dict[str, set[str]] = {}
    modules: dict[str, set[str]] = {}
    for category, leg in sorted(CATEGORY_LEG.items()):
        root = work / f"current-{leg}"
        if not root.is_dir():
            continue
        _, _, every = read_case_sets(root, category)
        if not every:
            continue
        cases[category] = every
        modules[category] = set(module_counts(every))
    return cases, modules


def audit_issues(work: Path, report: dict) -> None:
    cases, modules = known_case_lines(work)
    if not cases:
        warn("no category rosters in the work directory; skipping the mute audit")
        return
    for number, body in sorted(bot_issue_bodies().items()):
        for line in sorted(parse_cases_block(body)):
            parts = line.split(",")
            if len(parts) < 3:
                report["unknown_case_lines"].append(
                    {"issue": number, "line": line, "reason": "not a case row"})
                continue
            category = parts[0]
            if category not in cases:
                # The category did not run tonight, so there is no roster to
                # check against and absence proves nothing.
                continue
            if line in cases[category]:
                continue
            case = Case(category, parts[1], ",".join(parts[2:]), "")
            # A whole-module row never appears in a roster - a healthy run
            # records a module's cases, never the module - so it is checked
            # one level up, against the modules the roster does contain.
            if case.is_collection_error and case.module in modules[category]:
                continue
            report["unknown_case_lines"].append(
                {"issue": number, "line": line,
                 "reason": "no such case in this run's roster"})
    if report["unknown_case_lines"]:
        offenders = sorted({u["issue"] for u in report["unknown_case_lines"]})
        warn(
            f"{len(report['unknown_case_lines'])} muting line(s) in "
            f"{len(offenders)} issue(s) name a case this run has never heard "
            f"of: {', '.join('#' + str(n) for n in offenders)}. Each one is a "
            "subtraction rule that matches nothing today and will silently "
            "mute a real failure the day a test of that name fails. Fix or "
            "delete the line; see the report artifact for which."
        )


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, type=int)
    parser.add_argument("--test-type", default="")
    parser.add_argument("--mode", default="emit-evidence",
                        choices=("emit-evidence", "audit"))
    parser.add_argument("--work-dir", default="ut_auto_issue_work")
    parser.add_argument("--report-dir", default="ut_auto_issue_report")
    parser.add_argument("--evidence-dir", default="")
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report = new_report(args)
    work = Path(args.work_dir)

    if args.mode == "audit":
        audit_issues(work, report)
        return finish(report, report_dir)

    if not args.evidence_dir:
        raise SystemExit("::error::--mode emit-evidence needs --evidence-dir")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    evidence = collect_evidence(args, work, report)
    emit_evidence(evidence, Path(args.evidence_dir))
    print(f"Wrote evidence for {len(evidence.cases)} new failure(s) to "
          f"{args.evidence_dir}")
    return finish(report, report_dir)


def finish(report: dict, report_dir: Path) -> int:
    name = "report.json" if report["mode"] == "emit-evidence" else "audit.json"
    (report_dir / name).write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    lines = [f"## UT auto-issue - {report['mode']}", "", f"Run `{report['run_id']}`", ""]
    if report["categories"]:
        lines += ["| Category | State | Cases | Expected |", "|---|---|---|---|"]
        lines += [
            f"| {c['category']} | {c['state']} | {c['actual']} | {c['expected']} |"
            for c in report["categories"]
        ]
        lines.append("")
    for skipped in report["skipped_legs"]:
        lines.append(f"- Skipped `{skipped['leg']}`: {skipped['reason']}")
    if report["vanished_modules"]:
        lines += [
            "",
            "### Modules that produced no cases in this run",
            "",
            "These passed in their baseline and are absent here, so they did not "
            "fail - they did not run.",
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
    if report["unknown_case_lines"]:
        lines += [
            "",
            "### Muting lines that name no known case",
            "",
            "Each of these subtracts nothing today and will subtract a real "
            "failure the day a test of that name fails.",
            "",
            "| Issue | Line | Reason |",
            "|---|---|---|",
        ]
        lines += [
            f"| #{u['issue']} | `{u['line']}` | {u['reason']} |"
            for u in report["unknown_case_lines"]
        ]
        lines.append("")
    summary = "\n".join(lines) + "\n"
    print(summary)
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as handle:
            handle.write(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
