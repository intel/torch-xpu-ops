#!/usr/bin/env python3
"""Facts for the nightly UT auto-issue pipeline.

Nightly failures become GitHub issues in three steps: this script states what
happened, the ut-issue-authoring skill groups the failures and drafts the
issues, and ut_create_issues.py files them. Only the last of the three writes
anything.

What is stated here is the part that is not a judgement: which cases failed,
and how each compares with its category's baseline - passed there and fails now
(`regression`), absent there (`new_case_failure`), already failing
(`persistent`), or no usable baseline (`unknown`). That comparison is exact set
membership over ~180,000 cases, so it belongs in code rather than in a model.

Run by hand against a past nightly to see what it collected:

    python .github/scripts/ut_collect_evidence.py --run-id <run_id> \
        --evidence-dir ./evidence
"""

from __future__ import annotations

import argparse
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

# Four states, from comparing a failing case against its category's baseline:
#   regression        - passed in the baseline, fails now
#   new_case_failure  - absent from the baseline, or present but skipped
#   persistent        - already failing in the baseline; onset predates it
#   unknown           - no usable baseline, or the case cannot be compared
CLS_REGRESSION = "regression"
CLS_NEW_CASE = "new_case_failure"
CLS_PERSISTENT = "persistent"
CLS_UNKNOWN = "unknown"

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
# Above this many new failures the night is a question about the machine rather
# than about which bug is which, and the evidence stops being something a model
# can read in one pass. Collection stops here: resolving baselines means
# downloading five past nightlies to answer a question already settled.
TOO_MANY_THRESHOLD = 1000
# How many distinct (test file, exact message) strata get a traceback captured.
# Sampling, not grouping: two rows with byte-identical messages are one message.
MAX_TRACEBACK_SAMPLES = 300
# The evidence is one document and it is read whole, so the failure text has to
# be bounded. A JUnit <failure> carries the whole longrepr, and an OpInfo case
# dumps its SampleInput into it: 300 of those unabridged is megabytes, more
# than anything downstream can read. The ends of one say why the test failed;
# the middle is the tensors.
MAX_TRACEBACK_LINES = 40
# And a ceiling on all of them together, spent on the most repeated messages
# first, so that one pathological file cannot crowd out the rest.
MAX_TRACEBACK_CHARS = 300_000
# Names shown per module that lost or gained cases. Enough to see whether one
# name became another; a dtype parametrization alone runs to dozens.
NAME_SAMPLE = 20
HEALTH_RATIO = 0.95

# Covered UT jobs. xpu_distributed is deliberately excluded: it reports through
# run_distributed_tests in ut_result_check.sh, which produces neither the
# per-category passed/failed logs nor a case count, so neither the health
# gate nor the baseline comparison has anything to read.
UT_JOB_CATEGORIES = {
    "basic": ["op_regression", "op_regression_dev1", "op_extended"],
    "op_ut": ["op_ut"],
    "xpu_profiling": ["xpu_profiling"],
}
CATEGORY_UT_JOB = {c: job for job, cats in UT_JOB_CATEGORIES.items() for c in cats}

# Mirrors EXPECTED_CASES in ut_result_check.sh (linux column). Only a fallback:
# runs predating run_health.jsonl carry no recorded verdict of their own.
EXPECTED_CASES = {
    "op_extended": 5349,
    "op_regression": 268,
    "op_regression_dev1": 1,
    "op_ut": 178548,
}


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
    def ut_job(self) -> str:
        return CATEGORY_UT_JOB.get(self.category, "unknown")

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
    ut_job: str
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
    job_urls: dict[str, str]  # ut_job -> job url
    torch: dict[str, str]  # ut_job -> sha
    torch_xpu_ops: dict[str, str]
    collect_env: dict[str, str]
    runners: dict[str, str] = field(default_factory=dict)  # ut_job -> runner name


@dataclass
class Evidence:
    """Everything the rest of the pipeline is allowed to treat as true.

    Self-contained by design: it holds the baseline-derived numbers the issue
    bodies quote rather than the baselines themselves, so nothing after this
    step needs to download a past nightly.
    """
    run: RunInfo
    cases: list[Case]
    classification: dict[str, str]
    cls_reason: dict[str, str]
    collection_context: dict[str, dict]
    baselines: dict[str, BaselineMeta]
    tracebacks: dict[str, list[str]]
    reproduce: dict[str, dict]
    ut_job_health: dict[str, dict]
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


def pick_artifact(names: list[tuple[str, bool]], prefix: str, ut_job: str, run_id: int):
    """Highest run attempt of `<prefix>-<sha>-<ut_job>-<run_id>-<attempt>`."""
    pat = re.compile(rf"^{re.escape(prefix)}-.+-{re.escape(ut_job)}-{run_id}-(\d+)")
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
    ut_log/<ut_job>/, so their depth differs before and after it runs."""
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
# The gate that runs before anything else. Its checks are cited as H1-H6 in the
# warnings and in the report artifact, so they are enumerated here:
#
#   H1  build job conclusion          not success -> nothing downstream can be
#                                     trusted; abort the whole run
#   H2  UT job conclusion             cancelled or skipped
#   H3  UT data artifact              missing, or fails to download -> skip the UT job
#   H4  category present at all       no health record and no category log means
#                                     the category never ran: a quiet skip, not
#                                     an error
#   H5  case-count health             actual < HEALTH_RATIO * expected: the run
#                                     is truncated and the machine is suspect
#   H6  new-failure CSV row count     disagrees with new_failure_list.txt, so
#                                     some failures lost their error message
#
# H2 needs no code of its own - a cancelled or skipped UT job uploads no artifact,
# so H3 catches it. Evaluation is per category rather than per UT job, because the
# `basic` UT job carries three and they fail independently.
#
# All six are facts about the artifacts. Whether a machine misbehaved is not,
# so nothing here decides it: the failures and their messages go out as they
# are, and the skill reads them.
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


def jobs_for(jobs: list[tuple[int, str, str, str]], ut_job: str) -> list[tuple]:
    """The workflow jobs of one UT job, the container one first when there is
    one."""
    cands = [j for j in jobs if f"({ut_job})" in j[1]]
    return sorted(cands, key=lambda j: (not j[1].endswith("test-in-container"), j[0]))


def job_url(run_id: int, jobs: list[tuple[int, str, str, str]], ut_job: str) -> str:
    """Job-level link, so the reader lands on the UT job's log rather than a matrix
    summary page. Falls back to the run URL."""
    run_url = f"{SERVER}/{REPO}/actions/runs/{run_id}"
    cands = jobs_for(jobs, ut_job)
    return f"{run_url}/job/{cands[0][0]}" if cands else run_url


def job_runner(jobs: list[tuple[int, str, str, str]], ut_job: str) -> str:
    """Which machine ran the UT job.

    One error on one box reads differently from the same error on two, and
    nothing else in the artifacts says which box a UT job landed on.
    """
    cands = jobs_for(jobs, ut_job)
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


def trim_traceback(lines: list[str]) -> list[str]:
    """Both ends of a long one, which is where it says what went wrong."""
    if len(lines) <= MAX_TRACEBACK_LINES:
        return lines
    head = MAX_TRACEBACK_LINES * 3 // 4
    tail = MAX_TRACEBACK_LINES - head
    dropped = len(lines) - head - tail
    return [*lines[:head], f"... {dropped} line(s) omitted ...", *lines[-tail:]]


def within_budget(ranked: list[Case], found: dict[str, list[str]]) -> dict:
    """Trimmed, and as many as the budget allows, most-repeated message first."""
    kept: dict[str, list[str]] = {}
    spent = 0
    for case in ranked:
        lines = found.get(case.line)
        if not lines:
            continue
        lines = trim_traceback(lines)
        size = sum(len(line) + 1 for line in lines)
        if spent + size > MAX_TRACEBACK_CHARS:
            break
        kept[case.line] = lines
        spent += size
    if len(kept) < len(found):
        print(f"note: kept {len(kept)} of {len(found)} traceback(s), "
              f"{spent} chars, against a {MAX_TRACEBACK_CHARS} budget")
    return kept


def read_reproduce(root: Path, category: str) -> dict:
    """The `cd` and the pytest invocation linux-uttest/action.yml recorded.

    Written by every UT job and never read until now, which is why the issues
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


def roster(work: Path, category: str) -> set[str]:
    """Every case this run's category produced, skipped ones included."""
    root = work / f"current-{CATEGORY_UT_JOB[category]}"
    if not root.is_dir():
        return set()
    return read_case_sets(root, category)[2]


def case_from_line(line: str) -> Case | None:
    """A `category,class_name,test_name` roster line, back as a Case."""
    parts = line.split(",")
    if len(parts) < 3:
        return None
    return Case(parts[0], parts[1], ",".join(parts[2:]), "")


def module_counts(lines: set[str]) -> dict[str, int]:
    """`category,class_name,test_name` lines, counted per test module."""
    counts: dict[str, int] = {}
    for line in lines:
        case = case_from_line(line)
        if case and case.module:
            counts[case.module] = counts.get(case.module, 0) + 1
    return counts


def resolve_baselines(run_id: int, categories: set[str], work: Path,
                      report: dict) -> dict[str, Baseline]:
    """One pass over candidates, accumulating per category.

    There is no such thing as "the last good nightly" - only "the last nightly
    in which this category completed healthily". A run truncated in op_ut is
    still a perfectly good op_extended baseline, and one download of a
    candidate's `basic` artifact can resolve up to three categories at once.

    So the lookback is spent per category rather than per run: a category whose
    UT job keeps failing goes on looking after its neighbours have settled, and a
    candidate that produced no readable artifact for it costs it nothing.
    """
    pending = set(categories)
    looked = dict.fromkeys(categories, 0)
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
        for ut_job in {CATEGORY_UT_JOB[c] for c in pending}:
            artifact = pick_artifact(names, "Inductor-XPU-UT-Data", ut_job, cand_id)
            dest = work / f"baseline-{cand_id}-{ut_job}"
            if artifact and download(cand_id, artifact, dest):
                dirs[ut_job] = dest
        for category in sorted(pending):
            root = dirs.get(CATEGORY_UT_JOB[category])
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
            ut_job = CATEGORY_UT_JOB[category]
            passed, failed, every = read_case_sets(root, category)
            torch, tpo = read_versions(root)
            baselines[category] = Baseline(
                meta=BaselineMeta(
                    run_id=cand_id,
                    created_at=cand["createdAt"][:10],
                    age_in_runs=age,
                    ut_job=ut_job,
                    job_url=job_url(cand_id, jobs, ut_job),
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


def classify_case(case: Case, baselines: dict[str, Baseline],
                  churned: set[tuple[str, str]]) -> tuple[str, str]:
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
        return CLS_UNKNOWN, "no usable baseline for this category"
    if case.is_collection_error:
        if baseline.passed_by_module.get(case.module):
            return CLS_REGRESSION, "the module's cases passed in the baseline"
        if case.module in baseline.all_by_module:
            return CLS_PERSISTENT, "the baseline knew the module and passed none of it"
        return CLS_NEW_CASE, "the baseline had never seen this module"
    if case.line in baseline.passed:
        return CLS_REGRESSION, "passed in the baseline"
    if case.line in baseline.failed:
        return CLS_PERSISTENT, "already failing in the baseline"
    if case.line in baseline.all_cases:
        return CLS_NEW_CASE, "present in the baseline but skipped there"
    # Absent from the baseline, which only means "new test" if the module's
    # names are otherwise unchanged. A test renamed in stock pytorch is absent
    # under its new name and present under its old one, and calling that a new
    # case claims it has never been observed working when it may have been
    # passing for years.
    if (case.category, case.module) in churned:
        return CLS_UNKNOWN, (
            "absent from the baseline, but the module lost case names between "
            "the baseline and this run, so this may be a test renamed upstream"
        )
    return CLS_NEW_CASE, "absent from the baseline"


# --------------------------------------------------------------------------- #
# What stopped running
#
# A case can leave the run without failing: a module that will not import
# erases its cases rather than failing them, and a test removed or renamed in
# stock pytorch is simply not there. Neither reaches new_ut_failure_list.csv,
# and a few hundred missing cases sit far below the 5% count gate in
# ut_result_check.sh:check_test_cases. Comparing this run's roster against the
# baseline's is the only thing here that sees them.
#
# It matters twice. The count of cases a module used to pass is the blast
# radius an issue about that module has to state. And a module whose names
# changed is one where "absent from the baseline" stops meaning "new test" -
# see classify_case.
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


def record_vanished_cases(work: Path, categories: set[str],
                          baselines: dict[str, Baseline],
                          report: dict) -> set[tuple[str, str]]:
    """Cases the baseline ran and this run does not have at all.

    Three things leave this trace and only this one sees them: a module that
    will not import, a skip pattern wide enough to empty a file - deselected
    cases are absent from the JUnit XML rather than recorded as skipped - and a
    test removed or renamed in stock pytorch.

    Each row is one of three kinds, which differ in what they cost:

      module_gone  the module produced no cases at all. Its cases are dark and
                   nothing reports them as failures, because they did not fail.
      removed      names went and none arrived. Nothing here can be a rename,
                   so no classification is disturbed. What it does leave behind
                   is any open issue still muting one of those names, which now
                   subtracts nothing.
      moved        names went and names arrived. This is the one that can make
                   `new_case_failure` a false claim, so cases in these modules
                   are classified `unknown` instead.

    Which arrived name is which departed one, if any, is a judgement about two
    strings rather than a set operation, so it is not made here: both halves
    are recorded for the skill to read.
    """
    churned: set[tuple[str, str]] = set()
    for category in sorted(categories):
        base = baselines.get(category)
        tonight = roster(work, category)
        if base is None or not tonight:
            continue
        lost = names_by_module(base.all_cases - tonight)
        gained = names_by_module(tonight - base.all_cases)
        live_modules = set(module_counts(tonight))
        for module, names in sorted(lost.items(),
                                    key=lambda kv: (-len(kv[1]), kv[0])):
            arrived = gained.get(module, [])
            if module not in live_modules:
                kind = "module_gone"
            elif arrived:
                kind = "moved"
                churned.add((category, module))
            else:
                kind = "removed"
            report["vanished_cases"].append({
                "category": category,
                "module": module,
                "kind": kind,
                "cases": len(names),
                "baseline_passed": base.passed_by_module.get(module, 0),
                "baseline_run": base.meta.run_id,
                "lost_names": names[:NAME_SAMPLE],
                "gained_names": arrived[:NAME_SAMPLE],
            })
    if report["vanished_cases"]:
        total = sum(v["cases"] for v in report["vanished_cases"])
        warn(
            f"{total} case(s) the baseline ran are absent from this run, across "
            f"{len(report['vanished_cases'])} module(s); see the report "
            "artifact. They did not fail - they did not run, whether because a "
            "module stopped importing or because a test was removed or renamed "
            "upstream. Reported only; nothing filed, nothing muted."
        )
    return churned


def names_by_module(lines: set[str]) -> dict[str, list[str]]:
    """Test names, sorted, per test module."""
    out: dict[str, list[str]] = {}
    for line in lines:
        case = case_from_line(line)
        if case and case.module:
            out.setdefault(case.module, []).append(case.test_name)
    return {module: sorted(names) for module, names in out.items()}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def collect_ut_job(run_id: int, ut_job: str, names: list[tuple[str, bool]], work: Path,
                jobs: list, report: dict, current: RunInfo,
                ut_job_health: dict) -> list[Case]:
    """Artifact health checks plus this run's new failures, for one UT job."""
    data_artifact = pick_artifact(names, "Inductor-XPU-UT-Data", ut_job, run_id)
    if data_artifact is None:
        report["skipped_ut_jobs"].append(
            {"ut_job": ut_job, "reason": "no UT data artifact"})
        warn(f"{ut_job}: no usable Inductor-XPU-UT-Data artifact; filing nothing")
        return []
    root = work / f"current-{ut_job}"
    if not download(run_id, data_artifact, root):
        report["skipped_ut_jobs"].append(
            {"ut_job": ut_job, "reason": "artifact download failed"})
        return []

    current.job_urls[ut_job] = job_url(run_id, jobs, ut_job)
    current.runners[ut_job] = job_runner(jobs, ut_job)
    torch, tpo = read_versions(root)
    current.torch[ut_job] = torch
    current.torch_xpu_ops[ut_job] = tpo
    current.collect_env[ut_job] = read_collect_env(root)

    healthy_categories = set()
    for category in UT_JOB_CATEGORIES[ut_job]:
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
            print(f"note: {category} never ran in this UT job; nothing to file")

    failures_artifact = pick_artifact(names, "New-UT-Failures", ut_job, run_id)
    if failures_artifact is None:
        print(f"note: {ut_job} produced no new failures")
        return []
    csv_dir = work / f"current-{ut_job}-newfail"
    if not download(run_id, failures_artifact, csv_dir):
        return []
    cases = parse_failure_csv(find_file(csv_dir, "new_ut_failure_list.csv"))

    # H6: the CSV is built by grepping ut_failure_list.csv per filtered-log line,
    # so a mismatch means some failures lost their error message.
    expected_rows = len(read_lines(find_file(root, "new_failure_list.txt")))
    if expected_rows and expected_rows != len(cases):
        warn(
            f"{ut_job}: new failure count mismatch: filtered={expected_rows}, "
            f"csv={len(cases)}, so some failures lost their error message"
        )

    kept = [c for c in cases if c.category in healthy_categories]
    dropped = len(cases) - len(kept)
    if dropped:
        print(f"note: dropped {dropped} {ut_job} cases from unhealthy categories")

    ut_job_health[ut_job] = {
        "runner_name": current.runners.get(ut_job, ""),
        "new_failures": len(kept),
    }
    return kept


def new_report(args) -> dict:
    return {
        "run_id": args.run_id,
        "test_type": args.test_type,
        "categories": [],
        "skipped_ut_jobs": [],
        "vanished_cases": [],
        "baseline_walk": [],
    }


# --------------------------------------------------------------------------- #
# Collect - facts only
# --------------------------------------------------------------------------- #


def collect_evidence(args, work: Path, report: dict) -> Evidence:
    """Everything that can be read off the artifacts, and nothing else.

    No grouping, no infra verdict, no GitHub write. What comes out is meant to
    be enough for the skill and the filing step to work from alone, which is
    why the baseline-derived numbers are computed here rather than the
    baselines carried across.
    """
    run_meta = gh_json(f"repos/{REPO}/actions/runs/{args.run_id}")
    current = RunInfo(
        run_id=args.run_id,
        created_at=run_meta.get("created_at", "")[:10],
        job_urls={}, torch={}, torch_xpu_ops={}, collect_env={}, runners={},
    )
    gates = {"build_failed": False, "too_many": False}
    ut_job_health: dict[str, dict] = {}
    cases: list[Case] = []

    jobs = resolve_jobs(args.run_id)
    # H1: if the build failed nothing downstream can be trusted.
    build_jobs = [j for j in jobs if j[1].startswith("linux-build")]
    if any(j[2] in ("failure", "cancelled") for j in build_jobs):
        warn(
            "build job did not succeed, so nothing downstream can be trusted; "
            "filing nothing for this run"
        )
        report["skipped_ut_jobs"].append(
            {"ut_job": "*", "reason": "build not successful"})
        gates["build_failed"] = True
        return Evidence(
            run=current, cases=[], classification={}, cls_reason={},
            collection_context={}, baselines={}, tracebacks={}, reproduce={},
            ut_job_health=ut_job_health, gates=gates, report=carried_report(report),
        )

    names = list_artifacts(args.run_id)
    for ut_job in UT_JOB_CATEGORIES:
        cases.extend(collect_ut_job(args.run_id, ut_job, names, work, jobs, report,
                                 current, ut_job_health))

    if len(cases) > TOO_MANY_THRESHOLD:
        warn(
            f"{len(cases)} new failures is past the {TOO_MANY_THRESHOLD} at "
            "which a night is a question about the machine rather than about "
            "which bug is which; collecting the count and nothing else"
        )
        report["skipped_ut_jobs"].append({"ut_job": "*", "reason": "too many failures"})
        gates["too_many"] = True
        return Evidence(
            run=current, cases=cases, classification={}, cls_reason={},
            collection_context={}, baselines={}, tracebacks={}, reproduce={},
            ut_job_health=ut_job_health, gates=gates, report=carried_report(report),
        )

    # Every healthy category, not just the ones with something to file: a
    # category whose only symptom is that a file stopped producing cases
    # reports no failure at all, so a night that is otherwise green is exactly
    # the night the vanished-case check has to survive to.
    healthy = {c["category"] for c in report["categories"] if c["state"] == "complete"}
    baselines = resolve_baselines(
        args.run_id, healthy | {c.category for c in cases}, work, report)
    churned = record_vanished_cases(work, healthy, baselines, report)

    verdicts = {c.line: classify_case(c, baselines, churned) for c in cases}
    classification = {line: cls for line, (cls, _) in verdicts.items()}
    cls_reason = {line: why for line, (_, why) in verdicts.items()}
    context = {
        c.line: collection_error_context(c, baselines)
        for c in cases if c.is_collection_error
    }

    tracebacks: dict[str, list[str]] = {}
    samples = sample_traceback_targets(cases, MAX_TRACEBACK_SAMPLES)
    for ut_job in sorted({c.ut_job for c in samples}):
        root = work / f"current-{ut_job}"
        if root.is_dir():
            tracebacks.update(extract_tracebacks(root, {
                (c.class_name, c.test_name): c.line
                for c in samples if c.ut_job == ut_job
            }))
    tracebacks = within_budget(samples, tracebacks)

    reproduce: dict[str, dict] = {}
    for category in sorted({c.category for c in cases}):
        root = work / f"current-{CATEGORY_UT_JOB[category]}"
        entry = read_reproduce(root, category) if root.is_dir() else {}
        if entry:
            reproduce[category] = entry

    return Evidence(
        run=current, cases=cases, classification=classification,
        cls_reason=cls_reason, collection_context=context,
        baselines={cat: b.meta for cat, b in baselines.items()},
        tracebacks=tracebacks, reproduce=reproduce, ut_job_health=ut_job_health,
        gates=gates, report=carried_report(report),
    )


CARRIED_SECTIONS = ("categories", "skipped_ut_jobs", "vanished_cases",
                    "baseline_walk")


def carried_report(report: dict) -> dict:
    return {key: report[key] for key in CARRIED_SECTIONS}


# --------------------------------------------------------------------------- #
# Evidence on disk
# --------------------------------------------------------------------------- #


def emit_evidence(evidence: Evidence, out: Path) -> None:
    """One document: everything the skill reads about this run."""
    out.mkdir(parents=True, exist_ok=True)
    run = evidence.run
    write_json(out / "evidence.json", {
        "run": {
            "run_id": run.run_id,
            "created_at": run.created_at,
            "digest": evidence.digest,
            # Per UT job throughout, because a bisect range is per UT job: the
            # baseline sha and tonight's sha have to come from the same one or
            # the compare link spans the wrong commits.
            "job_urls": run.job_urls,
            "torch": run.torch,
            "torch_xpu_ops": run.torch_xpu_ops,
            "runners": run.runners,
            "collect_env": run.collect_env,
            "category_ut_job": CATEGORY_UT_JOB,
            "gates": evidence.gates,
            "ut_jobs": evidence.ut_job_health,
            "baselines": {cat: vars(meta)
                          for cat, meta in evidence.baselines.items()},
            "report": evidence.report,
        },
        "count": len(evidence.cases),
        "counts_by_cls": class_counts(evidence.classification),
        "cases": [
            {
                "line": c.line,
                "category": c.category,
                "ut_job": c.ut_job,
                "class_name": c.class_name,
                "test_name": c.test_name,
                "test_file": c.test_file,
                "module": c.module,
                "is_collection_error": c.is_collection_error,
                "message": c.message,
                "cls": evidence.classification.get(c.line, CLS_UNKNOWN),
                "cls_reason": evidence.cls_reason.get(c.line, ""),
                "runner_name": run.runners.get(c.ut_job, ""),
            }
            for c in evidence.cases
        ],
        "collection_context": list(evidence.collection_context.values()),
        "reproduce": evidence.reproduce,
        "tracebacks": evidence.tracebacks,
    })


def class_counts(classification: dict[str, str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for cls in classification.values():
        counts[cls] = counts.get(cls, 0) + 1
    return counts


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, type=int)
    parser.add_argument("--test-type", default="")
    parser.add_argument("--work-dir", default="ut_auto_issue_work")
    parser.add_argument("--report-dir", default="ut_auto_issue_report")
    parser.add_argument("--evidence-dir", required=True)
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report = new_report(args)
    work = Path(args.work_dir)

    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    evidence = collect_evidence(args, work, report)
    emit_evidence(evidence, Path(args.evidence_dir))
    report["counts_by_cls"] = class_counts(evidence.classification)
    print(f"Wrote evidence for {len(evidence.cases)} new failure(s) to "
          f"{args.evidence_dir}")
    return finish(report, report_dir)


def finish(report: dict, report_dir: Path) -> int:
    (report_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    lines = ["## UT auto-issue - evidence", "", f"Run `{report['run_id']}`", ""]
    if report["categories"]:
        lines += ["| Category | State | Cases | Expected |", "|---|---|---|---|"]
        lines += [
            f"| {c['category']} | {c['state']} | {c['actual']} | {c['expected']} |"
            for c in report["categories"]
        ]
        lines.append("")
    for skipped in report["skipped_ut_jobs"]:
        lines.append(f"- Skipped `{skipped['ut_job']}`: {skipped['reason']}")
    if report.get("counts_by_cls"):
        lines += ["", "| Classification | New failures |", "|---|---|"]
        lines += [
            f"| {cls} | {count} |"
            for cls, count in sorted(report["counts_by_cls"].items())
        ]
        lines.append("")
    if report["vanished_cases"]:
        lines += [
            "",
            "### Cases the baseline ran and this run does not have",
            "",
            "These did not fail - they did not run. `module_gone` is a file "
            "that produced nothing, `removed` is names that went with none "
            "arriving, and `moved` is names that went while others arrived: "
            "only the last can be a rename, so only its modules have their "
            "failing cases classified `unknown` rather than `new_case_failure`.",
            "",
            "| Category | Module | Kind | Missing | Passing in baseline |",
            "|---|---|---|---|---|",
        ]
        lines += [
            f"| {v['category']} | `{v['module']}` | {v['kind']} | {v['cases']} "
            f"| {v['baseline_passed']} |"
            for v in report["vanished_cases"]
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
