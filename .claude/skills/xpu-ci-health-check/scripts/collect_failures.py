#!/usr/bin/env python3
"""Collect XPU CI failure evidence for AI root-cause analysis.

This module now bundles the script entrypoint, collector helpers, and the small
data model used by the xpu-ci-health-check skill.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
import subprocess
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import quote
from urllib.request import Request, urlopen

DEFAULT_API_BASE = "https://api.github.com"
DEFAULT_WORKFLOW_FILE = "xpu.yml"
DEFAULT_RAW_JOB_LOG_BASE = "https://ossci-raw-job-status.s3.amazonaws.com/log"

FAILURE_PATTERNS = [
    re.compile(r"^FAILED\s+\[(?P<case>(?=[^\]]*(?:/|::))[^\]]+)\](?:\s+-\s+(?P<detail>.*))?$"),
    re.compile(r"^FAILED\s+\[(?P<duration>\d[^\]]*)\]\s+(?P<case>.+?)(?:\s+-\s+(?P<detail>.*))?$"),
    re.compile(r"^FAILED\s+(?P<case>[^\s]+)(?:\s+-\s+(?P<detail>.*))?$"),
    re.compile(r"^FAILED\s+(?P<case>.+?)(?:\s+-\s+(?P<detail>.*))?$"),
    re.compile(r"^ERROR\s+\[(?P<case>(?=[^\]]*(?:/|::))[^\]]+)\](?:\s+-\s+(?P<detail>.*))?$"),
    re.compile(r"^ERROR\s+\[(?P<duration>\d[^\]]*)\]\s+(?P<case>.+?)(?:\s+-\s+(?P<detail>.*))?$"),
    re.compile(r"^ERROR\s+(?P<case>[^\s]+)(?:\s+-\s+(?P<detail>.*))?$"),
    re.compile(r"^ERROR\s+(?P<case>.+?)(?:\s+-\s+(?P<detail>.*))?$"),
    re.compile(r"^\s*_{2,}\s*(?P<case>[^_]+?)\s*_{2,}\s*$"),
    re.compile(r"^\s*(?P<case>[^\s].*?::.*)\s+FAILED\b(?:\s+-\s+(?P<detail>.*))?$"),
    re.compile(r"^\s*(?P<case>[^\s].*?::.*?)\s+FAILED\b.*$"),
]

VALID_CASE_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9_./-]+\.py::[A-Za-z0-9_.\[\]-]+(?:::[A-Za-z0-9_.\[\]-]+)*$"
)

TIMESTAMP_PREFIX_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\S+\s+")

TARGET_PLATFORM = "xpu"
TORCH_CI_FAILURE_BASE = "https://www.torch-ci.com/failure?failureCaptures="
DISABLE_LABELS = ["module: xpu", "triaged"]
DISABLE_CC_LINE = (
    "cc [@gujinghui](https://github.com/gujinghui) "
    "[@EikanWang](https://github.com/EikanWang) "
    "[@fengyuan14](https://github.com/fengyuan14) "
    "[@guangyey](https://github.com/guangyey)"
)


@dataclass(frozen=True)
class WorkflowRun:
    run_id: int
    run_number: int
    branch: str
    status: str
    conclusion: str
    html_url: str
    logs_url: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    head_sha: Optional[str] = None


@dataclass(frozen=True)
class WorkflowJob:
    job_id: int
    name: str
    conclusion: str
    html_url: str
    logs_url: str = ""


@dataclass(frozen=True)
class FailureCase:
    case_id: str
    run_id: int
    run_number: int
    source: str
    raw_line: str
    detail: str = ""
    root_cause: str = ""
    cuda_status: str = "unknown"


@dataclass(frozen=True)
class DisableCandidate:
    case_id: str
    occurrences: list[FailureCase] = field(default_factory=list)

    @property
    def repeat_count(self) -> int:
        return len({occurrence.run_id for occurrence in self.occurrences})


@dataclass(frozen=True)
class IssueDraft:
    title: str
    body: str
    template_reference: str
    labels: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MonitorReport:
    runs: list[WorkflowRun]
    failures_by_run: dict[int, list[FailureCase]]
    candidates: list[DisableCandidate]
    window_complete: bool
    is_green: bool
    summary: str


def normalize_case_id(raw_case: str) -> str:
    case = raw_case.strip()
    case = case.rstrip(':,.;)]"')
    case = case.replace("\\", "/")
    return case


def strip_timestamp_prefix(line: str) -> str:
    return TIMESTAMP_PREFIX_PATTERN.sub("", line, count=1)


def is_plausible_case_id(case_id: str) -> bool:
    if not case_id or case_id.startswith("[") or case_id.endswith("%]"):
        return False
    if "::" not in case_id:
        return False
    return bool(VALID_CASE_ID_PATTERN.match(case_id))


def _recent_examples_url(case_id: str) -> str:
    failure_capture = case_id if case_id.startswith("test/") else f"test/{case_id}"
    encoded = quote(f'["{failure_capture}"]', safe="")
    return f"{TORCH_CI_FAILURE_BASE}{encoded}"


def _case_id_to_issue_title(case_id: str) -> str:
    parts = case_id.split("::")
    if len(parts) >= 3:
        return f"{parts[-1]} (__main__.{parts[-2]})"
    if len(parts) == 2:
        module_name = parts[-2].rsplit("/", maxsplit=1)[-1].removesuffix(".py")
        return f"{parts[-1]} (__main__.{module_name})"
    return case_id


def build_disable_issue(case_id: str) -> dict:
    title = f"DISABLED {_case_id_to_issue_title(case_id)}"
    body = "\n".join(
        [
            f"Platforms: {TARGET_PLATFORM}",
            "",
            (
                "This test was disabled because it is failing on main branch "
                f"([recent examples]({_recent_examples_url(case_id)}))."
            ),
            "",
            DISABLE_CC_LINE,
        ]
    )
    return {"title": title, "body": body, "labels": list(DISABLE_LABELS)}


def build_github_new_issue_url(owner: str, repo: str, issue: dict) -> str:
    title = quote(issue["title"], safe="")
    body = quote(issue["body"], safe="")
    query = f"title={title}&body={body}"
    if issue.get("labels"):
        labels = quote(",".join(issue["labels"]), safe=",")
        query += f"&labels={labels}"
    return f"https://github.com/{owner}/{repo}/issues/new?{query}"


def extract_failed_lines_from_raw_log_text(log_text: str) -> list[str]:
    failed_lines = [line.strip() for line in log_text.splitlines() if "FAILED [" in line]
    if failed_lines:
        return failed_lines
    return [line.strip() for line in log_text.splitlines() if line.lstrip().startswith("FAILED ")]


def extract_failure_cases_from_raw_log_text(
    raw_log_text: str,
    *,
    run_id: int,
    run_number: int,
    source: str,
) -> list[FailureCase]:
    failures: list[FailureCase] = []
    raw_lines = raw_log_text.splitlines()
    for index, raw_line in enumerate(raw_lines):
        normalized_line = strip_timestamp_prefix(raw_line)
        if not ("FAILED [" in normalized_line or normalized_line.lstrip().startswith("FAILED ")):
            continue
        for pattern in FAILURE_PATTERNS:
            match = pattern.match(normalized_line)
            if not match:
                continue
            case_id = normalize_case_id(match.group("case"))
            if not is_plausible_case_id(case_id):
                break
            detail = (match.groupdict().get("detail") or "").strip()
            root_cause = detail
            failures.append(
                FailureCase(
                    case_id=case_id,
                    run_id=run_id,
                    run_number=run_number,
                    source=source,
                    raw_line=normalized_line.strip(),
                    detail=detail,
                    root_cause=root_cause,
                    cuda_status="unknown",
                )
            )
            break
    unique: dict[str, FailureCase] = {}
    for failure in failures:
        unique.setdefault(failure.case_id, failure)
    return list(unique.values())




class GitHubActionsClient:
    def __init__(
        self,
        owner: str,
        repo: str,
        workflow_file: str = DEFAULT_WORKFLOW_FILE,
        api_base: str = DEFAULT_API_BASE,
        token: str | None = None,
    ) -> None:
        self.owner = owner
        self.repo = repo
        self.workflow_file = workflow_file
        self.api_base = api_base.rstrip("/")
        self.token = token

    def _request_json(self, url: str) -> dict:
        if self.token:
            request = Request(url)
            request.add_header("Accept", "application/vnd.github+json")
            request.add_header("X-GitHub-Api-Version", "2022-11-28")
            request.add_header("Authorization", f"Bearer {self.token}")
            with urlopen(request, timeout=10) as response:
                return json.loads(response.read().decode("utf-8"))

        gh_payload = self._gh_api_json(url)
        if gh_payload is not None:
            return gh_payload

        request = Request(url)
        request.add_header("Accept", "application/vnd.github+json")
        request.add_header("X-GitHub-Api-Version", "2022-11-28")
        with urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))

    def _request_bytes(self, url: str) -> bytes:
        if self.token:
            request = Request(url)
            request.add_header("Accept", "application/vnd.github+json")
            request.add_header("Authorization", f"Bearer {self.token}")
            with urlopen(request, timeout=10) as response:
                return response.read()

        if url.startswith(self.api_base.rstrip("/") + "/"):
            gh_bytes = self._gh_api_bytes(url)
            if gh_bytes is not None:
                return gh_bytes

        request = Request(url)
        with urlopen(request, timeout=10) as response:
            return response.read()

    def _gh_api_endpoint(self, url: str) -> str:
        if not url.startswith(self.api_base.rstrip("/") + "/"):
            raise ValueError("GitHub CLI fallback only supports the configured GitHub API base URL")
        return url.removeprefix(self.api_base.rstrip("/") + "/")

    def _gh_api_json(self, url: str) -> dict | None:
        if self.api_base.rstrip("/") != DEFAULT_API_BASE or shutil.which("gh") is None:
            return None
        endpoint = self._gh_api_endpoint(url)
        completed = subprocess.run(["gh", "api", endpoint], check=True, capture_output=True)
        return json.loads(completed.stdout.decode("utf-8"))

    def _gh_api_bytes(self, url: str) -> bytes | None:
        if self.api_base.rstrip("/") != DEFAULT_API_BASE or shutil.which("gh") is None:
            return None
        endpoint = self._gh_api_endpoint(url)
        completed = subprocess.run(["gh", "api", endpoint], check=True, capture_output=True)
        return completed.stdout

    def _request_json_page(self, url: str, page: int) -> dict:
        separator = "&" if "?" in url else "?"
        return self._request_json(f"{url}{separator}per_page=100&page={page}")

    def _raw_job_log_url(self, job_id: int) -> str:
        return f"{DEFAULT_RAW_JOB_LOG_BASE}/{job_id}"

    def list_run_jobs(self, run: WorkflowRun) -> list[WorkflowJob]:
        jobs: list[WorkflowJob] = []
        page = 1
        while True:
            url = f"{self.api_base}/repos/{self.owner}/{self.repo}/actions/runs/{run.run_id}/jobs"
            payload = self._request_json_page(url, page)
            job_items = payload.get("jobs", [])
            if not job_items:
                break
            for item in job_items:
                jobs.append(
                    WorkflowJob(
                        job_id=item["id"],
                        name=item.get("name", ""),
                        conclusion=item.get("conclusion", ""),
                        html_url=item.get("html_url", ""),
                        logs_url=self._raw_job_log_url(item["id"]),
                    )
                )
            if len(job_items) < 100:
                break
            page += 1
        return jobs

    def list_failed_jobs(self, run: WorkflowRun) -> list[WorkflowJob]:
        return [job for job in self.list_run_jobs(run) if job.conclusion == "failure"]

    @staticmethod
    def _is_test_job(job: WorkflowJob) -> bool:
        return " / test " in job.name

    def download_raw_log_text(self, logs_url: str) -> str:
        if not logs_url:
            raise ValueError("The failed subjob does not expose a raw log link")
        raw_bytes = self._request_bytes(logs_url)
        if raw_bytes.startswith(b"\x1f\x8b"):
            raw_bytes = gzip.decompress(raw_bytes)
        return raw_bytes.decode("utf-8-sig", errors="replace")

    def list_recent_completed_main_runs(self, limit: int = 3) -> list[WorkflowRun]:
        base_url = (
            f"{self.api_base}/repos/{self.owner}/{self.repo}"
            f"/actions/workflows/{self.workflow_file}/runs?branch=main&status=completed"
        )
        runs: list[WorkflowRun] = []
        page = 1
        while len(runs) < limit:
            payload = self._request_json_page(base_url, page)
            workflow_runs = payload.get("workflow_runs", [])
            if not workflow_runs:
                break
            for item in workflow_runs:
                if item.get("head_branch") != "main":
                    continue
                if item.get("status") != "completed":
                    continue
                runs.append(
                    WorkflowRun(
                        run_id=item["id"],
                        run_number=item.get("run_number", 0),
                        head_sha=item.get("head_sha"),
                        branch=item.get("head_branch", ""),
                        status=item.get("status", ""),
                        conclusion=item.get("conclusion", ""),
                        html_url=item.get("html_url", ""),
                        logs_url=item.get("logs_url", ""),
                        created_at=item.get("created_at"),
                        updated_at=item.get("updated_at"),
                    )
                )
                if len(runs) >= limit:
                    break
            if len(workflow_runs) < 100:
                break
            page += 1
        return runs

    def fetch_run_logs(self, run: WorkflowRun) -> bytes:
        if not run.logs_url:
            detail_url = f"{self.api_base}/repos/{self.owner}/{self.repo}/actions/runs/{run.run_id}"
            payload = self._request_json(detail_url)
            logs_url = payload.get("logs_url")
            if not logs_url:
                raise ValueError(f"Run {run.run_id} does not expose a logs URL")
            return self._request_bytes(logs_url)
        return self._request_bytes(run.logs_url)

    def collect_failure_cases(self, run: WorkflowRun) -> list[FailureCase]:
        failures: list[FailureCase] = []
        for job in self.list_failed_jobs(run):
            if not self._is_test_job(job):
                continue
            raw_log_text = self.download_raw_log_text(job.logs_url)
            failures.extend(
                extract_failure_cases_from_raw_log_text(
                    raw_log_text,
                    run_id=run.run_id,
                    run_number=run.run_number,
                    source=job.logs_url,
                )
            )
        unique: dict[str, FailureCase] = {}
        for failure in failures:
            unique.setdefault(failure.case_id, failure)
        return list(unique.values())


def collect(owner: str, repo: str, workflow_file: str, run_limit: int, token: str | None) -> dict:
    client = GitHubActionsClient(
        owner=owner,
        repo=repo,
        workflow_file=workflow_file,
        token=token,
    )
    runs = client.list_recent_completed_main_runs(limit=run_limit)

    grouped: OrderedDict[str, dict] = OrderedDict()
    runs_meta = []

    for run in runs:
        runs_meta.append(
            {
                "run_number": run.run_number,
                "run_id": run.run_id,
                "head_sha": run.head_sha,
                "html_url": run.html_url,
            }
        )
        for job in client.list_failed_jobs(run):
            if not client._is_test_job(job):
                continue
            raw_log_text = client.download_raw_log_text(job.logs_url)
            raw_lines = raw_log_text.splitlines()
            failures = extract_failure_cases_from_raw_log_text(
                raw_log_text,
                run_id=run.run_id,
                run_number=run.run_number,
                source=job.logs_url,
            )
            for failure in failures:
                if failure.case_id in grouped:
                    continue
                issue = build_disable_issue(failure.case_id)
                short_sha = (run.head_sha or "")[:7]
                grouped[failure.case_id] = {
                    "case_id": failure.case_id,
                    "run_number": run.run_number,
                    "run_id": run.run_id,
                    "commit_sha": run.head_sha,
                    "commit_short": short_sha,
                    "hud_url": (
                        f"https://hud.pytorch.org/pytorch/pytorch/commit/{run.head_sha}"
                        if run.head_sha
                        else None
                    ),
                    "commit_url": (
                        f"https://github.com/{owner}/{repo}/commit/{run.head_sha}"
                        if run.head_sha
                        else None
                    ),
                    "job_name": job.name,
                    "failure_line": failure.raw_line,
                    "error_excerpt": capture_error_excerpt(raw_lines, failure.case_id),
                    "issue_title": issue["title"],
                    "issue_body": issue["body"],
                    "issue_labels": issue["labels"],
                    "issue_url": build_github_new_issue_url(owner, repo, issue),
                }

    return {
        "owner": owner,
        "repo": repo,
        "workflow_file": workflow_file,
        "run_limit": run_limit,
        "runs": runs_meta,
        "cases": list(grouped.values()),
    }


_PYTEST_HEADER = re.compile(r"^_{3,}.*_{3,}$")
_SECTION_BREAK = re.compile(r"^={3,}.*={3,}$")


def _short_name(case_id: str) -> str:
    return case_id.split("::")[-1].split("[")[0]


def capture_error_excerpt(raw_lines: list[str], case_id: str, max_lines: int = 60) -> str:
    short = _short_name(case_id)
    stripped = [strip_timestamp_prefix(line).rstrip() for line in raw_lines]

    for idx, line in enumerate(stripped):
        if _PYTEST_HEADER.match(line) and short in line:
            block: list[str] = [line]
            for follow in stripped[idx + 1 :]:
                if _PYTEST_HEADER.match(follow) or _SECTION_BREAK.match(follow):
                    break
                block.append(follow)
                if len(block) >= max_lines:
                    break
            excerpt = "\n".join(b for b in block).strip()
            if excerpt:
                return excerpt

    for idx, line in enumerate(stripped):
        if (
                short in line
                and "FAILED" in line
                and not line.strip().endswith("FAILED")
           ):

            block = []
            for follow in stripped[idx : idx + max_lines]:
                if _SECTION_BREAK.match(follow):
                    break
                block.append(follow)
            excerpt = "\n".join(block).strip()
            if excerpt:
                return excerpt

    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect XPU CI failure evidence for AI root-cause analysis.")
    parser.add_argument("--owner", default="pytorch")
    parser.add_argument("--repo", default="pytorch")
    parser.add_argument("--workflow-file", default="xpu.yml")
    parser.add_argument("--run-limit", type=int, default=1, help="Number of completed main runs to inspect")
    parser.add_argument("--token", default=None, help="GitHub token (optional; falls back to gh CLI)")
    args = parser.parse_args()

    bundle = collect(
        owner=args.owner,
        repo=args.repo,
        workflow_file=args.workflow_file,
        run_limit=args.run_limit,
        token=args.token,
    )
    print(json.dumps(bundle, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
