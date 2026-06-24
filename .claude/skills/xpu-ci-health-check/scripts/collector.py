from __future__ import annotations

import gzip
import json
import re
import shutil
import subprocess
from typing import List
from urllib.request import Request, urlopen

try:
    from .models import FailureCase, WorkflowJob, WorkflowRun
except ImportError:
    from models import FailureCase, WorkflowJob, WorkflowRun  # type: ignore[no-redef]

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

ROOT_CAUSE_PATTERN = re.compile(
    r"^(?P<cause>(?:[A-Za-z_][\w.]*?(?:Error|Exception)|AssertionError):.*)$"
)

TIMESTAMP_PREFIX_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\S+\s+")


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


def extract_root_cause_from_context(raw_lines: List[str], failure_index: int) -> str:
    traceback_lines: List[str] = []
    for raw_line in raw_lines[failure_index + 1 :]:
        normalized_line = strip_timestamp_prefix(raw_line).strip()
        if not normalized_line:
            continue
        if normalized_line.startswith(("FAILED ", "ERROR ")):
            break
        traceback_lines.append(normalized_line)
        if normalized_line.startswith(("===", "---")):
            continue
        if normalized_line.startswith("Traceback"):
            continue
        if normalized_line.startswith("E "):
            normalized_line = normalized_line[1:].strip()
        match = ROOT_CAUSE_PATTERN.match(normalized_line)
        if match:
            raw_cause = match.group("cause").strip()
            return _summarize_root_cause(raw_cause, traceback_lines, raw_lines[failure_index])
    return _summarize_root_cause("", traceback_lines, raw_lines[failure_index])


def _summarize_root_cause(raw_cause: str, traceback_lines: List[str], failed_line: str) -> str:
    combined = " \n".join([failed_line, raw_cause, *traceback_lines]).lower()
    arch_match = re.search(r"arch\s+(\d+)", combined)
    arch_text = f"Arch {arch_match.group(1)}" if arch_match else "the detected architecture"

    if "cutlass" in combined and ("notimplementederror" in combined or "loweringexception" in combined):
        return (
            f"XPU CUTLASS lowering does not support {arch_text}; "
            "the failure is in the backend lowering path, not the test body."
        )

    if "inductor" in combined and "loweringexception" in combined:
        return "XPU Inductor lowering fails in the backend compilation path."

    if "assertionerror" in combined:
        return "XPU execution hits an assertion failure."

    return raw_cause or (traceback_lines[-1] if traceback_lines else "")


def extract_failure_cases_from_log_text(
    log_text: str,
    *,
    run_id: int,
    run_number: int,
    source: str,
) -> List[FailureCase]:
    seen: set[str] = set()
    failures: List[FailureCase] = []
    for line in log_text.splitlines():
        normalized_line = strip_timestamp_prefix(line)
        for pattern in FAILURE_PATTERNS:
            match = pattern.match(normalized_line)
            if not match:
                continue
            case_id = normalize_case_id(match.group("case"))
            if not is_plausible_case_id(case_id) or case_id in seen:
                break
            seen.add(case_id)
            failures.append(
                FailureCase(
                    case_id=case_id,
                    run_id=run_id,
                    run_number=run_number,
                    source=source,
                    raw_line=normalized_line.strip(),
                    detail=(match.groupdict().get("detail") or "").strip(),
                    cuda_status="unknown",
                )
            )
            break
    return failures


def extract_failed_lines_from_raw_log_text(log_text: str) -> List[str]:
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
) -> List[FailureCase]:
    failures: List[FailureCase] = []
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
            root_cause = detail or extract_root_cause_from_context(raw_lines, index)
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
            with urlopen(request) as response:
                return json.loads(response.read().decode("utf-8"))

        gh_payload = self._gh_api_json(url)
        if gh_payload is not None:
            return gh_payload

        request = Request(url)
        request.add_header("Accept", "application/vnd.github+json")
        request.add_header("X-GitHub-Api-Version", "2022-11-28")
        with urlopen(request) as response:
            return json.loads(response.read().decode("utf-8"))

    def _request_bytes(self, url: str) -> bytes:
        if self.token:
            request = Request(url)
            request.add_header("Accept", "application/vnd.github+json")
            request.add_header("Authorization", f"Bearer {self.token}")
            with urlopen(request) as response:
                return response.read()

        if url.startswith(self.api_base.rstrip("/") + "/"):
            gh_bytes = self._gh_api_bytes(url)
            if gh_bytes is not None:
                return gh_bytes

        request = Request(url)
        with urlopen(request) as response:
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

    def list_run_jobs(self, run: WorkflowRun) -> List[WorkflowJob]:
        jobs: List[WorkflowJob] = []
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

    def list_failed_jobs(self, run: WorkflowRun) -> List[WorkflowJob]:
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

    def list_recent_completed_main_runs(self, limit: int = 3) -> List[WorkflowRun]:
        base_url = (
            f"{self.api_base}/repos/{self.owner}/{self.repo}"
            f"/actions/workflows/{self.workflow_file}/runs?branch=main&status=completed"
        )
        runs: List[WorkflowRun] = []
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

    def collect_failure_cases(self, run: WorkflowRun) -> List[FailureCase]:
        failures: List[FailureCase] = []
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
