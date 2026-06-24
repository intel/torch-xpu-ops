from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


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
    occurrences: List[FailureCase] = field(default_factory=list)

    @property
    def repeat_count(self) -> int:
        return len({occurrence.run_id for occurrence in self.occurrences})


@dataclass(frozen=True)
class IssueDraft:
    title: str
    body: str
    template_reference: str
    labels: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class MonitorReport:
    runs: List[WorkflowRun]
    failures_by_run: dict[int, List[FailureCase]]
    candidates: List[DisableCandidate]
    window_complete: bool
    is_green: bool
    summary: str
