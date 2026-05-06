#!/usr/bin/env python3
"""
JUnit XML Test Comparison Tool — Target vs Baseline.

Compares test results between target and baseline with markdown reporting
and optional GitHub issue tracking.

Usage:
    python compare-ut.py -i results/target results/baseline -o comparison.xlsx
    python compare-ut.py -i results/*.xml -o comparison.csv -m
    python compare-ut.py -i target/ baseline/ -o out.xlsx --check-changes
    python compare-ut.py -i target/ baseline/ -o out.csv -m --testfile test_ops
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import glob
import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

try:
    from github import Github, Auth
    from github.Issue import Issue
    GITHUB_AVAILABLE = True
except ImportError:
    GITHUB_AVAILABLE = False

log = logging.getLogger(__name__)


# ── Enums ─────────────────────────────────────────────────────────────
class TestStatus(Enum):
    PASSED = "passed"
    XFAIL = "xfail"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, s: str) -> TestStatus:
        if not s or (isinstance(s, float) and pd.isna(s)):
            return cls.UNKNOWN
        s = str(s).lower().strip()
        for key, status in (
            ("pass", cls.PASSED), ("success", cls.PASSED),
            ("xfail", cls.XFAIL), ("fail", cls.FAILED),
            ("error", cls.ERROR), ("skip", cls.SKIPPED),
        ):
            if key in s:
                return status
        return cls.UNKNOWN

    @property
    def priority(self) -> int:
        return {
            self.PASSED: 5, self.XFAIL: 4, self.FAILED: 3,
            self.ERROR: 2, self.SKIPPED: 1, self.UNKNOWN: 0,
        }[self]

    @property
    def emoji(self) -> str:
        return {
            self.PASSED: "✅", self.XFAIL: "⚠️", self.FAILED: "❌",
            self.ERROR: "💥", self.SKIPPED: "⏭️", self.UNKNOWN: "❓",
        }[self]


class TestDevice(Enum):
    BASELINE = "baseline"
    TARGET = "target"
    UNKNOWN = "unknown"

    @classmethod
    def from_test_type(cls, test_type: str) -> TestDevice:
        t = test_type.lower()
        if "baseline" in t:
            return cls.BASELINE
        if "target" in t:
            return cls.TARGET
        return cls.UNKNOWN


# ── Data Classes ──────────────────────────────────────────────────────
@dataclasses.dataclass(frozen=True)
class TestCase:
    uniqname: str
    testfile: str
    classname: str
    name: str
    device: TestDevice
    testtype: str
    status: TestStatus
    time: float
    message: str = ""
    raw_file: str = ""
    raw_class: str = ""
    raw_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "uniqname": self.uniqname,
            "testfile": self.testfile,
            "classname": self.classname,
            "name": self.name,
            "device": self.device.value,
            "testtype": self.testtype,
            "status": self.status.value,
            "time": float(self.time),
            "message": self.message,
            "raw_file": self.raw_file,
            "raw_class": self.raw_class,
            "raw_name": self.raw_name,
        }


# ── File Pattern Matcher ─────────────────────────────────────────────
class FilePatternMatcher:
    _CLASSNAME_RE = re.compile(r".*\.")
    _CASENAME_RE = re.compile(r"[^a-zA-Z0-9_.-]")
    _TESTFILE_RE = re.compile(r".*torch-xpu-ops\.test\.")
    _TESTFILE_CPP_RE = re.compile(r".*/test/xpu/")
    _NORMALIZE_RE = re.compile(r".*\.\./test/")
    _GPU_RE = re.compile(r"(?:xpu|cuda)", re.IGNORECASE)

    TEST_TYPE_PATTERNS = {
        "xpu-baseline": [re.compile(r"/baseline/")],
        "xpu-target": [re.compile(r".*")],
    }

    FILE_REPLACEMENTS = [
        ("test/test/", "test/"),
        ("test_c10d_xccl.py", "test_c10d_nccl.py"),
        ("test_c10d_ops_xccl.py", "test_c10d_ops_nccl.py"),
    ]

    @lru_cache(maxsize=1024)
    def determine_test_type(self, xml_file: Path) -> str:
        s = str(xml_file)
        for test_type, patterns in self.TEST_TYPE_PATTERNS.items():
            if any(p.search(s) for p in patterns):
                return test_type
        return "unknown"

    @lru_cache(maxsize=1024)
    def normalize_filepath(self, filepath: str) -> str:
        if not filepath:
            return "unknown_file.py"
        n = filepath
        if self._NORMALIZE_RE.search(n):
            n = self._NORMALIZE_RE.sub("test/", n)
        for old, new in self.FILE_REPLACEMENTS:
            n = n.replace(old, new)
        n = n.replace("_xpu_xpu.py", ".py").replace("_xpu.py", ".py")
        n = re.sub(r'.*/jenkins/workspace/', '', n, flags=re.IGNORECASE)
        return n or "unknown_file.py"

    @lru_cache(maxsize=1024)
    def extract_testfile(self, classname: str, filename: str, xml_file: Path) -> str:
        if filename:
            if filename.endswith(".cpp"):
                testfile = self._TESTFILE_CPP_RE.sub("test/", filename)
            elif filename.endswith(".py"):
                testfile = f"test/{filename}"
            else:
                testfile = filename
        elif classname:
            testfile = self._TESTFILE_RE.sub("test/", classname).replace(".", "/")
            if "/" in testfile:
                testfile = f"{testfile.rsplit('/', 1)[0]}.py"
            else:
                testfile = f"{testfile}.py"
        else:
            s = str(xml_file)
            testfile = (
                re.sub(r".*op_ut_with_[a-zA-Z0-9]+\.", "test.", s)
                .replace(".", "/").replace("/py/xml", ".py").replace("/xml", ".py")
            )
        return self.normalize_filepath(testfile)

    @lru_cache(maxsize=1024)
    def extract_classname(self, full_classname: str) -> str:
        if not full_classname:
            return "UnknownClass"
        return self._CLASSNAME_RE.sub("", full_classname)

    def extract_casename(self, casename: str) -> str:
        if not casename:
            return "unknown_name"
        return self._CASENAME_RE.sub("", casename) or "error_name"

    @lru_cache(maxsize=2048)
    def generate_uniqname(self, filename: str, classname: str, name: str) -> str:
        return self._GPU_RE.sub("cuda", f"{filename}{classname}{name}")


# ── XML Extractor ─────────────────────────────────────────────────────
class TestDetailsExtractor:
    def __init__(self, pattern_matcher: FilePatternMatcher | None = None):
        self.pm = pattern_matcher or FilePatternMatcher()
        self.test_cases: list[TestCase] = []
        self.stats = {"files_processed": 0, "test_cases_found": 0,
                      "empty_files": 0, "failed_files": 0}

    @staticmethod
    def _determine_status(elem: ET.Element) -> tuple[TestStatus, str]:
        failure = elem.find("failure")
        if failure is not None:
            msg = failure.get("message", "")
            return (TestStatus.XFAIL if "pytest.xfail" in msg else TestStatus.FAILED), msg
        skipped = elem.find("skipped")
        if skipped is not None:
            msg = skipped.get("message", "")
            typ = skipped.get("type", "")
            if "pytest.xfail" in typ or "pytest.xfail" in msg:
                return TestStatus.XFAIL, msg
            return TestStatus.SKIPPED, msg
        error = elem.find("error")
        if error is not None:
            return TestStatus.ERROR, error.get("message", "")
        return TestStatus.PASSED, ""

    def _parse_testcase(self, elem: ET.Element, xml_file: Path, test_type: str) -> TestCase | None:
        try:
            device = TestDevice.from_test_type(test_type)
            if device == TestDevice.UNKNOWN:
                return None

            classname = elem.get("classname", "")
            filename = elem.get("file", "")
            name = elem.get("name", "")

            s_class = self.pm.extract_classname(classname)
            s_name = self.pm.extract_casename(name)
            testfile = self.pm.extract_testfile(classname, filename, xml_file)
            uniqname = self.pm.generate_uniqname(testfile, s_class, s_name)

            status, message = self._determine_status(elem)
            short_msg = message[:200].replace('\r\n', ';').replace('\n', ';').replace('\r', ';')

            try:
                t = float(elem.get("time", "0"))
            except (ValueError, TypeError):
                t = 0.0

            return TestCase(
                uniqname=uniqname, testfile=testfile, classname=s_class, name=s_name,
                device=device, testtype=test_type, status=status, time=t,
                message=short_msg, raw_file=filename, raw_class=classname, raw_name=name,
            )
        except Exception as e:
            log.debug("Error parsing test case in %s: %s", xml_file, e)
            return None

    def process_xml(self, xml_file: Path) -> list[TestCase]:
        try:
            test_type = self.pm.determine_test_type(xml_file)
            cases = []
            for _, elem in ET.iterparse(xml_file, events=('end',)):
                if elem.tag == 'testcase':
                    tc = self._parse_testcase(elem, xml_file, test_type)
                    if tc:
                        cases.append(tc)
                    elem.clear()
            return cases
        except Exception as e:
            log.error("Error processing %s: %s", xml_file, e)
            self.stats["failed_files"] += 1
            return []

    def find_xml_files(self, input_paths: list[str]) -> list[Path]:
        xml_files: set[Path] = set()
        for inp in input_paths:
            p = Path(inp).expanduser().resolve()
            if p.is_file() and p.suffix.lower() == ".xml":
                xml_files.add(p)
            elif p.is_dir():
                xml_files.update(p.rglob("*.xml"))
            else:
                for fp in glob.glob(str(p), recursive=True):
                    fp = Path(fp)
                    if fp.is_file() and fp.suffix.lower() == ".xml":
                        xml_files.add(fp.resolve())
        return sorted(xml_files)

    def process(self, input_paths: list[str], max_workers: int = None) -> bool:
        if max_workers is None:
            max_workers = max(1, os.cpu_count() // 2)

        xml_files = self.find_xml_files(input_paths)
        if not xml_files:
            log.error("No XML files found")
            return False

        log.info("Found %d XML files, using %d workers", len(xml_files), max_workers)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(self.process_xml, f): f for f in xml_files}
            done = 0
            for fut in concurrent.futures.as_completed(futures):
                done += 1
                if done % 10 == 0 or done == len(xml_files):
                    log.info("Processed %d/%d files", done, len(xml_files))
                try:
                    cases = fut.result()
                    if cases:
                        self.test_cases.extend(cases)
                        self.stats["test_cases_found"] += len(cases)
                    else:
                        self.stats["empty_files"] += 1
                except Exception as e:
                    log.error("Error processing %s: %s", futures[fut], e)
                    self.stats["failed_files"] += 1
                finally:
                    self.stats["files_processed"] += 1

        return bool(self.test_cases)


# ── GitHub Issue Tracker ──────────────────────────────────────────────
class GitHubIssueTracker:
    CASES_RE = re.compile(r'(?:Cases|Test Cases):\s*\n(.*?)(?:\n\n|\Z)', re.DOTALL | re.IGNORECASE)

    def __init__(self, repo: str = None, token: str = None,
                 cache_path: str = None, pattern_matcher: FilePatternMatcher | None = None):
        self.repo_name = repo or os.environ.get('GITHUB_REPOSITORY', '')
        self.token = token or os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN', '')
        self.cache_path = Path(cache_path) if cache_path else None
        self.github = None
        self.repository = None
        self.issues_cache: dict[int, dict[str, Any]] = {}
        self.test_to_issues: dict[str, list[dict[str, Any]]] = {}
        self.pm = pattern_matcher or FilePatternMatcher()

        if not GITHUB_AVAILABLE:
            log.warning("PyGithub not installed. GitHub integration disabled.")

    def load_cache(self) -> bool:
        if not self.cache_path:
            return False
        matches = sorted(glob.glob(f"./**/{self.cache_path.name}", recursive=True), reverse=True)
        cache_file = matches[-1] if matches else None
        if not cache_file or not Path(cache_file).exists():
            return False
        try:
            with open(cache_file, encoding='utf-8') as f:
                data = json.load(f)
            self.issues_cache = {int(k): v for k, v in data.get('issues_cache', {}).items()}
            self.test_to_issues = data.get('test_to_issues', {})
            log.info("Loaded %d issues from cache: %s", len(self.issues_cache), cache_file)
            return True
        except Exception as e:
            log.warning("Failed to load cache from %s: %s", cache_file, e)
            return False

    def save_cache(self) -> bool:
        if not self.cache_path:
            return False
        try:
            data = {
                'issues_cache': {str(k): v for k, v in self.issues_cache.items()},
                'test_to_issues': self.test_to_issues,
            }
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            log.info("Saved %d issues to cache: %s", len(self.issues_cache), self.cache_path)
            return True
        except Exception as e:
            log.warning("Failed to save cache: %s", e)
            return False

    def fetch_issues(self, state: str = 'all', labels: list[str] = None,
                     force_refresh: bool = False) -> bool:
        if not force_refresh and self.load_cache():
            return True
        if not self.repository and not self._init_github():
            return False
        log.info("Fetching issues from %s (state=%s)", self.repo_name, state)
        try:
            kwargs = {'state': state, 'direction': 'desc'}
            if labels:
                kwargs['labels'] = labels
            self.issues_cache.clear()
            self.test_to_issues.clear()
            count = 0
            for issue in self.repository.get_issues(**kwargs):
                if issue.pull_request:
                    continue
                self._parse_issue(issue)
                count += 1
            log.info("Fetched %d issues, %d test mappings", count, len(self.test_to_issues))
            self.save_cache()
            return True
        except Exception as e:
            log.error("Error fetching issues: %s", e)
            return False

    def _init_github(self) -> bool:
        try:
            if self.token:
                self.github = Github(auth=Auth.Token(self.token))
            else:
                self.github = Github()
            self.repository = self.github.get_repo(self.repo_name)
            log.info("Connected to GitHub: %s", self.repo_name)
            return True
        except Exception as e:
            log.error("Failed to init GitHub client: %s", e)
            return False

    def _parse_issue(self, issue: Issue) -> None:
        issue_id = issue.number
        labels = [l.name for l in issue.labels]
        self.issues_cache[issue_id] = {
            'id': issue_id, 'title': issue.title, 'state': issue.state,
            'labels': labels, 'url': issue.html_url,
            'created_at': issue.created_at.isoformat() if issue.created_at else '',
            'updated_at': issue.updated_at.isoformat() if issue.updated_at else '',
        }
        for tc_line in self._extract_test_cases(issue.body or ''):
            stripped = tc_line.strip()
            case_state = "close" if re.match(r'^~~.*~~$', stripped) else "open"
            if case_state == "close":
                stripped = stripped[2:-2]
            parts = stripped.split(',')
            if len(parts) < 3:
                continue
            test_file = self.pm.extract_testfile(parts[1].strip(), None, None)
            test_class = self.pm.extract_classname(parts[1].strip())
            test_name = self.pm.extract_casename(parts[2].strip())
            uniqname = self.pm.generate_uniqname(test_file, test_class, test_name)
            self.test_to_issues.setdefault(uniqname, []).append({
                'id': issue_id, 'state': issue.state,
                'case_state': case_state, 'labels': labels,
            })

    def _extract_test_cases(self, body: str) -> list[str]:
        if not body:
            return []
        match = self.CASES_RE.search(body)
        if not match:
            return []
        return [p.strip() for p in match.group(1).strip().split('\n') if p.strip()]

    def find_issues_for_test(self, uniqname: str) -> list[dict[str, Any]]:
        return self.test_to_issues.get(uniqname, [])

    def enhance_comparison(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        enhanced = df.copy()
        for col in ('issue_ids', 'issue_labels', 'issue_statuses', 'case_statuses'):
            enhanced[col] = ''
        for idx, row in enhanced.iterrows():
            uniqname = row.get('uniqname', '')
            if not uniqname:
                continue
            issues = self.find_issues_for_test(uniqname)
            if issues:
                enhanced.at[idx, 'issue_ids'] = '|'.join(str(i['id']) for i in issues)
                enhanced.at[idx, 'issue_statuses'] = '|'.join(i['state'] for i in issues)
                enhanced.at[idx, 'case_statuses'] = '|'.join(i['case_state'] for i in issues)
                all_labels = {l for i in issues for l in i['labels']}
                enhanced.at[idx, 'issue_labels'] = '|'.join(sorted(all_labels))
        return enhanced

    def mark_passed_issues(self, tracked_pass_df: pd.DataFrame) -> int:
        """Cross out passed test cases in GitHub issues and add comments.

        Uses `gh` CLI to update issue bodies (strikethrough) and add comments.
        Returns the number of issues updated.
        """
        if tracked_pass_df.empty:
            log.info("No tracked passes to update")
            return 0

        pass_uniqnames = set(tracked_pass_df['uniqname'].dropna())
        if not pass_uniqnames:
            return 0

        # Map issue_id -> set of passing uniqnames
        issue_to_passes: dict[int, set[str]] = {}
        for uname in pass_uniqnames:
            for entry in self.test_to_issues.get(uname, []):
                issue_to_passes.setdefault(entry['id'], set()).add(uname)

        if not issue_to_passes:
            log.info("No issue mappings found for tracked passes")
            return 0

        repo = self.repo_name
        run_id = os.environ.get('GITHUB_RUN_ID', '')
        gh_repo = os.environ.get('GITHUB_REPOSITORY', '')
        updated = 0

        for issue_id, passing_unames in issue_to_passes.items():
            try:
                result = subprocess.run(
                    ['gh', '--repo', repo, 'issue', 'view', str(issue_id),
                     '--json', 'body', '-q', '.body'],
                    capture_output=True, text=True, timeout=30,
                )
                if result.returncode != 0:
                    log.warning("Failed to get issue #%d: %s", issue_id, result.stderr.strip())
                    continue

                body = result.stdout
                new_lines = []
                cases_section = False
                modified = False
                passed_names: list[str] = []

                for line in body.split('\n'):
                    stripped = line.strip()
                    if re.match(r'(?:Cases|Test Cases):', stripped, re.IGNORECASE):
                        cases_section = True
                        new_lines.append(line)
                        continue
                    if not stripped:
                        cases_section = False

                    if cases_section and stripped and not stripped.startswith('~~'):
                        parts = stripped.split(',')
                        if len(parts) >= 3:
                            tf = self.pm.extract_testfile(parts[1].strip(), None, None)
                            tc = self.pm.extract_classname(parts[1].strip())
                            tn = self.pm.extract_casename(parts[2].strip())
                            uname = self.pm.generate_uniqname(tf, tc, tn)
                            if uname in passing_unames:
                                new_lines.append(f'~~{stripped}~~')
                                passed_names.append(parts[0].strip())
                                modified = True
                                continue

                    new_lines.append(line)

                if not modified:
                    continue

                new_body = '\n'.join(new_lines)
                body_file = Path(f'/tmp/issue-body-{issue_id}.txt')
                body_file.write_text(new_body, encoding='utf-8')
                subprocess.run(
                    ['gh', '--repo', repo, 'issue', 'edit', str(issue_id),
                     '--body-file', str(body_file)],
                    capture_output=True, timeout=30,
                )
                body_file.unlink(missing_ok=True)

                cases_str = ' '.join(sorted(set(passed_names)))
                if run_id and gh_repo:
                    comment = (
                        f"\u2705 {cases_str} Passed in "
                        f"[nightly testing](https://github.com/{gh_repo}/actions/runs/{run_id})"
                    )
                else:
                    comment = f"\u2705 {cases_str} Passed in nightly testing"
                subprocess.run(
                    ['gh', '--repo', repo, 'issue', 'comment', str(issue_id),
                     '--body', comment],
                    capture_output=True, timeout=30,
                )

                updated += 1
                log.info("Updated issue #%d: crossed out %d cases", issue_id, len(passed_names))

            except subprocess.TimeoutExpired:
                log.warning("Timeout updating issue #%d", issue_id)
            except Exception as e:
                log.warning("Failed to update issue #%d: %s", issue_id, e)

        log.info("Updated %d/%d issues", updated, len(issue_to_passes))
        return updated

    def get_summary_stats(self) -> dict[str, Any]:
        stats = {
            'total_issues': len(self.issues_cache),
            'open_issues': sum(1 for i in self.issues_cache.values() if i['state'] == 'open'),
            'closed_issues': sum(1 for i in self.issues_cache.values() if i['state'] != 'open'),
            'unique_test_cases': len(self.test_to_issues),
        }
        return stats


# ── Result Analyzer ───────────────────────────────────────────────────
class ResultAnalyzer:
    def __init__(self, test_cases: list[TestCase]):
        self.test_cases = test_cases
        self.df = self._create_df()
        self._deduped: pd.DataFrame | None = None
        self.issue_tracker: GitHubIssueTracker | None = None

    def set_issue_tracker(self, tracker: GitHubIssueTracker):
        self.issue_tracker = tracker

    def _create_df(self) -> pd.DataFrame:
        if not self.test_cases:
            return pd.DataFrame()
        df = pd.DataFrame([tc.to_dict() for tc in self.test_cases])
        if not df.empty:
            df['status'] = pd.Categorical(df['status'], categories=[e.value for e in TestStatus])
        return df

    def deduplicate(self) -> pd.DataFrame:
        if self._deduped is not None:
            return self._deduped
        if self.df.empty:
            self._deduped = pd.DataFrame()
            return self._deduped
        df = self.df.copy()
        df["_prio"] = df["status"].apply(lambda x: TestStatus.from_string(x).priority)
        df.sort_values("_prio", ascending=False, inplace=True)
        self._deduped = df.drop_duplicates(subset=["device", "uniqname"], keep="first") \
                          .drop(columns=["_prio"]).reset_index(drop=True)
        return self._deduped

    def split_by_device(self, df: pd.DataFrame = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        if df is None:
            df = self.df
        if df.empty or "device" not in df.columns:
            return pd.DataFrame(), pd.DataFrame()
        return df[df["device"] == "baseline"].copy(), df[df["device"] == "target"].copy()

    def merge_results(self, baseline_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
        if baseline_df.empty and target_df.empty:
            return pd.DataFrame()

        merged = pd.merge(baseline_df, target_df, on='uniqname', how='outer',
                          suffixes=('_baseline', '_target'))

        for col in merged.columns:
            if col.startswith('status_'):
                merged[col] = merged[col].fillna('unknown')
        merged[merged.select_dtypes(include='number').columns] = merged.select_dtypes(include='number').fillna(0)
        str_cols = [c for c in merged.columns
                    if not c.startswith('status_') and c not in merged.select_dtypes(include='number').columns]
        merged[str_cols] = merged[str_cols].fillna('')

        wanted = [
            "uniqname",
            "testfile_baseline", "classname_baseline", "name_baseline",
            "testfile_target", "classname_target", "name_target",
            "device_baseline", "testtype_baseline", "status_baseline", "time_baseline", "message_baseline",
            "device_target", "testtype_target", "status_target", "time_target", "message_target",
            "raw_file_target", "raw_class_target", "raw_name_target",
        ]
        existing = [c for c in wanted if c in merged.columns]
        merged = merged[existing]

        if self.issue_tracker:
            merged = self.issue_tracker.enhance_comparison(merged)
        return merged

    def find_changes(self, merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if merged.empty:
            return pd.DataFrame(), pd.DataFrame()

        bsl_pass = merged["status_baseline"].isin(["passed", "xfail"])
        tgt_pass = merged["status_target"].isin(["passed", "xfail"])

        new_fail = merged[bsl_pass & ~tgt_pass].copy()
        new_pass = merged[~bsl_pass & tgt_pass].copy()

        if not new_fail.empty:
            new_fail["change_type"] = "failure"
            new_fail["message_target"] = new_fail["message_target"].str[:50]
            new_fail["reason"] = np.select(
                [new_fail["status_target"] == "skipped",
                 new_fail["status_target"] == "failed",
                 new_fail["status_target"] == "error"],
                ["Skipped on Target", "Failed on Target", "Error on Target"],
                default="Not Run on Target",
            )

        if not new_pass.empty:
            new_pass["change_type"] = "pass"
            new_pass["message_baseline"] = new_pass["message_baseline"].str[:50]
            new_pass["reason"] = np.select(
                [new_pass["status_baseline"] == "skipped",
                 new_pass["status_baseline"] == "failed",
                 new_pass["status_baseline"] == "error"],
                ["Was skipped on Baseline", "Was failing on Baseline", "Was error on Baseline"],
                default="Now passing on Target",
            )

        return new_fail, new_pass

    def file_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        bsl, tgt = self.split_by_device(df)
        if bsl.empty or tgt.empty:
            return pd.DataFrame()

        def _agg(grp):
            return pd.Series({
                'Total': len(grp),
                'Passed': grp['status'].isin(['passed', 'xfail']).sum(),
                'Failed': (grp['status'] == 'failed').sum(),
                'Error': (grp['status'] == 'error').sum(),
                'Skipped': (grp['status'] == 'skipped').sum(),
            })

        bsl_s = bsl.groupby('testfile').apply(_agg, include_groups=False).reset_index().rename(columns={'testfile': 'Test File'})
        tgt_s = tgt.groupby('testfile').apply(_agg, include_groups=False).reset_index().rename(columns={'testfile': 'Test File'})

        bsl_s = bsl_s.rename(columns={c: f'Baseline_{c}' for c in bsl_s.columns if c != 'Test File'})
        tgt_s = tgt_s.rename(columns={c: f'Target_{c}' for c in tgt_s.columns if c != 'Test File'})

        summary = pd.merge(bsl_s, tgt_s, on='Test File', how='outer').fillna(0)
        for c in summary.columns:
            if c != 'Test File':
                summary[c] = summary[c].astype(int)

        def _rate(passed, total):
            return f"{passed/total*100:.2f}%" if total > 0 else "N/A"

        summary['Baseline Pass Rate'] = summary.apply(
            lambda r: _rate(r['Baseline_Passed'], r['Baseline_Total']), axis=1)
        summary['Target Pass Rate'] = summary.apply(
            lambda r: _rate(r['Target_Passed'], r['Target_Total']), axis=1)
        summary['Pass Rate Delta'] = summary.apply(
            lambda r: f"{(r['Target_Passed']/r['Target_Total'] - r['Baseline_Passed']/r['Baseline_Total'])*100:+.2f}%"
            if r['Baseline_Total'] > 0 and r['Target_Total'] > 0 else 'N/A', axis=1)
        return summary

    def device_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        rows = []
        for dev, label in [("baseline", "Baseline"), ("target", "Target")]:
            sub = df[df["device"] == dev]
            if sub.empty:
                continue
            total = len(sub)
            passed = sub["status"].isin(["passed", "xfail"]).sum()
            rows.append({
                "Device": label, "Total": total,
                "Passed": int(passed),
                "Failed": int((sub["status"] == "failed").sum()),
                "Error": int((sub["status"] == "error").sum()),
                "Skipped": int((sub["status"] == "skipped").sum()),
                "XFAIL": int((sub["status"] == "xfail").sum()),
                "Pass Rate": f"{passed / total * 100:.2f}%" if total else "0%",
            })
        return pd.DataFrame(rows)


# ── Markdown Report ───────────────────────────────────────────────────
def _first_nonempty(*vals: str, default: str = "Unknown") -> str:
    for v in vals:
        if v and str(v).strip():
            return v
    return default


def _display_status(status: str) -> str:
    """Display 'unknown' as 'not run' in markdown output."""
    return "not run" if status == "unknown" else status


def _issue_link(issue_ids: str, repo: str) -> str:
    if not issue_ids:
        return ""
    ids = issue_ids.split('|')
    if repo:
        return ', '.join(f"[#{i}](https://github.com/{repo}/issues/{i})" for i in ids)
    return issue_ids


def _write_case_table_target(md: list[str], df: pd.DataFrame, repo: str,
                             max_rows: int = 0) -> None:
    """Write a target-status markdown table. max_rows=0 means show all."""
    md.append("| Test File | Test Name | Status | Issues | Message |")
    md.append("|-----------|-----------|--------|--------|---------|")
    rows = df if max_rows == 0 else df.head(max_rows)
    for _, row in rows.iterrows():
        tf = _first_nonempty(row.get('testfile_target', ''), row.get('testfile_baseline', ''))
        tn = _first_nonempty(row.get('name_target', ''), row.get('name_baseline', ''))
        st = _display_status(str(row.get('status_target', '')))
        emoji = TestStatus.from_string(row.get('status_target', '')).emoji
        issues = _issue_link(str(row.get('issue_ids', '')), repo)
        msg = str(row.get('message_target', ''))[:50].replace('|', '\\|')
        md.append(f"| `{tf}` | `{tn}` | {emoji} {st} | {issues} | {msg} |")
    if max_rows > 0 and len(df) > max_rows:
        md.append(f"| ... | ... | ... | ... | *{len(df) - max_rows} more* |")
    md.append("")


def _write_case_table_baseline(md: list[str], df: pd.DataFrame, repo: str,
                               max_rows: int = 0) -> None:
    """Write a baseline-status markdown table. max_rows=0 means show all."""
    md.append("| Test File | Test Name | Baseline Status | Issues |")
    md.append("|-----------|-----------|-----------------|--------|")
    rows = df if max_rows == 0 else df.head(max_rows)
    for _, row in rows.iterrows():
        tf = _first_nonempty(row.get('testfile_target', ''), row.get('testfile_baseline', ''))
        tn = _first_nonempty(row.get('name_target', ''), row.get('name_baseline', ''))
        bst = _display_status(str(row.get('status_baseline', '')))
        emoji = TestStatus.from_string(row.get('status_baseline', '')).emoji
        issues = _issue_link(str(row.get('issue_ids', '')), repo)
        md.append(f"| `{tf}` | `{tn}` | {emoji} {bst} | {issues} |")
    if max_rows > 0 and len(df) > max_rows:
        md.append(f"| ... | ... | ... | *{len(df) - max_rows} more* |")
    md.append("")


def _write_new_failures_section(md: list[str], new_fail: pd.DataFrame | None,
                                n_fail: int, repo: str) -> None:
    """New failures expanded, Failed/Error first, then Not Run, then Skipped. All cases shown."""
    if new_fail is None or n_fail == 0:
        return

    md.append(f"## ❌ New Failures ({n_fail} tests)\n")
    md.append("Tests that passed on Baseline but failed/skipped/errored on Target:\n")

    nf = new_fail.copy()
    nf['_reason'] = nf['reason'].map({
        "Failed on Target": "Failed/Error on Target",
        "Error on Target": "Failed/Error on Target",
    }).fillna(nf['reason'])

    # Deterministic order: Failed/Error first, then Not Run, then Skipped
    reason_order = ["Failed/Error on Target", "Not Run on Target", "Skipped on Target"]
    for reason in reason_order:
        sub = nf[nf['_reason'] == reason]
        if sub.empty:
            continue
        md.append(f"### {reason} ({len(sub)} tests)\n")
        _write_case_table_target(md, sub, repo, max_rows=0)

    # Any remaining reasons not in the ordered list
    for reason in nf['_reason'].unique():
        if reason in reason_order:
            continue
        sub = nf[nf['_reason'] == reason]
        if sub.empty:
            continue
        md.append(f"### {reason} ({len(sub)} tests)\n")
        _write_case_table_target(md, sub, repo, max_rows=0)


def _write_untracked_section(md: list[str], untracked_fail: pd.DataFrame | None,
                             n_untracked: int, repo: str,
                             collapsed: bool = True) -> None:
    if untracked_fail is None or untracked_fail.empty:
        return
    if collapsed:
        md.append("<details>")
        md.append(f"<summary><b>⚠️ Untracked Failures ({n_untracked} tests)</b></summary>\n")
    else:
        md.append(f"## ⚠️ Untracked Failures ({n_untracked} tests)\n")
    md.append("Target failures with no matching GitHub issue:\n")
    _write_case_table_target(md, untracked_fail, repo, max_rows=0)
    if collapsed:
        md.append("</details>\n")


def _write_tracked_section(md: list[str], tracked_pass: pd.DataFrame | None,
                           n_tracked: int, repo: str,
                           collapsed: bool = True) -> None:
    if tracked_pass is None or tracked_pass.empty:
        return
    if collapsed:
        md.append("<details>")
        md.append(f"<summary><b>✅ Tracked Passes ({n_tracked} tests)</b></summary>\n")
    else:
        md.append(f"## ✅ Tracked Passes ({n_tracked} tests)\n")
    md.append("Target passes with open GitHub issues (consider updating):\n")
    md.append("| Test File | Test Name | Target Status | Issues |")
    md.append("|-----------|-----------|---------------|--------|")
    for _, row in tracked_pass.iterrows():
        tf = _first_nonempty(row.get('testfile_target', ''), row.get('testfile_baseline', ''))
        tn = _first_nonempty(row.get('name_target', ''), row.get('name_baseline', ''))
        st = _display_status(str(row.get('status_target', '')))
        emoji = TestStatus.from_string(row.get('status_target', '')).emoji
        issues = _issue_link(str(row.get('issue_ids', '')), repo)
        md.append(f"| `{tf}` | `{tn}` | {emoji} {st} | {issues} |")
    md.append("")
    if collapsed:
        md.append("</details>\n")


def _write_new_passes_section(md: list[str], new_pass: pd.DataFrame | None,
                              n_pass: int, repo: str,
                              collapsed: bool = True) -> None:
    if new_pass is None or new_pass.empty:
        return
    if collapsed:
        md.append("<details>")
        md.append(f"<summary><b>✅ New Passes ({n_pass} tests)</b></summary>\n")
    else:
        md.append(f"## ✅ New Passes ({n_pass} tests)\n")
    md.append("Tests now passing on Target (were failing/skipped on Baseline):\n")

    np_df = new_pass.copy()
    np_df['_reason'] = np_df['reason'].map({
        "Was failing on Baseline": "Was Failing/Error on Baseline",
        "Was error on Baseline": "Was Failing/Error on Baseline",
    }).fillna(np_df['reason'])

    for reason in np_df['_reason'].unique():
        sub = np_df[np_df['_reason'] == reason]
        md.append(f"### {reason} ({len(sub)} tests)\n")
        _write_case_table_baseline(md, sub, repo, max_rows=20 if collapsed else 0)

    if collapsed:
        md.append("</details>\n")


def _write_file_summary(md: list[str], file_sum: pd.DataFrame) -> None:
    if file_sum.empty:
        return
    md.append("<details>")
    md.append(f"<summary><b>📁 File-Level Summary ({len(file_sum)} files)</b></summary>\n")

    fs = file_sum.copy()
    fs['_fail_score'] = fs['Baseline_Failed'].astype(int) + fs['Baseline_Error'].astype(int)
    fs.sort_values(['_fail_score', 'Baseline_Total'], ascending=[False, False], inplace=True)

    md.append("| Test File | Baseline | Target | Delta |")
    md.append("|-----------|----------|--------|-------|")
    for _, r in fs.head(30).iterrows():
        bf = int(r['Baseline_Failed']) + int(r['Baseline_Error'])
        tf_cnt = int(r['Target_Failed']) + int(r['Target_Error'])
        bsl = f"✅{r['Baseline_Passed']}"
        if bf > 0:
            bsl += f" ❌{bf}"
        bsl += f" ⏭️{r['Baseline_Skipped']}"
        tgt = f"✅{r['Target_Passed']}"
        if tf_cnt > 0:
            tgt += f" ❌{tf_cnt}"
        tgt += f" ⏭️{r['Target_Skipped']}"
        delta = r['Pass Rate Delta']
        md.append(f"| `{r['Test File']}` | {bsl} | {tgt} | {delta} |")
    if len(file_sum) > 30:
        md.append(f"| ... | ... | ... | *{len(file_sum)-30} more* |")
    md.append("")
    md.append("</details>\n")


def write_markdown(
    analyzer: ResultAnalyzer,
    new_fail: pd.DataFrame | None, new_pass: pd.DataFrame | None,
    untracked_fail: pd.DataFrame | None,
    tracked_pass: pd.DataFrame | None,
    filename: str,
    event_type: str = "",
) -> None:
    unique_df = analyzer.deduplicate()
    if unique_df.empty:
        Path(filename).write_text("# Unit Test Results\n\nNo test data available.\n", encoding='utf-8')
        return

    stats_df = analyzer.device_summary(unique_df)
    file_sum = analyzer.file_summary(unique_df)
    n_fail = len(new_fail) if new_fail is not None else 0
    n_pass = len(new_pass) if new_pass is not None else 0
    n_untracked = len(untracked_fail) if untracked_fail is not None else 0
    n_tracked = len(tracked_pass) if tracked_pass is not None else 0
    repo = analyzer.issue_tracker.repo_name if analyzer.issue_tracker else ""
    is_pr = event_type == "pull_request"

    md = []

    # ── Header & Summary ──
    md.append("# Unit Test Comparison Report\n")
    md.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    md.append("## Summary\n")

    if not stats_df.empty:
        if is_pr:
            md.append("| Device | Total | Passed | Failed | Skipped | XFAIL | Pass Rate |")
            md.append("|--------|-------|--------|--------|---------|-------|-----------|")
            for _, r in stats_df.iterrows():
                fail_total = r['Failed'] + r['Error']
                md.append(
                    f"| {r['Device']} | {r['Total']} | {r['Passed']} | {fail_total} "
                    f"| {r['Skipped']} | {r['XFAIL']} | {r['Pass Rate']} |"
                )
        else:
            md.append("| Device | Total | Passed | Failed | ❌ New Fail | ✅ New Pass | Skipped | XFAIL | Pass Rate |")
            md.append("|--------|-------|--------|--------|-------------|-------------|---------|-------|-----------|")
            for _, r in stats_df.iterrows():
                nf = n_fail if r['Device'] == 'Target' else 0
                np_ = n_pass if r['Device'] == 'Target' else 0
                fail_total = r['Failed'] + r['Error']
                md.append(
                    f"| {r['Device']} | {r['Total']} | {r['Passed']} | {fail_total} "
                    f"| {nf} | {np_} | {r['Skipped']} | {r['XFAIL']} | {r['Pass Rate']} |"
                )
        md.append("")

    if not is_pr and (n_fail > 0 or n_pass > 0):
        net = n_pass - n_fail
        md.append(f"**Net Change:** {net:+d} tests ({n_fail} failures, {n_pass} passes)\n")

    if is_pr:
        # PR mode: expanded untracked failures & tracked passes, hide rest
        _write_untracked_section(md, untracked_fail, n_untracked, repo, collapsed=False)
        _write_tracked_section(md, tracked_pass, n_tracked, repo, collapsed=False)
        _write_file_summary(md, file_sum)
    else:
        # Non-PR mode: expanded new failures, hide untracked/tracked/passes/file-summary
        _write_new_failures_section(md, new_fail, n_fail, repo)
        _write_untracked_section(md, untracked_fail, n_untracked, repo, collapsed=True)
        _write_tracked_section(md, tracked_pass, n_tracked, repo, collapsed=True)
        _write_new_passes_section(md, new_pass, n_pass, repo, collapsed=True)
        _write_file_summary(md, file_sum)

    # ── GitHub issue stats ──
    if analyzer.issue_tracker:
        s = analyzer.issue_tracker.get_summary_stats()
        md.append("<details>")
        md.append(f"<summary><b>🏷️ GitHub Issues ({s['total_issues']} tracked)</b></summary>\n")
        md.append(f"- Open: {s['open_issues']}, Closed: {s['closed_issues']}")
        md.append(f"- Unique test cases tracked: {s['unique_test_cases']}")
        md.append("\n</details>\n")

    # Write file
    content = "\n".join(md)
    Path(filename).write_text(content, encoding='utf-8')
    log.info("Markdown written to %s", filename)


# ── File Output ───────────────────────────────────────────────────────
def export_excel(analyzer: ResultAnalyzer, path: Path) -> None:
    unique = analyzer.deduplicate()
    if unique.empty:
        log.warning("No data to export")
        return
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        stats = analyzer.device_summary(unique)
        if not stats.empty:
            stats.to_excel(w, sheet_name="Summary", index=False)
        bsl, tgt = analyzer.split_by_device(unique)
        if not bsl.empty or not tgt.empty:
            merged = analyzer.merge_results(bsl, tgt)
            fail, pass_ = analyzer.find_changes(merged)
            if not fail.empty:
                fail.to_excel(w, sheet_name="New failures", index=False)
            if not pass_.empty:
                pass_.to_excel(w, sheet_name="New passes", index=False)
            merged.to_excel(w, sheet_name="Comparison Details", index=False)
        fsum = analyzer.file_summary(unique)
        if not fsum.empty:
            fsum.to_excel(w, sheet_name="Files summary", index=False)
    log.info("Excel written to %s", path)


def export_csv(analyzer: ResultAnalyzer, path: Path) -> None:
    unique = analyzer.deduplicate()
    if unique.empty:
        log.warning("No data to export")
        return
    base = path.parent / path.stem
    bsl, tgt = analyzer.split_by_device(unique)
    if not bsl.empty or not tgt.empty:
        merged = analyzer.merge_results(bsl, tgt)
        merged.to_csv(f"{base}_comparison.csv", index=False)
        fail, pass_ = analyzer.find_changes(merged)
        if not fail.empty:
            fail.to_csv(f"{base}_new_failures.csv", index=False)
        if not pass_.empty:
            pass_.to_csv(f"{base}_new_passes.csv", index=False)
    fsum = analyzer.file_summary(unique)
    if not fsum.empty:
        fsum.to_csv(f"{base}_files_summary.csv", index=False)
    stats = analyzer.device_summary(unique)
    if not stats.empty:
        stats.to_csv(f"{base}_summary.csv", index=False)
    log.info("CSV files written to %s*", base)


def export_check_changes(
    analyzer: ResultAnalyzer, output_path: Path,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    unique = analyzer.deduplicate()
    bsl, tgt = analyzer.split_by_device(unique)
    if bsl.empty and tgt.empty:
        log.info("Skipping check-changes: no data")
        return None, None

    merged = analyzer.merge_results(bsl, tgt)

    def classify_row(row):
        for pattern, label in [
            ('test/regressions/', 'op_regression'),
            ('test/xpu/extended/', 'op_extended'),
            ('test/distributed/', 'xpu_distributed'),
        ]:
            if pattern in str(row.get('testfile_target', '')).lower() or \
               pattern in str(row.get('testfile_baseline', '')).lower():
                return label
        return 'op_ut'

    merged['raw_file_target'] = merged.apply(classify_row, axis=1)
    for col in ('issue_ids', 'issue_labels', 'issue_statuses', 'case_statuses'):
        if col not in merged.columns:
            merged[col] = ''

    has_open = lambda s: s.str.contains('open', case=False, na=False)

    untracked = merged[
        merged['status_target'].isin(['failed', 'error']) &
        ~has_open(merged['issue_statuses']) & ~has_open(merged['case_statuses'])
    ].copy()

    tracked_pass = merged[
        merged['status_target'].str.contains('pass', case=False, na=False) &
        has_open(merged['issue_statuses']) & has_open(merged['case_statuses'])
    ].copy()

    export_cols = [
        'raw_file_target', 'raw_class_target', 'raw_name_target',
        'status_target', 'time_target', 'message_target',
        'issue_ids', 'issue_labels', 'issue_statuses', 'case_statuses',
    ]

    def _export(df, suffix, sheet):
        if df.empty:
            return
        edf = df.reindex(columns=export_cols)
        if 'message_target' in edf.columns:
            edf['message_target'] = edf['message_target'].astype(str).str[:50]
        if output_path.suffix.lower() in ('.xlsx', '.xls'):
            if output_path.exists():
                with pd.ExcelWriter(output_path, engine='openpyxl', mode='a',
                                    if_sheet_exists='replace') as w:
                    edf.to_excel(w, sheet_name=sheet, index=False)
            else:
                with pd.ExcelWriter(output_path, engine='openpyxl') as w:
                    edf.to_excel(w, sheet_name=sheet, index=False)
        else:
            edf.to_csv(output_path.parent / f"{output_path.stem}{suffix}.csv", index=False)
        log.info("Exported %d rows (%s)", len(edf), sheet)

    _export(tracked_pass, "_tracked_passes", "Tracked Passes")
    _export(untracked, "_untracked_failures", "Untracked Failures")

    return untracked if not untracked.empty else None, tracked_pass if not tracked_pass.empty else None


# ── Console Output ────────────────────────────────────────────────────
def print_report(
    extractor: TestDetailsExtractor, analyzer: ResultAnalyzer,
    output_path: Path, md_path: str | None,
    n_fail: int, n_pass: int,
    untracked_n: int | None, tracked_n: int | None,
    elapsed: float,
) -> None:
    W = 64
    sep = "=" * W
    thin = "-" * W

    print(f"\n{sep}")
    print(f"{'UNIT TEST COMPARISON':^{W}}")
    print(sep)

    print(f"  {'Files processed:':<28} {extractor.stats['files_processed']:>6}")
    print(f"  {'Test cases found:':<28} {extractor.stats['test_cases_found']:>6}")
    print(f"  {'Time:':<28} {elapsed:>5.1f}s")
    print(thin)

    unique = analyzer.deduplicate()
    if not unique.empty:
        bsl_n = int((unique["device"] == "baseline").sum())
        tgt_n = int((unique["device"] == "target").sum())
        print(f"  {'Target tests:':<28} {tgt_n:>6}")
        print(f"  {'Baseline tests:':<28} {bsl_n:>6}")

        fsum = analyzer.file_summary(unique)
        if not fsum.empty:
            print(f"  {'Test files:':<28} {len(fsum):>6}")
        print(thin)

        if n_fail:
            print(f"  {'❌ New failures:':<28} {n_fail:>6}")
        if n_pass:
            print(f"  {'✅ New passes:':<28} {n_pass:>6}")
        if n_fail or n_pass:
            print(f"  {'Net change:':<28} {n_pass - n_fail:>+6}")

    if untracked_n is not None or tracked_n is not None:
        print(thin)
        if tracked_n:
            print(f"  {'✅ Tracked issues now pass:':<28} {tracked_n:>6}")
        if untracked_n:
            print(f"  {'⚠️  Untracked failures:':<28} {untracked_n:>6}")

    print(thin)
    print(f"  Output: {output_path}")
    if md_path:
        print(f"  Markdown: {md_path}")
    print(f"{sep}\n")


# ── Main ──────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(
        description="JUnit XML Test Comparison — Target vs Baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s -i target/ baseline/ -o comparison.xlsx
  %(prog)s -i results/*.xml -o comparison.csv -m
  %(prog)s -i target/ baseline/ -o out.xlsx --check-changes
  %(prog)s -i target/ baseline/ -o out.csv -m --testfile test_ops
""",
    )
    parser.add_argument("-i", "--input", nargs="+", required=True,
                        help="XML file paths, directories, or glob patterns")
    parser.add_argument("-o", "--output", default="test_comparison.xlsx",
                        help="Output file (.xlsx or .csv)")
    parser.add_argument("-m", "--markdown", action="store_true",
                        help="Generate markdown report")
    parser.add_argument("--markdown-output",
                        help="Markdown output path (default: {stem}_report.md)")
    parser.add_argument("-w", "--workers", type=int,
                        help="Parallel workers (default: CPU/2)")
    parser.add_argument("--testfile", nargs="*",
                        help="Filter to tests from specific file(s) (substring match)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")

    # GitHub
    gh = parser.add_argument_group("GitHub integration")
    gh.add_argument("--github-repo", default="intel/torch-xpu-ops")
    gh.add_argument("--github-token", help="GitHub token (or GH_TOKEN env)")
    gh.add_argument("--github-issue-state", default="open",
                    choices=["open", "closed", "all"])
    gh.add_argument("--github-labels", nargs="+", default=["skipped"])
    gh.add_argument("--no-github", action="store_true")
    gh.add_argument("--github-issue-cache", default="selected_issues.json")
    gh.add_argument("--refresh-issues", action="store_true")
    gh.add_argument("--check-changes", action="store_true",
                    help="Find untracked failures and tracked passes")
    gh.add_argument("--update-issues", action="store_true",
                    help="Cross out passed cases in GitHub issues and add comments")

    parser.add_argument("--event-type", default="",
                        help="GitHub event type (pull_request, schedule, etc.)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S",
    )

    try:
        t0 = time.time()

        # Extract
        extractor = TestDetailsExtractor()
        if not extractor.process(args.input, max_workers=args.workers):
            log.error("No test cases found")
            return 1

        log.info("Found %d test cases", len(extractor.test_cases))
        analyzer = ResultAnalyzer(extractor.test_cases)

        # GitHub
        if not args.no_github:
            repo = args.github_repo or os.environ.get('GITHUB_REPOSITORY')
            token = args.github_token or os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN')
            if repo:
                tracker = GitHubIssueTracker(
                    repo=repo, token=token, cache_path=args.github_issue_cache)
                if tracker.fetch_issues(state=args.github_issue_state,
                                        labels=args.github_labels,
                                        force_refresh=args.refresh_issues):
                    analyzer.set_issue_tracker(tracker)
                    log.info("GitHub issue tracking enabled")

        # Export main output
        output_path = Path(args.output)
        if output_path.suffix.lower() in (".xlsx", ".xls"):
            export_excel(analyzer, output_path)
        else:
            export_csv(analyzer, output_path)

        # Check changes (untracked failures / tracked passes)
        untracked_fail, tracked_pass = None, None
        if args.check_changes:
            untracked_fail, tracked_pass = export_check_changes(analyzer, output_path)

        # Find changes for report
        unique = analyzer.deduplicate()
        new_fail_df, new_pass_df = pd.DataFrame(), pd.DataFrame()
        bsl, tgt = analyzer.split_by_device(unique)
        if not bsl.empty and not tgt.empty:
            merged = analyzer.merge_results(bsl, tgt)

            # Apply testfile filter if requested
            if args.testfile:
                patterns = args.testfile
                mask = pd.Series(False, index=merged.index)
                for pat in patterns:
                    mask |= merged.get('testfile_target', pd.Series('', index=merged.index)).str.contains(pat, case=False, na=False)
                    mask |= merged.get('testfile_baseline', pd.Series('', index=merged.index)).str.contains(pat, case=False, na=False)
                merged = merged[mask]
                log.info("Filtered to %d tests matching %s", len(merged), patterns)

            new_fail_df, new_pass_df = analyzer.find_changes(merged)

        n_fail = len(new_fail_df)
        n_pass = len(new_pass_df)

        # Markdown
        md_path = None
        if args.markdown:
            if args.markdown_output:
                md_path = args.markdown_output
            else:
                md_path = str(output_path.parent / f"{output_path.stem}_report.md")
            write_markdown(
                analyzer,
                new_fail=new_fail_df if n_fail else None,
                new_pass=new_pass_df if n_pass else None,
                untracked_fail=untracked_fail,
                tracked_pass=tracked_pass,
                filename=md_path,
                event_type=args.event_type,
            )

        # Update GitHub issues (cross out passed cases, add comments)
        if args.update_issues and tracked_pass is not None and analyzer.issue_tracker:
            analyzer.issue_tracker.mark_passed_issues(tracked_pass)

        # Console report
        elapsed = time.time() - t0
        print_report(
            extractor, analyzer, output_path, md_path,
            n_fail, n_pass,
            len(untracked_fail) if untracked_fail is not None else None,
            len(tracked_pass) if tracked_pass is not None else None,
            elapsed,
        )

        return 0

    except KeyboardInterrupt:
        log.info("Interrupted")
        return 130
    except Exception as e:
        log.error("Unexpected error: %s", e)
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
