#!/usr/bin/env python3
"""
JUnit XML Test Details Extractor - Target/Baseline Comparison Tool

Compares test results between target and baseline with GitHub markdown reporting
and GitHub issue tracking integration.

Usage:
    python compare_tests.py --input "results/*.xml" --output comparison.xlsx
    python compare_tests.py --input file1.xml file2.xml --output comparison.csv --markdown
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import glob
import logging
import os
import re
import sys
import time
import json
from datetime import datetime
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

# Try to import PyGithub, but provide fallback if not available
try:
    from github import Github, Auth
    from github.Issue import Issue
    from github.Repository import Repository
    GITHUB_AVAILABLE = True
except ImportError:
    GITHUB_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("PyGithub not installed. GitHub integration disabled. Install with: pip install PyGithub")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================

class TestStatus(Enum):
    """Test status enumeration with priority for deduplication."""
    PASSED = "passed"
    XFAIL = "xfail"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, status_str: str) -> TestStatus:
        """Convert string to TestStatus enum."""
        if not status_str or pd.isna(status_str):
            return cls.UNKNOWN

        status_str = str(status_str).lower().strip()

        status_mapping = {
            "pass": cls.PASSED,
            "success": cls.PASSED,
            "xfail": cls.XFAIL,
            "fail": cls.FAILED,
            "error": cls.ERROR,
            "skip": cls.SKIPPED,
        }

        for key, status in status_mapping.items():
            if key in status_str:
                return status

        return cls.UNKNOWN

    @property
    def priority(self) -> int:
        """Priority for deduplication (higher = more important)."""
        priorities = {
            self.PASSED: 5,
            self.XFAIL: 4,
            self.FAILED: 3,
            self.ERROR: 2,
            self.SKIPPED: 1,
            self.UNKNOWN: 0,
        }
        return priorities[self]

    @property
    def emoji(self) -> str:
        """Get emoji for test status."""
        emojis = {
            self.PASSED: "✅",
            self.XFAIL: "⚠️",
            self.FAILED: "❌",
            self.ERROR: "💥",
            self.SKIPPED: "⏭️",
            self.UNKNOWN: "❓",
        }
        return emojis[self]

    @property
    def color(self) -> str:
        """Get color for test status (for markdown)."""
        colors = {
            self.PASSED: "green",
            self.XFAIL: "yellow",
            self.FAILED: "red",
            self.ERROR: "red",
            self.SKIPPED: "gray",
            self.UNKNOWN: "gray",
        }
        return colors[self]


class TestDevice(Enum):
    """Test device enumeration."""
    BASELINE = "baseline"
    TARGET = "target"
    UNKNOWN = "unknown"

    @classmethod
    def from_test_type(cls, test_type: str) -> TestDevice:
        """Extract device from test type string."""
        test_type_lower = test_type.lower()
        if "baseline" in test_type_lower:
            return cls.BASELINE
        elif "target" in test_type_lower:
            return cls.TARGET
        return cls.UNKNOWN

    @property
    def display_name(self) -> str:
        """Get display name for device."""
        return self.value.capitalize()


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclasses.dataclass(frozen=True)
class TestCase:
    """Immutable data class representing a single test case."""

    # Core identifiers
    uniqname: str
    testfile: str
    classname: str
    name: str

    # Test properties
    device: TestDevice
    testtype: str
    status: TestStatus
    time: float

    # Metadata
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        return {
            "uniqname": self.uniqname,
            "testfile": self.testfile,
            "classname": self.classname,
            "name": self.name,
            "device": self.device.value,  # 'baseline' or 'target'
            "testtype": self.testtype,
            "status": self.status.value,
            "time": float(self.time),
            "message": self.message,
        }


@dataclasses.dataclass
class Comparison:
    """Enhanced comparison data class with GitHub issue information."""

    # Original comparison data
    uniqname: str
    testfile_baseline: str = ""
    classname_baseline: str = ""
    name_baseline: str = ""
    testfile_target: str = ""
    classname_target: str = ""
    name_target: str = ""
    device_baseline: str = ""
    testtype_baseline: str = ""
    status_baseline: str = ""
    time_baseline: float = 0.0
    message_baseline: str = ""
    device_target: str = ""
    testtype_target: str = ""
    status_target: str = ""
    time_target: float = 0.0
    message_target: str = ""

    # GitHub issue fields
    issue_ids: str = ""  # Comma-separated issue IDs
    issue_labels: str = ""  # Comma-separated labels
    issue_statuses: str = ""  # Comma-separated statuses (open/closed)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return dataclasses.asdict(self)


# ============================================================================
# GITHUB ISSUE TRACKER (using PyGithub)
# ============================================================================
# ... (previous code remains the same until GitHubIssueTracker)

class GitHubIssueTracker:
    """Fetches and parses GitHub issues using PyGithub, with local caching."""

    CASES_PATTERN = re.compile(r'Cases:\s*\n(.*?)(?:\n\n|\Z)', re.DOTALL | re.IGNORECASE)
    TEST_CASE_SPLIT_PATTERN = re.compile(r'[\n]+')

    def __init__(self, repo: str = None, token: str = None, cache_path: str = None, pattern_matcher: Optional[FilePatternMatcher] = None):
        self.repo_name = repo or os.environ.get('GITHUB_REPOSITORY', '')
        self.token = token or os.environ.get('GITHUB_TOKEN', '')
        self.cache_path = Path(cache_path) if cache_path else None
        self.github = None
        self.repository = None
        self.issues_cache: Dict[int, Dict[str, Any]] = {}
        self.test_to_issues: Dict[str, List[Dict[str, Any]]] = {}
        self.pattern_matcher = pattern_matcher or FilePatternMatcher()

        if not GITHUB_AVAILABLE:
            logger.error("PyGithub is not installed. GitHub integration disabled.")
            return

    def load_cache(self) -> bool:
        """Load issues from local cache file."""
        if not self.cache_path or not self.cache_path.exists():
            return False

        try:
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.issues_cache = {int(k): v for k, v in data.get('issues_cache', {}).items()}
            self.test_to_issues = data.get('test_to_issues', {})
            logger.info(f"Loaded {len(self.issues_cache)} issues from cache: {self.cache_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load cache from {self.cache_path}: {e}")
            return False

    def save_cache(self) -> bool:
        """Save issues to local cache file."""
        if not self.cache_path:
            return False

        try:
            # Convert issues_cache keys to strings for JSON
            data = {
                'issues_cache': {str(k): v for k, v in self.issues_cache.items()},
                'test_to_issues': self.test_to_issues
            }
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self.issues_cache)} issues to cache: {self.cache_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to save cache to {self.cache_path}: {e}")
            return False

    def fetch_issues(self, state: str = 'all', labels: List[str] = None, force_refresh: bool = False) -> bool:
        """
        Fetch issues from GitHub repository, using cache if available and not forced refresh.

        Args:
            state: 'open', 'closed', or 'all'
            labels: List of labels to filter by
            force_refresh: If True, ignore cache and fetch fresh data

        Returns:
            True if successful (either from cache or fresh fetch), False otherwise
        """
        # Try cache first if not forced refresh
        if not force_refresh and self.load_cache():
            return True

        # Otherwise, fetch fresh from GitHub
        if not self.repository:
            if not self._init_github():
                return False

        logger.info(f"Fetching issues from {self.repo_name} (state={state})")

        try:
            github_state = 'all' if state == 'all' else state
            kwargs = {'state': github_state, 'direction': 'desc'}
            if labels:
                kwargs['labels'] = labels

            issues = self.repository.get_issues(**kwargs)

            # Clear any old data
            self.issues_cache.clear()
            self.test_to_issues.clear()

            issue_count = 0
            for issue in issues:
                if issue.pull_request:
                    continue
                self._parse_issue(issue)
                issue_count += 1

            logger.info(f"Fetched {issue_count} issues, found {len(self.test_to_issues)} test mappings")

            # Save to cache
            self.save_cache()
            return True

        except Exception as e:
            logger.error(f"Error fetching issues from GitHub: {e}")
            return False

    def _init_github(self) -> bool:
        """Initialize GitHub client and repository."""
        try:
            if self.token:
                auth = Auth.Token(self.token)
                self.github = Github(auth=auth)
            else:
                self.github = Github()

            self.repository = self.github.get_repo(self.repo_name)
            logger.info(f"Connected to GitHub repository: {self.repo_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize GitHub client: {e}")
            return False

    def _parse_issue(self, issue: Issue) -> None:
        """Parse a single GitHub issue and update caches."""
        issue_id = issue.number
        issue_body = issue.body or ''
        issue_labels = [label.name for label in issue.labels]
        issue_state = issue.state
        issue_title = issue.title

        self.issues_cache[issue_id] = {
            'id': issue_id,
            'title': issue_title,
            'state': issue_state,
            'labels': issue_labels,
            'url': issue.html_url,
            'created_at': issue.created_at.isoformat() if issue.created_at else '',
            'updated_at': issue.updated_at.isoformat() if issue.updated_at else '',
        }

        test_cases = self._extract_test_cases(issue_body)
        for test_case in test_cases:
            # Safely split the test case string
            parts = test_case.split(',')
            if len(parts) < 3:
                logger.debug(f"Issue #{issue_id}: test case '{test_case}' has fewer than 3 comma-separated parts, skipping")
                continue

            # Assume format: test_file, class_name, test_name (or similar)
            class_name_raw = parts[1].strip()
            test_name_raw = parts[2].strip()

            # Use pattern matcher to normalize
            test_file = self.pattern_matcher.extract_testfile(class_name_raw, None, None)
            test_class = self.pattern_matcher.extract_classname(class_name_raw)
            test_name = self.pattern_matcher.extract_casename(test_name_raw)
            uniqname = self.pattern_matcher.generate_uniqname(test_file, test_class, test_name)

            self.test_to_issues.setdefault(uniqname, []).append({
                'id': issue_id,
                'state': issue_state,
                'labels': issue_labels
            })

    def _extract_test_cases(self, body: str) -> List[str]:
        if not body:
            return []
        match = self.CASES_PATTERN.search(body)
        if not match:
            return []
        cases_text = match.group(1).strip()
        cases = self.TEST_CASE_SPLIT_PATTERN.split(cases_text)
        logger.debug(f"{cases}")
        return [case.strip() for case in cases if case.strip()]

    def find_issues_for_test(self, test_uniqname: str) -> List[Dict[str, Any]]:
        return self.test_to_issues.get(test_uniqname, [])

    def enhance_comparison(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        if merged_df.empty:
            return merged_df

        enhanced_df = merged_df.copy()
        enhanced_df['issue_ids'] = ''
        enhanced_df['issue_labels'] = ''
        enhanced_df['issue_statuses'] = ''

        for idx, row in enhanced_df.iterrows():
            uniqname = row.get('uniqname', '')
            if not uniqname:
                continue
            issues = self.find_issues_for_test(uniqname)
            if issues:
                logger.debug(f"{issues}")
                issue_ids = [str(issue['id']) for issue in issues]
                issue_statuses = [issue['state'] for issue in issues]
                all_labels = set()
                for issue in issues:
                    all_labels.update(issue['labels'])
                enhanced_df.at[idx, 'issue_ids'] = ','.join(issue_ids)
                enhanced_df.at[idx, 'issue_labels'] = ','.join(sorted(all_labels))
                enhanced_df.at[idx, 'issue_statuses'] = ','.join(issue_statuses)

        return enhanced_df

    def get_issue_summary_stats(self) -> Dict[str, Any]:
        """Compute summary statistics about issues and their association with test cases."""
        stats = {
            'total_issues': len(self.issues_cache),
            'open_issues': 0,
            'closed_issues': 0,
            'issues_with_test_cases': len(set(
                issue['id'] for mappings in self.test_to_issues.values() for issue in mappings
            )),
            'unique_test_cases': len(self.test_to_issues),
            'labels': {},
        }
        for issue in self.issues_cache.values():
            if issue['state'] == 'open':
                stats['open_issues'] += 1
            else:
                stats['closed_issues'] += 1
            for label in issue['labels']:
                stats['labels'][label] = stats['labels'].get(label, 0) + 1
        return stats


# ============================================================================
# FILE PATTERN MATCHER
# ============================================================================

class FilePatternMatcher:
    """Handles file pattern matching and normalization."""

    # Compiled regex patterns
    _CLASSNAME_PATTERN = re.compile(r".*\.")
    _CASENAME_PATTERN = re.compile(r"[^a-zA-Z0-9_.-]")
    _TESTFILE_PATTERN = re.compile(r".*torch-xpu-ops\.test\.")
    _TESTFILE_PATTERN_CPP = re.compile(r".*/test/xpu/")
    _NORMALIZE_PATTERN = re.compile(r".*\.\./test/")
    _GPU_PATTERN = re.compile(r"(?:xpu|cuda)", re.IGNORECASE)

    # Test type detection patterns
    TEST_TYPE_PATTERNS = {
        "xpu-target": [r"/target/"],
        "xpu-baseline": [r"/baseline/"],
    }

    # File replacements
    FILE_REPLACEMENTS = [
        ("test/test/", "test/"),
        ("test_c10d_xccl.py", "test_c10d_nccl.py"),
        ("test_c10d_ops_xccl.py", "test_c10d_ops_nccl.py"),
    ]

    def __init__(self):
        self._compiled_test_type_patterns = self._compile_patterns()

    def _compile_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Compile regex patterns."""
        return {
            test_type: [re.compile(pattern) for pattern in patterns]
            for test_type, patterns in self.TEST_TYPE_PATTERNS.items()
        }

    @lru_cache(maxsize=1024)
    def determine_test_type(self, xml_file: Path) -> str:
        """Determine test type based on XML file path."""
        xml_file_str = str(xml_file)

        for test_type, patterns in self._compiled_test_type_patterns.items():
            if any(pattern.search(xml_file_str) for pattern in patterns):
                return test_type

        return "unknown"

    @lru_cache(maxsize=1024)
    def normalize_filepath(self, filepath: str) -> str:
        """Normalize test file path."""
        if not filepath:
            return "unknown_file.py"

        normalized = filepath

        # Apply regex normalization
        if self._NORMALIZE_PATTERN.search(normalized):
            normalized = self._NORMALIZE_PATTERN.sub("test/", normalized)

        # Apply string replacements
        for old, new in self.FILE_REPLACEMENTS:
            if old in normalized:
                normalized = normalized.replace(old, new)

        normalized = normalized.replace("_xpu_xpu.py", ".py").replace("_xpu.py", ".py")
        normalized = re.sub(r'.*/jenkins/workspace/', '', normalized, flags=re.IGNORECASE)

        return normalized or "unknown_file.py"

    @lru_cache(maxsize=1024)
    def extract_testfile(self, classname: str, filename: str, xml_file: Path) -> str:
        """Extract and normalize test file path."""
        # Priority 1: Use filename from XML
        if filename:
            if filename.endswith(".cpp"):
                testfile = self._TESTFILE_PATTERN_CPP.sub("test/", filename)
            elif filename.endswith(".py"):
                testfile = f"test/{filename}"
            else:
                testfile = filename
        # Priority 2: Extract from classname
        elif classname:
            testfile = self._TESTFILE_PATTERN.sub("test/", classname).replace(".", "/")
            if "/" in testfile:
                testfile = f"{testfile.rsplit('/', 1)[0]}.py"
            else:
                testfile = f"{testfile}.py"
        # Priority 3: Extract from XML filename
        else:
            xml_file_str = str(xml_file)
            testfile = (
                re.sub(r".*op_ut_with_[a-zA-Z0-9]+\.", "test.", xml_file_str)
                .replace(".", "/")
                .replace("/py/xml", ".py")
                .replace("/xml", ".py")
            )

        return self.normalize_filepath(testfile)

    @lru_cache(maxsize=1024)
    def extract_classname(self, full_classname: str) -> str:
        """Extract simplified classname."""
        if not full_classname:
            return "UnknownClass"
        return self._CLASSNAME_PATTERN.sub("", full_classname)

    def extract_casename(self, casename: str) -> str:
        """Extract normalized test case name."""
        if not casename:
            return "unknown_name"
        return self._CASENAME_PATTERN.sub("", casename) or "error_name"

    @lru_cache(maxsize=2048)
    def generate_uniqname(self, filename: str, classname: str, name: str) -> str:
        """Generate unique identifier for test case."""
        combined = f"{filename}{classname}{name}"
        return self._GPU_PATTERN.sub("cuda", combined)


# ============================================================================
# TEST DETAILS EXTRACTOR
# ============================================================================

class TestDetailsExtractor:
    """Extracts test details from JUnit XML files."""

    def __init__(self, pattern_matcher: Optional[FilePatternMatcher] = None):
        self.pattern_matcher = pattern_matcher or FilePatternMatcher()
        self.test_cases: List[TestCase] = []
        self.stats = {
            "files_processed": 0,
            "test_cases_found": 0,
            "empty_files": 0,
            "failed_files": 0,
        }

    def _determine_test_status(self, testcase: ET.Element) -> Tuple[TestStatus, str]:
        """Determine test status and extract message."""
        # Check for failure
        failure = testcase.find("failure")
        if failure is not None:
            message = failure.get("message", "")
            if "pytest.xfail" in message:
                return TestStatus.XFAIL, message
            return TestStatus.FAILED, message

        # Check for skip
        skipped = testcase.find("skipped")
        if skipped is not None:
            message = skipped.get("message", "")
            skip_type = skipped.get("type", "")
            if "pytest.xfail" in skip_type or "pytest.xfail" in message:
                return TestStatus.XFAIL, message
            return TestStatus.SKIPPED, message

        # Check for error
        error = testcase.find("error")
        if error is not None:
            return TestStatus.ERROR, error.get("message", "")

        return TestStatus.PASSED, ""

    def _parse_testcase(self, testcase: ET.Element, xml_file: Path) -> Optional[TestCase]:
        """Parse a single testcase element."""
        try:
            classname = testcase.get("classname", "")
            filename = testcase.get("file", "")
            name = testcase.get("name", "")
            time_str = testcase.get("time", "0")

            # Determine test type and device
            test_type = self.pattern_matcher.determine_test_type(xml_file)
            device = TestDevice.from_test_type(test_type)

            # Skip if not BASELINE or TARGET
            if device == TestDevice.UNKNOWN:
                return None

            # Extract and normalize
            simplified_classname = self.pattern_matcher.extract_classname(classname)
            simplified_casename = self.pattern_matcher.extract_casename(name)
            testfile = self.pattern_matcher.extract_testfile(classname, filename, xml_file)

            # Generate unique identifier
            uniqname = self.pattern_matcher.generate_uniqname(
                testfile, simplified_classname, simplified_casename
            )

            # Determine status
            status, message = self._determine_test_status(testcase)

            # Convert time
            try:
                time_val = float(time_str)
            except (ValueError, TypeError):
                time_val = 0.0

            return TestCase(
                uniqname=uniqname,
                testfile=testfile,
                classname=simplified_classname,
                name=simplified_casename,
                device=device,
                testtype=test_type,
                status=status,
                time=time_val,
                message=message,
            )

        except Exception as e:
            logger.debug(f"Error parsing test case in {xml_file}: {e}")
            return None

    def process_xml(self, xml_file: Path) -> List[TestCase]:
        """Process a single XML file and return test cases."""
        try:
            test_cases = []

            # Use iterparse for memory efficiency
            for event, elem in ET.iterparse(xml_file, events=('end',)):
                if elem.tag == 'testcase':
                    test_case = self._parse_testcase(elem, xml_file)
                    if test_case:
                        test_cases.append(test_case)
                    elem.clear()

            return test_cases

        except Exception as e:
            logger.error(f"Error processing {xml_file}: {e}")
            self.stats["failed_files"] += 1
            return []

    def find_xml_files(self, input_paths: List[str]) -> List[Path]:
        """Find all XML files from input specifications."""
        xml_files: Set[Path] = set()

        for input_path in input_paths:
            path = Path(input_path).expanduser().resolve()

            if path.is_file() and path.suffix.lower() == ".xml":
                xml_files.add(path)
            elif path.is_dir():
                xml_files.update(path.rglob("*.xml"))
            else:
                for file_path in glob.glob(str(path), recursive=True):
                    file_path = Path(file_path)
                    if file_path.is_file() and file_path.suffix.lower() == ".xml":
                        xml_files.add(file_path.resolve())

        return sorted(xml_files)

    def process(self, input_paths: List[str], max_workers: int = None) -> bool:
        """Process all XML files in parallel."""
        if max_workers is None:
            max_workers = max(1, os.cpu_count() - 2)

        # Find XML files
        xml_files = self.find_xml_files(input_paths)

        if not xml_files:
            logger.error("No XML files found")
            return False

        logger.info(f"Found {len(xml_files)} XML files, using {max_workers} workers")

        # Process files in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self.process_xml, xml_file): xml_file
                for xml_file in xml_files
            }

            completed = 0
            for future in concurrent.futures.as_completed(future_to_file):
                xml_file = future_to_file[future]
                completed += 1

                # Log progress every 10 files or at completion
                if completed % 10 == 0 or completed == len(xml_files):
                    logger.info(f"Processed {completed}/{len(xml_files)} files")

                try:
                    test_cases = future.result()
                    if test_cases:
                        self.test_cases.extend(test_cases)
                        self.stats["test_cases_found"] += len(test_cases)
                    else:
                        self.stats["empty_files"] += 1
                except Exception as e:
                    logger.error(f"Error processing {xml_file}: {e}")
                    self.stats["failed_files"] += 1
                finally:
                    self.stats["files_processed"] += 1

        return len(self.test_cases) > 0


# ============================================================================
# RESULT ANALYZER
# ============================================================================

class ResultAnalyzer:
    """Analyze and compare test results between BASELINE and TARGET."""

    def __init__(self, test_cases: List[TestCase]):
        self.test_cases = test_cases
        self.df = self._create_dataframe()
        self.issue_tracker: Optional[GitHubIssueTracker] = None

    def set_issue_tracker(self, issue_tracker: GitHubIssueTracker):
        """Set GitHub issue tracker for enhanced analysis."""
        self.issue_tracker = issue_tracker

    def _create_dataframe(self) -> pd.DataFrame:
        """Create DataFrame from test cases."""
        if not self.test_cases:
            return pd.DataFrame()

        # Convert test cases to DataFrame
        data = [tc.to_dict() for tc in self.test_cases]
        return pd.DataFrame(data)

    def deduplicate_by_priority(self) -> pd.DataFrame:
        """
        Deduplicate test cases keeping highest priority status.
        Priority: PASSED > XFAIL > FAILED > ERROR > SKIPPED > UNKNOWN
        """
        if self.df.empty:
            return pd.DataFrame()

        # Add priority column
        df = self.df.copy()
        df["_priority"] = df["status"].apply(
            lambda x: TestStatus.from_string(x).priority
        )

        # Sort by priority (highest first) and deduplicate
        df_sorted = df.sort_values("_priority", ascending=False)
        result = df_sorted.drop_duplicates(subset=["device", "uniqname"], keep="first")

        return result.drop(columns=["_priority"]).reset_index(drop=True)

    def split_by_device(self, df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split DataFrame by device (BASELINE/TARGET)."""
        if df is None:
            df = self.df

        if df.empty or "device" not in df.columns:
            return pd.DataFrame(), pd.DataFrame()

        baseline_mask = df["device"] == "baseline"
        target_mask = df["device"] == "target"

        return df[baseline_mask].copy(), df[target_mask].copy()

    def merge_results(self, baseline_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
        """Merge BASELINE and TARGET results for comparison."""
        if baseline_df.empty and target_df.empty:
            return pd.DataFrame()

        # Prepare dataframes with consistent naming
        baseline_clean = baseline_df.add_suffix("_baseline").rename(columns={"uniqname_baseline": "uniqname"})
        target_clean = target_df.add_suffix("_target").rename(columns={"uniqname_target": "uniqname"})

        # Merge
        merged = pd.merge(
            baseline_clean,
            target_clean,
            on="uniqname",
            how="outer",
            suffixes=("", "_dup"),
        ).fillna("")

        # Select and order columns for easy comparison
        columns = [
            "uniqname",
            "testfile_baseline", "classname_baseline", "name_baseline",
            "testfile_target", "classname_target", "name_target",
            "device_baseline", "testtype_baseline", "status_baseline", "time_baseline", "message_baseline",
            "device_target", "testtype_target", "status_target", "time_target", "message_target",
        ]

        # Keep only existing columns
        existing_cols = [col for col in columns if col in merged.columns]
        merged_df = merged[existing_cols]

        # Enhance with GitHub issue information if available
        if self.issue_tracker:
            logger.info("Enhancing comparison with GitHub issue information")
            merged_df = self.issue_tracker.enhance_comparison(merged_df)

        return merged_df

    def find_target_changes(self, merged_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Find tests where status changed between BASELINE and TARGET.
        Returns two DataFrames: new_failures and new_passes.
        """
        if merged_df.empty:
            return pd.DataFrame(), pd.DataFrame()

        # Convert status strings to TestStatus for comparison
        merged_df = merged_df.copy()

        # Define conditions for new failures (baseline passed, target didn't pass)
        baseline_passed = merged_df["status_baseline"].isin(["passed", "xfail"])
        target_not_passed = ~merged_df["status_target"].isin(["passed", "xfail"])
        new_failures = merged_df[baseline_passed & target_not_passed].copy()

        # Define conditions for new passes (baseline didn't pass, target passed)
        baseline_not_passed = ~merged_df["status_baseline"].isin(["passed", "xfail"])
        target_passed = merged_df["status_target"].isin(["passed", "xfail"])
        new_passes = merged_df[baseline_not_passed & target_passed].copy()

        if not new_failures.empty:
            # Add reason column for new failures
            new_failures["change_type"] = "failure"
            new_failures["reason"] = np.select(
                [
                    new_failures["status_target"].isin(["skipped"]),
                    new_failures["status_target"].isin(["failed"]),
                    new_failures["status_target"].isin(["error"]),
                ],
                [
                    "Skipped on Target",
                    "Failed on Target",
                    "Error on Target",
                ],
                default="Unknown issue"
            )

        if not new_passes.empty:
            # Add reason column for new passes
            new_passes["change_type"] = "pass"
            new_passes["reason"] = np.select(
                [
                    new_passes["status_baseline"].isin(["skipped"]),
                    new_passes["status_baseline"].isin(["failed"]),
                    new_passes["status_baseline"].isin(["error"]),
                ],
                [
                    "Was skipped on Baseline",
                    "Was failing on Baseline",
                    "Was error on Baseline",
                ],
                default="Now passing on Target"
            )

        return new_failures, new_passes

    def generate_file_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summary statistics grouped by testfile.
        Shows comparison between baseline and target for each test file.
        """
        if df.empty:
            return pd.DataFrame()

        # Split by device first
        baseline_df, target_df = self.split_by_device(df)

        if baseline_df.empty or target_df.empty:
            return pd.DataFrame()

        # Get unique test files from both
        all_testfiles = sorted(set(baseline_df["testfile"].unique()) | set(target_df["testfile"].unique()))

        file_summaries = []

        for testfile in all_testfiles:
            # Get data for this test file
            baseline_file_df = baseline_df[baseline_df["testfile"] == testfile]
            target_file_df = target_df[target_df["testfile"] == testfile]

            summary = {
                "Test File": testfile,

                # Baseline stats
                "Baseline Total": len(baseline_file_df),
                "Baseline Passed": len(baseline_file_df[baseline_file_df["status"].isin(["passed", "xfail"])]),
                "Baseline Failed": len(baseline_file_df[baseline_file_df["status"] == "failed"]),
                "Baseline Error": len(baseline_file_df[baseline_file_df["status"] == "error"]),
                "Baseline Skipped": len(baseline_file_df[baseline_file_df["status"] == "skipped"]),

                # Target stats
                "Target Total": len(target_file_df),
                "Target Passed": len(target_file_df[target_file_df["status"].isin(["passed", "xfail"])]),
                "Target Failed": len(target_file_df[target_file_df["status"] == "failed"]),
                "Target Error": len(target_file_df[target_file_df["status"] == "error"]),
                "Target Skipped": len(target_file_df[target_file_df["status"] == "skipped"]),
            }

            # Calculate pass rates (excluding failed and error)
            baseline_total = summary["Baseline Total"]
            if baseline_total > 0:
                baseline_passed_count = summary["Baseline Passed"]
                summary["Baseline Pass Rate"] = f"{(baseline_passed_count / baseline_total * 100):.2f}%"
            else:
                summary["Baseline Pass Rate"] = "N/A"

            target_total = summary["Target Total"]
            if target_total > 0:
                target_passed_count = summary["Target Passed"]
                summary["Target Pass Rate"] = f"{(target_passed_count / target_total * 100):.2f}%"
            else:
                summary["Target Pass Rate"] = "N/A"

            # Calculate delta (Target - Baseline)
            if baseline_total > 0 and target_total > 0:
                baseline_passed_count = summary["Baseline Passed"]
                target_passed_count = summary["Target Passed"]
                baseline_passed_pct = baseline_passed_count / baseline_total * 100
                target_passed_pct = target_passed_count / target_total * 100
                summary["Pass Rate Delta"] = f"{(target_passed_pct - baseline_passed_pct):+.2f}%"
            else:
                summary["Pass Rate Delta"] = "N/A"

            file_summaries.append(summary)

        return pd.DataFrame(file_summaries)

    def generate_summary_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate summary statistics by device."""
        if df.empty:
            return pd.DataFrame()

        stats = []

        for device_value in ["baseline", "target"]:
            device_df = df[df["device"] == device_value]

            if device_df.empty:
                continue

            total = len(device_df)
            failed = len(device_df[device_df["status"] == "failed"])
            error = len(device_df[device_df["status"] == "error"])

            # Pass rate = (total - failed - error) / total
            passed_count = total - failed - error
            pass_rate = (passed_count / total * 100) if total > 0 else 0

            device_name = "Baseline" if device_value == "baseline" else "Target"

            stats.append({
                "Device": device_name,
                "Total": total,
                "Passed": passed_count,
                "Failed": failed,
                "Error": error,
                "Skipped": len(device_df[device_df["status"] == "skipped"]),
                "XFAIL": len(device_df[device_df["status"] == "xfail"]),
                "Pass Rate": f"{pass_rate:.2f}%",
            })

        return pd.DataFrame(stats)

    def generate_issue_summary_section(self) -> str:
        """Generate markdown section for GitHub issue summary."""
        if not self.issue_tracker:
            return ""

        stats = self.issue_tracker.get_issue_summary_stats()

        md = []
        md.append("## 🏷️ GitHub Issues Summary\n")

        md.append(f"- **Total Issues:** {stats['total_issues']}")
        md.append(f"  - 🔓 Open: {stats['open_issues']}")
        md.append(f"  - 🔒 Closed: {stats['closed_issues']}")
        md.append(f"- **Issues with Test Cases:** {stats['issues_with_test_cases']}")
        md.append(f"- **Unique Test Cases Tracked:** {stats['unique_test_cases']}")

        if stats['labels']:
            # Show top 5 labels
            top_labels = sorted(stats['labels'].items(), key=lambda x: x[1], reverse=True)[:5]
            md.append("\n**Top Labels:**")
            for label, count in top_labels:
                md.append(f"  - `{label}`: {count} issues")

        md.append("")
        return "\n".join(md)

    def generate_markdown_summary(self, df: pd.DataFrame, new_failures_df: pd.DataFrame = None, new_passes_df: pd.DataFrame = None) -> str:
        """
        Generate a comprehensive markdown summary for GitHub issues.
        Includes both new failures and new passes.
        """
        if df.empty:
            return "# Test Comparison Report\n\nNo test data available."

        # Get summary stats
        stats_df = self.generate_summary_stats(df)
        file_summary_df = self.generate_file_summary(df)

        # Get current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Start building markdown
        md = []

        # Header
        md.append("# Test Comparison Report\n")
        md.append(f"**Generated:** {timestamp}\n")

        # GitHub Issues Summary (if available)
        issue_summary = self.generate_issue_summary_section()
        if issue_summary:
            md.append(issue_summary)

        # Overall summary section
        md.append("## 📊 Overall Summary\n")

        if not stats_df.empty:
            # Create a nice table for overall stats
            md.append("| Device | Total | ✅ Passed | ❌ Failed | 💥 Error | ⏭️ Skipped | ⚠️ XFAIL | 📈 Pass Rate |")
            md.append("|--------|-------|----------|----------|---------|-----------|---------|--------------|")

            for _, row in stats_df.iterrows():
                md.append(
                    f"| {row['Device']} | {row['Total']} | {row['Passed']} | "
                    f"{row['Failed']} | {row['Error']} | {row['Skipped']} | "
                    f"{row['XFAIL']} | {row['Pass Rate']} |"
                )
            md.append("")

        # Comparison metrics
        if len(stats_df) == 2:
            baseline_row = stats_df[stats_df['Device'] == 'Baseline'].iloc[0]
            target_row = stats_df[stats_df['Device'] == 'Target'].iloc[0]

            total_delta = target_row['Total'] - baseline_row['Total']
            pass_rate_delta = float(target_row['Pass Rate'].rstrip('%')) - float(baseline_row['Pass Rate'].rstrip('%'))

            delta_emoji = "✅" if pass_rate_delta >= 0 else "❌"

            md.append("### 🔄 Comparison Metrics\n")
            md.append(f"- **Test Count Delta:** {total_delta:+.0f} tests")
            md.append(f"- **Pass Rate Delta:** {delta_emoji} {pass_rate_delta:+.2f}%\n")

        # Summary of changes
        total_new_failures = len(new_failures_df) if new_failures_df is not None else 0
        total_new_passes = len(new_passes_df) if new_passes_df is not None else 0

        md.append("### 📊 Change Summary\n")
        md.append(f"- **New Failures:** {total_new_failures} tests")
        md.append(f"- **New Passes:** {total_new_passes} tests")
        md.append(f"- **Net Change:** {total_new_passes - total_new_failures:+.0f} tests\n")

        # New Failures Section
        md.append("## 🚨 New Failures on Target\n")

        if new_failures_df is not None and not new_failures_df.empty:
            md.append(f"Found **{len(new_failures_df)}** tests that passed on Baseline but failed on Target:\n")

            # Group by reason for better organization
            for reason in new_failures_df['reason'].unique():
                reason_issues = new_failures_df[new_failures_df['reason'] == reason]
                reason_emoji = "❌" if "Failed" in reason else "💥" if "Error" in reason else "⏭️"

                md.append(f"### {reason_emoji} {reason} ({len(reason_issues)})\n")

                # Create table for this category
                md.append("| Test File | Test Name | Status | Issue IDs | Message |")
                md.append("|-----------|-----------|--------|-----------|---------|")

                for _, issue in reason_issues.head(10).iterrows():  # Limit to 10 per category
                    test_file = issue.get('testfile_target', issue.get('testfile_baseline', 'Unknown'))
                    test_name = issue.get('name_target', issue.get('name_baseline', 'Unknown'))
                    status = issue['status_target']

                    # Get status emoji
                    status_emoji = TestStatus.from_string(status).emoji

                    # Get issue IDs
                    issue_ids = issue.get('issue_ids', '')
                    issue_display = ""
                    if issue_ids and self.issue_tracker and self.issue_tracker.repo_name:
                        # Format as links if we have issue IDs and repo
                        ids = issue_ids.split(',')
                        issue_links = [f"[#{id}](https://github.com/{self.issue_tracker.repo_name}/issues/{id})" for id in ids]
                        issue_display = ', '.join(issue_links)
                    elif issue_ids:
                        issue_display = issue_ids

                    # Truncate message if too long
                    message = issue.get('message_target', '')
                    if len(message) > 100:
                        message = message[:97] + "..."
                    message = message.replace('\n', ' ').replace('|', '\\|')

                    md.append(f"| `{test_file}` | `{test_name}` | {status_emoji} {status} | {issue_display} | {message} |")

                if len(reason_issues) > 10:
                    md.append(f"| ... | ... | ... | ... | *{len(reason_issues) - 10} more issues* |")

                md.append("")
        else:
            md.append("✅ **No new failures found!** All tests that passed on Baseline also pass on Target.\n")

        # New Passes Section
        md.append("## ✨ New Passes on Target\n")

        if new_passes_df is not None and not new_passes_df.empty:
            md.append(f"Found **{len(new_passes_df)}** tests that now pass on Target (were failing/skipped on Baseline):\n")

            # Group by reason for better organization
            for reason in new_passes_df['reason'].unique():
                reason_passes = new_passes_df[new_passes_df['reason'] == reason]
                reason_emoji = "✅"

                md.append(f"### {reason_emoji} {reason} ({len(reason_passes)})\n")

                # Create table for this category
                md.append("| Test File | Test Name | Baseline Status | Issue IDs |")
                md.append("|-----------|-----------|-----------------|-----------|")

                for _, issue in reason_passes.head(10).iterrows():  # Limit to 10 per category
                    test_file = issue.get('testfile_target', issue.get('testfile_baseline', 'Unknown'))
                    test_name = issue.get('name_target', issue.get('name_baseline', 'Unknown'))
                    baseline_status = issue['status_baseline']

                    # Get status emoji
                    status_emoji = TestStatus.from_string(baseline_status).emoji

                    # Get issue IDs
                    issue_ids = issue.get('issue_ids', '')
                    issue_display = ""
                    if issue_ids and self.issue_tracker and self.issue_tracker.repo_name:
                        ids = issue_ids.split(',')
                        issue_links = [f"[#{id}](https://github.com/{self.issue_tracker.repo_name}/issues/{id})" for id in ids]
                        issue_display = ', '.join(issue_links)
                    elif issue_ids:
                        issue_display = issue_ids

                    md.append(f"| `{test_file}` | `{test_name}` | {status_emoji} {baseline_status} | {issue_display} |")

                if len(reason_passes) > 10:
                    md.append(f"| ... | ... | ... | *{len(reason_passes) - 10} more passes* |")

                md.append("")
        else:
            md.append("No new passes detected.\n")

        # File-level summary
        md.append("## 📁 File-Level Summary\n")

        if not file_summary_df.empty:
            # Create a copy for sorting
            file_summary_sorted = file_summary_df.copy()

            # Calculate failure score for sorting (Baseline Failed + Baseline Error)
            file_summary_sorted['Baseline Failures'] = (
                file_summary_sorted['Baseline Failed'].astype(int) +
                file_summary_sorted['Baseline Error'].astype(int)
            )

            # Sort by:
            # 1. Baseline Failures (descending) - files with most baseline failures first
            # 2. Baseline Total (descending) - then by total tests
            file_summary_sorted = file_summary_sorted.sort_values(
                by=['Baseline Failures', 'Baseline Total'],
                ascending=[False, False]
            )

            # Calculate delta numeric for display
            file_summary_sorted['Delta Numeric'] = file_summary_sorted['Pass Rate Delta'].apply(
                lambda x: float(x.rstrip('%')) if x != 'N/A' and x != 'N/A%' else 0
            )

            md.append("| Test File | Baseline Stats | Target Stats | Delta | Details |")
            md.append("|-----------|---------------|--------------|-------|---------|")

            for _, row in file_summary_sorted.head(10).iterrows():  # Limit to 10 files
                test_file = row['Test File']
                baseline_rate = row['Baseline Pass Rate']
                target_rate = row['Target Pass Rate']
                delta = row['Pass Rate Delta']

                # Add emoji based on delta
                delta_emoji = ""
                if delta != 'N/A':
                    delta_val = float(delta.rstrip('%'))
                    if delta_val < -5:
                        delta_emoji = "🔻"
                    elif delta_val > 5:
                        delta_emoji = "🔺"

                # Create baseline stats string with failures highlighted
                baseline_failures = int(row['Baseline Failed']) + int(row['Baseline Error'])
                baseline_stats = f"✅:{row['Baseline Passed']}"
                if baseline_failures > 0:
                    baseline_stats += f" ❌:{row['Baseline Failed']} 💥:{row['Baseline Error']}"
                baseline_stats += f" ⏭️:{row['Baseline Skipped']}"

                # Create target stats string
                target_failures = int(row['Target Failed']) + int(row['Target Error'])
                target_stats = f"✅:{row['Target Passed']}"
                if target_failures > 0:
                    target_stats += f" ❌:{row['Target Failed']} 💥:{row['Target Error']}"
                target_stats += f" ⏭️:{row['Target Skipped']}"

                # Create details string
                details = f"Total: {row['Baseline Total']} tests"

                md.append(
                    f"| `{test_file}` | {baseline_stats} | {target_stats} | {delta_emoji} {delta} | {details} |"
                )

            if len(file_summary_df) > 10:
                md.append(f"| ... | ... | ... | ... | *{len(file_summary_df) - 10} more files* |")
            md.append("")

            # Add a note about the sorting
            md.append("> 📝 *Files are sorted by Baseline Failures (Failed + Error) descending, then by total test count.*\n")
        else:
            md.append("No file-level summary available.\n")

        # Top failures section
        md.append("## 🔥 Top Failures by File\n")

        if not file_summary_df.empty:
            # Find files with most failures on target
            target_failures = file_summary_df.nlargest(5, 'Target Failed')[['Test File', 'Target Failed', 'Target Error']]

            if not target_failures.empty and (target_failures['Target Failed'].sum() > 0 or target_failures['Target Error'].sum() > 0):
                md.append("| Test File | Failed | Error |")
                md.append("|-----------|--------|-------|")

                for _, row in target_failures.iterrows():
                    if row['Target Failed'] > 0 or row['Target Error'] > 0:
                        md.append(f"| `{row['Test File']}` | {row['Target Failed']} | {row['Target Error']} |")
                md.append("")
            else:
                md.append("No failures found on Target.\n")
        else:
            md.append("No failure data available.\n")

        # Top improvements section
        md.append("## 📈 Top Improvements by File\n")

        if not file_summary_df.empty and new_passes_df is not None and not new_passes_df.empty:
            # Find files with most new passes
            if not new_passes_df.empty:
                new_passes_by_file = new_passes_df.groupby('testfile_target').size().reset_index(name='new_passes_count')
                new_passes_by_file = new_passes_by_file.sort_values('new_passes_count', ascending=False).head(5)

                md.append("| Test File | New Passes |")
                md.append("|-----------|------------|")

                for _, row in new_passes_by_file.iterrows():
                    md.append(f"| `{row['testfile_target']}` | {row['new_passes_count']} |")
                md.append("")
        else:
            md.append("No significant improvements detected.\n")

        # Recommendations
        md.append("## 💡 Recommendations\n")

        if new_failures_df is not None and not new_failures_df.empty:
            md.append("Based on the analysis, here are some recommendations:\n")

            md.append("1. **🔥 Focus on new failures first** - The {} new failures should be investigated urgently".format(len(new_failures_df)))
            md.append("2. **Check error messages** - Review the error messages for patterns in the new failures")
            md.append("3. **Verify test environment** - Ensure Target environment is properly configured")
            md.append("4. **Review recent changes** - Changes between Baseline and Target may have introduced these issues")

            # Add recommendation based on baseline failures
            if not file_summary_df.empty:
                high_failure_files = file_summary_df[
                    (file_summary_df['Baseline Failed'].astype(int) + file_summary_df['Baseline Error'].astype(int)) > 10
                ]
                if not high_failure_files.empty:
                    md.append(f"5. **Address baseline failures** - {len(high_failure_files)} files have >10 failures in baseline")

            if new_passes_df is not None and not new_passes_df.empty:
                md.append(f"6. **✨ Celebrate improvements** - {len(new_passes_df)} tests are now passing on Target!")
        else:
            md.append("✅ All tests are passing! No immediate action required.\n")

            if new_passes_df is not None and not new_passes_df.empty:
                md.append(f"✨ **Note:** {len(new_passes_df)} tests that were previously failing are now passing on Target!\n")

        # Footer with instructions
        md.append("---\n")
        md.append("*This report was automatically generated by the Test Comparison Tool.*")
        md.append("*For more details, check the attached Excel/CSV files.*")

        return "\n".join(md)


# ============================================================================
# REPORT EXPORTER
# ============================================================================

class ReportExporter:
    """Export comparison results to various formats."""

    def __init__(self, markdown_output: Optional[Path] = None):
        self.markdown_output = markdown_output

    def export_excel(self, analyzer: ResultAnalyzer, output_path: Path) -> None:
        """Export results to Excel with comparison sheets."""
        # Get unique test cases
        unique_df = analyzer.deduplicate_by_priority()

        if unique_df.empty:
            logger.warning("No data to export")
            return

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            # Split and compare
            baseline_df, target_df = analyzer.split_by_device(unique_df)

            if not baseline_df.empty and not target_df.empty:
                # Merge results (includes GitHub issue info if available)
                merged_df = analyzer.merge_results(baseline_df, target_df)

                # Write comparison
                merged_df.to_excel(writer, sheet_name="Comparison", index=False)

                # Find target changes (New fail/pass)
                new_failures_df, new_passes_df = analyzer.find_target_changes(merged_df)

                if not new_failures_df.empty:
                    new_failures_df.to_excel(writer, sheet_name="New failures", index=False)

                if not new_passes_df.empty:
                    new_passes_df.to_excel(writer, sheet_name="New passes", index=False)

            # Generate and write file summary
            file_summary_df = analyzer.generate_file_summary(unique_df)
            if not file_summary_df.empty:
                file_summary_df.to_excel(writer, sheet_name="Files summary", index=False)

            # Add summary statistics
            stats_df = analyzer.generate_summary_stats(unique_df)
            if not stats_df.empty:
                stats_df.to_excel(writer, sheet_name="Summary", index=False)

            # Export case-to-issue mapping if available
            case_issue_df = self._generate_case_issue_df(analyzer)
            if not case_issue_df.empty:
                case_issue_df.to_excel(writer, sheet_name="Case to issue", index=False)
                logger.info("Exported case-to-issue mapping to sheet 'Case to issue'")

        logger.info(f"Exported comparison results to {output_path}")

    def export_csv(self, analyzer: ResultAnalyzer, output_path: Path) -> None:
        """Export results to CSV files."""
        # Get unique test cases
        unique_df = analyzer.deduplicate_by_priority()

        if unique_df.empty:
            logger.warning("No data to export")
            return

        base_path = output_path.parent / output_path.stem

        # Split and compare
        baseline_df, target_df = analyzer.split_by_device(unique_df)

        if not baseline_df.empty and not target_df.empty:
            # Merge results (includes GitHub issue info if available)
            merged_df = analyzer.merge_results(baseline_df, target_df)

            # Save comparison
            merged_df.to_csv(f"{base_path}_comparison.csv", index=False)

            # Find and save target changes (New fail/pass)
            new_failures_df, new_passes_df = analyzer.find_target_changes(merged_df)

            if not new_failures_df.empty:
                new_failures_df.to_csv(f"{base_path}_new_failures.csv", index=False)

            if not new_passes_df.empty:
                new_passes_df.to_csv(f"{base_path}_new_passes.csv", index=False)

        # Generate and save file summary
        file_summary_df = analyzer.generate_file_summary(unique_df)
        if not file_summary_df.empty:
            file_summary_df.to_csv(f"{base_path}_files_summary.csv", index=False)

        # Save summary
        stats_df = analyzer.generate_summary_stats(unique_df)
        if not stats_df.empty:
            stats_df.to_csv(f"{base_path}_summary.csv", index=False)

        # Export case-to-issue mapping if available
        case_issue_df = self._generate_case_issue_df(analyzer)
        if not case_issue_df.empty:
            case_issue_path = f"{base_path}_case_to_issue.csv"
            case_issue_df.to_csv(case_issue_path, index=False)
            logger.info(f"Exported case-to-issue mapping to {case_issue_path}")

        logger.info(f"Exported comparison results to {output_path}")

    def export_markdown(self, analyzer: ResultAnalyzer, output_path: Path) -> None:
        """Export results to Markdown format for GitHub issues."""
        # Get unique test cases
        unique_df = analyzer.deduplicate_by_priority()

        if unique_df.empty:
            logger.warning("No data to export to markdown")
            return

        # Get changes for the markdown report
        baseline_df, target_df = analyzer.split_by_device(unique_df)
        new_failures_df = None
        new_passes_df = None

        if not baseline_df.empty and not target_df.empty:
            merged_df = analyzer.merge_results(baseline_df, target_df)
            new_failures_df, new_passes_df = analyzer.find_target_changes(merged_df)

        # Generate markdown content
        markdown_content = analyzer.generate_markdown_summary(unique_df, new_failures_df, new_passes_df)

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        logger.info(f"Exported markdown report to {output_path}")

    def _generate_case_issue_df(self, analyzer: ResultAnalyzer) -> pd.DataFrame:
        """Create a DataFrame from test_to_issues mapping."""
        if not analyzer.issue_tracker or not analyzer.issue_tracker.test_to_issues:
            return pd.DataFrame()

        tracker = analyzer.issue_tracker
        rows = []
        repo_name = tracker.repo_name

        for uniqname, issues in tracker.test_to_issues.items():
            # Collect information for all issues of this test
            issue_ids = [str(issue['id']) for issue in issues]
            issue_states = [issue['state'] for issue in issues]
            # Labels are stored as a list per issue; join them for display
            issue_labels_list = [', '.join(issue.get('labels', [])) for issue in issues]

            # Combine multiple issues into readable strings
            combined_ids = ', '.join(issue_ids)
            combined_states = ', '.join(issue_states)
            combined_labels = ' | '.join(issue_labels_list)  # separate issues with a pipe

            # Build GitHub URLs if repo name is known
            if repo_name:
                urls = [f"https://github.com/{repo_name}/issues/{id}" for id in issue_ids]
                combined_urls = ', '.join(urls)
            else:
                combined_urls = ''

            rows.append({
                'Test Case (uniqname)': uniqname,
                'Issue IDs': combined_ids,
                'Issue States': combined_states,
                'Issue Labels': combined_labels,
                'Issue URLs': combined_urls,
            })

        return pd.DataFrame(rows)

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="JUnit XML Test Details Extractor - Target/Baseline Comparison with GitHub Issue Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-i", "--input",
        nargs="+",
        required=True,
        help="XML file paths, directories, or glob patterns",
    )

    parser.add_argument(
        "-o", "--output",
        default="test_comparison.xlsx",
        help="Output file path (.xlsx or .csv)",
    )

    parser.add_argument(
        "-m", "--markdown",
        action="store_true",
        help="Generate a markdown summary file for GitHub issues",
    )

    parser.add_argument(
        "--markdown-output",
        help="Output path for markdown file (default: {output_stem}_report.md)",
    )

    parser.add_argument(
        "-w", "--workers",
        type=int,
        help="Number of parallel workers (default: CPU count - 2)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    # GitHub integration arguments
    parser.add_argument(
        "--github-repo",
        default="intel/torch-xpu-ops",
        help="GitHub repository in format 'owner/repo' (default: from GITHUB_REPOSITORY env)",
    )

    parser.add_argument(
        "--github-token",
        help="GitHub personal access token (default: from GITHUB_TOKEN env)",
    )

    parser.add_argument(
        "--github-issue-state",
        default="open",
        choices=["open", "closed", "all"],
        help="State of issues to fetch (default: all)",
    )

    parser.add_argument(
        "--github-labels",
        nargs="+",
        default="skipped",
        help="Filter issues by labels",
    )

    parser.add_argument(
        "--no-github",
        action="store_true",
        help="Disable GitHub integration even if credentials are available",
    )

    parser.add_argument(
        "--github-issue-cache",
        default="selected_issues.json",
        help="Path to cache file for GitHub issues (default: selected_issues.json)",
    )

    parser.add_argument(
        "--refresh-issues",
        action="store_true",
        help="Force refresh GitHub issues even if cache exists",
    )
    args = parser.parse_args()

    # Configure logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        start_time = time.time()

        # Set default workers if not specified
        if args.workers is None:
            args.workers = max(1, os.cpu_count() - 2)
            logger.info(f"Using {args.workers} workers (CPU count - 2)")

        # Initialize extractor
        extractor = TestDetailsExtractor()

        # Process files
        logger.info("Starting test extraction...")
        success = extractor.process(args.input, max_workers=args.workers)

        if not success:
            logger.error("No test cases found")
            return 1

        # Initialize analyzer
        logger.info(f"Found {len(extractor.test_cases)} test cases")
        analyzer = ResultAnalyzer(extractor.test_cases)

        # Initialize and fetch GitHub issues if not disabled
        if not args.no_github:
            github_repo = args.github_repo or os.environ.get('GITHUB_REPOSITORY')
            github_token = args.github_token or os.environ.get('GITHUB_TOKEN')

            if github_repo:
                logger.info(f"Initializing GitHub issue tracker for {github_repo}")
                issue_tracker = GitHubIssueTracker(
                    repo=github_repo,
                    token=github_token,
                    cache_path=args.github_issue_cache
                )

                # Fetch issues (using cache if available and not refreshing)
                if issue_tracker.fetch_issues(
                    state=args.github_issue_state,
                    labels=args.github_labels,
                    force_refresh=args.refresh_issues
                ):
                    analyzer.set_issue_tracker(issue_tracker)
                    logger.info("GitHub issue tracking enabled")
                else:
                    logger.warning("Failed to fetch GitHub issues, continuing without issue tracking")
            else:
                logger.info("GitHub repository not configured, skipping issue tracking")

        # Export results
        output_path = Path(args.output)
        exporter = ReportExporter()

        # Export main format (Excel/CSV)
        if output_path.suffix.lower() in [".xlsx", ".xls"]:
            exporter.export_excel(analyzer, output_path)
        else:
            exporter.export_csv(analyzer, output_path)

        # Export markdown if requested
        if args.markdown:
            # Determine markdown output path
            if args.markdown_output:
                markdown_path = Path(args.markdown_output)
            else:
                # Default: same as output but with _report.md suffix
                markdown_path = output_path.parent / f"{output_path.stem}_report.md"

            exporter.export_markdown(analyzer, markdown_path)

        # Print summary
        elapsed = time.time() - start_time

        print("\n" + "=" * 60)
        print("COMPARISON COMPLETE")
        print("=" * 60)
        print(f"📊 Files processed: {extractor.stats['files_processed']}")
        print(f"🧪 Test cases found: {extractor.stats['test_cases_found']}")
        print(f"⏱️  Time: {elapsed:.2f}s")
        print(f"📁 Output: {output_path}")

        if args.markdown:
            if args.markdown_output:
                print(f"📝 Markdown report: {markdown_path}")
            else:
                print(f"📝 Markdown report: {output_path.parent / f'{output_path.stem}_report.md'}")

        unique_df = analyzer.deduplicate_by_priority()
        if not unique_df.empty:
            baseline_count = len(unique_df[unique_df["device"] == "baseline"])
            target_count = len(unique_df[unique_df["device"] == "target"])

            print(f"📱 Baseline tests: {baseline_count}, Target tests: {target_count}")

            # Show file summary stats
            file_summary = analyzer.generate_file_summary(unique_df)
            if not file_summary.empty:
                print(f"📂 Test files: {len(file_summary)}")

            # Show changes summary
            baseline_df, target_df = analyzer.split_by_device(unique_df)
            if not baseline_df.empty and not target_df.empty:
                merged_df = analyzer.merge_results(baseline_df, target_df)
                new_failures_df, new_passes_df = analyzer.find_target_changes(merged_df)
                if not new_failures_df.empty:
                    print(f"🚨 New failures: {len(new_failures_df)}")
                if not new_passes_df.empty:
                    print(f"✨ New passes: {len(new_passes_df)}")

        print("=" * 60)

        return 0

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
