#!/usr/bin/env python3
"""
Test Case Status Monitor for torch-xpu-ops
This script tracks test case statuses from GitHub issues and JUnit XML files.
"""

import os
import re
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
import asyncio
import aiohttp
import aiofiles
import xml.etree.ElementTree as ET
from github import Github
from github.Issue import Issue
import csv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TestCase:
    """Represents a test case with its status."""
    name: str
    class_name: str = ""
    method_name: str = ""
    status: str = "unknown"  # passed, failed, skipped, error
    duration: float = 0.0
    message: str = ""
    traceback: str = ""

    @property
    def full_name(self) -> str:
        """Get the full test case name."""
        if self.class_name and self.method_name:
            return f"{self.class_name}::{self.method_name}"
        return self.name

@dataclass
class TestStatus:
    """Represents test status at different points."""
    current: str = "unknown"
    previous: str = "unknown"
    github: str = "unknown"  # Status from GitHub issues

    def has_improved(self) -> bool:
        """Check if test improved from failed to passed."""
        return (self.previous not in ["passed", "unknown"] and self.current == "passed") or \
               (self.github not in ["passed", "unknown"] and self.current == "passed")

    def has_regressed(self) -> bool:
        """Check if test regressed from passed to failed."""
        return (self.previous == "passed" and self.current not in ["passed", "unknown"])

class CSVExporter:
    """Handles CSV export functionality."""

    @staticmethod
    def save_detailed_status_to_csv(statuses: Dict[str, TestStatus],
                                   current_tests: Dict[str, TestCase],
                                   previous_tests: Dict[str, TestCase],
                                   filename: str) -> None:
        """Save detailed status of all tests to CSV file.

        Args:
            statuses: Dictionary mapping test names to TestStatus objects
            current_tests: Current test results
            previous_tests: Previous test results
            filename: Output CSV filename
        """
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'test_name',
                    'current_status',
                    'previous_status',
                    'github_status',
                    'current_duration',
                    'previous_duration',
                    'has_improved',
                    'has_regressed',
                    'current_message',
                    'previous_message'
                ]

                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                # Sort tests by name for consistency
                sorted_tests = sorted(statuses.items(), key=lambda x: x[0])

                for test_name, test_status in sorted_tests:
                    current_test = current_tests.get(test_name)
                    previous_test = previous_tests.get(test_name)

                    row = {
                        'test_name': test_name,
                        'current_status': test_status.current,
                        'previous_status': test_status.previous,
                        'github_status': test_status.github,
                        'current_duration': current_test.duration if current_test else 0.0,
                        'previous_duration': previous_test.duration if previous_test else 0.0,
                        'has_improved': 'Yes' if test_status.has_improved() else 'No',
                        'has_regressed': 'Yes' if test_status.has_regressed() else 'No',
                        'current_message': CSVExporter._sanitize_csv_field(
                            current_test.message if current_test else ''
                        ),
                        'previous_message': CSVExporter._sanitize_csv_field(
                            previous_test.message if previous_test else ''
                        )
                    }
                    writer.writerow(row)

            logger.info(f"Detailed status saved to {filename} ({len(statuses)} tests)")

        except Exception as e:
            logger.error(f"Error saving detailed status to CSV: {e}")

    @staticmethod
    def save_new_passed_to_csv(new_passed_tests: List[str],
                              statuses: Dict[str, TestStatus],
                              current_tests: Dict[str, TestCase],
                              previous_tests: Dict[str, TestCase],
                              filename: str) -> None:
        """Save newly passed tests to CSV file.

        Args:
            new_passed_tests: List of newly passed test names
            statuses: Dictionary mapping test names to TestStatus objects
            current_tests: Current test results
            previous_tests: Previous test results
            filename: Output CSV filename
        """
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'test_name',
                    'improvement_type',
                    'current_status',
                    'previous_status',
                    'github_status',
                    'current_duration',
                    'current_message',
                    'previous_message'
                ]

                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for test_name in sorted(new_passed_tests):
                    test_status = statuses[test_name]
                    current_test = current_tests.get(test_name)
                    previous_test = previous_tests.get(test_name)

                    # Determine improvement type
                    if test_status.previous != "passed" and test_status.current == "passed":
                        improvement_type = "failed_to_passed"
                    elif test_status.github != "passed" and test_status.current == "passed":
                        improvement_type = "github_issue_to_passed"
                    else:
                        improvement_type = "other_improvement"

                    row = {
                        'test_name': test_name,
                        'improvement_type': improvement_type,
                        'current_status': test_status.current,
                        'previous_status': test_status.previous,
                        'github_status': test_status.github,
                        'current_duration': current_test.duration if current_test else 0.0,
                        'current_message': CSVExporter._sanitize_csv_field(
                            current_test.message if current_test else ''
                        ),
                        'previous_message': CSVExporter._sanitize_csv_field(
                            previous_test.message if previous_test else ''
                        )
                    }
                    writer.writerow(row)

            logger.info(f"Newly passed tests saved to {filename} ({len(new_passed_tests)} tests)")

        except Exception as e:
            logger.error(f"Error saving newly passed tests to CSV: {e}")

    @staticmethod
    def save_new_failed_to_csv(new_failed_tests: List[str],
                              statuses: Dict[str, TestStatus],
                              current_tests: Dict[str, TestCase],
                              previous_tests: Dict[str, TestCase],
                              filename: str) -> None:
        """Save newly failed tests to CSV file.

        Args:
            new_failed_tests: List of newly failed test names
            statuses: Dictionary mapping test names to TestStatus objects
            current_tests: Current test results
            previous_tests: Previous test results
            filename: Output CSV filename
        """
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'test_name',
                    'regression_type',
                    'current_status',
                    'previous_status',
                    'current_duration',
                    'current_message',
                    'current_traceback_preview',
                    'requires_github_issue'
                ]

                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for test_name in sorted(new_failed_tests):
                    test_status = statuses[test_name]
                    current_test = current_tests.get(test_name)
                    previous_test = previous_tests.get(test_name)

                    # Determine regression type
                    if test_status.previous == "passed" and test_status.current == "failed":
                        regression_type = "passed_to_failed"
                    else:
                        regression_type = "other_regression"

                    # Get traceback preview (first 200 chars)
                    traceback_preview = ""
                    if current_test and current_test.traceback:
                        traceback_preview = current_test.traceback[:200]
                        if len(current_test.traceback) > 200:
                            traceback_preview += "..."

                    row = {
                        'test_name': test_name,
                        'regression_type': regression_type,
                        'previous_status': test_status.previous,
                        'current_status': test_status.current,
                        'current_duration': current_test.duration if current_test else 0.0,
                        'current_message': CSVExporter._sanitize_csv_field(
                            current_test.message if current_test else ''
                        ),
                        'current_traceback_preview': CSVExporter._sanitize_csv_field(traceback_preview),
                        'requires_github_issue': 'Yes' if regression_type == "passed_to_failed" else 'Maybe'
                    }
                    writer.writerow(row)

            logger.info(f"Newly failed tests saved to {filename} ({len(new_failed_tests)} tests)")

        except Exception as e:
            logger.error(f"Error saving newly failed tests to CSV: {e}")

    @staticmethod
    def save_summary_to_csv(statuses: Dict[str, TestStatus],
                           new_passed: List[str],
                           new_failed: List[str],
                           filename: str) -> None:
        """Save summary statistics to CSV file.

        Args:
            statuses: Dictionary mapping test names to TestStatus objects
            new_passed: List of newly passed test names
            new_failed: List of newly failed test names
            filename: Output CSV filename
        """
        try:
            # Calculate statistics
            total_tests = len(statuses)

            current_status_counts = defaultdict(int)
            previous_status_counts = defaultdict(int)

            for test_status in statuses.values():
                current_status_counts[test_status.current] += 1
                previous_status_counts[test_status.previous] += 1

            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'metric',
                    'value',
                    'percentage'
                ]

                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                # Write summary rows
                summary_data = [
                    ('total_tests', total_tests, 100.0),
                    ('newly_passed', len(new_passed), (len(new_passed) / total_tests * 100) if total_tests > 0 else 0),
                    ('newly_failed', len(new_failed), (len(new_failed) / total_tests * 100) if total_tests > 0 else 0),
                ]

                for metric, value, percentage in summary_data:
                    writer.writerow({
                        'metric': metric,
                        'value': value,
                        'percentage': f"{percentage:.2f}%"
                    })

                # Write current status distribution
                writer.writerow({'metric': 'current_status_distribution', 'value': '', 'percentage': ''})
                for status, count in current_status_counts.items():
                    percentage = (count / total_tests * 100) if total_tests > 0 else 0
                    writer.writerow({
                        'metric': f'current_{status}',
                        'value': count,
                        'percentage': f"{percentage:.2f}%"
                    })

                # Write previous status distribution
                writer.writerow({'metric': 'previous_status_distribution', 'value': '', 'percentage': ''})
                for status, count in previous_status_counts.items():
                    if status != "unknown":  # Skip unknown in previous if it's too many
                        percentage = (count / total_tests * 100) if total_tests > 0 else 0
                        writer.writerow({
                            'metric': f'previous_{status}',
                            'value': count,
                            'percentage': f"{percentage:.2f}%"
                        })

            logger.info(f"Summary saved to {filename}")

        except Exception as e:
            logger.error(f"Error saving summary to CSV: {e}")

    @staticmethod
    def _sanitize_csv_field(text: str) -> str:
        """Sanitize text for CSV export (remove newlines, quotes if needed).

        Args:
            text: Input text to sanitize

        Returns:
            Sanitized text
        """
        if not text:
            return ""

        # Replace newlines with spaces, remove extra whitespace
        sanitized = ' '.join(text.split())

        # Remove quotes that could break CSV parsing
        sanitized = sanitized.replace('"', "'")

        # Truncate if too long for CSV
        if len(sanitized) > 1000:
            sanitized = sanitized[:1000] + "...[truncated]"

        return sanitized

class TestCaseMonitor:
    """Main class for monitoring test cases."""

    def __init__(self, github_token: str, repo_name: str = "intel/torch-xpu-ops"):
        """Initialize the monitor.

        Args:
            github_token: GitHub API token
            repo_name: Repository name in format "owner/repo"
        """
        self.github = Github(github_token) if github_token else None
        self.repo = self.github.get_repo(repo_name) if github_token else None
        self.session = None
        self.csv_exporter = CSVExporter()

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    def parse_test_name(self, test_name: str) -> Tuple[str, str]:
        """Parse test name into class and method names.

        Args:
            test_name: Full test name like "test_ops_xpu.py::TestFakeTensorXPU::test_fake_dot_xpu_float32"

        Returns:
            Tuple of (class_name, method_name)
        """
        parts = test_name.split("::")
        if len(parts) >= 3:
            return parts[1], parts[2]
        elif len(parts) == 2:
            return parts[0], parts[1]
        return "", test_name

    async def parse_junit_xml(self, xml_path: Path) -> Dict[str, TestCase]:
        """Parse JUnit XML file and extract test cases.

        Args:
            xml_path: Path to JUnit XML file

        Returns:
            Dictionary mapping test names to TestCase objects
        """
        test_cases = {}

        try:
            async with aiofiles.open(xml_path) as f:
                content = await f.read()

            root = ET.fromstring(content)

            # Handle different JUnit XML formats
            for testcase_elem in root.findall('.//testcase'):
                test_name = testcase_elem.get('name', '')
                class_name = testcase_elem.get('classname', '')

                # Parse full test name
                if not class_name and "::" in test_name:
                    class_name, method_name = self.parse_test_name(test_name)
                else:
                    method_name = test_name

                # Create test case
                tc = TestCase(
                    name=test_name,
                    class_name=class_name,
                    method_name=method_name,
                    status="passed",  # Default assumption
                    duration=float(testcase_elem.get('time', '0'))
                )

                # Check for failures, errors, or skips
                failure = testcase_elem.find('failure')
                error = testcase_elem.find('error')
                skipped = testcase_elem.find('skipped')

                if failure is not None:
                    tc.status = "failed"
                    tc.message = failure.get('message', '')
                    tc.traceback = failure.text or ''
                elif error is not None:
                    tc.status = "error"
                    tc.message = error.get('message', '')
                    tc.traceback = error.text or ''
                elif skipped is not None:
                    tc.status = "skipped"
                    tc.message = skipped.get('message', '')

                # Use full name as key
                key = tc.full_name if tc.full_name else tc.name
                test_cases[key] = tc

        except Exception as e:
            logger.error(f"Error parsing {xml_path}: {e}")

        return test_cases

    async def parse_junit_dir(self, dir_path: Path) -> Dict[str, TestCase]:
        """Parse all JUnit XML files in a directory with batching."""
        test_cases = {}

        if not dir_path.exists():
            logger.warning(f"Directory does not exist: {dir_path}")
            return test_cases

        # Find all XML files
        xml_files = list(dir_path.glob("**/*.xml"))
        logger.info(f"Found {len(xml_files)} XML files in {dir_path}")

        if len(xml_files) > 1000:
            logger.warning(f"Large number of files ({len(xml_files)}). Using batched processing.")

        # Process in batches
        batch_size = 100
        for i in range(0, len(xml_files), batch_size):
            batch = xml_files[i:i + batch_size]
            logger.debug(f"Processing batch {i//batch_size + 1}/{(len(xml_files) + batch_size - 1)//batch_size}")

            # Create tasks for current batch
            tasks = [self.parse_junit_xml(xml_file) for xml_file in batch]

            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results
                for idx, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Error processing {batch[idx]}: {result}")
                        continue

                    for test_name, test_case in result.items():
                        # Merge logic...
                        if test_name in test_cases:
                            if test_case.status == "passed" and test_cases[test_name].status != "passed":
                                test_cases[test_name] = test_case
                        else:
                            test_cases[test_name] = test_case

            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                continue

        return test_cases

    def get_github_issues(self, label: str = None, state: str = "open") -> List[Issue]:
        """Get GitHub issues related to test cases.

        Args:
            label: Filter by label (e.g., "test-failure")
            state: Issue state ("open", "closed", "all")

        Returns:
            List of GitHub issues
        """
        if not self.github:
            return []

        query = f"repo:{self.repo.full_name} is:issue"
        if state != "all":
            query += f" is:{state}"
        if label:
            query += f" label:{label}"

        return list(self.repo.get_issues(state=state))

    def extract_test_from_issue(self, issue: Issue) -> Optional[str]:
        """Extract test case name from issue title or body.

        Args:
            issue: GitHub issue

        Returns:
            Test case name if found, None otherwise
        """
        # Check title first
        title = issue.title
        body = issue.body or ""

        # Common patterns for test case references
        patterns = [
            r'test_[a-zA-Z0-9_]+\.py::[A-Za-z0-9_]+::[a-zA-Z0-9_]+',
            r'::[A-Za-z0-9_]+::[a-zA-Z0-9_]+',
            r'Test[A-Za-z0-9_]+::test_[a-zA-Z0-9_]+'
        ]

        for text in [title, body]:
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    return match.group(0)

        return None

    async def get_github_test_statuses(self) -> Dict[str, str]:
        """Get test statuses from GitHub issues.

        Returns:
            Dictionary mapping test names to their status from GitHub
        """
        test_statuses = {}

        if not self.github:
            return test_statuses

        try:
            # Get issues with test failure label
            issues = self.get_github_issues(label="test-failure", state="open")
            logger.info(f"Found {len(issues)} open test-failure issues")

            for issue in issues:
                test_name = self.extract_test_from_issue(issue)
                if test_name:
                    # Parse test name to standard format
                    class_name, method_name = self.parse_test_name(test_name)
                    if class_name and method_name:
                        key = f"{class_name}::{method_name}"
                        test_statuses[key] = "failed"  # Assuming open issue means test is failing

        except Exception as e:
            logger.error(f"Error fetching GitHub issues: {e}")

        return test_statuses

    def compare_statuses(self,
                        current_tests: Dict[str, TestCase],
                        previous_tests: Dict[str, TestCase],
                        github_tests: Dict[str, str]) -> Dict[str, TestStatus]:
        """Compare test statuses across different sources.

        Args:
            current_tests: Current test results
            previous_tests: Previous test results
            github_tests: Test statuses from GitHub

        Returns:
            Dictionary mapping test names to TestStatus objects
        """
        all_tests = set()
        all_tests.update(current_tests.keys())
        all_tests.update(previous_tests.keys())
        all_tests.update(github_tests.keys())

        results = {}

        for test_name in all_tests:
            status = TestStatus()

            # Get current status
            if test_name in current_tests:
                status.current = current_tests[test_name].status

            # Get previous status
            if test_name in previous_tests:
                status.previous = previous_tests[test_name].status

            # Get GitHub status
            if test_name in github_tests:
                status.github = github_tests[test_name]

            results[test_name] = status

        return results

    def find_new_passed_tests(self, statuses: Dict[str, TestStatus]) -> List[str]:
        """Find tests that passed now but were failing before.

        Args:
            statuses: Test status comparison results

        Returns:
            List of test names that newly passed
        """
        new_passed = []

        for test_name, status in statuses.items():
            if status.has_improved():
                new_passed.append(test_name)

        return new_passed

    def find_new_failed_tests(self, statuses: Dict[str, TestStatus]) -> List[str]:
        """Find tests that failed now but were passing before.

        Args:
            statuses: Test status comparison results

        Returns:
            List of test names that newly failed
        """
        new_failed = []

        for test_name, status in statuses.items():
            if status.has_regressed():
                new_failed.append(test_name)

        return new_failed

    async def comment_on_issue(self, test_name: str, current_status: TestCase):
        """Comment on existing GitHub issue when test passes.

        Args:
            test_name: Name of the test
            current_status: Current test status object
        """
        if not self.github:
            return

        try:
            # Find issue for this test
            issues = self.get_github_issues(state="all")

            for issue in issues:
                extracted_test = self.extract_test_from_issue(issue)
                if extracted_test:
                    class_name, method_name = self.parse_test_name(extracted_test)
                    if f"{class_name}::{method_name}" == test_name and issue.state == "open":
                        # Create comment
                        comment = f"✅ Test **{test_name}** has passed in the latest run!\n\n"
                        comment += "**Details:**\n"
                        comment += f"- Status: {current_status.status}\n"
                        comment += f"- Duration: {current_status.duration:.2f}s\n"
                        comment += f"- Timestamp: {datetime.now().isoformat()}\n\n"
                        comment += "Consider closing this issue if the fix is verified."

                        issue.create_comment(comment)
                        logger.info(f"Commented on issue #{issue.number} for test {test_name}")
                        break

        except Exception as e:
            logger.error(f"Error commenting on issue for test {test_name}: {e}")

    async def create_new_issue(self, test_name: str, test_case: TestCase):
        """Create a new GitHub issue for a newly failing test.

        Args:
            test_name: Name of the test
            test_case: Test case object with failure details
        """
        if not self.github:
            return None

        try:
            # Parse test name for better issue title
            class_name, method_name = self.parse_test_name(test_name)

            title = f"Test Failure: {test_name}"

            body = "## Test Failure Detected\n\n"
            body += f"**Test:** `{test_name}`\n\n"
            body += f"**Status:** {test_case.status}\n"
            body += f"**Duration:** {test_case.duration:.2f}s\n\n"

            if test_case.message:
                body += f"**Error Message:**\n```\n{test_case.message}\n```\n\n"

            if test_case.traceback:
                # Limit traceback length
                traceback_preview = '\n'.join(test_case.traceback.split('\n')[:20])
                body += f"**Traceback (first 20 lines):**\n```python\n{traceback_preview}\n```\n\n"
                if len(test_case.traceback.split('\n')) > 20:
                    body += "*Full traceback available in test logs*\n\n"

            body += f"**Detected:** {datetime.now().isoformat()}\n\n"
            body += "## Next Steps\n"
            body += "1. Investigate the failure\n"
            body += "2. Fix the issue or update test expectations\n"
            body += "3. Re-run tests to verify fix\n"

            # Create issue
            issue = self.repo.create_issue(
                title=title,
                body=body,
                labels=["bug", "test-failure", "ci"]
            )

            logger.info(f"Created new issue #{issue.number} for test {test_name}")
            return issue

        except Exception as e:
            logger.error(f"Error creating issue for test {test_name}: {e}")
            return None

    def generate_report(self, statuses: Dict[str, TestStatus],
                       new_passed: List[str],
                       new_failed: List[str]) -> str:
        """Generate a comprehensive report.

        Args:
            statuses: All test statuses
            new_passed: Newly passed tests
            new_failed: Newly failed tests

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("TEST STATUS REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append(f"Total tests analyzed: {len(statuses)}")
        report.append("")

        # Summary
        report.append("SUMMARY")
        report.append("-" * 40)
        report.append(f"Newly passed tests: {len(new_passed)}")
        report.append(f"Newly failed tests: {len(new_failed)}")
        report.append("")

        # Newly passed tests
        if new_passed:
            report.append("🎉 NEWLY PASSED TESTS (consider closing related issues)")
            report.append("-" * 40)
            for test in sorted(new_passed):
                report.append(f"  ✅ {test}")
            report.append("")

        # Newly failed tests (highlighted)
        if new_failed:
            report.append("🔥 NEWLY FAILED TESTS (need investigation)")
            report.append("-" * 40)
            for test in sorted(new_failed):
                report.append(f"  ❌ {test}")
            report.append("")

        # Detailed status table
        report.append("DETAILED STATUS")
        report.append("-" * 40)
        report.append(f"{'Test Name':<60} {'Current':<10} {'Previous':<10} {'GitHub':<10}")
        report.append("-" * 100)

        # Sort tests by status for better readability
        sorted_tests = sorted(statuses.items(),
                            key=lambda x: (x[1].current != "passed", x[0]))

        for test_name, status in sorted_tests:
            # Highlight regressions
            if status.has_regressed():
                line = f"❌ {test_name:<58} {status.current:<10} {status.previous:<10} {status.github:<10}"
            elif status.has_improved():
                line = f"✅ {test_name:<58} {status.current:<10} {status.previous:<10} {status.github:<10}"
            else:
                line = f"  {test_name:<60} {status.current:<10} {status.previous:<10} {status.github:<10}"
            report.append(line)

        return '\n'.join(report)

    def save_all_csv_files(self,
                          statuses: Dict[str, TestStatus],
                          new_passed: List[str],
                          new_failed: List[str],
                          current_tests: Dict[str, TestCase],
                          previous_tests: Dict[str, TestCase],
                          output_dir: Path) -> Dict[str, Path]:
        """Save all CSV files to the specified directory.

        Args:
            statuses: Dictionary mapping test names to TestStatus objects
            new_passed: List of newly passed test names
            new_failed: List of newly failed test names
            current_tests: Current test results
            previous_tests: Previous test results
            output_dir: Directory to save CSV files

        Returns:
            Dictionary mapping file types to file paths
        """
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_paths = {}

        # Save detailed status
        detailed_file = output_dir / f"detailed_status_{timestamp}.csv"
        self.csv_exporter.save_detailed_status_to_csv(
            statuses, current_tests, previous_tests, str(detailed_file)
        )
        file_paths['detailed'] = detailed_file

        # Save newly passed tests
        if new_passed:
            passed_file = output_dir / f"newly_passed_{timestamp}.csv"
            self.csv_exporter.save_new_passed_to_csv(
                new_passed, statuses, current_tests, previous_tests, str(passed_file)
            )
            file_paths['passed'] = passed_file

        # Save newly failed tests
        if new_failed:
            failed_file = output_dir / f"newly_failed_{timestamp}.csv"
            self.csv_exporter.save_new_failed_to_csv(
                new_failed, statuses, current_tests, previous_tests, str(failed_file)
            )
            file_paths['failed'] = failed_file

        # Save summary
        summary_file = output_dir / f"summary_{timestamp}.csv"
        self.csv_exporter.save_summary_to_csv(
            statuses, new_passed, new_failed, str(summary_file)
        )
        file_paths['summary'] = summary_file

        # Create a manifest file
        manifest_file = output_dir / f"manifest_{timestamp}.txt"
        with open(manifest_file, 'w') as f:
            f.write(f"Test Report Generated: {datetime.now().isoformat()}\n")
            f.write(f"Total tests: {len(statuses)}\n")
            f.write(f"Newly passed: {len(new_passed)}\n")
            f.write(f"Newly failed: {len(new_failed)}\n\n")
            f.write("Generated Files:\n")
            for file_type, file_path in file_paths.items():
                f.write(f"  {file_type}: {file_path.name}\n")

        file_paths['manifest'] = manifest_file

        return file_paths

async def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Monitor test case statuses for torch-xpu-ops",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --current ./current_results --previous ./previous_results --github-token YOUR_TOKEN
  %(prog)s --current ./results --github-token YOUR_TOKEN --dry-run
        """
    )

    parser.add_argument(
        "--current",
        type=Path,
        required=True,
        help="Directory containing current JUnit XML files"
    )
    parser.add_argument(
        "--previous",
        type=Path,
        help="Directory containing previous JUnit XML files (optional)"
    )
    parser.add_argument(
        "--github-token",
        type=str,
        default=None,
        help="GitHub API token (can also use GITHUB_TOKEN env var)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't create comments or issues, just generate report"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory for CSV files (default: ./test_reports)"
    )
    parser.add_argument(
        "--console-report",
        action="store_true",
        default=False,
        help="Print report to console (default: False)"
    )
    parser.add_argument(
        "--no-console-report",
        action="store_false",
        dest="console_report",
        help="Don't print report to console"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Use environment variable for token if not provided
    token = args.github_token if args.github_token is not None else os.getenv("GITHUB_TOKEN")

    # Set default output directory
    output_dir = args.output if args.output else Path("./test_reports")

    # Initialize monitor
    async with TestCaseMonitor(token) as monitor:
        logger.info("Starting test case monitoring...")

        # Parse test results
        logger.info(f"Parsing current test results from {args.current}")
        current_tests = await monitor.parse_junit_dir(args.current)
        logger.info(f"Found {len(current_tests)} current tests")

        previous_tests = {}
        if args.previous:
            logger.info(f"Parsing previous test results from {args.previous}")
            previous_tests = await monitor.parse_junit_dir(args.previous)
            logger.info(f"Found {len(previous_tests)} previous tests")

        # Get GitHub statuses
        github_tests = {}
        if token:
            logger.info("Fetching test statuses from GitHub...")
            github_tests = await monitor.get_github_test_statuses()
            logger.info(f"Found {len(github_tests)} tests with GitHub status")

        # Compare statuses
        logger.info("Comparing test statuses...")
        all_statuses = monitor.compare_statuses(current_tests, previous_tests, github_tests)

        # Find changes
        new_passed = monitor.find_new_passed_tests(all_statuses)
        new_failed = monitor.find_new_failed_tests(all_statuses)

        # Generate and optionally display console report
        if args.console_report:
            report = monitor.generate_report(all_statuses, new_passed, new_failed)
            print(report)

        # Save all CSV files
        logger.info(f"Saving CSV files to {output_dir}")
        csv_files = monitor.save_all_csv_files(
            all_statuses, new_passed, new_failed,
            current_tests, previous_tests, output_dir
        )

        # Log generated files
        logger.info("Generated CSV files:")
        for file_type, file_path in csv_files.items():
            logger.info(f"  {file_type}: {file_path}")

        # Take actions if not dry run
        if not args.dry_run and token:
            # Comment on issues for newly passed tests
            if new_passed:
                logger.info(f"Adding comments for {len(new_passed)} newly passed tests...")
                for test_name in new_passed:
                    if test_name in current_tests:
                        await monitor.comment_on_issue(test_name, current_tests[test_name])

            # Create issues for newly failed tests
            if new_failed:
                logger.info(f"Creating issues for {len(new_failed)} newly failed tests...")
                for test_name in new_failed:
                    if test_name in current_tests:
                        await monitor.create_new_issue(test_name, current_tests[test_name])

        # Exit with appropriate code
        if new_failed:
            logger.warning(f"Found {len(new_failed)} new test failures")
            sys.exit(1)
        else:
            logger.info("No new test failures found")
            sys.exit(0)

if __name__ == "__main__":
    asyncio.run(main())
