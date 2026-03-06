#!/usr/bin/env python3
"""
JUnit XML Test Details Extractor - Target/Baseline Comparison Tool

Compares test results between target and baseline

Usage:
    python compare_tests.py --input "results/*.xml" --output comparison.xlsx
    python compare_tests.py --input file1.xml file2.xml --output comparison.csv
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
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

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


# ============================================================================
# FILE PATTERN MATCHER
# ============================================================================

class FilePatternMatcher:
    """Handles file pattern matching and normalization."""

    # Compiled regex patterns
    _CLASSNAME_PATTERN = re.compile(r".*\.")
    _CASENAME_PATTERN = re.compile(r"[^a-zA-Z0-9_.-]")
    _TESTFILE_PATTERN = re.compile(r".*torch-xpu-ops\.test\.xpu\.")
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
        return merged[existing_cols]

    def find_target_issues(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """
        Find tests where BASELINE passed but TARGET failed/skipped.
        These are "New failed" cases.
        """
        if merged_df.empty:
            return pd.DataFrame()

        # Define conditions
        baseline_passed = merged_df["status_baseline"].isin(["passed", "xfail"])
        target_not_passed = ~merged_df["status_target"].isin(["passed", "xfail"])

        issues = merged_df[baseline_passed & target_not_passed].copy()

        if issues.empty:
            return issues

        # Add reason column
        issues["reason"] = np.select(
            [
                issues["status_target"].isin(["skipped"]),
                issues["status_target"].isin(["failed"]),
                issues["status_target"].isin(["error"]),
            ],
            [
                "Skipped on Target",
                "Failed on Target",
                "Error on Target",
            ],
            default="Unknown issue"
        )

        return issues

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
                "Pass Rate": f"{pass_rate:.3f}%",
            })

        return pd.DataFrame(stats)


# ============================================================================
# REPORT EXPORTER
# ============================================================================

class ReportExporter:
    """Export comparison results to various formats."""

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
                # Merge results
                merged_df = analyzer.merge_results(baseline_df, target_df)

                # Write comparison
                merged_df.to_excel(writer, sheet_name="Comparison", index=False)

                # Find target issues (New failed)
                issues_df = analyzer.find_target_issues(merged_df)

                if not issues_df.empty:
                    issues_df.to_excel(writer, sheet_name="New failed", index=False)

            # Generate and write file summary
            file_summary_df = analyzer.generate_file_summary(unique_df)
            if not file_summary_df.empty:
                file_summary_df.to_excel(writer, sheet_name="Files summary", index=False)

            # Add summary statistics
            stats_df = analyzer.generate_summary_stats(unique_df)
            if not stats_df.empty:
                stats_df.to_excel(writer, sheet_name="Summary", index=False)

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
            # Merge results
            merged_df = analyzer.merge_results(baseline_df, target_df)

            # Save comparison
            merged_df.to_csv(f"{base_path}_comparison.csv", index=False)

            # Find and save target issues (New failed)
            issues_df = analyzer.find_target_issues(merged_df)

            if not issues_df.empty:
                issues_df.to_csv(f"{base_path}_new_failed.csv", index=False)

        # Generate and save file summary
        file_summary_df = analyzer.generate_file_summary(unique_df)
        if not file_summary_df.empty:
            file_summary_df.to_csv(f"{base_path}_files_summary.csv", index=False)

        # Save summary
        stats_df = analyzer.generate_summary_stats(unique_df)
        if not stats_df.empty:
            stats_df.to_csv(f"{base_path}_summary.csv", index=False)

        logger.info(f"Exported comparison results to {output_path}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="JUnit XML Test Details Extractor - Target/Baseline Comparison",
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
        "-w", "--workers",
        type=int,
        help="Number of parallel workers (default: CPU count - 2)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
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

        # Analyze results
        logger.info(f"Found {len(extractor.test_cases)} test cases")
        analyzer = ResultAnalyzer(extractor.test_cases)

        # Export results
        output_path = Path(args.output)
        exporter = ReportExporter()

        if output_path.suffix.lower() in [".xlsx", ".xls"]:
            exporter.export_excel(analyzer, output_path)
        else:
            exporter.export_csv(analyzer, output_path)

        # Print summary
        elapsed = time.time() - start_time

        print("\n" + "=" * 60)
        print("COMPARISON COMPLETE")
        print("=" * 60)
        print(f"📊 Files processed: {extractor.stats['files_processed']}")
        print(f"🧪 Test cases found: {extractor.stats['test_cases_found']}")
        print(f"⏱️  Time: {elapsed:.2f}s")
        print(f"📁 Output: {output_path}")

        unique_df = analyzer.deduplicate_by_priority()
        if not unique_df.empty:
            baseline_count = len(unique_df[unique_df["device"] == "baseline"])
            target_count = len(unique_df[unique_df["device"] == "target"])

            print(f"📱 Baseline tests: {baseline_count}, Target tests: {target_count}")

            # Show file summary stats
            file_summary = analyzer.generate_file_summary(unique_df)
            if not file_summary.empty:
                print(f"📂 Test files: {len(file_summary)}")

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
