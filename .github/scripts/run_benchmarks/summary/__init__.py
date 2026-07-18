"""summary — Compare PyTorch Dynamo Benchmark results (target vs baseline).

Usage:
    python -m run_benchmarks.summary -t target/ -b baseline/ -o comparison.xlsx
    python -m run_benchmarks.summary -t target/ -o out.csv -m report.md
"""

from .loader import find_result_files, load_results, parse_filename
from .merge import (
    generate_summary,
    merge_accuracy,
    merge_performance,
    merge_pt2e_accuracy,
    merge_pt2e_performance,
)
from .report import print_report, write_csv, write_excel, write_markdown

__all__ = [
    "find_result_files",
    "load_results",
    "parse_filename",
    "merge_accuracy",
    "merge_performance",
    "merge_pt2e_accuracy",
    "merge_pt2e_performance",
    "generate_summary",
    "write_markdown",
    "write_excel",
    "write_csv",
    "print_report",
]
