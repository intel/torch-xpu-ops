#!/usr/bin/env python3
"""
GitHub Issues Data Extractor - Efficient Version
Fetches GitHub issues and extracts table data with optimized performance.
"""

import os
import re
import json
import argparse
from github import Github


def get_github_issues(repo_owner, repo_name, labels, state='open'):
    """
    Efficiently get GitHub issues and extract table data.
    Uses batch processing and optimized filtering.
    """
    repo = g.get_repo(f"{repo_owner}/{repo_name}")
    issues_data = []

    # Use generator to avoid loading all issues into memory at once
    issues = repo.get_issues(state=state, labels=labels)

    for issue in issues:
        # Quick filter for issues with body content
        if not issue.body:
            continue

        # Fast table extraction with pre-compiled regex
        table_rows = fast_extract_table_rows(issue.body)

        if table_rows:
            issues_data.append({
                'issue_number': issue.number,
                'issue_title': issue.title,
                'table_rows': table_rows
            })

    return issues_data


def fast_extract_table_rows(issue_body):
    """
    Fast table extraction using pre-compiled regex patterns.
    """
    # Pre-compile regex patterns for better performance
    TABLE_ROW_PATTERN = re.compile(r'^(?=.*\|)(?!.*Suite)(?=.*[a-z]).+$', re.MULTILINE)
    WHITESPACE_PATTERN = re.compile(r'\s+')

    # Find all table rows in one pass
    rows = TABLE_ROW_PATTERN.findall(issue_body)

    # Process rows in batch
    clean_rows = []
    for row in rows:
        # Fast cleaning: replace multiple spaces with single space and split
        clean_cells = [cell.strip() for cell in WHITESPACE_PATTERN.sub(' ', row).split('|')]
        clean_cells = [cell for cell in clean_cells if cell]  # Filter empty cells

        if len(clean_cells) > 1:  # Only add rows with multiple cells
            clean_rows.append(clean_cells)

    return clean_rows


def save_issues_json(issues_data, output_file):
    """
    Efficient JSON saving with direct file writing.
    """
    with open(output_file, 'w') as f:
        # Write JSON manually for better control
        f.write('[\n')
        for i, issue in enumerate(issues_data):
            if i > 0:
                f.write(',\n')
            json.dump(issue, f, separators=(',', ':'))  # Compact JSON
        f.write('\n]')


def main():
    """Optimized main function."""
    parser = argparse.ArgumentParser(description="Efficient GitHub issues exporter")
    parser.add_argument("--repo_owner", default="intel", help="Repo owner")
    parser.add_argument("--repo_name", default="torch-xpu-ops", help="Repo name")
    parser.add_argument('--labels', nargs='*', help='Filter by labels')
    parser.add_argument("--output", default="issues.json", help="Output file")
    parser.add_argument("--state", default="open", help="Issue state")

    args = parser.parse_args()

    # Quick token check
    token = os.getenv('GH_TOKEN')
    global g
    g = Github(token)  # Increase page size for fewer API calls

    print(f"Fetching known issues from {args.repo_owner}/{args.repo_name}...")

    # Time the operation
    issues_data = get_github_issues(
        repo_owner=args.repo_owner,
        repo_name=args.repo_name,
        labels=args.labels,
        state=args.state
    )

    # Save results
    with open(args.output, "w") as f:
        json.dump(issues_data, f, indent=2)

    # Results summary
    print(f"✅ Done: {len(issues_data)} issues -> {args.output}")

    if issues_data:
        total_rows = sum(len(issue['table_rows']) for issue in issues_data)
        print(f"📊 {total_rows} table rows extracted")


if __name__ == "__main__":
    main()
