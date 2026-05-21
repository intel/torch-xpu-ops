#!/usr/bin/env python3
"""Print status of all tracked issues."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pytorch_agent.utils.state import get_all_tracked


def main() -> None:
    tracked = get_all_tracked()
    if not tracked:
        print("No tracked issues found.")
        return

    print(f"{'#':>5}  {'Stage':<15}  {'Branch':<25}  {'PR':>5}  {'Pub':>5}  {'Rev':>3}  {'Att':>3}  Title")
    print("-" * 100)
    for t in sorted(tracked, key=lambda x: x.source_number):
        print(
            f"{t.source_number:>5}  {t.stage:<15}  {t.branch or 'N/A':<25}  "
            f"{t.tracking_pr_number or '-':>5}  {t.public_pr_number or '-':>5}  "
            f"{t.review_iteration:>3}  {t.attempt_count:>3}  {t.title[:40]}"
        )


if __name__ == "__main__":
    main()
