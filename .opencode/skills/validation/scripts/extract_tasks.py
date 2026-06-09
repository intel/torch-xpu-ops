#!/usr/bin/env python3
"""
extract_tasks.py — Extract rows needing classification from an Excel sheet.

Phase 1 of classify_ut workflow:
1. Reads an Excel sheet with columns: testfile_cuda, classname_cuda, name_cuda, message_xpu, status_xpu
2. Deduplicates rows that share the same classname_cuda + similar error message as an already-analyzed row
3. Outputs tasks.json (rows needing classification) and already_resolved.json (deduplicated rows)

Usage:
    python3 extract_tasks.py <excel_path> [sheet_name] > tasks.json
"""

import json
import re
import sys

try:
    from openpyxl import load_workbook
except ImportError:
    print("ERROR: openpyxl is required. Install with: pip install openpyxl", file=sys.stderr)
    sys.exit(1)


def extract_operators(message: str) -> set:
    """Extract operator/API references from an error message."""
    if not message:
        return set()
    ops = set()
    for m in re.finditer(r'aten::(\w+)', message):
        ops.add(f'aten::{m.group(1)}')
    for m in re.finditer(r'torch\.(\w+(?:\.\w+)*)', message):
        ops.add(f'torch.{m.group(1)}')
    for kw in ['CUDA error', 'cuBLAS', 'cuDNN', 'NCCL', 'XPU', 'SYCL']:
        if kw.lower() in message.lower():
            ops.add(kw)
    return ops


def normalize_message(msg: str) -> str:
    """Normalize a message for similarity comparison."""
    if not msg:
        return ""
    msg = msg.lower().strip()
    msg = re.sub(r'0x[0-9a-f]+', '', msg)
    msg = re.sub(r'/\S+', '', msg)
    msg = re.sub(r'\b\d{10,}\b', '', msg)
    msg = re.sub(r'\s+', ' ', msg).strip()
    return msg


def levenshtein_similarity(a: str, b: str) -> float:
    """Compute Levenshtein similarity (0.0 to 1.0), capped at 200 chars."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    a, b = a[:200], b[:200]
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    max_len = max(n, m)
    return 1.0 - (dp[n][m] / max_len) if max_len > 0 else 1.0


def messages_similar(msg_a: str, msg_b: str, threshold: float = 0.7) -> bool:
    """Check if two error messages are similar (share operators OR high string similarity)."""
    ops_a = extract_operators(msg_a)
    ops_b = extract_operators(msg_b)
    if ops_a and ops_b and ops_a & ops_b:
        return True
    sim = levenshtein_similarity(normalize_message(msg_a), normalize_message(msg_b))
    return sim >= threshold


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 extract_tasks.py <excel_path> [sheet_name]", file=sys.stderr)
        sys.exit(1)

    excel_path = sys.argv[1]
    sheet_name = sys.argv[2] if len(sys.argv) > 2 else None

    wb = load_workbook(excel_path, data_only=True)
    ws = wb[sheet_name] if sheet_name else wb.active

    headers = [cell.value for cell in ws[1]]
    col_map = {h: i for i, h in enumerate(headers) if h}

    required = ["testfile_cuda", "classname_cuda", "name_cuda"]
    missing = [c for c in required if c not in col_map]
    if missing:
        print(f"ERROR: Missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    has_status = "status_xpu" in col_map
    has_reason = "Reason" in col_map
    has_detail = "DetailReason" in col_map
    has_analyzed = "Analyzed" in col_map

    rows = []
    for row_idx, row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
        if row[0] is None:
            continue
        entry = {"_row": row_idx}
        for col_name, col_idx in col_map.items():
            val = row[col_idx] if col_idx < len(row) else None
            entry[col_name] = str(val) if val is not None else ""
        rows.append(entry)

    analyzed_rows = [r for r in rows if has_analyzed and r.get("Analyzed", "").lower() == "true"]

    tasks = []
    already_resolved = []

    for r in rows:
        already_analyzed = has_analyzed and r.get("Analyzed", "").lower() == "true"
        if already_analyzed:
            already_resolved.append({
                "testfile_cuda": r.get("testfile_cuda", ""),
                "classname_cuda": r.get("classname_cuda", ""),
                "name_cuda": r.get("name_cuda", ""),
                "message_xpu": r.get("message_xpu", ""),
                "Analyzed": True,
                "Reason": r.get("Reason", ""),
                "DetailReason": r.get("DetailReason", ""),
                "ReuseSource": "",
            })
            continue

        cls = r.get("classname_cuda", "")
        msg = r.get("message_xpu", "")
        reused = False
        for ar in analyzed_rows:
            if ar.get("classname_cuda", "") == cls and messages_similar(ar.get("message_xpu", ""), msg):
                already_resolved.append({
                    "testfile_cuda": r.get("testfile_cuda", ""),
                    "classname_cuda": r.get("classname_cuda", ""),
                    "name_cuda": r.get("name_cuda", ""),
                    "message_xpu": r.get("message_xpu", ""),
                    "Analyzed": True,
                    "Reason": ar.get("Reason", ""),
                    "DetailReason": ar.get("DetailReason", ""),
                    "ReuseSource": ar.get("name_cuda", ""),
                })
                reused = True
                break

        if reused:
            continue

        tasks.append({
            "testfile_cuda": r.get("testfile_cuda", ""),
            "classname_cuda": r.get("classname_cuda", ""),
            "name_cuda": r.get("name_cuda", ""),
            "message_xpu": r.get("message_xpu", ""),
            "status_xpu": r.get("status_xpu", ""),
        })

    output = {
        "tasks": tasks,
        "already_resolved": already_resolved,
        "summary": {
            "total_rows": len(rows),
            "already_analyzed": sum(1 for r in rows if has_analyzed and r.get("Analyzed", "").lower() == "true"),
            "deduplicated": len(already_resolved) - sum(1 for r in rows if has_analyzed and r.get("Analyzed", "").lower() == "true"),
            "needs_classification": len(tasks),
        },
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
