#!/usr/bin/env python3
"""
write_results.py — Write classification results to a new sheet in the Excel file.

Phase 3 of classify_ut workflow:
1. Reads the original Excel sheet and a results.json file
2. Creates a new sheet (default: "agent") with original columns + Analyzed, Reason,
   DetailReason, ReuseSource, Confidence
3. Confidence is auto-computed: High if DetailReason contains exact evidence
   (commit hash, issue/PR URL, PR reference), otherwise Medium
4. Preserves rows that were already Analyzed=TRUE
5. Fills in classification results for new rows

Usage:
    python3 write_results.py <excel_path> <results.json> [sheet_name] [--output_sheet=agent] [--output-excel=output.xlsx]
"""

import json
import re
import sys

try:
    from openpyxl import load_workbook
except ImportError:
    print("ERROR: openpyxl is required. Install with: pip install openpyxl", file=sys.stderr)
    sys.exit(1)


EVIDENCE_PATTERNS = [
    re.compile(r'github\.com/[\w.-]+/[\w.-]+/issues/\d+'),
    re.compile(r'github\.com/[\w.-]+/[\w.-]+/pull/\d+'),
    re.compile(r'\b[0-9a-f]{7,40}\b'),
    re.compile(r'#\d+'),
]


def compute_confidence(detail_reason: str) -> str:
    """Return 'High' if detail_reason contains exact evidence, else 'Medium'."""
    if not detail_reason:
        return "Medium"
    for pat in EVIDENCE_PATTERNS:
        if pat.search(detail_reason):
            return "High"
    return "Medium"


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 write_results.py <excel_path> <results.json> [sheet_name] [--output_sheet=agent] [--output-excel=output.xlsx]", file=sys.stderr)
        sys.exit(1)

    excel_path = sys.argv[1]
    results_path = sys.argv[2]
    sheet_name = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else None
    output_sheet = "agent"
    output_excel = None
    for arg in sys.argv[1:]:
        if arg.startswith("--output_sheet="):
            output_sheet = arg.split("=", 1)[1]
        if arg.startswith("--output-excel="):
            output_excel = arg.split("=", 1)[1]

    with open(results_path) as f:
        results = json.load(f)

    result_map = {}
    for r in results:
        result_map[r.get("name_cuda", "")] = r

    wb = load_workbook(excel_path)
    ws = wb[sheet_name] if sheet_name else wb.active

    headers = [cell.value for cell in ws[1]]
    data_rows = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[0] is not None:
            data_rows.append(row)

    output_headers = list(headers) + ["Analyzed", "Reason", "DetailReason", "ReuseSource", "Confidence"]

    if output_excel:
        from openpyxl import Workbook
        out_wb = Workbook()
        out_ws = out_wb.active
        out_ws.title = output_sheet
    else:
        if output_sheet in wb.sheetnames:
            del wb[output_sheet]
        out_ws = wb.create_sheet(title=output_sheet)

    for col_idx, h in enumerate(output_headers, start=1):
        out_ws.cell(row=1, column=col_idx, value=h)

    name_col = headers.index("name_cuda") if "name_cuda" in headers else -1
    analyzed_col_orig = headers.index("Analyzed") if "Analyzed" in headers else -1
    reason_col_orig = headers.index("Reason") if "Reason" in headers else -1
    detail_col_orig = headers.index("DetailReason") if "DetailReason" in headers else -1

    for row_idx, row in enumerate(data_rows, start=2):
        for col_idx, val in enumerate(row, start=1):
            out_ws.cell(row=row_idx, column=col_idx, value=val)

        name = str(row[name_col]) if name_col >= 0 and name_col < len(row) and row[name_col] is not None else ""
        result = result_map.get(name, {})

        ac = len(headers) + 1
        rc = len(headers) + 2
        dc = len(headers) + 3
        rsc = len(headers) + 4
        cc = len(headers) + 5

        orig_analyzed = ""
        if analyzed_col_orig >= 0 and analyzed_col_orig < len(row) and row[analyzed_col_orig] is not None:
            orig_analyzed = str(row[analyzed_col_orig]).lower()

        if orig_analyzed == "true":
            out_ws.cell(row=row_idx, column=ac, value="TRUE")
            orig_reason = str(row[reason_col_orig]) if reason_col_orig >= 0 and reason_col_orig < len(row) and row[reason_col_orig] is not None else ""
            orig_detail = str(row[detail_col_orig]) if detail_col_orig >= 0 and detail_col_orig < len(row) and row[detail_col_orig] is not None else ""
            out_ws.cell(row=row_idx, column=rc, value=orig_reason)
            out_ws.cell(row=row_idx, column=dc, value=orig_detail)
            out_ws.cell(row=row_idx, column=rsc, value="")
            out_ws.cell(row=row_idx, column=cc, value=compute_confidence(orig_detail))
        elif result:
            out_ws.cell(row=row_idx, column=ac, value="TRUE")
            out_ws.cell(row=row_idx, column=rc, value=result.get("Reason", ""))
            detail = result.get("DetailReason", "")
            out_ws.cell(row=row_idx, column=dc, value=detail)
            out_ws.cell(row=row_idx, column=rsc, value=result.get("ReuseSource", ""))
            out_ws.cell(row=row_idx, column=cc, value=compute_confidence(detail))
        else:
            out_ws.cell(row=row_idx, column=ac, value="FALSE")
            out_ws.cell(row=row_idx, column=rc, value="")
            out_ws.cell(row=row_idx, column=dc, value="")
            out_ws.cell(row=row_idx, column=rsc, value="")
            out_ws.cell(row=row_idx, column=cc, value="")

    if output_excel:
        out_wb.save(output_excel)
        print(f"Written to '{output_excel}' (sheet '{output_sheet}')")
    else:
        wb.save(excel_path)
        print(f"Written to sheet '{output_sheet}' in {excel_path}")


if __name__ == "__main__":
    main()
