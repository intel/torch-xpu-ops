"""Markdown, console, and file (Excel/CSV) output for benchmark comparison."""

import logging
from datetime import datetime
from html import escape as html_escape
from typing import Any

import pandas as pd

from .constants import FAIL_LABELS, PASS_LABELS
from .merge import _geomean, _is_acc_pass, _is_positive

log = logging.getLogger(__name__)


# ── Markdown Helpers ──────────────────────────────────────────────────
def _fmt_ratio(val: Any, threshold: float) -> str:
    """Format ratio with visual indicator when outside threshold."""
    if pd.isna(val) or val == "":
        return ""
    try:
        num = float(val)
        if num < 1 - threshold:
            return f"**{num:.3f}** 🔴"
        if num > 1 + threshold:
            return f"**{num:.3f}** 🟢"
        return f"{num:.3f}"
    except (TypeError, ValueError):
        return str(val)


def _html_table(
    df: pd.DataFrame, columns: list[str], cond_col: str,
    fail_vals: set[str], pass_vals: set[str],
) -> str:
    """Render an HTML table with conditional row background colours."""
    lines = ["<table>", "<thead><tr>"]
    for c in columns:
        lines.append(f"  <th>{html_escape(c)}</th>")
    lines.append("</tr></thead>")
    lines.append("<tbody>")

    for _, row in df.iterrows():
        val = row.get(cond_col, "")
        if val in fail_vals:
            style = ' style="background-color: #f8d7da;"'
        elif val in pass_vals:
            style = ' style="background-color: #d4edda;"'
        else:
            style = ""
        lines.append(f"<tr{style}>")
        for c in columns:
            cell = html_escape(str(row.get(c, "")) if pd.notna(row.get(c)) else "")
            lines.append(f"  <td>{cell}</td>")
        lines.append("</tr>")

    lines.append("</tbody></table>")
    return "\n".join(lines)


def _md_table(df: pd.DataFrame, columns: list[str]) -> str:
    """Render a plain Markdown table."""
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in df.iterrows():
        cells = [
            str(row.get(c, "")) if pd.notna(row.get(c)) else ""
            for c in columns
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ── Markdown Report ───────────────────────────────────────────────────
def write_markdown(
    summary: pd.DataFrame,
    acc_df: pd.DataFrame, perf_df: pd.DataFrame,
    threshold: float, filename: str,
) -> None:
    """Write a single Markdown report: summary -> failures -> improvements."""

    with open(filename, "w", encoding="utf-8") as f:
        f.write("# Dynamo Benchmark Comparison Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # ── Overall summary table ──
        f.write("## Summary\n\n")
        overall = summary[summary["Category"] == "Overall"] if not summary.empty else pd.DataFrame()
        if not overall.empty:
            f.write(
                "| Type | Total | Target Pass | Baseline Pass "
                "| Target % | Baseline % "
                "| ❌ Fail | 📉 Drop | ✅ Pass | 📈 Improve "
                "| Ind Ratio | Eag Ratio |\n"
            )
            f.write(
                "|------|-------|-------------|---------------"
                "|----------|------------"
                "|---------|---------|---------|------------"
                "|-----------|----------|\n"
            )
            for _, row in overall.iterrows():
                typ = row["Type"]
                is_perf = typ == "Performance"
                nd = row.get("new_drop", 0) if is_perf else "/"
                ni = row.get("new_improve", 0) if is_perf else "/"
                ir = _fmt_ratio(row.get("ind_ratio"), threshold) if is_perf else "/"
                er = _fmt_ratio(row.get("eag_ratio"), threshold) if is_perf else "/"
                f.write(
                    f"| {typ} | {row['total']} "
                    f"| {row['target_passed']} | {row['baseline_passed']} "
                    f"| {row['target_passrate']}% | {row['baseline_passrate']}% "
                    f"| {row['new_fail']} | {nd} | {row['new_pass']} | {ni} "
                    f"| {ir} | {er} |\n"
                )
            f.write("\n")
        else:
            f.write("No summary data available.\n\n")

        # ── Suite-level overview (collapsible) ──
        by_suite = summary[summary["Level"] == "By Suite"] if not summary.empty else pd.DataFrame()
        if not by_suite.empty:
            f.write("<details>\n<summary><b>Overview by Suite</b></summary>\n\n")
            f.write(
                "| Type | Suite | Total | Target | Baseline "
                "| ❌ Fail | 📉 Drop | ✅ Pass | 📈 Improve "
                "| Ind Ratio | Eag Ratio |\n"
            )
            f.write(
                "|------|-------|-------|--------|----------"
                "|---------|---------|---------|------------"
                "|-----------|----------|\n"
            )
            for _, row in by_suite.iterrows():
                typ = row["Type"]
                is_perf = typ == "Performance"
                nd = row.get("new_drop", 0) if is_perf else "/"
                ni = row.get("new_improve", 0) if is_perf else "/"
                ir = _fmt_ratio(row.get("ind_ratio"), threshold) if is_perf else "/"
                er = _fmt_ratio(row.get("eag_ratio"), threshold) if is_perf else "/"
                f.write(
                    f"| {typ} | {row['Category']} | {row['total']} "
                    f"| {row['target_passed']} | {row['baseline_passed']} "
                    f"| {row['new_fail']} | {nd} | {row['new_pass']} | {ni} "
                    f"| {ir} | {er} |\n"
                )
            f.write("\n</details>\n\n")

        # ── NEW FAILURES & REGRESSIONS (shown first, most important) ──
        acc_fails = (
            acc_df[acc_df["comparison"] == "new_failed"]
            if not acc_df.empty else pd.DataFrame()
        )
        perf_dropped = (
            perf_df[perf_df["comparison"] == "new_dropped"]
            if not perf_df.empty else pd.DataFrame()
        )
        perf_failed = (
            perf_df[perf_df["comparison"] == "new_failed"]
            if not perf_df.empty else pd.DataFrame()
        )
        n_issues = len(acc_fails) + len(perf_dropped) + len(perf_failed)

        f.write(f"## ❌ New Failures & Regressions ({n_issues} models)\n\n")

        if n_issues == 0:
            f.write("None! 🎉\n\n")
        else:
            if not acc_fails.empty:
                f.write(f"### Accuracy Failures ({len(acc_fails)} models)\n\n")
                cols = [c for c in [
                    "suite", "data_type", "mode", "model",
                    "accuracy_target", "accuracy_baseline", "comparison",
                ] if c in acc_fails.columns]
                f.write(_html_table(acc_fails, cols, "comparison", FAIL_LABELS, PASS_LABELS))
                f.write("\n\n")

            perf_cols = [c for c in [
                "suite", "data_type", "mode", "model",
                "inductor_target", "eager_target",
                "inductor_baseline", "eager_baseline",
                "inductor_ratio", "eager_ratio", "comparison",
            ] if c in perf_df.columns] if not perf_df.empty else []

            if not perf_dropped.empty:
                f.write(
                    f"### Performance Regressions "
                    f"({len(perf_dropped)} models, ratio < {(1 - threshold) * 100:.0f}%)\n\n"
                )
                f.write(_html_table(perf_dropped, perf_cols, "comparison", FAIL_LABELS, PASS_LABELS))
                f.write("\n\n")

            if not perf_failed.empty:
                f.write(f"### Performance Failures ({len(perf_failed)} models)\n\n")
                f.write(_html_table(perf_failed, perf_cols, "comparison", FAIL_LABELS, PASS_LABELS))
                f.write("\n\n")

        # ── NEW PASSES & IMPROVEMENTS (shown second) ──
        acc_passes = (
            acc_df[acc_df["comparison"] == "new_passed"]
            if not acc_df.empty else pd.DataFrame()
        )
        perf_improved = (
            perf_df[perf_df["comparison"] == "new_improved"]
            if not perf_df.empty else pd.DataFrame()
        )
        perf_new_pass = (
            perf_df[perf_df["comparison"] == "new_passed"]
            if not perf_df.empty else pd.DataFrame()
        )
        n_good = len(acc_passes) + len(perf_improved) + len(perf_new_pass)

        f.write(f"## ✅ New Passes & Improvements ({n_good} models)\n\n")

        if n_good == 0:
            f.write("None.\n\n")
        else:
            if not acc_passes.empty:
                f.write(f"### Accuracy New Passes ({len(acc_passes)} models)\n\n")
                cols = [c for c in [
                    "suite", "data_type", "mode", "model",
                    "accuracy_target", "accuracy_baseline", "comparison",
                ] if c in acc_passes.columns]
                f.write(_html_table(acc_passes, cols, "comparison", FAIL_LABELS, PASS_LABELS))
                f.write("\n\n")

            perf_cols = [c for c in [
                "suite", "data_type", "mode", "model",
                "inductor_target", "eager_target",
                "inductor_baseline", "eager_baseline",
                "inductor_ratio", "eager_ratio", "comparison",
            ] if c in perf_df.columns] if not perf_df.empty else []

            if not perf_improved.empty:
                f.write(
                    f"### Performance Improvements "
                    f"({len(perf_improved)} models, ratio > {(1 + threshold) * 100:.0f}%)\n\n"
                )
                f.write(_html_table(perf_improved, perf_cols, "comparison", FAIL_LABELS, PASS_LABELS))
                f.write("\n\n")

            if not perf_new_pass.empty:
                f.write(f"### Performance New Passes ({len(perf_new_pass)} models)\n\n")
                f.write(_html_table(perf_new_pass, perf_cols, "comparison", FAIL_LABELS, PASS_LABELS))
                f.write("\n\n")

        # ── Not-run models (collapsible) ──
        acc_missing = (
            acc_df[acc_df["comparison"] == "not_run"]
            if not acc_df.empty else pd.DataFrame()
        )
        if not acc_missing.empty:
            f.write(
                f"<details>\n"
                f"<summary><b>⚠️ Not Run in Target ({len(acc_missing)} models)</b></summary>\n\n"
            )
            cols = [c for c in [
                "suite", "data_type", "mode", "model", "accuracy_baseline",
            ] if c in acc_missing.columns]
            f.write(_md_table(acc_missing, cols))
            f.write("\n\n</details>\n\n")

    log.info("Markdown report written to %s", filename)


# ── Console Output ────────────────────────────────────────────────────
def print_report(
    target_acc_n: int, target_perf_n: int,
    baseline_acc_n: int, baseline_perf_n: int,
    acc_merged: pd.DataFrame, perf_merged: pd.DataFrame,
    output_file: str, title: str = "DYNAMO BENCHMARK COMPARISON",
) -> None:
    """Print a structured summary to stdout."""
    W = 64
    sep = "=" * W
    thin = "-" * W

    print(f"\n{sep}")
    print(f"{title:^{W}}")
    print(sep)

    print(f"  {'Target records:':<28} acc={target_acc_n:<6} perf={target_perf_n}")
    print(f"  {'Baseline records:':<28} acc={baseline_acc_n:<6} perf={baseline_perf_n}")
    print(thin)

    if not acc_merged.empty:
        comp = acc_merged["comparison"]
        total = len(acc_merged)
        tgt_pass = int(acc_merged["accuracy_target"].apply(_is_acc_pass).sum())
        bsl_pass = int(acc_merged["accuracy_baseline"].apply(_is_acc_pass).sum())

        print(f"  ACCURACY ({total} models)")
        if total:
            print(f"    Target pass rate:    {tgt_pass}/{total} ({tgt_pass / total * 100:.1f}%)")
            print(f"    Baseline pass rate:  {bsl_pass}/{total} ({bsl_pass / total * 100:.1f}%)")

        for label, emoji in [
            ("new_failed", "❌ New failures"),
            ("new_passed", "✅ New passes"),
            ("not_run", "⚠️  Not run"),
        ]:
            count = int((comp == label).sum())
            if count:
                print(f"    {emoji + ':':<24} {count}")
        print(thin)

    if not perf_merged.empty:
        comp = perf_merged["comparison"]
        total = len(perf_merged)
        tgt_pass = int(perf_merged["inductor_target"].apply(_is_positive).sum())
        bsl_pass = int(perf_merged["inductor_baseline"].apply(_is_positive).sum())

        print(f"  PERFORMANCE ({total} models)")
        if total:
            print(f"    Target pass rate:    {tgt_pass}/{total} ({tgt_pass / total * 100:.1f}%)")
            print(f"    Baseline pass rate:  {bsl_pass}/{total} ({bsl_pass / total * 100:.1f}%)")

        for label, emoji in [
            ("new_failed", "❌ New failures"),
            ("new_dropped", "📉 Dropped Cases"),
            ("new_passed", "✅ New passes"),
            ("new_improved", "📈 Improved Cases"),
        ]:
            count = int((comp == label).sum())
            if count:
                print(f"    {emoji + ':':<24} {count}")

        ind_gm = _geomean(
            perf_merged["inductor_ratio"] if "inductor_ratio" in perf_merged.columns
            else pd.Series(dtype=float)
        )
        eag_gm = _geomean(
            perf_merged["eager_ratio"] if "eager_ratio" in perf_merged.columns
            else pd.Series(dtype=float)
        )
        if pd.notna(ind_gm):
            print(f"    Inductor ratio (geomean): {ind_gm:.3f}")
        if pd.notna(eag_gm):
            print(f"    Eager ratio (geomean):    {eag_gm:.3f}")
        print(thin)

    print(f"  Output: {output_file}")
    print(f"{sep}\n")


# ── File Output (Excel / CSV) ────────────────────────────────────────
def write_excel(
    summary: pd.DataFrame, acc: pd.DataFrame, perf: pd.DataFrame, path: str,
    pt2e_acc: pd.DataFrame | None = None, pt2e_perf: pd.DataFrame | None = None,
) -> None:
    placeholder = pd.DataFrame({"Info": ["No data available"]})
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        (summary if not summary.empty else placeholder).to_excel(
            w, sheet_name="Summary", index=False,
        )
        (acc if not acc.empty else placeholder).to_excel(
            w, sheet_name="Accuracy Details", index=False,
        )
        (perf if not perf.empty else placeholder).to_excel(
            w, sheet_name="Performance Details", index=False,
        )
        if pt2e_acc is not None and not pt2e_acc.empty:
            pt2e_acc.to_excel(w, sheet_name="PT2E Accuracy", index=False)
        if pt2e_perf is not None and not pt2e_perf.empty:
            pt2e_perf.to_excel(w, sheet_name="PT2E Performance", index=False)
    log.info("Excel written to %s", path)


def write_csv(
    summary: pd.DataFrame, acc: pd.DataFrame, perf: pd.DataFrame, base: str,
) -> None:
    for df, suffix in [
        (summary, "_summary"),
        (acc, "_accuracy"),
        (perf, "_performance"),
    ]:
        if not df.empty:
            path = f"{base}{suffix}.csv"
            df.to_csv(path, index=False, na_rep="")
            log.info("Written %s", path)
