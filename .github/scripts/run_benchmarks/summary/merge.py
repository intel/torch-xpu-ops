"""Merge, classification, and summary logic for benchmark comparison."""

import numpy as np
import pandas as pd

from .constants import (
    ACC_OUTPUT_COLS,
    MERGE_KEYS,
    PERF_OUTPUT_COLS,
    PT2E_ACC_OUTPUT_COLS,
    PT2E_ACC_THRESHOLDS,
    PT2E_PERF_OUTPUT_COLS,
    SUMMARY_LEVELS,
)


# ── Comparison Helpers ────────────────────────────────────────────────
def _is_acc_pass(val) -> bool:
    return pd.notna(val) and "pass" in str(val)


def _is_acc_fail(val) -> bool:
    return pd.notna(val) and "pass" not in str(val) and str(val).strip() != ""


def _extract_pt2e_top1(val) -> float:
    """Extract numeric top1 value from PT2E accuracy string like 'pass_top1=56.556'."""
    if pd.isna(val):
        return np.nan
    s = str(val)
    if "top1=" in s:
        try:
            return float(s.split("top1=")[1])
        except (ValueError, IndexError):
            return np.nan
    return np.nan


def _classify_pt2e_accuracy(row: pd.Series) -> str:
    """Classify PT2E accuracy by comparing top1 values with dtype-specific thresholds."""
    tgt, bsl = row.get("accuracy_target"), row.get("accuracy_baseline")
    tgt_top1 = _extract_pt2e_top1(tgt)
    bsl_top1 = _extract_pt2e_top1(bsl)
    data_type = str(row.get("data_type", "float32"))

    # Determine threshold based on dtype (int8 variants all use int8 threshold)
    dt_key = "int8" if "int8" in data_type else data_type
    threshold = PT2E_ACC_THRESHOLDS.get(dt_key, 0.0)

    tgt_valid = pd.notna(tgt_top1) and tgt_top1 > 0
    bsl_valid = pd.notna(bsl_top1) and bsl_top1 > 0

    if tgt_valid and bsl_valid:
        # Compare: if target drops more than threshold below baseline, it's a regression
        if bsl_top1 > 0 and (bsl_top1 - tgt_top1) / bsl_top1 > threshold:
            return "new_failed"
        if bsl_top1 > 0 and (tgt_top1 - bsl_top1) / bsl_top1 > threshold:
            return "new_passed"
        return "no_changed"
    if tgt_valid and not bsl_valid:
        return "new_case"
    if bsl_valid and not tgt_valid:
        return "not_run"
    return "no_changed"


def _is_positive(val) -> bool:
    return pd.notna(val) and isinstance(val, (int, float)) and val > 0


def _geomean(series: pd.Series) -> float:
    """Geometric mean of positive values; NaN if none."""
    vals = series.dropna()
    vals = vals[vals > 0]
    if vals.empty:
        return np.nan
    return float(np.exp(np.log(vals).mean()))


# ── Accuracy Merge ────────────────────────────────────────────────────
def _classify_accuracy(row: pd.Series) -> str:
    """
    Classify an accuracy comparison row.

    Labels:
        no_changed       – same status in both
        new_passed       – was failing in baseline, now passing (improvement)
        new_failed       – was passing in baseline, now failing (regression)
        new_case         – not in baseline, now passing in target
        new_case_failed  – not in baseline, failing in target
        not_run          – was passing in baseline, absent from target
    """
    tgt, bsl = row.get("accuracy_target"), row.get("accuracy_baseline")
    tgt_pass, bsl_pass = _is_acc_pass(tgt), _is_acc_pass(bsl)
    tgt_fail, bsl_fail = _is_acc_fail(tgt), _is_acc_fail(bsl)

    if tgt_pass and bsl_pass:
        return "no_changed"
    if tgt_pass and bsl_fail:
        return "new_passed"       # was failing, now passes
    if tgt_pass and not bsl_fail:
        return "new_case"         # target passes, baseline absent
    if bsl_pass and tgt_fail:
        return "new_failed"       # REGRESSION
    if bsl_pass and not tgt_fail:
        return "not_run"          # baseline passes, target absent
    if tgt_fail and not bsl_fail and not bsl_pass:
        return "new_case_failed"  # new test that fails, baseline absent
    return "no_changed"


def merge_accuracy(
    target_records: list[dict], baseline_records: list[dict],
) -> pd.DataFrame:
    """Outer-join target and baseline accuracy records, classify each model."""
    tgt = pd.DataFrame(target_records)
    bsl = pd.DataFrame(baseline_records)

    if tgt.empty and bsl.empty:
        return pd.DataFrame(columns=ACC_OUTPUT_COLS)

    if tgt.empty:
        tgt = pd.DataFrame(columns=["suite", "data_type", "mode", "model", "batch_size", "accuracy"])
    if bsl.empty:
        bsl = pd.DataFrame(columns=["suite", "data_type", "mode", "model", "batch_size", "accuracy"])

    merged = pd.merge(
        tgt, bsl, on=MERGE_KEYS, how="outer", suffixes=("_target", "_baseline"),
    )
    for col in ("batch_size_target", "batch_size_baseline"):
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").astype("Int64")

    # Use PT2E-specific classifier for pt2e suite (numeric top1 comparison)
    is_pt2e = merged["suite"] == "pt2e" if "suite" in merged.columns else pd.Series(False, index=merged.index)
    merged["comparison"] = ""
    if is_pt2e.any():
        merged.loc[is_pt2e, "comparison"] = merged[is_pt2e].apply(_classify_pt2e_accuracy, axis=1)
    if (~is_pt2e).any():
        merged.loc[~is_pt2e, "comparison"] = merged[~is_pt2e].apply(_classify_accuracy, axis=1)

    for c in ACC_OUTPUT_COLS:
        if c not in merged.columns:
            merged[c] = None
    return merged[ACC_OUTPUT_COLS].sort_values(MERGE_KEYS).reset_index(drop=True)


# ── PT2E Accuracy Merge (pivoted by dtype) ────────────────────────────
def _pt2e_acc_compare(tgt: float, bsl: float, threshold: float) -> str:
    """Compare two top1/top5 values with a relative threshold."""
    tgt_valid = pd.notna(tgt) and tgt > 0
    bsl_valid = pd.notna(bsl) and bsl > 0
    if tgt_valid and bsl_valid:
        if bsl > 0 and (bsl - tgt) / bsl > threshold:
            return "new_failed"
        if bsl > 0 and (tgt - bsl) / bsl > threshold:
            return "new_passed"
        return "no_changed"
    if tgt_valid and not bsl_valid:
        return "new_case"
    if bsl_valid and not tgt_valid:
        return "not_run"
    return "no_changed"


def merge_pt2e_accuracy(
    target_records: list[dict], baseline_records: list[dict],
) -> pd.DataFrame:
    """Merge PT2E accuracy records into a pivoted format by dtype (fp32/int8)."""
    if not target_records and not baseline_records:
        return pd.DataFrame(columns=PT2E_ACC_OUTPUT_COLS)

    tgt = pd.DataFrame(target_records) if target_records else pd.DataFrame()
    bsl = pd.DataFrame(baseline_records) if baseline_records else pd.DataFrame()

    # Build lookup: (mode, model, data_type) -> {top1, top5}
    def _build_map(df: pd.DataFrame) -> dict:
        m: dict[tuple, dict] = {}
        if df.empty:
            return m
        for _, row in df.iterrows():
            key = (row["mode"], row["model"], row.get("data_type", ""))
            m[key] = {"top1": row.get("top1", np.nan), "top5": row.get("top5", np.nan)}
        return m

    tgt_map = _build_map(tgt)
    bsl_map = _build_map(bsl)

    # Collect all (mode, model) pairs
    all_keys: set[tuple[str, str]] = set()
    for mode, model, _ in list(tgt_map.keys()) + list(bsl_map.keys()):
        all_keys.add((mode, model))

    rows = []
    for mode, model in sorted(all_keys):
        # For each metric (top1, top5)
        for category in ("top1", "top5"):
            fp32_tgt = tgt_map.get((mode, model, "float32"), {}).get(category, np.nan)
            int8_tgt = tgt_map.get((mode, model, "int8"), {}).get(category, np.nan)
            fp32_bsl = bsl_map.get((mode, model, "float32"), {}).get(category, np.nan)
            int8_bsl = bsl_map.get((mode, model, "int8"), {}).get(category, np.nan)

            # Skip rows where all values are NaN
            if all(pd.isna(v) for v in [fp32_tgt, int8_tgt, fp32_bsl, int8_bsl]):
                continue

            # Compute int8/fp32 ratios
            int8_fp32_tgt = (int8_tgt / fp32_tgt) if (
                pd.notna(int8_tgt) and pd.notna(fp32_tgt) and fp32_tgt > 0
            ) else np.nan
            int8_fp32_bsl = (int8_bsl / fp32_bsl) if (
                pd.notna(int8_bsl) and pd.notna(fp32_bsl) and fp32_bsl > 0
            ) else np.nan

            # Per-dtype comparison
            fp32_thresh = PT2E_ACC_THRESHOLDS.get("float32", 0.0)
            int8_thresh = PT2E_ACC_THRESHOLDS.get("int8", 0.05)
            fp32_comp = _pt2e_acc_compare(fp32_tgt, fp32_bsl, fp32_thresh)
            int8_comp = _pt2e_acc_compare(int8_tgt, int8_bsl, int8_thresh)

            rows.append({
                "suite": "pt2e", "mode": mode, "model": model,
                "category": category,
                "fp32_target": round(fp32_tgt, 3) if pd.notna(fp32_tgt) else np.nan,
                "int8_target": round(int8_tgt, 3) if pd.notna(int8_tgt) else np.nan,
                "int8/fp32_target": round(int8_fp32_tgt, 4) if pd.notna(int8_fp32_tgt) else np.nan,
                "fp32_baseline": round(fp32_bsl, 3) if pd.notna(fp32_bsl) else np.nan,
                "int8_baseline": round(int8_bsl, 3) if pd.notna(int8_bsl) else np.nan,
                "int8/fp32_baseline": round(int8_fp32_bsl, 4) if pd.notna(int8_fp32_bsl) else np.nan,
                "fp32_comparison": fp32_comp,
                "int8_comparison": int8_comp,
            })

    return pd.DataFrame(rows, columns=PT2E_ACC_OUTPUT_COLS)


# ── PT2E Performance Merge (pivoted by quantization) ──────────────────
def merge_pt2e_performance(
    target_records: list[dict], baseline_records: list[dict],
    threshold: float,
) -> pd.DataFrame:
    """Merge PT2E performance records into a pivoted format by quantization."""
    if not target_records and not baseline_records:
        return pd.DataFrame(columns=PT2E_PERF_OUTPUT_COLS)

    tgt = pd.DataFrame(target_records) if target_records else pd.DataFrame()
    bsl = pd.DataFrame(baseline_records) if baseline_records else pd.DataFrame()

    # Build lookup: (mode, model) -> {fp32: throughput, symm: throughput, asymm: throughput}
    def _build_map(df: pd.DataFrame) -> dict:
        m: dict[tuple, dict] = {}
        if df.empty:
            return m
        for _, row in df.iterrows():
            key = (row["mode"], row["model"])
            dt = str(row.get("data_type", ""))
            quant = str(row.get("quantization", "")).strip()
            throughput = row.get("throughput", np.nan)

            if dt == "float32":
                slot = "fp32"
            elif quant in ("symm", "symmetric"):
                slot = "symm"
            elif quant in ("asymm", "asymmetric"):
                slot = "asymm"
            else:
                slot = "fp32" if "float32" in dt else "symm"

            m.setdefault(key, {})[slot] = throughput
        return m

    tgt_map = _build_map(tgt)
    bsl_map = _build_map(bsl)

    all_keys: set[tuple[str, str]] = set()
    for k in list(tgt_map.keys()) + list(bsl_map.keys()):
        all_keys.add(k)

    rows = []
    for mode, model in sorted(all_keys):
        tgt_vals = tgt_map.get((mode, model), {})
        bsl_vals = bsl_map.get((mode, model), {})

        fp32_t = tgt_vals.get("fp32", np.nan)
        symm_t = tgt_vals.get("symm", np.nan)
        asymm_t = tgt_vals.get("asymm", np.nan)
        fp32_b = bsl_vals.get("fp32", np.nan)
        symm_b = bsl_vals.get("symm", np.nan)
        asymm_b = bsl_vals.get("asymm", np.nan)

        # Compute ratios (throughput: higher is better, so ratio = target/baseline)
        def _ratio(val, fp32):
            if pd.notna(val) and pd.notna(fp32) and fp32 > 0:
                return round(val / fp32, 4)
            return np.nan

        symm_fp32_t = _ratio(symm_t, fp32_t)
        asymm_fp32_t = _ratio(asymm_t, fp32_t)
        symm_fp32_b = _ratio(symm_b, fp32_b)
        asymm_fp32_b = _ratio(asymm_b, fp32_b)

        # Overall comparison: check if any throughput regressed
        comp = "no_changed"
        for t_val, b_val in [(fp32_t, fp32_b), (symm_t, symm_b), (asymm_t, asymm_b)]:
            t_ok = pd.notna(t_val) and t_val > 0
            b_ok = pd.notna(b_val) and b_val > 0
            if t_ok and b_ok:
                ratio = t_val / b_val
                if ratio < 1 - threshold:
                    comp = "new_dropped"
                    break
                if ratio > 1 + threshold:
                    comp = "new_improved"
            elif t_ok and not b_ok:
                comp = "new_passed"
            elif b_ok and not t_ok:
                comp = "new_failed"
                break

        rows.append({
            "suite": "pt2e", "mode": mode, "model": model,
            "fp32_target": round(fp32_t, 4) if pd.notna(fp32_t) else np.nan,
            "symm_target": round(symm_t, 4) if pd.notna(symm_t) else np.nan,
            "asymm_target": round(asymm_t, 4) if pd.notna(asymm_t) else np.nan,
            "symm/fp32_target": symm_fp32_t,
            "asymm/fp32_target": asymm_fp32_t,
            "fp32_baseline": round(fp32_b, 4) if pd.notna(fp32_b) else np.nan,
            "symm_baseline": round(symm_b, 4) if pd.notna(symm_b) else np.nan,
            "asymm_baseline": round(asymm_b, 4) if pd.notna(asymm_b) else np.nan,
            "symm/fp32_baseline": symm_fp32_b,
            "asymm/fp32_baseline": asymm_fp32_b,
            "comparison": comp,
        })

    return pd.DataFrame(rows, columns=PT2E_PERF_OUTPUT_COLS)


# ── Performance Merge ─────────────────────────────────────────────────
def _classify_performance(row: pd.Series, threshold: float) -> str:
    """
    Classify a performance comparison row.

    Labels:
        no_changed    – within threshold or both invalid
        new_dropped   – significantly slower (regression)
        new_improved  – significantly faster (improvement)
        new_passed    – target valid, baseline invalid (fix)
        new_failed    – baseline valid, target invalid (regression)
    """
    ind_tgt = row.get("inductor_target")
    ind_bsl = row.get("inductor_baseline")
    tgt_ok, bsl_ok = _is_positive(ind_tgt), _is_positive(ind_bsl)

    if tgt_ok and bsl_ok:
        ind_r = row.get("inductor_ratio")
        eag_r = row.get("eager_ratio")
        if (pd.notna(ind_r) and ind_r < 1 - threshold) or \
           (pd.notna(eag_r) and eag_r < 1 - threshold):
            return "new_dropped"
        if (pd.notna(ind_r) and ind_r > 1 + threshold) or \
           (pd.notna(eag_r) and eag_r > 1 + threshold):
            return "new_improved"
        return "no_changed"
    if tgt_ok:
        return "new_passed"
    if bsl_ok:
        return "new_failed"
    return "no_changed"


def merge_performance(
    target_records: list[dict], baseline_records: list[dict],
    threshold: float,
) -> pd.DataFrame:
    """Outer-join target and baseline performance records, compute ratios, classify."""
    tgt = pd.DataFrame(target_records)
    bsl = pd.DataFrame(baseline_records)

    if tgt.empty and bsl.empty:
        return pd.DataFrame(columns=PERF_OUTPUT_COLS)

    if tgt.empty:
        tgt = pd.DataFrame(columns=["suite", "data_type", "mode", "model", "batch_size", "eager", "inductor", "speedup"])
    if bsl.empty:
        bsl = pd.DataFrame(columns=["suite", "data_type", "mode", "model", "batch_size", "eager", "inductor", "speedup"])

    merged = pd.merge(
        tgt, bsl, on=MERGE_KEYS, how="outer", suffixes=("_target", "_baseline"),
    )
    for col in ("batch_size_target", "batch_size_baseline"):
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").astype("Int64")

    # Ratios: baseline / target  (>1 means target is faster)
    for metric in ("inductor", "eager"):
        tgt_col = f"{metric}_target"
        bsl_col = f"{metric}_baseline"
        ratio_col = f"{metric}_ratio"
        if tgt_col in merged.columns and bsl_col in merged.columns:
            mask = merged[tgt_col].notna() & (merged[tgt_col] > 0)
            merged[ratio_col] = np.where(
                mask,
                merged[bsl_col].astype(float) / merged[tgt_col].astype(float),
                np.nan,
            )

    for col in ("inductor_target", "inductor_baseline", "eager_target", "eager_baseline"):
        if col in merged.columns:
            merged[col] = merged[col].round(4)
    for col in ("inductor_ratio", "eager_ratio"):
        if col in merged.columns:
            merged[col] = merged[col].round(3)

    merged["comparison"] = merged.apply(
        lambda r: _classify_performance(r, threshold), axis=1,
    )

    for c in PERF_OUTPUT_COLS:
        if c not in merged.columns:
            merged[c] = None
    return merged[PERF_OUTPUT_COLS].sort_values(MERGE_KEYS).reset_index(drop=True)


# ── Summary Generation ────────────────────────────────────────────────
def _group_metrics(group: pd.DataFrame, is_perf: bool) -> pd.Series:
    """Compute summary metrics for a group of rows."""
    comp = group["comparison"]
    if is_perf:
        tgt_passed = group["inductor_target"].apply(_is_positive).sum()
        bsl_passed = group["inductor_baseline"].apply(_is_positive).sum()
    else:
        tgt_passed = group["accuracy_target"].apply(_is_acc_pass).sum()
        bsl_passed = group["accuracy_baseline"].apply(_is_acc_pass).sum()

    result = {
        "target_passed": tgt_passed,
        "baseline_passed": bsl_passed,
        "total": len(group),
        "new_fail": int((comp == "new_failed").sum()),
        "new_pass": int((comp == "new_passed").sum()),
        "new_drop": int((comp == "new_dropped").sum()) if is_perf else 0,
        "new_improve": int((comp == "new_improved").sum()) if is_perf else 0,
    }
    if is_perf:
        result["ind_ratio"] = _geomean(
            group["inductor_ratio"] if "inductor_ratio" in group.columns
            else pd.Series(dtype=float)
        )
        result["eag_ratio"] = _geomean(
            group["eager_ratio"] if "eager_ratio" in group.columns
            else pd.Series(dtype=float)
        )
    if not is_perf:
        result["not_run"] = int((comp == "not_run").sum())
    return pd.Series(result)


def generate_summary(
    acc_merged: pd.DataFrame, perf_merged: pd.DataFrame,
) -> pd.DataFrame:
    """Generate summary at all grouping levels and combine."""
    frames = []
    for level_name, group_cols in SUMMARY_LEVELS:
        for df, label, is_perf in [
            (acc_merged, "Accuracy", False),
            (perf_merged, "Performance", True),
        ]:
            if df.empty:
                continue
            if group_cols:
                grp = (
                    df.groupby(group_cols, dropna=False)
                    .apply(lambda g: _group_metrics(g, is_perf), include_groups=False)
                    .reset_index()
                )
                grp["Category"] = grp[group_cols].astype(str).agg("_".join, axis=1)
            else:
                grp = _group_metrics(df, is_perf).to_frame().T
                grp["Category"] = "Overall"
            grp["Level"] = level_name
            grp["Type"] = label
            frames.append(grp)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)

    # Pass rates
    combined["target_passrate"] = (
        combined["target_passed"] / combined["total"] * 100
    ).round(2)
    combined["baseline_passrate"] = (
        combined["baseline_passed"] / combined["total"] * 100
    ).round(2)

    # Integer columns
    for c in ("target_passed", "baseline_passed", "total",
              "new_fail", "new_drop", "new_pass", "new_improve"):
        if c in combined.columns:
            combined[c] = pd.to_numeric(combined[c], errors="coerce").astype("Int64")
    for c in ("ind_ratio", "eag_ratio"):
        if c in combined.columns:
            combined[c] = combined[c].round(3)

    # Sort by level priority, then type, then category
    level_order = {name: i for i, (name, _) in enumerate(SUMMARY_LEVELS)}
    combined["_sort"] = combined["Level"].map(level_order)
    combined.sort_values(["_sort", "Type", "Category"], inplace=True)

    cols = [
        "Level", "Type", "Category",
        "target_passed", "baseline_passed", "total",
        "target_passrate", "baseline_passrate",
        "new_fail", "new_drop", "new_pass", "new_improve",
        "ind_ratio", "eag_ratio",
    ]
    for c in cols:
        if c not in combined.columns:
            combined[c] = np.nan
    return combined[cols].reset_index(drop=True)
