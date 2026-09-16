#!/usr/bin/env python3
"""Component acceptance comparison report generator.

Compares a baseline vs a target build of one component (e.g. oneDNN, Triton)
across:
  1. Unit Tests  - JUnit xml files, key = (test file, test class, test name)
  2. Accuracy    - *accuracy.csv,   key = (suite, dtype, mode, name, scenario)
  3. Performance - *performance.csv, key = (suite, dtype, mode, name, scenario)

Component and versions come from ACC_COMPONENT / ACC_TARGET_LABEL /
ACC_BASE_LABEL (a label may be "<component> <version>"). Emits an HTML report,
an XLSX workbook, and a brief GITHUB_STEP_SUMMARY. Standard library only.
"""

import csv
import html
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.environ.get("ACC_BASE_DIR") or os.path.join(HERE, "3.7.2")
TARGET_DIR = os.environ.get("ACC_TARGET_DIR") or os.path.join(HERE, "3.8.0")
BASE_LABEL = os.environ.get("ACC_BASE_LABEL", "baseline")
TARGET_LABEL = os.environ.get("ACC_TARGET_LABEL", "target")
OUT_DIR = os.environ.get("ACC_OUT_DIR") or HERE


def _split_component_version(label):
    """Split a label like 'onednn v3.13.2' into (component, version)."""
    toks = (label or "").split()
    vers = [t for t in toks if re.fullmatch(r"v?\d[\w.+-]*", t)]
    comp = " ".join(t for t in toks if t not in vers).strip()
    return comp, (" ".join(vers).strip() or (label or ""))


# Component under test and its two versions, derived from the labels (or set
# ACC_COMPONENT explicitly) so the report is not tied to any specific component.
COMPONENT = os.environ.get("ACC_COMPONENT") or _split_component_version(TARGET_LABEL)[0] or "component"
TARGET_VERSION = _split_component_version(TARGET_LABEL)[1]
BASE_VERSION = _split_component_version(BASE_LABEL)[1]

# Blank-target classification: a UT case that passed in baseline but is blank in
# target is REMOVED (deselected: pytest did not collect it, fed back via
# REMOVED_LIST_FILE) or NOT-RUN / lost (collectable but the run skipped it).
REMOVED_MSG = "Deselected on target (pytest did not collect this case)"
CRASH_MSG = "Collection failure: file crashed at import, no cases run"
NOT_RUN_MSG = "Baseline passed but target did not run (collectable but skipped, not a code regression)"
TIMEOUT_MSG = ("xdist worker crashed on setup (hung / timed-out case on the same "
               "worker); tracked as a timeout / hang, not a code regression")
# pytest-xdist reports a hung/timed-out case (and its collateral cases on the
# same worker) as `failed on setup with "worker 'gwN' crashed while running ..."`.
WORKER_CRASH_RE = re.compile(
    r"worker '?gw\d+'? crashed while running|failed on setup with", re.IGNORECASE)


def load_removed_list(path):
    """Set of (test_file, short_class, name) that pytest could not collect on the
    target node (deselected => removed), from a TSV of `file<TAB>class<TAB>name`.
    Missing/empty path -> empty set."""
    keys = set()
    if not path or not os.path.isfile(path):
        return keys
    try:
        with open(path, errors="replace") as fh:
            for ln in fh:
                parts = ln.rstrip("\n").split("\t")
                if len(parts) == 3:
                    f, cls, name = parts
                    keys.add((f, cls.rsplit(".", 1)[-1], name))
    except OSError:
        pass
    return keys


# Path to a TSV (file<TAB>class<TAB>name) of baseline-pass / target-blank cases
# that pytest could NOT collect on the target node (deselected => removed). Such
# cases are labelled REMOVED; every other baseline-pass / target-blank case is
# NOT-RUN (lost) and can be re-run and merged back. "" disables.
REMOVED_LIST_FILE = ""


PASS_ACC = {"pass", "pass_due_to_skip"}
# accuracy statuses where the model never really ran / couldn't be compared
NOTRUN_ACC = {"eager_fail_to_run", "fail_to_run", "model_fail_to_load",
              "eager_1st_run_OOM", "eager_2nd_run_OOM", "out of memory",
              "timeout", "Memory>0.9", "UR_RESULT_ERROR"}
PERF_REG_THRESHOLD = 0.90   # target/base below this = regression
PERF_IMP_THRESHOLD = 1.10   # target/base above this = improvement


# --------------------------------------------------------------------------- #
# Parsers
# --------------------------------------------------------------------------- #
def find_files(root, pattern):
    rx = re.compile(pattern)
    out = []
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if rx.search(f):
                out.append(os.path.join(dirpath, f))
    return out


# E2E accuracy/perf metadata may live in columns (legacy schema) or be encoded
# in the per-suite filename: inductor_<suite>_<dtype>_<mode>_xpu_<scenario>.csv
# (the source filename is also appended as the last column of every row).
_E2E_FNAME_RX = re.compile(
    r"inductor_(huggingface|timm_models|torchbench|pt2e)_(.+)_(inference|training)_xpu_(accuracy|performance)")


def _e2e_meta(row, path):
    """(suite, dtype, mode, scenario) from columns if present, else the filename."""
    suite = (row.get("suite") or "").strip()
    dtype = (row.get("dtype") or "").strip()
    mode = (row.get("mode") or "").strip()
    scenario = (row.get("scenario") or "").strip()
    if suite and dtype and mode:
        return suite, dtype, mode, scenario
    for cand in list(row.values()) + [os.path.basename(path)]:
        m = _E2E_FNAME_RX.search(str(cand or ""))
        if m:
            return m.group(1), m.group(2), m.group(3), m.group(4)
    return suite, dtype, mode, scenario



def _brief_msg(text, limit=280):
    """First non-empty line of a failure/error message, trimmed."""
    if not text:
        return ""
    for line in text.replace("\r", "\n").split("\n"):
        line = line.strip()
        if line:
            return line[:limit]
    return ""


def _ut_norm_class(cls):
    """Short test-class name: the final dotted component. A Cases-block / JUnit
    class may be a full module path + class (e.g.
    `ops.test.xpu.export.test_retraceability_xpu.RetraceExportNonStrictTestExport`
    or `...third_party.torch-xpu-ops....Foo`); only the last component identifies
    the class, so every module-path form (with or without the `_xpu` file suffix)
    maps to the same key."""
    return cls.rsplit(".", 1)[-1] if "." in cls else cls


def _ut_norm_name(name):
    """Test-name key: drop a trailing `_xpu` device suffix so a case matches
    whether or not the name carries it (e.g. `test_pool_backward_xpu` ==
    `test_pool_backward`). Applied symmetrically to the test and issue sides."""
    return name[:-4] if name.endswith("_xpu") else name


def _ut_key(cls, name):
    """Canonical UT match key: short class + device-suffix-stripped name."""
    return (_ut_norm_class(cls), _ut_norm_name(name))


def _canon_xpu_file(segments):
    """Build the canonical xpu mirror file 'test/xpu/<sub>/<name>_xpu.py' from
    path or module segments. Any root-dir prefix before 'test' is dropped, a
    redundant 'xpu' segment is collapsed, and the basename is given a single
    '_xpu' suffix. Returns None when the segments are not a 'test.*' path."""
    segs = [s for s in segments if s]
    if "test" in segs:
        segs = segs[segs.index("test"):]
    if len(segs) < 2 or segs[0] != "test":
        return None
    sub = segs[1:]
    if sub[0] == "xpu":
        sub = sub[1:]
    if not sub:
        return None
    name = sub[-1]
    if name.endswith(".py"):
        name = name[:-3]
    if not name.endswith("_xpu"):
        name += "_xpu"
    return "test/xpu/" + "/".join(sub[:-1] + [name]) + ".py"


def _norm_ut_file(path):
    """Normalise a test-file path so the same case matches regardless of the root
    directory it was run from (and of the '_xpu' filename suffix)."""
    p = (path or "").replace("\\", "/").strip()
    return _canon_xpu_file(p.split("/")) or p


def _ut_file_class(tc, xml_file):
    """Determine a canonical (test_file, test_class) for a JUnit <testcase> so the
    same case matches across runs even when run from a different root directory
    (which otherwise makes the file path and classname differ):
      1. `file` attribute present -> use it (root-normalised);
      2. classname under `third_party.torch-xpu-ops.` -> derive the file path;
      3. upstream-style dotted classname (e.g. `test.dynamo.test_functions.Foo`,
         emitted when the same case runs from a different root) -> map to the xpu
         mirror file `test/xpu/dynamo/test_functions_xpu.py`;
      4. otherwise fall back to the xml file name.
    The test class is always reduced to its short (last) component; test files are
    canonicalised to their `test/xpu/.../<name>_xpu.py` mirror form.
    """
    cls = tc.get("classname", "")
    klass = cls.rsplit(".", 1)[-1] if "." in cls else cls
    file_attr = (tc.get("file") or "").strip()
    if file_attr:
        return _norm_ut_file(file_attr), klass
    marker = "third_party.torch-xpu-ops."
    if marker in cls:
        rest = cls.split(marker, 1)[1]
        if "." in rest:
            path, klass = rest.rsplit(".", 1)
            return _norm_ut_file(path.replace(".", "/") + ".py"), klass
    if "." in cls:
        canon = _canon_xpu_file(cls.rsplit(".", 1)[0].split("."))
        if canon:
            return canon, klass
    return xml_file, klass


def _collection_file(cls, name):
    """If a JUnit (class, name) pair is a whole-file collection-failure marker
    (empty class + dotted module path as the name, emitted when a test file
    crashes at import so none of its cases run), return the canonical crashed
    test file 'test/xpu/.../<name>_xpu.py', else None."""
    if cls or not name:
        return None
    n = name.strip()
    if "::" in n or "/" in n or " " in n:
        return None
    marker = "third_party.torch-xpu-ops."
    if marker in n:
        n = n.split(marker, 1)[1]
    if "." not in n:
        return None
    return _canon_xpu_file(n.split("."))


def parse_ut(root):
    """Return (status_map, msg_map).

    status_map: {(test_file, class, name): status}
                status in passed / failure / error / skipped / xfail / others.
    msg_map:    {(test_file, class, name): brief message} for failure/error only.
    """
    # priority when a case appears more than once:
    # passed > skipped > xfail > others > failed(failure) > error  (higher kept)
    order = {"passed": 6, "skipped": 5, "xfail": 4, "others": 3, "failure": 2, "error": 1}
    result = {}
    messages = {}
    for path in find_files(root, r"\.xml$"):
        xml_file = os.path.basename(path)[:-4]  # drop .xml (fallback test-file name)
        try:
            tree = ET.parse(path)
        except ET.ParseError:
            continue
        root_el = tree.getroot()
        for tc in root_el.iter("testcase"):
            name = tc.get("name", "")
            if not name.strip():
                continue  # skip bare <testcase time="0.000"/> placeholder artifacts
            test_file, cls = _ut_file_class(tc, xml_file)
            status = "passed"
            msg = ""
            for child in tc:
                tag = child.tag.lower()
                if tag == "failure":
                    status = "failure"
                    msg = _brief_msg(child.get("message") or child.text)
                elif tag == "error":
                    status = "error"
                    msg = _brief_msg(child.get("message") or child.text)
                elif tag == "skipped":
                    typ = (child.get("type") or "").lower()
                    status = "xfail" if "xfail" in typ else "skipped"
                elif status == "passed":
                    status = "others"
            key = (test_file, cls, name)
            if key not in result or order[status] > order[result[key]]:
                result[key] = status
                if status in ("failure", "error"):
                    messages[key] = msg
                else:
                    messages.pop(key, None)
    return result, messages


def ut_status_summary(status_map):
    """Per-version status breakdown."""
    from collections import Counter
    c = Counter(status_map.values())
    total = sum(c.values())
    failure = c.get("failure", 0)
    error = c.get("error", 0)
    passed = c.get("passed", 0)
    passrate = (total - failure - error) / total if total else 0.0
    return {
        "Total": total,
        "Passed": passed,
        "Passrate": passrate,
        "Skipped": c.get("skipped", 0),
        "Failure": failure,
        "Error": error,
        "Xfail": c.get("xfail", 0),
        "Others": c.get("others", 0),
    }


def acc_bucket(value):
    """Classify an accuracy status into Passed / Failed / Notrun."""
    if value in PASS_ACC:
        return "Passed"
    if value in NOTRUN_ACC:
        return "Notrun"
    return "Failed"


def acc_value_map(acc_map):
    return {k: v["value"] for k, v in acc_map.items()}


def acc_status_summary(value_map):
    """Status columns: Total, Passed, Passrate, Failed, Notrun."""
    from collections import Counter
    c = Counter(acc_bucket(v) for v in value_map.values())
    total = sum(c.values())
    passed = c.get("Passed", 0)
    failed = c.get("Failed", 0)
    notrun = c.get("Notrun", 0)
    passrate = (total - failed - notrun) / total if total else 0.0
    return {"Total": total, "Passed": passed, "Passrate": passrate,
            "Failed": failed, "Notrun": notrun}


def acc_suite_summary(value_map):
    """Per-suite breakdown -> {suite: summary}."""
    from collections import defaultdict
    groups = defaultdict(dict)
    for key, v in value_map.items():
        groups[key[0]][key] = v
    return {suite: acc_status_summary(vm) for suite, vm in groups.items()}


def compare_acc(base_vmap, target_vmap):
    """Pass-based comparison for E2E accuracy.

    improvement : target pass, baseline not pass (or null)
    regression  : baseline pass, target not pass (or null)
    pass        : both pass          (== "No Change")
    fail        : both not pass       (== "Both Fail")
    """
    rows = []
    counts = defaultdict(int)
    for key in sorted(set(base_vmap) | set(target_vmap)):
        b = base_vmap.get(key)
        t = target_vmap.get(key)
        bp = b in PASS_ACC if b is not None else False
        tp = t in PASS_ACC if t is not None else False
        if tp and not bp:
            cat = "improvement"
        elif bp and not tp:
            cat = "regression"
        elif bp and tp:
            cat = "pass"
        else:
            cat = "fail"
        counts[cat] += 1
        rows.append({"key": key, "base": b, "target": t, "cat": cat})
    return rows, counts


def _strip_ts(line):
    return re.sub(r'^\[\d{4}-\d\d-\d\d[ T]\d\d:\d\d:\d\d\]\s*', '', line).rstrip()


def _extract_log_msg(path, limit=300):
    """Best-effort brief error message from an E2E accuracy log."""
    try:
        with open(path, errors="replace") as fh:
            lines = [_strip_ts(l) for l in fh]
    except OSError:
        return ""
    err_re = re.compile(r'^[A-Za-z_][\w.]*(?:Error|Exception)\b')
    for l in lines:                       # root-cause exception (first in the chain)
        s = l.strip()
        if err_re.match(s):
            return s[:limit]
    for l in lines:                       # accuracy failures report an RMSE line
        if "RMSE" in l:
            return l[l.index("RMSE"):].strip()[:limit]
    nonempty = [l.strip() for l in lines if l.strip()]

    def _is_noise_tail(t):
        # progress bars (tqdm), trailing status tokens, speedup ratios, bare numbers
        if "%|" in t or "█" in t or re.search(r'\d+/\d+\s*\[\d+:\d+', t) or "it/s]" in t:
            return True
        return (bool(re.fullmatch(r'[A-Za-z0-9_>.%]+', t)) and "_" in t) \
            or bool(re.fullmatch(r'-?\d+(?:\.\d+)?x?', t)) \
            or bool(re.fullmatch(r'-?\d+(?:\.\d+)?\s*(?:x|ms|s|us|it/s)?', t))

    while nonempty and _is_noise_tail(nonempty[-1]):
        nonempty.pop()                    # drop trailing bare status / speedup tokens
    return nonempty[-1][:limit] if nonempty else ""


def build_log_index(root, leaf):
    """Map (suite, dtype, mode, model) -> logs-*.log path under */<leaf>/ (accuracy|performance)."""
    idx = {}
    for path in find_files(root, r'^logs-.*\.log$'):
        parts = path.split(os.sep)
        if len(parts) < 5 or parts[-2] != leaf:
            continue
        suite, dtype, mode = parts[-5], parts[-4], parts[-3]
        m = re.match(r'^logs-(.*?)-worker\d+-card\d+\.log$', parts[-1]) \
            or re.match(r'^logs-(.*?)\.log$', parts[-1])
        model = m.group(1) if m else parts[-1]
        idx[(suite, dtype, mode, model)] = path
    return idx


def build_acc_log_index(root):
    return build_log_index(root, "accuracy")


_STATUS_TOKENS = PASS_ACC | NOTRUN_ACC | {"fail_accuracy", "eager_two_runs_differ", "fail"}


def log_status_token(path):
    """Last line of the log that is exactly a known status token, else None."""
    tok = None
    try:
        with open(path, errors="replace") as fh:
            for line in fh:
                s = _strip_ts(line).strip()
                if s in _STATUS_TOKENS:
                    tok = s
    except OSError:
        return None
    return tok


def parse_acc_messages(root, value_map):
    """{key: brief message} for non-passing accuracy cases, read from logs-*.log."""
    idx = build_acc_log_index(root)
    msgs = {}
    for key, v in value_map.items():
        if v in PASS_ACC:
            continue
        suite, dtype, mode, name, scenario = key
        path = idx.get((suite, dtype, mode, name.replace("/", "_")))
        if path:
            m = _extract_log_msg(path)
            if m:
                msgs[key] = m
    return msgs


# ---- performance (E2E) helpers -------------------------------------------- #
def _pos(x):
    return x is not None and x > 0


def _numf(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _verified_rank(path):
    """Verified-source rank for E2E dedup (higher wins). Within a 'verified' path
    segment, a trailing rerun counter '-N' (N a small int, 1-999, no leading
    zero) ranks N, so verified-2 > verified-1 > verified (no counter). A segment
    with no such trailing counter ranks 0; a non-verified path ranks -1. Only a
    bare small trailing '-N' is a counter, so an embedded date/version like
    'verified-perf-0805' or 'verified-perf-2.14' stays rank 0 while
    'verified-perf-0805-2' / 'verified-perf-2.14-2' are rank 2. Thus any verified
    source beats non-verified, and the latest (largest counter) rerun wins."""
    rank = -1
    for comp in re.split(r"[\\/]", path.lower()):
        if "verified" in comp:
            m = re.search(r"-([1-9]\d{0,2})$", comp)
            rank = max(rank, int(m.group(1)) if m else 0)
    return rank


def _prio_pos_small(raw, vrank=-1):
    """Dedup priority (higher kept): latest verified rank > >0 smaller > >0 larger > =0 > failed > others."""
    x = _numf(raw)
    if x is not None and x > 0:
        base = (3, -x)
    elif x is not None and x == 0:
        base = (2, 0.0)
    elif str(raw).strip().lower() == "failed":
        base = (1, 0.0)
    else:
        base = (0, 0.0)
    return (vrank,) + base


def _prio_pos_large(raw, vrank=-1):
    """Dedup priority (higher kept): latest verified rank > >0 larger > >0 smaller > =0 > failed > others."""
    x = _numf(raw)
    if x is not None and x > 0:
        base = (3, x)
    elif x is not None and x == 0:
        base = (2, 0.0)
    elif str(raw).strip().lower() == "failed":
        base = (1, 0.0)
    else:
        base = (0, 0.0)
    return (vrank,) + base


def _prio_accuracy(value, vrank=-1):
    """Dedup priority (higher kept): latest verified rank > pass(contains) > fail_accuracy > out of memory > others."""
    v = (value or "").strip().lower()
    if "pass" in v:
        base = 3
    elif "fail_accuracy" in v:
        base = 2
    elif "out of memory" in v:
        base = 1
    else:
        base = 0
    return (vrank, base)


def parse_perf_e2e(root):
    """{(suite,dtype,mode,name,scenario): {'abs','speedup','bs','raw'}} for non-pt2e perf.

    inductor latency = abs_latency ; eager latency = abs_latency * speedup ; passed = abs_latency > 0.
    """
    result = {}
    best = {}
    for path in find_files(root, r'performance\.csv$'):
        if "pt2e" in os.path.basename(path):
            continue
        vrank = _verified_rank(path)
        for row in _read_csv(path):
            suite, dtype, mode, scenario = _e2e_meta(row, path)
            key = (suite, dtype, mode, row.get("name", ""), scenario)

            def f(x):
                try:
                    return float(x)
                except (TypeError, ValueError):
                    return None
            pr = _prio_pos_small(row.get("abs_latency"), vrank)
            if key not in result or pr > best[key]:
                result[key] = {"abs": f(row.get("abs_latency")), "speedup": f(row.get("speedup")),
                               "bs": row.get("batch_size", ""), "raw": row}
                best[key] = pr
    return result


def perf_metrics(entry):
    """Return (bs, inductor_latency, eager_latency); latencies None when not passed."""
    if entry is None:
        return ("", None, None)
    abs_l, spd, bs = entry["abs"], entry["speedup"], entry["bs"]
    ind = abs_l if _pos(abs_l) else None
    eag = abs_l * spd if (_pos(abs_l) and spd is not None and spd > 0) else None
    return (bs, ind, eag)


PERF_CATS = ["stable", "both_fail", "bs_change", "new_pass", "new_fail", "improve", "drop"]
PERF_CAT_LABEL = {
    "bs_change": "BS Change", "both_fail": "Both Fail", "stable": "Stable",
    "new_pass": "New Pass", "new_fail": "New Fail",
    "improve": "Improves", "drop": "Drops",
}
PERF_CAT_CLS = {
    "bs_change": "unk", "both_fail": "fail", "stable": "pass",
    "new_pass": "imp", "new_fail": "reg", "improve": "imp", "drop": "reg",
}


def perf_status(bt, bb, ind_t, ind_b, eag_t, eag_b):
    it, ib = _pos(ind_t), _pos(ind_b)
    et, eb = _pos(eag_t), _pos(eag_b)
    if it and ib and str(bt) != str(bb):
        return "bs_change"
    # a pass/fail transition hits inductor and eager together (eager = abs*speedup)
    if ib and not it:
        return "new_fail"
    if it and not ib:
        return "new_pass"
    if not (it and ib):        # both failed
        return "both_fail"
    # both passed -> a drop/improve on EITHER metric classifies the row (drop wins)
    ind_r = ind_b / ind_t
    eag_r = (eag_b / eag_t) if (et and eb) else None
    dropped = ind_r < PERF_REG_THRESHOLD or (eag_r is not None and eag_r < PERF_REG_THRESHOLD)
    improved = ind_r > PERF_IMP_THRESHOLD or (eag_r is not None and eag_r > PERF_IMP_THRESHOLD)
    if dropped:
        return "drop"
    if improved:
        return "improve"
    return "stable"


def compare_perf2(base_map, target_map):
    rows = []
    counts = defaultdict(int)
    for key in sorted(set(base_map) | set(target_map)):
        bb, ind_b, eag_b = perf_metrics(base_map.get(key))
        bt, ind_t, eag_t = perf_metrics(target_map.get(key))
        cat = perf_status(bt, bb, ind_t, ind_b, eag_t, eag_b)
        counts[cat] += 1
        rows.append({"key": key, "bt": bt, "bb": bb, "ind_t": ind_t, "ind_b": ind_b,
                     "eag_t": eag_t, "eag_b": eag_b, "cat": cat})
    return rows, counts


def _geomean(ratios):
    rs = [r for r in ratios if r is not None and r > 0]
    if not rs:
        return None
    return math.exp(sum(math.log(r) for r in rs) / len(rs))


def perf_geomeans(rows):
    """(eager_geomean, inductor_geomean) of baseline/target over rows passed in both."""
    eag = _geomean([r["eag_b"] / r["eag_t"] for r in rows if _pos(r["eag_t"]) and _pos(r["eag_b"])])
    ind = _geomean([r["ind_b"] / r["ind_t"] for r in rows if _pos(r["ind_t"]) and _pos(r["ind_b"])])
    return eag, ind


def perf_notrun_status(root, perf_map):
    """{key: 'Notrun'|'Failed'} for non-passed perf rows, from the perf log token."""
    idx = build_log_index(root, "performance")
    out = {}
    for key, e in perf_map.items():
        if _pos(e["abs"]):
            continue
        suite, dtype, mode, name, scenario = key
        lp = idx.get((suite, dtype, mode, name.replace("/", "_")))
        tok = log_status_token(lp) if lp else None
        out[key] = "Notrun" if tok in NOTRUN_ACC else "Failed"
    return out


def perf_ver_summary(perf_map, notrun_status):
    total = len(perf_map)
    passed = failed = notrun = 0
    for key, e in perf_map.items():
        if _pos(e["abs"]):
            passed += 1
        elif notrun_status.get(key) == "Notrun":
            notrun += 1
        else:
            failed += 1
    passrate = (total - failed - notrun) / total if total else 0.0
    return {"Total": total, "Passed": passed, "Passrate": passrate,
            "Failed": failed, "Notrun": notrun}


def perf_suite_summary(perf_map, notrun_status):
    from collections import defaultdict as _dd
    groups = _dd(dict)
    for key, e in perf_map.items():
        groups[key[0]][key] = e
    return {s: perf_ver_summary(vm, notrun_status) for s, vm in groups.items()}


def parse_perf_messages(root, perf_map):
    """{key: brief message} for non-passing target perf rows, from performance logs."""
    idx = build_log_index(root, "performance")
    msgs = {}
    for key, e in perf_map.items():
        if _pos(e["abs"]):
            continue
        suite, dtype, mode, name, scenario = key
        lp = idx.get((suite, dtype, mode, name.replace("/", "_")))
        if lp:
            m = _extract_log_msg(lp)
            if m:
                msgs[key] = m
    return msgs


def _read_csv(path):
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            yield row


def parse_accuracy(root, pt2e=False):
    """Return {(suite,dtype,mode,name,scenario): {'value':.., 'raw':row}}.

    Dedups repeated keys by priority: pass(contains) > fail_accuracy
    > out of memory > others.
    """
    result = {}
    best = {}
    for path in find_files(root, r"accuracy\.csv$"):
        is_pt2e = "pt2e" in os.path.basename(path)
        if is_pt2e != pt2e:
            continue
        vrank = _verified_rank(path)
        for row in _read_csv(path):
            suite, dtype, mode, scenario = _e2e_meta(row, path)
            key = (suite, dtype, mode, row.get("name", ""), scenario)
            if pt2e:
                value = row.get("top1", "")
                pr = _prio_pos_large(value, vrank)
            else:
                value = row.get("accuracy", "")
                pr = _prio_accuracy(value, vrank)
            if key not in result or pr > best[key]:
                result[key] = {"value": value, "raw": row}
                best[key] = pr
    return result


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
CATEGORY_ORDER = ["regression", "improvement", "new", "removed", "crash", "timeout", "not_run", "fail", "pass", "others", "unknown"]
CATEGORY_LABEL = {
    "regression": "Regression",
    "improvement": "Improvement",
    "new": "New",
    "removed": "Deselected",
    "crash": "Collection Failure",
    "timeout": "Timeout / Hang",
    "not_run": "Not Run",
    "fail": "Both Fail",
    "pass": "Both Pass",
    "others": "Other Change",
    "unknown": "Unknown",
}


def compare_ut(base_map, target_map):
    """UT-specific comparison using the richer status vocabulary.

    improvement : target passed AND baseline not passed (incl. baseline missing/null)
    regression  : baseline passed AND target failed/error/null (a skip / xfail is
                  an intentional non-run, so it is reported as "others")
    pass        : passed in both
    new         : baseline missing, target also not passed
    removed     : target missing, baseline also not passed
    fail        : failure/error in both
    others      : any other transition (skip / xfail / status change)
    """
    fail_set = {"failure", "error"}
    rows = []
    counts = defaultdict(int)
    for key in sorted(set(base_map) | set(target_map)):
        b = base_map.get(key)
        t = target_map.get(key)
        bp, tp = b == "passed", t == "passed"
        if tp and not bp:
            cat = "improvement"          # target passed, baseline not passed / null
        elif bp and not tp:
            # a pass -> skipped / xfail is an intentional non-run, not a regression
            cat = "others" if t in ("skipped", "xfail") else "regression"
        elif bp and tp:
            cat = "pass"
        elif b is None:
            cat = "new"
        elif t is None:
            cat = "removed"
        elif b in fail_set and t in fail_set:
            cat = "fail"
        else:
            cat = "others"
        counts[cat] += 1
        rows.append({"key": key, "base": b, "target": t, "cat": cat})
    return rows, counts


# --------------------------------------------------------------------------- #
# HTML rendering
# --------------------------------------------------------------------------- #
def esc(x):
    return html.escape(str(x))


CAT_CLASS = {
    "regression": "reg", "improvement": "imp", "new": "new",
    "removed": "rem", "crash": "crash", "timeout": "timeout", "not_run": "notrun", "fail": "fail", "pass": "pass",
    "others": "unk", "unknown": "unk",
}


def ut_summary_table_html(base_sum, target_sum):
    cols = ["Total", "Passed", "Passrate", "Skipped", "Failure", "Error", "Xfail", "Others"]

    def fmt(sm, c):
        v = sm[c]
        return f"{v * 100:.2f}%" if c == "Passrate" else str(v)

    def row(label, sm, cls, worse=False):
        tds = "".join(
            f'<td class="pr-worse">{fmt(sm, c)}</td>' if (worse and c == "Passrate")
            else f"<td>{fmt(sm, c)}</td>" for c in cols)
        return f'<tr class="{cls}"><td class="rowlbl">{label}</td>{tds}</tr>'

    # delta row (target - baseline)
    def delta_cell(c):
        if c == "Passrate":
            d = (target_sum[c] - base_sum[c]) * 100
            sign = "+" if d >= 0 else ""
            cls = "d-up" if d > 0 else ("d-down" if d < 0 else "")
            return f'<td class="{cls}">{sign}{d:.2f}%</td>'
        d = target_sum[c] - base_sum[c]
        sign = "+" if d > 0 else ""
        good_up = c in ("Passed",)
        good_down = c in ("Failure", "Error")
        cls = ""
        if d != 0:
            up = d > 0
            cls = "d-up" if (up and good_up) or (not up and good_down) else \
                  ("d-down" if (up and good_down) or (not up and good_up) else "")
        return f'<td class="{cls}">{sign}{d}</td>'

    head = "".join(f"<th>{c}</th>" for c in cols)
    delta = "".join(delta_cell(c) for c in cols)
    worse = target_sum["Passrate"] < base_sum["Passrate"]
    return f'''
    <h3 class="subh">Status summary (Passrate = (Total − Failure − Error) / Total)</h3>
    <div class="table-wrap small">
      <table class="data summary-tbl">
        <thead><tr><th>Category</th>{head}</tr></thead>
        <tbody>
          {row(esc(TARGET_LABEL) + " (Target)", target_sum, "row-target" + (" worse" if worse else ""), worse)}
          {row(esc(BASE_LABEL) + " (Baseline)", base_sum, "row-base")}
          <tr class="row-delta"><td class="rowlbl">Δ Target − Baseline</td>{delta}</tr>
        </tbody>
      </table>
    </div>'''


ACC_SUM_COLS = ["Total", "Passed", "Passrate", "Failed", "Notrun"]


def acc_summary_table_html(base_sum, target_sum, title="Status summary"):
    cols = ACC_SUM_COLS

    def fmt(sm, c):
        return f"{sm[c] * 100:.2f}%" if c == "Passrate" else str(sm[c])

    def row(label, sm, cls, worse=False):
        tds = "".join(
            f'<td class="pr-worse">{fmt(sm, c)}</td>' if (worse and c == "Passrate")
            else f"<td>{fmt(sm, c)}</td>" for c in cols)
        return f'<tr class="{cls}"><td class="rowlbl">{label}</td>{tds}</tr>'

    def delta_cell(c):
        if c == "Passrate":
            d = (target_sum[c] - base_sum[c]) * 100
            cls = "d-up" if d > 0 else ("d-down" if d < 0 else "")
            return f'<td class="{cls}">{"+" if d >= 0 else ""}{d:.2f}%</td>'
        d = target_sum[c] - base_sum[c]
        good_up = c in ("Passed",)
        good_down = c in ("Failed", "Notrun")
        cls = ""
        if d != 0:
            up = d > 0
            cls = "d-up" if (up and good_up) or (not up and good_down) else \
                  ("d-down" if (up and good_down) or (not up and good_up) else "")
        return f'<td class="{cls}">{"+" if d > 0 else ""}{d}</td>'

    head = "".join(f"<th>{c}</th>" for c in cols)
    delta = "".join(delta_cell(c) for c in cols)
    worse = target_sum["Passrate"] < base_sum["Passrate"]
    return f'''
    <h3 class="subh">{esc(title)} <span class="hint">(Passrate = (Total − Failed − Notrun) / Total)</span></h3>
    <div class="table-wrap small">
      <table class="data summary-tbl">
        <thead><tr><th>Category</th>{head}</tr></thead>
        <tbody>
          {row(esc(TARGET_LABEL) + " (Target)", target_sum, "row-target" + (" worse" if worse else ""), worse)}
          {row(esc(BASE_LABEL) + " (Baseline)", base_sum, "row-base")}
          <tr class="row-delta"><td class="rowlbl">Δ Target − Baseline</td>{delta}</tr>
        </tbody>
      </table>
    </div>'''


def acc_suite_breakdown_html(base_vmap, target_vmap):
    base_by = acc_suite_summary(base_vmap)
    target_by = acc_suite_summary(target_vmap)
    suites = sorted(set(base_by) | set(target_by))
    cols = ACC_SUM_COLS

    def cells(sm, worse=False):
        return "".join(
            (f'<td class="pr-worse">{sm[c] * 100:.2f}%</td>' if worse else f'<td>{sm[c] * 100:.2f}%</td>')
            if c == "Passrate" else f"<td>{sm[c]}</td>"
            for c in cols)

    body = []
    empty = {"Total": 0, "Passed": 0, "Passrate": 0.0, "Failed": 0, "Notrun": 0}
    for suite in suites:
        tb = target_by.get(suite, empty)
        bb = base_by.get(suite, empty)
        worse = tb["Passrate"] < bb["Passrate"]
        body.append(
            f'<tr class="row-target{" worse" if worse else ""}"><td class="rowlbl" rowspan="2">{esc(suite)}</td>'
            f'<td>Target</td>{cells(tb, worse)}</tr>'
            f'<tr class="row-base"><td>Baseline</td>{cells(bb)}</tr>')
    head = "".join(f"<th>{c}</th>" for c in cols)
    return f'''
    <h3 class="subh">Breakdown by suite</h3>
    <div class="table-wrap small">
      <table class="data summary-tbl">
        <thead><tr><th>Suite</th><th>Category</th>{head}</tr></thead>
        <tbody>{''.join(body)}</tbody>
      </table>
    </div>'''


PERF_SUM_COLS = ["Total", "Passed", "Passrate", "Failed", "Notrun", "Eager", "Inductor"]


def _gm(x):
    return f"{x:.3f}" if x is not None else "/"


def _gm_cell(x):
    """Geomean baseline/target as a color-scaled <td> (5% criteria)."""
    if x is None:
        return "<td>/</td>"
    cls = "gm-good" if x > 1.05 else ("gm-bad" if x < 0.95 else "gm-mid")
    return f'<td class="{cls}">{x:.3f}</td>'


def perf_summary_table_html(base_sum, target_sum, eag_gm, ind_gm):
    metric_cols = ["Total", "Passed", "Passrate", "Failed", "Notrun"]

    def fmt(sm, c):
        return f"{sm[c] * 100:.2f}%" if c == "Passrate" else str(sm[c])

    def mrow(sm, worse=False):
        return "".join(
            f'<td class="pr-worse">{fmt(sm, c)}</td>' if (worse and c == "Passrate")
            else f"<td>{fmt(sm, c)}</td>" for c in metric_cols)

    def delta_cell(c):
        d = target_sum[c] - base_sum[c]
        if c == "Passrate":
            cls = "d-up" if d > 0 else ("d-down" if d < 0 else "")
            return f'<td class="{cls}">{"+" if d >= 0 else ""}{d * 100:.2f}%</td>'
        good_up = c in ("Passed",)
        good_down = c in ("Failed", "Notrun")
        cls = ""
        if d != 0:
            up = d > 0
            cls = "d-up" if (up and good_up) or (not up and good_down) else \
                  ("d-down" if (up and good_down) or (not up and good_up) else "")
        return f'<td class="{cls}">{"+" if d > 0 else ""}{d}</td>'

    head = "".join(f"<th>{c}</th>" for c in PERF_SUM_COLS)
    delta = "".join(delta_cell(c) for c in metric_cols)
    worse = target_sum["Passrate"] < base_sum["Passrate"]
    return f'''
    <h3 class="subh">Status summary <span class="hint">(Passrate=(Total−Failed−Notrun)/Total · Eager/Inductor = geomean baseline/target latency · Inductor latency=abs_latency, Eager=abs_latency×speedup)</span></h3>
    <div class="table-wrap small">
      <table class="data summary-tbl">
        <thead><tr><th>Category</th>{head}</tr></thead>
        <tbody>
          <tr class="row-target{' worse' if worse else ''}"><td class="rowlbl">{esc(TARGET_LABEL)} (Target)</td>{mrow(target_sum, worse)}{_gm_cell(eag_gm)}{_gm_cell(ind_gm)}</tr>
          <tr class="row-base"><td class="rowlbl">{esc(BASE_LABEL)} (Baseline)</td>{mrow(base_sum)}<td>/</td><td>/</td></tr>
          <tr class="row-delta"><td class="rowlbl">Δ Target − Baseline</td>{delta}<td></td><td></td></tr>
        </tbody>
      </table>
    </div>'''


def perf_suite_breakdown_html(base_map, target_map, rows, base_notrun, target_notrun):
    base_by = perf_suite_summary(base_map, base_notrun)
    target_by = perf_suite_summary(target_map, target_notrun)
    # per-suite geomeans
    from collections import defaultdict as _dd
    suite_rows = _dd(list)
    for r in rows:
        suite_rows[r["key"][0]].append(r)
    suites = sorted(set(base_by) | set(target_by))
    metric_cols = ["Total", "Passed", "Passrate", "Failed", "Notrun"]

    def cells(sm, worse=False):
        return "".join(
            (f'<td class="pr-worse">{sm[c] * 100:.2f}%</td>' if worse else f'<td>{sm[c] * 100:.2f}%</td>')
            if c == "Passrate" else f"<td>{sm[c]}</td>"
            for c in metric_cols)

    empty = {"Total": 0, "Passed": 0, "Passrate": 0.0, "Failed": 0, "Notrun": 0}
    body = []
    for suite in suites:
        tb, bb = target_by.get(suite, empty), base_by.get(suite, empty)
        eag_gm, ind_gm = perf_geomeans(suite_rows.get(suite, []))
        worse = tb["Passrate"] < bb["Passrate"]
        body.append(
            f'<tr class="row-target{" worse" if worse else ""}"><td class="rowlbl" rowspan="2">{esc(suite)}</td>'
            f'<td>Target</td>{cells(tb, worse)}{_gm_cell(eag_gm)}{_gm_cell(ind_gm)}</tr>'
            f'<tr class="row-base"><td>Baseline</td>{cells(bb)}<td>/</td><td>/</td></tr>')
    head = "".join(f"<th>{c}</th>" for c in PERF_SUM_COLS)
    return f'''
    <h3 class="subh">Breakdown by suite</h3>
    <div class="table-wrap small">
      <table class="data summary-tbl">
        <thead><tr><th>Suite</th><th>Category</th>{head}</tr></thead>
        <tbody>{''.join(body)}</tbody>
      </table>
    </div>'''


def summary_cards_html(title, counts, cards=None, onclick="utCardFilter", default="regression"):
    if cards is not None:
        # explicit cards mirroring the section's filters.
        # 3-tuple (label, value, css) -> static;  4-tuple adds a filter id -> clickable
        parts = []
        for c in cards:
            if len(c) == 4:
                label, val, cls, fid = c
                sel = " sel" if fid == default else ""
                parts.append(
                    f'<div class="card {cls} clickable{sel}" data-f="{fid}" '
                    f'onclick="{onclick}(this)"><span class="num">{val}</span>'
                    f'<span class="lbl">{esc(label)}</span></div>')
            else:
                label, val, cls = c
                parts.append(
                    f'<div class="card {cls}"><span class="num">{val}</span>'
                    f'<span class="lbl">{esc(label)}</span></div>')
        return f'<div class="cards">{"".join(parts)}</div>'
    total = sum(counts.values())
    html_cards = [f'<div class="card total"><span class="num">{total}</span><span class="lbl">Total</span></div>']
    for cat in CATEGORY_ORDER:
        if counts.get(cat, 0) == 0 and cat in ("unknown",):
            continue
        n = counts.get(cat, 0)
        html_cards.append(
            f'<div class="card {CAT_CLASS[cat]}"><span class="num">{n}</span>'
            f'<span class="lbl">{CATEGORY_LABEL[cat]}</span></div>')
    return f'<div class="cards">{"".join(html_cards)}</div>'


def _rate_tone(target, base):
    """Colour a pass rate by comparison with the baseline."""
    if target > base:
        return "imp"
    if target < base:
        return "reg"
    return "pass"


def _gm_tone(x, hi=1.05, lo=0.95):
    """Colour a geomean ratio: above hi = improvement, below lo = regression."""
    x = x if x is not None else 1.0
    return "imp" if x > hi else ("reg" if x < lo else "pass")


def overview_html(summary, caption=""):
    """Top-level 'Summary' block: one card per section with its focus metrics.

    `summary` is a list of {id, title, metrics:[(label, value, tone)], tag?}.
    The first metric is shown large (hero); the rest as compact stat pills.
    Zero values are not colour-highlighted. Rendered above the tabs.
    """
    if not summary:
        return ""
    valid = {"reg", "imp", "pass", "fail", "neutral"}

    def _t(t, v):
        if v == 0 or str(v) == "0":
            return "neutral"
        return t if t in valid else "neutral"

    cards = []
    for g in summary:
        ms = g["metrics"]
        hero_l, hero_v, hero_t = ms[0]
        tag = f'<span class="ov-tag">{esc(g["tag"])}</span>' if g.get("tag") else ""
        pills = "".join(
            f'<div class="ov-pill"><span class="ovp-v {_t(t, v)}">{esc(str(v))}</span>'
            f'<span class="ovp-l">{esc(l)}</span></div>'
            for l, v, t in ms[1:])
        cards.append(
            f'<div class="ov-card">'
            f'<div class="ov-top"><span class="ov-name">{esc(g["title"])}</span>{tag}</div>'
            f'<div class="ov-hero"><span class="ovh-v {_t(hero_t, hero_v)}">{esc(str(hero_v))}</span>'
            f'<span class="ovh-l">{esc(hero_l)}</span></div>'
            f'<div class="ov-pills">{pills}</div></div>')
    cap = f'<p class="ov-cap">{esc(caption)}</p>' if caption else ""
    return ('<section class="overview"><div class="ov-titlebar">'
            f'<h2 class="ov-title">Summary</h2>{cap}</div>'
            f'<div class="ov-grid">{"".join(cards)}</div></section>')


def compute_gate(scope, ut_rows, ut_counts, acc_rows, acc_counts, acc_passed,
                 perf_rows, perf_counts):
    """Reasons the acceptance job fails (empty list == pass). A section is only
    checked when it is in `scope` (the tested sections):
      1. UT regression                       4. tested section with TOTAL == 0
      2. Accuracy regression                 5. TOTAL > 0 but no passing baseline
      3. Performance new-fail or drop           (UT Both Pass / ACC passed / PERF Stable == 0)
    """
    reasons = []
    if "ut" in scope:
        if len(ut_rows) == 0:
            reasons.append("Unit Tests: no cases collected (TOTAL = 0)")
        else:
            if ut_counts.get("regression", 0) > 0:
                reasons.append(f"Unit Tests: {ut_counts['regression']} regression(s)")
            if ut_counts.get("pass", 0) == 0:
                reasons.append("Unit Tests: 0 Both Pass")
    if "acc" in scope:
        if len(acc_rows) == 0:
            reasons.append("Accuracy: no cases (TOTAL = 0)")
        else:
            if acc_counts.get("regression", 0) > 0:
                reasons.append(f"Accuracy: {acc_counts['regression']} regression(s)")
            if acc_passed == 0:
                reasons.append("Accuracy: no passed cases")
    if "perf" in scope:
        if len(perf_rows) == 0:
            reasons.append("Performance: no cases (TOTAL = 0)")
        else:
            nf, dr = perf_counts.get("new_fail", 0), perf_counts.get("drop", 0)
            if nf or dr:
                reasons.append(f"Performance: {nf} new fail, {dr} drop")
            if perf_counts.get("stable", 0) == 0:
                reasons.append("Performance: 0 Stable")
    return reasons


def gate_banner_html(reasons):
    if reasons:
        items = "".join(f"<li>{esc(r)}</li>" for r in reasons)
        return ('<section class="gate gate-fail"><div class="gate-hd">&#10060; Acceptance gate FAILED</div>'
                f'<ul class="gate-list">{items}</ul></section>')
    return '<section class="gate gate-pass"><div class="gate-hd">&#9989; Acceptance gate PASSED</div></section>'



def ut_detail_html(rows, issues, lookup, target_msg):
    """Full case-level table for UT, rendered client-side (all rows, default=regression).

    Rows are interned (files/classes) + emitted as compact JSON to keep the file small
    and the browser responsive even with hundreds of thousands of cases.
    Extra columns track intel/torch-xpu-ops issues and a brief target failure message.
    """
    import json
    ST = ["passed", "failure", "error", "skipped", "xfail", "others"]
    CATS = ["pass", "fail", "regression", "improvement", "new", "removed", "crash", "timeout", "not_run", "others"]
    st_i = {s: i for i, s in enumerate(ST)}
    cat_i = {c: i for i, c in enumerate(CATS)}
    files, fidx = [], {}
    classes, cidx = [], {}
    data = []
    for r in rows:
        f, c, n = r["key"]
        if f not in fidx:
            fidx[f] = len(files); files.append(f)
        if c not in cidx:
            cidx[c] = len(classes); classes.append(c)
        b = -1 if r["base"] is None else st_i[r["base"]]
        t = -1 if r["target"] is None else st_i[r["target"]]
        hit = lookup.get(_ut_key(c, n))
        issue_idx = hit[0] if hit else -1
        fixed = 1 if (hit and hit[1]) else 0
        msg = target_msg.get(r["key"], "")
        data.append([fidx[f], cidx[c], n, b, t, cat_i[r["cat"]], issue_idx, fixed, msg])

    issues_js = [[i["id"], i["url"], i["state"]] for i in issues]
    blob = (
        "const UT_FILES=" + json.dumps(files) + ";"
        "const UT_CLASSES=" + json.dumps(classes) + ";"
        "const UT_ST=" + json.dumps(ST) + ";"
        "const UT_CATS=" + json.dumps(CATS) + ";"
        "const UT_ISSUES=" + json.dumps(issues_js) + ";"
        "const UT_ROWS=" + json.dumps(data, separators=(",", ":")) + ";"
    ).replace("</", "<\\/")

    table = f'''
    <h3 class="subh">Case-level comparison <span class="hint">(all cases loaded · click a card above to filter · showing regressions by default)</span></h3>
    <div class="toolbar">
      <button class="exportbtn" onclick="utExportCSV()">⬇ Export CSV</button>
      <button class="exportbtn" onclick="utCopyCmd(this)">⧉ Copy command</button>
    </div>
    <div class="table-wrap">
      <table id="tbl_ut" class="data">
        <thead><tr><th>Test File</th><th>Test Class</th><th>Test Name</th>
          <th>{esc(BASE_LABEL)}</th><th>{esc(TARGET_LABEL)}</th><th class="status">Status</th>
          <th>Message</th></tr></thead>
        <tbody id="ut_body"></tbody>
      </table>
    </div>
    <div class="pager">Showing <b id="ut_shown">0</b> of <b id="ut_match">0</b> matched
      &nbsp;·&nbsp; <span id="ut_total"></span> total
      <button id="ut_more" class="loadmore" onclick="utMore()">Load more</button></div>'''

    script = f'<script>{blob}\n{UT_TABLE_JS}</script>'
    return table + script


UT_TABLE_JS = r'''
let UT_FILTER='regression', UT_SEARCH='', UT_LIMIT=1000;
let UT_COLF=[], UT_SC=-1, UT_SD=1; const UT_NUM=[];
const UT_CATLBL={pass:'Both Pass',fail:'Both Fail',regression:'Regression',
  improvement:'Improvement',new:'New',removed:'Deselected',crash:'Collection Failure',timeout:'Timeout / Hang',not_run:'Not Run',others:'Other Change'};
const UT_CATCLS={pass:'pass',fail:'fail',regression:'reg',improvement:'imp',
  new:'new',removed:'rem',crash:'crash',timeout:'timeout',not_run:'notrun',others:'unk'};
function utEsc(s){return String(s).replace(/[&<>]/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[m]));}
const UT_FAIL=[UT_ST.indexOf('failure'),UT_ST.indexOf('error')];
function utIsFail(c){return UT_FAIL.indexOf(c)>=0;}
const UT_FILTERS=['all','regression','improvement','new','removed','crash','timeout','not_run',
  'fail','pass','others','target_fail_issue','target_fail_noissue'];
function utRowInFilter(row,f){
  const bc=row[3], tc=row[4], cat=UT_CATS[row[5]];
  switch(f){
    case 'all': return true;
    case 'regression': return cat==='regression';
    case 'improvement': return cat==='improvement';
    case 'target_fail_issue': return utIsFail(tc)&&row[6]>=0;
    case 'target_fail_noissue': return utIsFail(tc)&&row[6]<0;
    default: return cat===f;
  }
}
function utSearchMatch(row){
  if(!UT_SEARCH) return true;
  const s=(UT_FILES[row[0]]+' '+UT_CLASSES[row[1]]+' '+row[2]+' '+(row[8]||'')).toLowerCase();
  return s.includes(UT_SEARCH);
}
function utIssueStr(row){ const ii=row[6]; return ii>=0? '#'+UT_ISSUES[ii][0] : ''; }
function utIssueSt(row){ const ii=row[6]; return ii>=0? (row[7]?'fixed':(UT_ISSUES[ii][2]||'opened')) : ''; }
function utCellText(row){ return [UT_FILES[row[0]],UT_CLASSES[row[1]],row[2],
  row[3]<0?'-':UT_ST[row[3]], row[4]<0?'-':UT_ST[row[4]], UT_CATLBL[UT_CATS[row[5]]],
  row[8]||'']; }
function utMatch(row){
  if(!utRowInFilter(row,UT_FILTER)) return false;
  if(!utSearchMatch(row)) return false;
  if(gAnyF(UT_COLF)&&!gColMatch(utCellText(row),UT_COLF)) return false;
  return true;
}
function utUpdateCounts(){
  const cnt={}; UT_FILTERS.forEach(f=>cnt[f]=0);
  const anyc=gAnyF(UT_COLF);
  for(const row of UT_ROWS){
    if(!utSearchMatch(row)) continue;
    if(anyc&&!gColMatch(utCellText(row),UT_COLF)) continue;
    for(const f of UT_FILTERS) if(utRowInFilter(row,f)) cnt[f]++;
  }
  document.querySelectorAll('.panel#ut .cards .card[data-f]').forEach(card=>{
    const f=card.getAttribute('data-f'); const el=card.querySelector('.num');
    if(el && cnt[f]!==undefined) el.textContent=cnt[f];
  });
}
function utCardFilter(card){
  UT_FILTER=card.getAttribute('data-f');
  card.closest('.cards').querySelectorAll('.card[data-f]').forEach(c=>c.classList.remove('sel'));
  card.classList.add('sel');
  utRender(true);
}
function utColF(i,v){ UT_COLF[i]=v; utRender(true); }
function utSortBy(i){ if(UT_SC===i){UT_SD=-UT_SD;}else{UT_SC=i;UT_SD=1;} gSortInd('tbl_ut',UT_SC,UT_SD); utRender(true); }
function utRowHtml(row){
  const cat=UT_CATS[row[5]];
  const b=row[3]<0?'-':UT_ST[row[3]];
  const t=row[4]<0?'-':UT_ST[row[4]];
  const cls=UT_CATCLS[cat];
  const msg=row[8]||'';
  return '<tr class="row-'+cls+'"><td>'+utEsc(UT_FILES[row[0]])+'</td><td>'+
    utEsc(UT_CLASSES[row[1]])+'</td><td>'+utEsc(row[2])+'</td><td>'+b+'</td><td>'+t+
    '</td><td class="status"><span class="badge '+cls+'">'+UT_CATLBL[cat]+'</span></td>'+
    '<td class="msg" title="'+utEsc(msg)+'">'+utEsc(msg)+'</td></tr>';
}
function utRender(reset){
  if(reset) UT_LIMIT=1000;
  const body=document.getElementById('ut_body');
  let arr=[];
  for(const row of UT_ROWS){ if(utMatch(row)) arr.push(row); }
  const matched=arr.length;
  arr=gSortRows(arr,utCellText,UT_SC,UT_SD,UT_NUM.indexOf(UT_SC)>=0);
  const shown=Math.min(UT_LIMIT,matched);
  let html='';
  for(let k=0;k<shown;k++) html+=utRowHtml(arr[k]);
  body.innerHTML = html || '<tr><td colspan="7" style="text-align:center;color:#94a3b8;padding:22px">No matching cases</td></tr>';
  document.getElementById('ut_shown').textContent=shown;
  document.getElementById('ut_match').textContent=matched;
  document.getElementById('ut_total').textContent=UT_ROWS.length+' total';
  document.getElementById('ut_more').style.display = matched>shown ? '' : 'none';
  utUpdateCounts();
}
function utSearchFn(v){ UT_SEARCH=v.toLowerCase(); utRender(true); }
function utMore(){ UT_LIMIT+=3000; utRender(false); }
function csvCell(s){ return '"'+String(s==null?'':s).replace(/"/g,'""')+'"'; }
function csvDownload(text,fname){
  const blob=new Blob(["\ufeff"+text],{type:'text/csv;charset=utf-8;'});
  const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download=fname;
  document.body.appendChild(a); a.click(); document.body.removeChild(a); URL.revokeObjectURL(a.href);
}
function utExportCSV(){
  const hdr=['Test File','Test Class','Test Name','Baseline','Target','Status','Message'];
  const lines=[hdr.map(csvCell).join(',')];
  for(const row of UT_ROWS){
    if(!utMatch(row)) continue;
    const cat=UT_CATS[row[5]];
    const b=row[3]<0?'':UT_ST[row[3]], t=row[4]<0?'':UT_ST[row[4]];
    lines.push([UT_FILES[row[0]],UT_CLASSES[row[1]],row[2],b,t,UT_CATLBL[cat],row[8]||''].map(csvCell).join(','));
  }
  csvDownload(lines.join('\n'),'ut_'+UT_FILTER+'.csv');
}
function utCopyText(text,btn){
  const done=()=>{ if(btn){const o=btn.textContent; btn.textContent='\u2713 Copied'; setTimeout(()=>{btn.textContent=o;},1500);} };
  if(navigator.clipboard && window.isSecureContext){
    navigator.clipboard.writeText(text).then(done,()=>utCopyFallback(text,done));
  } else { utCopyFallback(text,done); }
}
function utCopyFallback(text,done){
  const ta=document.createElement('textarea'); ta.value=text;
  ta.style.position='fixed'; ta.style.top='-1000px'; document.body.appendChild(ta);
  ta.focus(); ta.select();
  try{ document.execCommand('copy'); done&&done(); }catch(e){ prompt('Copy the command below:', text); }
  document.body.removeChild(ta);
}
function utCopyCmd(btn){
  const nodes=[];
  for(const row of UT_ROWS){ if(!utMatch(row)) continue;
    const f=UT_FILES[row[0]], c=UT_CLASSES[row[1]], n=row[2];
    nodes.push('"'+(c ? f+'::'+c+'::'+n : f+'::'+n)+'"');
  }
  if(!nodes.length){ alert('No matching cases to copy.'); return; }
  const cmd='python -m pytest -v \\\n  '+nodes.join(' \\\n  ')+' \\\n  --junit-xml=selected_cases.xml';
  utCopyText(cmd,btn);
}
document.addEventListener('DOMContentLoaded',()=>{ gMakeFilters('tbl_ut',7,utColF); gMakeSort('tbl_ut',utSortBy); utRender(true); });
'''


ACC_NS = {"P": "ACC", "p": "acc", "tid": "tbl_accd", "bid": "accd", "panel": "acc", "csv": "accuracy"}
PF_NS = {"P": "PF", "p": "pf", "tid": "tbl_pfd", "bid": "pfd", "panel": "perf", "csv": "performance"}


def _ns_js(js, old, new):
    """Rewrite a case-level table's JS to a distinct namespace (id / var / fn prefix)."""
    if old == new:
        return js
    js = js.replace(old["tid"], new["tid"])
    js = js.replace(old["bid"] + "_", new["bid"] + "_")
    js = js.replace(".panel#" + old["panel"] + " ", ".panel#" + new["panel"] + " ")
    js = js.replace("'" + old["csv"] + "_'", "'" + new["csv"] + "_'")
    js = js.replace(old["P"] + "_", new["P"] + "_")
    js = re.sub(r'\b' + re.escape(old["p"]) + r'([A-Z][A-Za-z0-9_]*)', new["p"] + r'\1', js)
    return js


def acc_detail_html(rows, issues, lookup, target_msg, ns=None):
    """Client-rendered accuracy case-level table with cards-as-filters."""
    import json
    ns = ns or ACC_NS
    P, p, tid, bid = ns["P"], ns["p"], ns["tid"], ns["bid"]
    CATS = ["pass", "fail", "regression", "improvement"]
    cat_i = {c: i for i, c in enumerate(CATS)}
    suites, sidx = [], {}
    dtypes, didx = [], {}
    modes, midx = [], {}
    vals, vidx = [], {}

    def vi(v):
        if v is None:
            return -1
        if v not in vidx:
            vidx[v] = len(vals); vals.append(v)
        return vidx[v]

    data = []
    for r in rows:
        suite, dtype, mode, name, scenario = r["key"]
        for lst, ix, val in ((suites, sidx, suite), (dtypes, didx, dtype), (modes, midx, mode)):
            if val not in ix:
                ix[val] = len(lst); lst.append(val)
        hit = lookup.get(r["key"])
        issue_idx = hit[0] if hit else -1
        fixed = 1 if (hit and hit[1]) else 0
        msg = target_msg.get(r["key"], "") if r["cat"] in ("regression", "fail") else ""
        data.append([sidx[suite], didx[dtype], midx[mode], name, scenario,
                     vi(r["base"]), vi(r["target"]), cat_i[r["cat"]], issue_idx, fixed, msg])

    issues_js = [[i["id"], i["url"], i["state"]] for i in issues]
    blob = (
        f"const {P}_SUITE=" + json.dumps(suites) + ";"
        f"const {P}_DTYPE=" + json.dumps(dtypes) + ";"
        f"const {P}_MODE=" + json.dumps(modes) + ";"
        f"const {P}_VALS=" + json.dumps(vals) + ";"
        f"const {P}_CATS=" + json.dumps(CATS) + ";"
        f"const {P}_ISSUES=" + json.dumps(issues_js) + ";"
        f"const {P}_ROWS=" + json.dumps(data, separators=(",", ":")) + ";"
    ).replace("</", "<\\/")

    table = f'''
    <h3 class="subh">Case-level comparison <span class="hint">(click a card above to filter · showing regressions by default)</span></h3>
    <div class="toolbar">
      <button class="exportbtn" onclick="{p}ExportCSV()">⬇ Export CSV</button>
    </div>
    <div class="table-wrap">
      <table id="{tid}" class="data">
        <thead><tr><th>Suite</th><th>Dtype</th><th>Mode</th><th>Name</th>
          <th>{esc(BASE_LABEL)}</th><th>{esc(TARGET_LABEL)}</th><th class="status">Status</th>
          <th>Message</th></tr></thead>
        <tbody id="{bid}_body"></tbody>
      </table>
    </div>
    <div class="pager">Showing <b id="{bid}_shown">0</b> of <b id="{bid}_match">0</b> matched
      &nbsp;·&nbsp; <span id="{bid}_total"></span> total
      <button id="{bid}_more" class="loadmore" onclick="{p}More()">Load more</button></div>'''

    script = f'<script>{blob}\n{_ns_js(ACC_TABLE_JS, ACC_NS, ns)}</script>'
    return table + script


ACC_TABLE_JS = r'''
let ACC_FILTER='regression', ACC_SEARCH='', ACC_LIMIT=1000;
let ACC_COLF=[], ACC_SC=-1, ACC_SD=1; const ACC_NUM=[];
const ACC_CATLBL={pass:'No Change',fail:'Both Fail',regression:'Regression',improvement:'Improvement'};
const ACC_CATCLS={pass:'pass',fail:'fail',regression:'reg',improvement:'imp'};
function accEsc(s){return String(s).replace(/[&<>]/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[m]));}
const ACC_FILTERS=['all','regression','improvement','target_fail_issue',
  'target_fail_noissue','both_fail','no_change'];
function accRowInFilter(row,f){
  const cat=ACC_CATS[row[7]];
  const tfail=(cat==='regression'||cat==='fail');
  switch(f){
    case 'all': return true;
    case 'regression': return cat==='regression';
    case 'improvement': return cat==='improvement';
    case 'target_fail_issue': return tfail&&row[8]>=0;
    case 'target_fail_noissue': return tfail&&row[8]<0;
    case 'both_fail': return cat==='fail';
    case 'no_change': return cat==='pass';
    default: return cat===f;
  }
}
function accSearchMatch(row){
  if(!ACC_SEARCH) return true;
  const s=(ACC_SUITE[row[0]]+' '+ACC_DTYPE[row[1]]+' '+ACC_MODE[row[2]]+' '+row[3]+' '+(row[10]||'')).toLowerCase();
  return s.includes(ACC_SEARCH);
}
function accIssueStr(row){ const ii=row[8]; return ii>=0? '#'+ACC_ISSUES[ii][0] : ''; }
function accIssueSt(row){ const ii=row[8]; return ii>=0? (row[9]?'fixed':(ACC_ISSUES[ii][2]||'opened')) : ''; }
function accCellText(row){ return [ACC_SUITE[row[0]],ACC_DTYPE[row[1]],ACC_MODE[row[2]],row[3],
  row[5]<0?'-':ACC_VALS[row[5]], row[6]<0?'-':ACC_VALS[row[6]], ACC_CATLBL[ACC_CATS[row[7]]],
  row[10]||'']; }
function accMatch(row){
  if(!accRowInFilter(row,ACC_FILTER)) return false;
  if(!accSearchMatch(row)) return false;
  if(gAnyF(ACC_COLF)&&!gColMatch(accCellText(row),ACC_COLF)) return false;
  return true;
}
function accUpdateCounts(){
  const cnt={}; ACC_FILTERS.forEach(f=>cnt[f]=0);
  const anyc=gAnyF(ACC_COLF);
  for(const row of ACC_ROWS){
    if(!accSearchMatch(row)) continue;
    if(anyc&&!gColMatch(accCellText(row),ACC_COLF)) continue;
    for(const f of ACC_FILTERS) if(accRowInFilter(row,f)) cnt[f]++;
  }
  document.querySelectorAll('.panel#acc .cards .card[data-f]').forEach(card=>{
    const f=card.getAttribute('data-f'); const el=card.querySelector('.num');
    if(el && cnt[f]!==undefined) el.textContent=cnt[f];
  });
}
function accCardFilter(card){
  ACC_FILTER=card.getAttribute('data-f');
  card.closest('.cards').querySelectorAll('.card[data-f]').forEach(c=>c.classList.remove('sel'));
  card.classList.add('sel');
  accRender(true);
}
function accColF(i,v){ ACC_COLF[i]=v; accRender(true); }
function accSortBy(i){ if(ACC_SC===i){ACC_SD=-ACC_SD;}else{ACC_SC=i;ACC_SD=1;} gSortInd('tbl_accd',ACC_SC,ACC_SD); accRender(true); }
function accRowHtml(row){
  const cat=ACC_CATS[row[7]]; const cls=ACC_CATCLS[cat];
  const b=row[5]<0?'-':ACC_VALS[row[5]];
  const t=row[6]<0?'-':ACC_VALS[row[6]];
  const msg=row[10]||'';
  return '<tr class="row-'+cls+'"><td>'+accEsc(ACC_SUITE[row[0]])+'</td><td>'+
    accEsc(ACC_DTYPE[row[1]])+'</td><td>'+accEsc(ACC_MODE[row[2]])+'</td><td>'+
    accEsc(row[3])+'</td><td>'+b+'</td><td>'+t+
    '</td><td class="status"><span class="badge '+cls+'">'+ACC_CATLBL[cat]+'</span></td>'+
    '<td class="msg" title="'+accEsc(msg)+'">'+accEsc(msg)+'</td></tr>';
}
function accRender(reset){
  if(reset) ACC_LIMIT=1000;
  const body=document.getElementById('accd_body');
  let arr=[];
  for(const row of ACC_ROWS){ if(accMatch(row)) arr.push(row); }
  const matched=arr.length;
  arr=gSortRows(arr,accCellText,ACC_SC,ACC_SD,ACC_NUM.indexOf(ACC_SC)>=0);
  const shown=Math.min(ACC_LIMIT,matched);
  let html='';
  for(let k=0;k<shown;k++) html+=accRowHtml(arr[k]);
  body.innerHTML = html || '<tr><td colspan="8" style="text-align:center;color:#94a3b8;padding:22px">No matching cases</td></tr>';
  document.getElementById('accd_shown').textContent=shown;
  document.getElementById('accd_match').textContent=matched;
  document.getElementById('accd_total').textContent=ACC_ROWS.length+' total';
  document.getElementById('accd_more').style.display = matched>shown ? '' : 'none';
  accUpdateCounts();
}
function accSearch(v){ ACC_SEARCH=v.toLowerCase(); accRender(true); }
function accMore(){ ACC_LIMIT+=3000; accRender(false); }
function accExportCSV(){
  const hdr=['Suite','Dtype','Mode','Name','Baseline','Target','Status','Message'];
  const lines=[hdr.map(csvCell).join(',')];
  for(const row of ACC_ROWS){
    if(!accMatch(row)) continue;
    const cat=ACC_CATS[row[7]];
    const b=row[5]<0?'':ACC_VALS[row[5]], t=row[6]<0?'':ACC_VALS[row[6]];
    lines.push([ACC_SUITE[row[0]],ACC_DTYPE[row[1]],ACC_MODE[row[2]],row[3],b,t,ACC_CATLBL[cat],row[10]||''].map(csvCell).join(','));
  }
  csvDownload(lines.join('\n'),'accuracy_'+ACC_FILTER+'.csv');
}
document.addEventListener('DOMContentLoaded',()=>{ gMakeFilters('tbl_accd',8,accColF); gMakeSort('tbl_accd',accSortBy); accRender(true); });
'''


def perf2_detail_html(rows, issues, lookup, target_msg, ns=None):
    """Client-rendered performance case-level table (eager/inductor) with cards-as-filters."""
    import json
    ns = ns or PF_NS
    P, p, tid, bid = ns["P"], ns["p"], ns["tid"], ns["bid"]
    cat_i = {c: i for i, c in enumerate(PERF_CATS)}
    suites, sidx = [], {}
    dtypes, didx = [], {}
    modes, midx = [], {}

    def num(x):
        return round(x, 4) if x is not None else 0

    data = []
    for r in rows:
        suite, dtype, mode, name, scenario = r["key"]
        for lst, ix, val in ((suites, sidx, suite), (dtypes, didx, dtype), (modes, midx, mode)):
            if val not in ix:
                ix[val] = len(lst); lst.append(val)
        hit = lookup.get(r["key"])
        issue_idx = hit[0] if hit else -1
        fixed = 1 if (hit and hit[1]) else 0
        tfail = not _pos(r["ind_t"])
        msg = target_msg.get(r["key"], "") if tfail else ""
        data.append([sidx[suite], didx[dtype], midx[mode], name,
                     str(r["bt"]), str(r["bb"]), num(r["ind_t"]), num(r["ind_b"]),
                     num(r["eag_t"]), num(r["eag_b"]), cat_i[r["cat"]], issue_idx, fixed, msg])

    issues_js = [[i["id"], i["url"], i["state"]] for i in issues]
    blob = (
        f"const {P}_SUITE=" + json.dumps(suites) + ";"
        f"const {P}_DTYPE=" + json.dumps(dtypes) + ";"
        f"const {P}_MODE=" + json.dumps(modes) + ";"
        f"const {P}_CATS=" + json.dumps(PERF_CATS) + ";"
        f"const {P}_CATLBL=" + json.dumps(PERF_CAT_LABEL) + ";"
        f"const {P}_CATCLS=" + json.dumps(PERF_CAT_CLS) + ";"
        f"const {P}_ISSUES=" + json.dumps(issues_js) + ";"
        f"const {P}_ROWS=" + json.dumps(data, separators=(",", ":")) + ";"
    ).replace("</", "<\\/")

    table = f'''
    <h3 class="subh">Case-level comparison <span class="hint">(click a card above to filter · Ratio = baseline/target latency)</span></h3>
    <div class="toolbar">
      <button class="exportbtn" onclick="{p}ExportCSV()">⬇ Export CSV</button>
    </div>
    <div class="table-wrap">
      <table id="{tid}" class="data">
        <thead><tr><th>Suite</th><th>Dtype</th><th>Mode</th><th>Model</th>
          <th class="num">BS(T)</th><th class="num">BS(B)</th>
          <th class="num">Inductor(T)</th><th class="num">Inductor(B)</th><th class="num">Ind Ratio</th>
          <th class="num">Eager(T)</th><th class="num">Eager(B)</th><th class="num">Eager Ratio</th>
          <th class="status">Status</th><th>Message</th></tr></thead>
        <tbody id="{bid}_body"></tbody>
      </table>
    </div>
    <div class="pager">Showing <b id="{bid}_shown">0</b> of <b id="{bid}_match">0</b> matched
      &nbsp;·&nbsp; <span id="{bid}_total"></span> total
      <button id="{bid}_more" class="loadmore" onclick="{p}More()">Load more</button></div>'''

    script = f'<script>{blob}\n{_ns_js(PERF2_TABLE_JS, PF_NS, ns)}</script>'
    return table + script


PERF2_TABLE_JS = r'''
let PF_FILTER='drop', PF_SEARCH='', PF_LIMIT=1000;
let PF_COLF=[], PF_SC=-1, PF_SD=1; const PF_NUM=[4,5,6,7,8,9,10,11];
function pfEsc(s){return String(s).replace(/[&<>]/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[m]));}
const PF_FILTERS=['all','bs_change','new_pass','new_fail','improve','drop',
  'stable','both_fail','target_fail_issue','target_fail_noissue'];
function pfIndRatio(row){ return (row[6]>0&&row[7]>0)? row[7]/row[6] : null; }
function pfEagRatio(row){ return (row[8]>0&&row[9]>0)? row[9]/row[8] : null; }
function pfRowInFilter(row,f){
  const cat=PF_CATS[row[10]];
  const tfail=!(row[6]>0);
  switch(f){
    case 'all': return true;
    case 'target_fail_issue': return tfail&&row[11]>=0;
    case 'target_fail_noissue': return tfail&&row[11]<0;
    default: return cat===f;
  }
}
function pfSearchMatch(row){
  if(!PF_SEARCH) return true;
  const s=(PF_SUITE[row[0]]+' '+PF_DTYPE[row[1]]+' '+PF_MODE[row[2]]+' '+row[3]+' '+(row[13]||'')).toLowerCase();
  return s.includes(PF_SEARCH);
}
function pfFmt(x){ return (x>0)? x.toFixed(3) : '-'; }
function pfIssueStr(row){ const ii=row[11]; return ii>=0? '#'+PF_ISSUES[ii][0] : ''; }
function pfIssueSt(row){ const ii=row[11]; return ii>=0? (row[12]?'fixed':(PF_ISSUES[ii][2]||'opened')) : ''; }
function pfCellText(row){
  const ir=pfIndRatio(row), er=pfEagRatio(row);
  return [PF_SUITE[row[0]],PF_DTYPE[row[1]],PF_MODE[row[2]],row[3],row[4],row[5],
    pfFmt(row[6]),pfFmt(row[7]), ir==null?'-':ir.toFixed(3), pfFmt(row[8]),pfFmt(row[9]),
    er==null?'-':er.toFixed(3), PF_CATLBL[PF_CATS[row[10]]], row[13]||''];
}
function pfMatch(row){
  if(!pfRowInFilter(row,PF_FILTER)) return false;
  if(!pfSearchMatch(row)) return false;
  if(gAnyF(PF_COLF)&&!gColMatch(pfCellText(row),PF_COLF)) return false;
  return true;
}
function pfUpdateCounts(){
  const cnt={}; PF_FILTERS.forEach(f=>cnt[f]=0);
  const anyc=gAnyF(PF_COLF);
  for(const row of PF_ROWS){
    if(!pfSearchMatch(row)) continue;
    if(anyc&&!gColMatch(pfCellText(row),PF_COLF)) continue;
    for(const f of PF_FILTERS) if(pfRowInFilter(row,f)) cnt[f]++;
  }
  document.querySelectorAll('.panel#perf .cards .card[data-f]').forEach(card=>{
    const f=card.getAttribute('data-f'); const el=card.querySelector('.num');
    if(el && cnt[f]!==undefined) el.textContent=cnt[f];
  });
}
function pfCardFilter(card){
  PF_FILTER=card.getAttribute('data-f');
  card.closest('.cards').querySelectorAll('.card[data-f]').forEach(c=>c.classList.remove('sel'));
  card.classList.add('sel');
  pfRender(true);
}
function pfColF(i,v){ PF_COLF[i]=v; pfRender(true); }
function pfSortBy(i){ if(PF_SC===i){PF_SD=-PF_SD;}else{PF_SC=i;PF_SD=1;} gSortInd('tbl_pfd',PF_SC,PF_SD); pfRender(true); }
function pfRatioCell(x){
  if(x==null) return '<td class="num">-</td>';
  const cls = x>1.10?' gm-good':(x<0.90?' gm-bad':'');
  return '<td class="num'+cls+'">'+x.toFixed(3)+'</td>';
}
function pfRowHtml(row){
  const cat=PF_CATS[row[10]]; const cls=PF_CATCLS[cat];
  const indR=pfIndRatio(row), eagR=pfEagRatio(row);
  const msg=row[13]||'';
  return '<tr class="row-'+cls+'"><td>'+pfEsc(PF_SUITE[row[0]])+'</td><td>'+pfEsc(PF_DTYPE[row[1]])+
    '</td><td>'+pfEsc(PF_MODE[row[2]])+'</td><td>'+pfEsc(row[3])+
    '</td><td class="num">'+pfEsc(row[4])+'</td><td class="num">'+pfEsc(row[5])+
    '</td><td class="num">'+pfFmt(row[6])+'</td><td class="num">'+pfFmt(row[7])+'</td>'+pfRatioCell(indR)+
    '<td class="num">'+pfFmt(row[8])+'</td><td class="num">'+pfFmt(row[9])+'</td>'+pfRatioCell(eagR)+
    '<td class="status"><span class="badge '+cls+'">'+PF_CATLBL[cat]+'</span></td>'+
    '<td class="msg" title="'+pfEsc(msg)+'">'+pfEsc(msg)+'</td></tr>';
}
function pfRender(reset){
  if(reset) PF_LIMIT=1000;
  const body=document.getElementById('pfd_body');
  let arr=[];
  for(const row of PF_ROWS){ if(pfMatch(row)) arr.push(row); }
  const matched=arr.length;
  arr=gSortRows(arr,pfCellText,PF_SC,PF_SD,PF_NUM.indexOf(PF_SC)>=0);
  const shown=Math.min(PF_LIMIT,matched);
  let html='';
  for(let k=0;k<shown;k++) html+=pfRowHtml(arr[k]);
  body.innerHTML = html || '<tr><td colspan="14" style="text-align:center;color:#94a3b8;padding:22px">No matching cases</td></tr>';
  document.getElementById('pfd_shown').textContent=shown;
  document.getElementById('pfd_match').textContent=matched;
  document.getElementById('pfd_total').textContent=PF_ROWS.length+' total';
  document.getElementById('pfd_more').style.display = matched>shown ? '' : 'none';
  pfUpdateCounts();
}
function pfSearch(v){ PF_SEARCH=v.toLowerCase(); pfRender(true); }
function pfMore(){ PF_LIMIT+=3000; pfRender(false); }
function pfExportCSV(){
  const hdr=['Suite','Dtype','Mode','Model','BS(T)','BS(B)','Inductor(T)','Inductor(B)','Ind Ratio','Eager(T)','Eager(B)','Eager Ratio','Status','Message'];
  const lines=[hdr.map(csvCell).join(',')];
  for(const row of PF_ROWS){
    if(!pfMatch(row)) continue;
    const cat=PF_CATS[row[10]];
    const indR=pfIndRatio(row), eagR=pfEagRatio(row);
    lines.push([PF_SUITE[row[0]],PF_DTYPE[row[1]],PF_MODE[row[2]],row[3],row[4],row[5],
      row[6]>0?row[6]:'',row[7]>0?row[7]:'',indR==null?'':indR.toFixed(4),
      row[8]>0?row[8]:'',row[9]>0?row[9]:'',eagR==null?'':eagR.toFixed(4),
      PF_CATLBL[cat],row[13]||''].map(csvCell).join(','));
  }
  csvDownload(lines.join('\n'),'performance_'+PF_FILTER+'.csv');
}
document.addEventListener('DOMContentLoaded',()=>{ gMakeFilters('tbl_pfd',14,pfColF); gMakeSort('tbl_pfd',pfSortBy); pfRender(true); });
'''


def build_html(sections, meta):
    nav = "".join(
        f'<button class="tab{" active" if i == 0 else ""}" data-tab="{s["id"]}" onclick="showTab(\'{s["id"]}\', this)">{esc(s["title"])}'
        f'<span class="tab-badge {"warn" if s.get("badge") else ""}">{s.get("badge", 0)}</span></button>'
        for i, s in enumerate(sections))

    panels = []
    for i, s in enumerate(sections):
        cards_html = ("" if s.get("cards") is False else
                      summary_cards_html(s["title"], s["counts"], s.get("cards"),
                                         s.get("card_onclick", "utCardFilter"),
                                         s.get("card_default", "regression")))
        panels.append(
            f'<section id="{s["id"]}" class="panel{" active" if i == 0 else ""}">'
            f'<h2>{esc(s["title"])}</h2>'
            f'<p class="desc">{esc(s["desc"])}</p>'
            f'{cards_html}'
            f'{s["table"]}'
            f'</section>')

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(meta["component"])} acceptance · {esc(meta["target_version"])} vs {esc(meta["base_version"])}</title>
<style>
:root {{
  --bg:#0f172a; --panel:#111c33; --card:#16233f;
  --line:#243043; --text:#e2e8f0; --muted:#94a3b8;
  --reg:#ef4444; --imp:#22c55e; --new:#3b82f6; --rem:#a855f7;
  --crash:#fb7185; --timeout:#eab308; --notrun:#64748b;
  --fail:#f59e0b; --pass:#334155; --accent:#38bdf8;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:'Segoe UI',system-ui,-apple-system,sans-serif;
  background:linear-gradient(160deg,#0b1220,#0f172a); color:var(--text); }}
header.top {{ padding:28px 40px 18px; border-bottom:1px solid var(--line);
  background:rgba(15,23,42,.7); backdrop-filter:blur(6px); position:sticky; top:0; z-index:20; }}
header.top h1 {{ margin:0 0 6px; font-size:24px; letter-spacing:.3px; }}
header.top .sub {{ color:var(--muted); font-size:13px; }}
.legend {{ display:flex; gap:14px; flex-wrap:wrap; margin-top:12px; font-size:12px; }}
.legend span {{ display:flex; align-items:center; gap:6px; color:var(--muted); }}
.legend .lg-label {{ color:var(--text); font-weight:600; }}
.legend .lg-sep {{ width:1px; height:14px; background:var(--line); margin:0 2px; }}
.inline-legend {{ margin:0 0 12px; padding:10px 14px; background:var(--card);
  border:1px solid var(--line); border-radius:10px; }}
.dot {{ width:11px; height:11px; border-radius:3px; display:inline-block; }}
.tabs {{ display:flex; gap:8px; flex-wrap:wrap; padding:16px 40px 0; }}
.tab {{ background:var(--card); color:var(--text); border:1px solid var(--line);
  padding:10px 16px; border-radius:10px 10px 0 0; cursor:pointer; font-size:14px;
  display:flex; align-items:center; gap:8px; transition:.15s; }}
.tab:hover {{ background:#1c2c4d; }}
.tab.active {{ background:var(--panel); border-bottom-color:var(--panel); color:#fff; }}
.tab-badge {{ background:#334155; color:#cbd5e1; font-size:11px; padding:1px 7px;
  border-radius:999px; }}
.tab-badge.warn {{ background:var(--reg); color:#fff; }}
main {{ padding:0 40px 60px; }}
.panel {{ display:none; background:var(--panel); border:1px solid var(--line);
  border-radius:0 12px 12px 12px; padding:26px; }}
.panel.active {{ display:block; }}
.panel h2 {{ margin:0 0 4px; font-size:19px; }}
.desc {{ color:var(--muted); margin:0 0 18px; font-size:13px; }}
.cards {{ display:flex; gap:14px; flex-wrap:wrap; margin-bottom:22px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
  padding:14px 20px; min-width:110px; display:flex; flex-direction:column; gap:2px;
  border-top:3px solid var(--line); }}
.card .num {{ font-size:26px; font-weight:700; }}
.card .lbl {{ font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:.5px; }}
.card.total {{ border-top-color:var(--accent); }}
.card.reg {{ border-top-color:var(--reg); }}
.card.imp {{ border-top-color:var(--imp); }}
.card.new {{ border-top-color:var(--new); }}
.card.rem {{ border-top-color:var(--rem); }}
.card.crash {{ border-top-color:var(--crash); }}
.card.timeout {{ border-top-color:var(--timeout); }}
.card.notrun {{ border-top-color:var(--notrun); }}
.card.fail {{ border-top-color:var(--fail); }}
.card.pass {{ border-top-color:var(--pass); }}
.card.other {{ border-top-color:var(--accent); }}
.card.clickable {{ cursor:pointer; transition:.15s; }}
.card.clickable:hover {{ background:#1c2c4d; transform:translateY(-1px); }}
.card.clickable.sel {{ background:#1c2c4d; box-shadow:0 0 0 2px var(--accent) inset; }}
.overview {{ padding:22px 40px 4px; }}
.ov-titlebar {{ display:flex; align-items:baseline; gap:14px; margin-bottom:14px; flex-wrap:wrap; }}
.ov-title {{ margin:0; font-size:19px; letter-spacing:.3px; }}
.ov-cap {{ margin:0; font-size:12px; color:var(--muted); }}
.ov-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:16px; }}
.ov-card {{ position:relative; background:linear-gradient(150deg,#16233f,#111c33);
  border:1px solid var(--line); border-radius:14px; padding:16px 18px; overflow:hidden; }}
.ov-card::before {{ content:""; position:absolute; top:0; bottom:0; left:0; width:3px;
  background:var(--accent); opacity:.7; }}
.ov-top {{ display:flex; align-items:center; gap:8px; margin-bottom:12px; }}
.ov-name {{ font-size:13px; font-weight:700; color:#fff; letter-spacing:.3px; }}
.ov-tag {{ font-size:10px; text-transform:uppercase; letter-spacing:.5px; color:var(--accent);
  border:1px solid var(--accent); border-radius:999px; padding:1px 8px; }}
.ov-hero {{ display:flex; flex-direction:column; gap:2px; margin-bottom:12px; }}
.ovh-v {{ font-size:30px; font-weight:800; line-height:1; }}
.ovh-l {{ font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.5px; }}
.ov-pills {{ display:flex; gap:8px; flex-wrap:wrap; }}
.ov-pill {{ display:flex; align-items:baseline; gap:6px; background:rgba(148,163,184,.08);
  border:1px solid var(--line); border-radius:8px; padding:5px 9px; }}
.ovp-v {{ font-size:14px; font-weight:700; }}
.ovp-l {{ font-size:10.5px; color:var(--muted); text-transform:uppercase; letter-spacing:.3px; }}
.ovh-v.reg, .ovp-v.reg {{ color:var(--reg); }}
.ovh-v.imp, .ovp-v.imp {{ color:var(--imp); }}
.ovh-v.pass, .ovp-v.pass {{ color:var(--accent); }}
.ovh-v.fail, .ovp-v.fail {{ color:var(--fail); }}
.ovh-v.neutral, .ovp-v.neutral {{ color:var(--text); }}
.gate {{ margin:16px 40px 0; border-radius:12px; padding:14px 20px; border:1px solid var(--line); }}
.gate-hd {{ font-size:16px; font-weight:700; }}
.gate-fail {{ background:rgba(239,68,68,.12); border-color:rgba(239,68,68,.5); }}
.gate-fail .gate-hd {{ color:#fca5a5; }}
.gate-pass {{ background:rgba(34,197,94,.10); border-color:rgba(34,197,94,.4); }}
.gate-pass .gate-hd {{ color:#86efac; }}
.gate-list {{ margin:8px 0 0; padding-left:22px; color:#fecaca; font-size:13px; }}
.gate-list li {{ margin:2px 0; }}
.toolbar {{ display:flex; gap:14px; align-items:center; flex-wrap:wrap; margin-bottom:12px; }}
.search {{ flex:1; min-width:220px; background:var(--card); border:1px solid var(--line);
  color:var(--text); padding:9px 14px; border-radius:9px; font-size:13px; }}
.search:focus {{ outline:none; border-color:var(--accent); }}
.chips {{ display:flex; gap:7px; flex-wrap:wrap; }}
.chip {{ background:var(--card); color:var(--muted); border:1px solid var(--line);
  padding:7px 13px; border-radius:999px; cursor:pointer; font-size:12px; }}
.chip.active {{ background:var(--accent); color:#04263a; border-color:var(--accent); font-weight:600; }}
.chip .cc {{ display:inline-block; margin-left:7px; padding:0 7px; border-radius:999px;
  background:rgba(148,163,184,.25); color:inherit; font-size:11px; font-weight:700; min-width:16px; text-align:center; }}
.chip.active .cc {{ background:rgba(4,38,58,.25); }}
.table-wrap {{ overflow:auto; border:1px solid var(--line); border-radius:10px; max-height:70vh; }}
table.data {{ border-collapse:collapse; width:100%; font-size:12.5px; }}
table.data thead th {{ position:sticky; top:0; background:#0d1830; color:#cbd5e1;
  text-align:left; padding:10px 12px; border-bottom:1px solid var(--line); white-space:nowrap;
  cursor:pointer; vertical-align:middle; }}
table.data tbody td {{ padding:8px 12px; border-bottom:1px solid #1a2740; vertical-align:middle; }}
table.data tbody tr:hover {{ background:#152341; }}
table.data th.num, table.data td.num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
table.data th.status, table.data td.status {{ text-align:center; white-space:nowrap; }}
table.data td.num.gm-good {{ background:rgba(34,197,94,.18); color:#86efac; font-weight:600; }}
table.data td.num.gm-bad {{ background:rgba(239,68,68,.18); color:#fca5a5; font-weight:600; }}
.badge {{ padding:2px 9px; border-radius:999px; font-size:11px; font-weight:600; white-space:nowrap; }}
.badge.reg {{ background:rgba(239,68,68,.18); color:#fca5a5; }}
.badge.imp {{ background:rgba(34,197,94,.18); color:#86efac; }}
.badge.new {{ background:rgba(59,130,246,.18); color:#93c5fd; }}
.badge.rem {{ background:rgba(168,85,247,.18); color:#d8b4fe; }}
.badge.crash {{ background:rgba(251,113,133,.20); color:#fda4af; }}
.badge.timeout {{ background:rgba(234,179,8,.18); color:#fde047; }}
.badge.notrun {{ background:rgba(100,116,139,.20); color:#cbd5e1; }}
.badge.fail {{ background:rgba(245,158,11,.18); color:#fcd34d; }}
.badge.pass {{ background:rgba(148,163,184,.16); color:#cbd5e1; }}
.badge.unk {{ background:rgba(56,189,248,.14); color:#7dd3fc; }}
.ibadge {{ padding:2px 9px; border-radius:999px; font-size:11px; font-weight:600; white-space:nowrap; }}
.ibadge.issue-open {{ background:rgba(245,158,11,.18); color:#fcd34d; }}
.ibadge.issue-closed {{ background:rgba(148,163,184,.18); color:#cbd5e1; }}
.ibadge.issue-fixed {{ background:rgba(34,197,94,.18); color:#86efac; }}
table.data td.status a {{ color:var(--accent); text-decoration:none; font-weight:600; }}
table.data td.status a:hover {{ text-decoration:underline; }}
table.data td.msg {{ max-width:520px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
  color:#fca5a5; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:11.5px; cursor:help; }}
.row-reg td:first-child {{ box-shadow:inset 3px 0 var(--reg); }}
.row-imp td:first-child {{ box-shadow:inset 3px 0 var(--imp); }}
.row-new td:first-child {{ box-shadow:inset 3px 0 var(--new); }}
.row-rem td:first-child {{ box-shadow:inset 3px 0 var(--rem); }}
.row-crash td:first-child {{ box-shadow:inset 3px 0 var(--crash); }}
.row-timeout td:first-child {{ box-shadow:inset 3px 0 var(--timeout); }}
.row-notrun td:first-child {{ box-shadow:inset 3px 0 var(--notrun); }}
.subh {{ margin:22px 0 8px; font-size:15px; color:#e2e8f0; }}
.subh .hint {{ font-size:12px; color:var(--muted); font-weight:400; }}
.table-wrap.small {{ max-height:none; }}
table.data.summary-tbl thead th, table.data.summary-tbl tbody td {{ text-align:right; font-variant-numeric:tabular-nums; }}
table.data.summary-tbl thead th:first-child, table.data.summary-tbl tbody td.rowlbl {{ text-align:left; font-weight:600; }}
.summary-tbl .row-target td {{ background:rgba(56,189,248,.08); }}
.summary-tbl .row-delta td {{ border-top:2px solid var(--line); color:var(--muted); }}
.summary-tbl td.pr-worse {{ background:rgba(239,68,68,.20); color:#fca5a5; font-weight:700; box-shadow:inset 0 0 0 1px rgba(239,68,68,.45); }}
.summary-tbl .d-up {{ color:#86efac; font-weight:600; }}
.summary-tbl .d-down {{ color:#fca5a5; font-weight:600; }}
.summary-tbl td.gm-good {{ background:rgba(34,197,94,.20); color:#86efac; font-weight:700; }}
.summary-tbl td.gm-bad {{ background:rgba(239,68,68,.20); color:#fca5a5; font-weight:700; }}
.summary-tbl td.gm-mid {{ color:#cbd5e1; font-weight:600; }}
tr.colfilters th {{ padding:3px 4px; background:#0e1626; position:sticky; top:0; }}
tr.colfilters input.colf {{ width:100%; box-sizing:border-box; font-size:11px; font-weight:400;
  padding:2px 5px; background:#0b1220; color:var(--text); border:1px solid var(--line); border-radius:4px; }}
tr.colfilters input.colf:focus {{ outline:none; border-color:#38bdf8; }}
table.data thead th.sortable {{ cursor:pointer; user-select:none; }}
table.data thead th.sortable:hover {{ color:#e2e8f0; }}
table.data thead th.sort-asc::after {{ content:' \\25B2'; font-size:9px; color:#38bdf8; }}
table.data thead th.sort-desc::after {{ content:' \\25BC'; font-size:9px; color:#38bdf8; }}
.pager {{ display:flex; align-items:center; gap:10px; margin-top:12px; font-size:13px; color:var(--muted); }}
.loadmore {{ background:var(--accent); color:#04263a; border:none; padding:7px 16px;
  border-radius:8px; cursor:pointer; font-size:13px; font-weight:600; }}
.loadmore:hover {{ filter:brightness(1.08); }}
.exportbtn {{ background:var(--card); color:var(--text); border:1px solid var(--line);
  padding:8px 14px; border-radius:9px; cursor:pointer; font-size:13px; font-weight:600; white-space:nowrap; }}
.exportbtn:hover {{ background:#1c2c4d; border-color:var(--accent); }}
footer {{ text-align:center; color:var(--muted); font-size:12px; padding:24px; }}
</style>
</head>
<body>
<header class="top">
  <h1>{esc(meta["component"])} acceptance <span style="color:var(--accent)">· {esc(meta["target_version"])} vs {esc(meta["base_version"])}</span></h1>
  <div class="sub">Component <b>{esc(meta["component"])}</b> &nbsp;·&nbsp; Baseline <b>{esc(meta["base_version"])}</b> &nbsp;→&nbsp; Target <b>{esc(meta["target_version"])}</b>
     &nbsp;·&nbsp; Generated {esc(meta["ts"])}</div>
</header>
{gate_banner_html(meta.get("gate_reasons", []))}
{overview_html(meta.get("summary", []), meta.get("summary_caption", ""))}
<nav class="tabs">{nav}</nav>
<main>{''.join(panels)}</main>
<footer>Generated by gen_report.py · perf regression &lt; {int(PERF_REG_THRESHOLD*100)}% · improvement &gt; {int(PERF_IMP_THRESHOLD*100)}% of baseline speedup</footer>
<script>
function showTab(id, btn) {{
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  btn.classList.add('active');
}}
function filterTable(id, q) {{
  q = q.toLowerCase();
  const t = document.getElementById(id);
  const f = t.getAttribute('data-chip') || 'all';
  const counts = {{ all:0, regression:0, improvement:0, new:0, removed:0, fail:0, pass:0, others:0 }};
  t.querySelectorAll('tbody tr').forEach(tr => {{
    const cat = tr.getAttribute('data-cat');
    const okText = tr.innerText.toLowerCase().includes(q);
    if (okText && cat) {{ counts.all++; if (counts[cat] !== undefined) counts[cat]++; }}
    const okCat = (f === 'all') || (cat === f);
    tr.style.display = (okText && okCat) ? '' : 'none';
  }});
  t.closest('.panel').querySelectorAll('.chips .chip').forEach(btn => {{
    const cf = btn.getAttribute('data-f'); const el = btn.querySelector('.cc');
    if (el && counts[cf] !== undefined) el.textContent = counts[cf];
  }});
}}
function chipFilter(id, btn) {{
  const t = document.getElementById(id);
  t.setAttribute('data-chip', btn.getAttribute('data-f'));
  btn.parentNode.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
  btn.classList.add('active');
  const search = btn.closest('.toolbar').querySelector('.search');
  filterTable(id, search ? search.value : '');
}}
// initialize chip counts for the static tables
document.addEventListener('DOMContentLoaded', () => {{
  const PAGINATED = ['tbl_ut','tbl_accd','tbl_pfd','tbl_qd','tbl_qpd','tbl_oacd','tbl_opfd'];
  document.querySelectorAll('table.data').forEach(tbl => {{
    if (tbl.id && tbl.id !== 'tbl_ut' && PAGINATED.indexOf(tbl.id) < 0) filterTable(tbl.id, '');
  }});
}});
// shared grid helpers for the paginated case-level tables (sort + per-column filter)
function gColMatch(t, f) {{
  for (let i = 0; i < f.length; i++) {{
    if (f[i] && String(t[i] == null ? '' : t[i]).toLowerCase().indexOf(f[i]) < 0) return false;
  }}
  return true;
}}
function gAnyF(f) {{ for (let i = 0; i < f.length; i++) if (f[i]) return true; return false; }}
function gSortRows(rows, tf, c, d, num) {{
  if (c < 0) return rows;
  const keyed = rows.map((r, i) => [tf(r)[c], i, r]);
  keyed.sort((x, y) => {{
    let cc;
    if (num) {{
      const p = parseFloat(x[0]), q = parseFloat(y[0]);
      const pn = isNaN(p), qn = isNaN(q);
      cc = (pn && qn) ? 0 : pn ? 1 : qn ? -1 : (p - q);
    }} else cc = String(x[0] == null ? '' : x[0]).localeCompare(String(y[0] == null ? '' : y[0]));
    return cc !== 0 ? cc * d : (x[1] - y[1]);
  }});
  return keyed.map(z => z[2]);
}}
function gMakeFilters(id, n, cb) {{
  const h = document.querySelector('#' + id + ' thead');
  if (!h || h.querySelector('.colfilters')) return;
  const tr = document.createElement('tr'); tr.className = 'colfilters';
  for (let i = 0; i < n; i++) {{
    const th = document.createElement('th');
    const inp = document.createElement('input');
    inp.className = 'colf'; inp.type = 'text'; inp.setAttribute('aria-label', 'filter column');
    inp.addEventListener('input', () => cb(i, inp.value.toLowerCase().trim()));
    inp.addEventListener('click', e => e.stopPropagation());
    th.appendChild(inp); tr.appendChild(th);
  }}
  h.appendChild(tr);
}}
function gMakeSort(id, cb) {{
  document.querySelectorAll('#' + id + ' thead tr:first-child th').forEach((th, i) => {{
    th.classList.add('sortable');
    th.addEventListener('click', () => cb(i));
  }});
}}
function gSortInd(id, c, d) {{
  document.querySelectorAll('#' + id + ' thead tr:first-child th').forEach((th, i) => {{
    th.classList.remove('sort-asc', 'sort-desc');
    if (i === c) th.classList.add(d > 0 ? 'sort-asc' : 'sort-desc');
  }});
}}
// column sort
document.querySelectorAll('table.data thead th').forEach((th, idx) => {{
  const tid = th.closest('table').id;
  if (['tbl_ut','tbl_accd','tbl_pfd','tbl_qd','tbl_qpd','tbl_oacd','tbl_opfd','tbl_issues','tbl_prev_issues'].indexOf(tid) >= 0) return;  // handled per-table
  th.addEventListener('click', () => {{
    const table = th.closest('table');
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    const asc = !(th.__asc);
    th.__asc = asc;
    rows.sort((a, b) => {{
      const x = a.children[idx].innerText.trim();
      const y = b.children[idx].innerText.trim();
      const nx = parseFloat(x), ny = parseFloat(y);
      if (!isNaN(nx) && !isNaN(ny)) return asc ? nx - ny : ny - nx;
      return asc ? x.localeCompare(y) : y.localeCompare(x);
    }});
    rows.forEach(r => tbody.appendChild(r));
  }});
}});
</script>
</body>
</html>'''


# --------------------------------------------------------------------------- #
# Minimal XLSX writer (Office Open XML, standard library only)
# --------------------------------------------------------------------------- #
class XlsxWriter:
    # style indices defined in _styles_xml
    S_DEFAULT = 0
    S_HEADER = 1
    S_REG = 2
    S_IMP = 3
    S_NEW = 4
    S_REM = 5
    S_FAIL = 6
    S_PASS = 7

    CAT_STYLE = {
        "regression": S_REG, "improvement": S_IMP, "new": S_NEW,
        "removed": S_REM, "crash": S_REM, "timeout": S_FAIL, "not_run": S_DEFAULT, "fail": S_FAIL, "pass": S_PASS,
        "others": S_DEFAULT, "unknown": S_DEFAULT,
    }

    def __init__(self):
        self.sheets = []  # list of (name, rows) ; rows = list of list of (value, style)

    def add_sheet(self, name, rows):
        self.sheets.append((name[:31], rows))

    @staticmethod
    def _col(n):
        s = ""
        n += 1
        while n:
            n, r = divmod(n - 1, 26)
            s = chr(65 + r) + s
        return s

    @staticmethod
    def _cell_xml(ref, value, style):
        if value is None:
            value = ""
        if isinstance(value, bool):
            value = str(value)
        if isinstance(value, (int, float)):
            return f'<c r="{ref}" s="{style}"><v>{value}</v></c>'
        text = html.escape(str(value))
        return f'<c r="{ref}" s="{style}" t="inlineStr"><is><t xml:space="preserve">{text}</t></is></c>'

    def _sheet_xml(self, rows):
        out = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
               '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
               '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" '
               'activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>',
               '<sheetData>']
        for ri, row in enumerate(rows, start=1):
            out.append(f'<row r="{ri}">')
            for ci, cell in enumerate(row):
                value, style = cell
                ref = f"{self._col(ci)}{ri}"
                out.append(self._cell_xml(ref, value, style))
            out.append('</row>')
        out.append('</sheetData><autoFilter ref="A1"/></worksheet>')
        return "".join(out)

    @staticmethod
    def _styles_xml():
        # fills: 0 none,1 gray125(reserved),2 header,3 reg,4 imp,5 new,6 rem,7 fail,8 pass
        return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<fonts count="2">
  <font><sz val="11"/><color theme="1"/><name val="Calibri"/></font>
  <font><b/><sz val="11"/><color rgb="FFFFFFFF"/><name val="Calibri"/></font>
</fonts>
<fills count="9">
  <fill><patternFill patternType="none"/></fill>
  <fill><patternFill patternType="gray125"/></fill>
  <fill><patternFill patternType="solid"><fgColor rgb="FF1F3A5F"/></patternFill></fill>
  <fill><patternFill patternType="solid"><fgColor rgb="FFFDE0E0"/></patternFill></fill>
  <fill><patternFill patternType="solid"><fgColor rgb="FFDFF6E3"/></patternFill></fill>
  <fill><patternFill patternType="solid"><fgColor rgb="FFDDEBFF"/></patternFill></fill>
  <fill><patternFill patternType="solid"><fgColor rgb="FFF0E3FB"/></patternFill></fill>
  <fill><patternFill patternType="solid"><fgColor rgb="FFFFF3D6"/></patternFill></fill>
  <fill><patternFill patternType="solid"><fgColor rgb="FFEFEFEF"/></patternFill></fill>
</fills>
<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
<cellXfs count="8">
  <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
  <xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1"/>
  <xf numFmtId="0" fontId="0" fillId="3" borderId="0" xfId="0" applyFill="1"/>
  <xf numFmtId="0" fontId="0" fillId="4" borderId="0" xfId="0" applyFill="1"/>
  <xf numFmtId="0" fontId="0" fillId="5" borderId="0" xfId="0" applyFill="1"/>
  <xf numFmtId="0" fontId="0" fillId="6" borderId="0" xfId="0" applyFill="1"/>
  <xf numFmtId="0" fontId="0" fillId="7" borderId="0" xfId="0" applyFill="1"/>
  <xf numFmtId="0" fontId="0" fillId="8" borderId="0" xfId="0" applyFill="1"/>
</cellXfs>
<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>'''

    def save(self, path):
        n = len(self.sheets)
        content_types = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
                         '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
                         '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
                         '<Default Extension="xml" ContentType="application/xml"/>',
                         '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
                         '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>']
        for i in range(n):
            content_types.append(
                f'<Override PartName="/xl/worksheets/sheet{i+1}.xml" '
                'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>')
        content_types.append('</Types>')

        root_rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                     '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                     '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
                     '</Relationships>')

        sheets_xml = "".join(
            f'<sheet name="{html.escape(name)}" sheetId="{i+1}" r:id="rId{i+1}"/>'
            for i, (name, _rows) in enumerate(self.sheets))
        workbook = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                    '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                    'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
                    f'<sheets>{sheets_xml}</sheets></workbook>')

        wb_rels = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
                   '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">']
        for i in range(n):
            wb_rels.append(
                f'<Relationship Id="rId{i+1}" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
                f'Target="worksheets/sheet{i+1}.xml"/>')
        wb_rels.append(
            f'<Relationship Id="rId{n+1}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
            'Target="styles.xml"/>')
        wb_rels.append('</Relationships>')

        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("[Content_Types].xml", "".join(content_types))
            z.writestr("_rels/.rels", root_rels)
            z.writestr("xl/workbook.xml", workbook)
            z.writestr("xl/_rels/workbook.xml.rels", "".join(wb_rels))
            z.writestr("xl/styles.xml", self._styles_xml())
            for i, (_name, rows) in enumerate(self.sheets):
                z.writestr(f"xl/worksheets/sheet{i+1}.xml", self._sheet_xml(rows))


def xlsx_ut_rows(key_cols, rows, base_label, target_label, issues, lookup, target_msg):
    H = XlsxWriter.S_HEADER
    header = ([(c, H) for c in key_cols]
              + [(base_label, H), (target_label, H), ("Status", H), ("Message", H)])
    out = [header]
    for r in rows:
        style = XlsxWriter.CAT_STYLE[r["cat"]]
        cells = [(k, style) for k in r["key"]]
        cells.append(("" if r["base"] is None else r["base"], style))
        cells.append(("" if r["target"] is None else r["target"], style))
        cells.append((CATEGORY_LABEL[r["cat"]], style))
        msg = target_msg.get(r["key"], "")
        cells.append((msg, style))
        out.append(cells)
    return out


def summary_sheet_rows(sections):
    H = XlsxWriter.S_HEADER
    rows = [[("Section", H)] + [(CATEGORY_LABEL[c], H) for c in CATEGORY_ORDER] + [("Total", H)]]
    for s in sections:
        c = s["counts"]
        style_for = lambda cat: XlsxWriter.CAT_STYLE[cat] if c.get(cat) else XlsxWriter.S_DEFAULT
        row = [(s["title"], XlsxWriter.S_DEFAULT)]
        for cat in CATEGORY_ORDER:
            row.append((c.get(cat, 0), style_for(cat)))
        row.append((sum(c.values()), XlsxWriter.S_DEFAULT))
        rows.append(row)
    return rows


def ut_summary_sheet_rows(base_sum, target_sum):
    H = XlsxWriter.S_HEADER
    cols = ["Total", "Passed", "Passrate", "Skipped", "Failure", "Error", "Xfail", "Others"]
    rows = [[("Category", H)] + [(c, H) for c in cols]]

    def val(sm, c):
        return f"{sm[c] * 100:.2f}%" if c == "Passrate" else sm[c]

    rows.append([(TARGET_LABEL + " (Target)", XlsxWriter.S_DEFAULT)]
                + [(val(target_sum, c), XlsxWriter.S_DEFAULT) for c in cols])
    rows.append([(BASE_LABEL + " (Baseline)", XlsxWriter.S_DEFAULT)]
                + [(val(base_sum, c), XlsxWriter.S_DEFAULT) for c in cols])
    return rows


def xlsx_acc_rows(rows, base_label, target_label, issues, lookup, target_msg):
    H = XlsxWriter.S_HEADER
    keys = ["Suite", "Dtype", "Mode", "Name", "Scenario"]
    header = ([(c, H) for c in keys]
              + [(base_label, H), (target_label, H), ("Status", H), ("Message", H)])
    out = [header]
    acc_lbl = {"pass": "No Change", "fail": "Both Fail",
               "regression": "Regression", "improvement": "Improvement"}
    for r in rows:
        style = XlsxWriter.CAT_STYLE[r["cat"]]
        cells = [(k, style) for k in r["key"]]
        cells.append(("" if r["base"] is None else r["base"], style))
        cells.append(("" if r["target"] is None else r["target"], style))
        cells.append((acc_lbl.get(r["cat"], r["cat"]), style))
        cells.append((target_msg.get(r["key"], "") if r["cat"] in ("regression", "fail") else "", style))
        out.append(cells)
    return out


def xlsx_perf2_rows(rows, issues, lookup, target_msg):
    H = XlsxWriter.S_HEADER
    header = [(c, H) for c in ("Suite", "Dtype", "Mode", "Model", "BS(T)", "BS(B)",
                               "Inductor(T)", "Inductor(B)", "Ind Ratio",
                               "Eager(T)", "Eager(B)", "Eager Ratio", "Status", "Message")]
    out = [header]
    S = XlsxWriter.CAT_STYLE
    cls_style = {"imp": S["improvement"], "reg": S["regression"], "fail": S["fail"],
                 "pass": S["pass"], "unk": XlsxWriter.S_DEFAULT}
    for r in rows:
        style = cls_style.get(PERF_CAT_CLS[r["cat"]], XlsxWriter.S_DEFAULT)
        ind_r = (r["ind_b"] / r["ind_t"]) if (_pos(r["ind_t"]) and _pos(r["ind_b"])) else ""
        eag_r = (r["eag_b"] / r["eag_t"]) if (_pos(r["eag_t"]) and _pos(r["eag_b"])) else ""
        suite, dtype, mode, name, scenario = r["key"]
        out.append([
            (suite, style), (dtype, style), (mode, style), (name, style),
            (str(r["bt"]), style), (str(r["bb"]), style),
            (round(r["ind_t"], 4) if _pos(r["ind_t"]) else "", style),
            (round(r["ind_b"], 4) if _pos(r["ind_b"]) else "", style),
            (round(ind_r, 4) if ind_r != "" else "", style),
            (round(r["eag_t"], 4) if _pos(r["eag_t"]) else "", style),
            (round(r["eag_b"], 4) if _pos(r["eag_b"]) else "", style),
            (round(eag_r, 4) if eag_r != "" else "", style),
            (PERF_CAT_LABEL[r["cat"]], style),
            (target_msg.get(r["key"], "") if not _pos(r["ind_t"]) else "", style)])
    return out


def perf_summary_sheet_rows(base_sum, target_sum, base_map, target_map, rows,
                            base_notrun, target_notrun, eag_gm, ind_gm):
    H = XlsxWriter.S_HEADER
    cols = PERF_SUM_COLS
    metric = ["Total", "Passed", "Passrate", "Failed", "Notrun"]

    def val(sm, c):
        return f"{sm[c] * 100:.2f}%" if c == "Passrate" else sm[c]

    rows_out = [[("Category", H)] + [(c, H) for c in cols]]
    rows_out.append([(TARGET_LABEL + " (Target)", XlsxWriter.S_DEFAULT)]
                    + [(val(target_sum, c), XlsxWriter.S_DEFAULT) for c in metric]
                    + [(_gm(eag_gm), XlsxWriter.S_DEFAULT), (_gm(ind_gm), XlsxWriter.S_DEFAULT)])
    rows_out.append([(BASE_LABEL + " (Baseline)", XlsxWriter.S_DEFAULT)]
                    + [(val(base_sum, c), XlsxWriter.S_DEFAULT) for c in metric]
                    + [("/", XlsxWriter.S_DEFAULT), ("/", XlsxWriter.S_DEFAULT)])
    rows_out.append([("", XlsxWriter.S_DEFAULT)])
    rows_out.append([("Suite", H), ("Category", H)] + [(c, H) for c in cols])
    from collections import defaultdict as _dd
    suite_rows = _dd(list)
    for r in rows:
        suite_rows[r["key"][0]].append(r)
    base_by = perf_suite_summary(base_map, base_notrun)
    target_by = perf_suite_summary(target_map, target_notrun)
    empty = {"Total": 0, "Passed": 0, "Passrate": 0.0, "Failed": 0, "Notrun": 0}
    for suite in sorted(set(base_by) | set(target_by)):
        eg, ig = perf_geomeans(suite_rows.get(suite, []))
        tb, bb = target_by.get(suite, empty), base_by.get(suite, empty)
        rows_out.append([(suite, XlsxWriter.S_DEFAULT), ("Target", XlsxWriter.S_DEFAULT)]
                        + [(val(tb, c), XlsxWriter.S_DEFAULT) for c in metric]
                        + [(_gm(eg), XlsxWriter.S_DEFAULT), (_gm(ig), XlsxWriter.S_DEFAULT)])
        rows_out.append([("", XlsxWriter.S_DEFAULT), ("Baseline", XlsxWriter.S_DEFAULT)]
                        + [(val(bb, c), XlsxWriter.S_DEFAULT) for c in metric]
                        + [("/", XlsxWriter.S_DEFAULT), ("/", XlsxWriter.S_DEFAULT)])
    return rows_out


def acc_summary_sheet_rows(base_sum, target_sum, base_vmap, target_vmap):
    H = XlsxWriter.S_HEADER
    cols = ACC_SUM_COLS

    def val(sm, c):
        return f"{sm[c] * 100:.2f}%" if c == "Passrate" else sm[c]

    rows = [[("Category", H)] + [(c, H) for c in cols]]
    rows.append([(TARGET_LABEL + " (Target)", XlsxWriter.S_DEFAULT)]
                + [(val(target_sum, c), XlsxWriter.S_DEFAULT) for c in cols])
    rows.append([(BASE_LABEL + " (Baseline)", XlsxWriter.S_DEFAULT)]
                + [(val(base_sum, c), XlsxWriter.S_DEFAULT) for c in cols])
    # per-suite breakdown
    rows.append([("", XlsxWriter.S_DEFAULT)])
    rows.append([("Suite", H), ("Category", H)] + [(c, H) for c in cols])
    base_by = acc_suite_summary(base_vmap)
    target_by = acc_suite_summary(target_vmap)
    for suite in sorted(set(base_by) | set(target_by)):
        empty = {"Total": 0, "Passed": 0, "Passrate": 0.0, "Failed": 0, "Notrun": 0}
        tb, bb = target_by.get(suite, empty), base_by.get(suite, empty)
        rows.append([(suite, XlsxWriter.S_DEFAULT), ("Target", XlsxWriter.S_DEFAULT)]
                    + [(val(tb, c), XlsxWriter.S_DEFAULT) for c in cols])
        rows.append([("", XlsxWriter.S_DEFAULT), ("Baseline", XlsxWriter.S_DEFAULT)]
                    + [(val(bb, c), XlsxWriter.S_DEFAULT) for c in cols])
    return rows


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    if not os.path.isdir(BASE_DIR) or not os.path.isdir(TARGET_DIR):
        sys.exit(f"Missing baseline/target dir under {HERE}")

    print("Parsing unit tests...")
    ut_base, _ut_base_msg = parse_ut(BASE_DIR)
    ut_target, ut_target_msg = parse_ut(TARGET_DIR)
    ut_issues, ut_issue_lookup = [], {}

    print("Parsing accuracy...")
    acc_base = parse_accuracy(BASE_DIR, pt2e=False)
    acc_target = parse_accuracy(TARGET_DIR, pt2e=False)

    print("Parsing performance...")
    perf_base = parse_perf_e2e(BASE_DIR)
    perf_target = parse_perf_e2e(TARGET_DIR)

    # ---- comparisons ----
    ut_rows, ut_counts = compare_ut(ut_base, ut_target)
    # Collection-failure (crash) files: a whole test file failed to import in
    # target, so every case is blank; group the marker + all its cases together.
    _crashed = set()
    _markers = []
    for r in ut_rows:
        if r["target"] in ("failure", "error"):
            cf = _collection_file(r["key"][1], r["key"][2])
            if cf:
                _crashed.add(cf)
                _markers.append((r, cf))
    if _crashed:
        _seen = set()
        for r, cf in _markers:
            if r["cat"] != "crash":
                ut_counts[r["cat"]] -= 1
                ut_counts["crash"] += 1
            r["cat"] = "crash"
            r["key"] = (cf, r["key"][1], r["key"][2])
            ut_target_msg[r["key"]] = CRASH_MSG
            _seen.add(id(r))
        _ncase = 0
        for r in ut_rows:
            if id(r) in _seen:
                continue
            if r["key"][0] in _crashed and r["base"] is not None and r["target"] is None:
                if r["cat"] != "crash":
                    ut_counts[r["cat"]] -= 1
                    ut_counts["crash"] += 1
                r["cat"] = "crash"
                ut_target_msg.setdefault(r["key"], CRASH_MSG)
                _ncase += 1
        print(f"  UT collection-failure: {len(_crashed)} crashed files, "
              f"{_ncase} member cases")
    # Timeout / hang: a case pytest-xdist reports as `failed on setup with
    # "worker 'gwN' crashed while running ..."` is the root (hung / timed-out) or
    # collateral of a case that took down the xdist worker, not a code
    # regression. Group these into a dedicated `timeout` category so they are not
    # counted as regressions. Runs after crash grouping so import crashes stay
    # separate.
    _ntmo = 0
    for r in ut_rows:
        if r["cat"] == "crash":
            continue
        if r["target"] in ("failure", "error") and WORKER_CRASH_RE.search(
                ut_target_msg.get(r["key"], "")):
            if r["cat"] != "timeout":
                ut_counts[r["cat"]] -= 1
                ut_counts["timeout"] += 1
            r["cat"] = "timeout"
            ut_target_msg.setdefault(r["key"], TIMEOUT_MSG)
            _ntmo += 1
    if _ntmo:
        print(f"  UT timeout / worker-crash reclassified: {_ntmo}")
    _removed_keys = load_removed_list(REMOVED_LIST_FILE)
    if _removed_keys:
        _nrem = 0
        for r in ut_rows:
            if r["cat"] == "crash":
                continue
            if r["base"] == "passed" and r["target"] is None and r["key"] in _removed_keys:
                if r["cat"] == "regression":
                    ut_counts["regression"] -= 1
                    ut_counts["removed"] += 1
                r["cat"] = "removed"
                ut_target_msg.setdefault(r["key"], REMOVED_MSG)
                _nrem += 1
        print(f"  UT removed (deselected on target): {_nrem}")
    # Whatever remains as a baseline-pass -> target-blank is not a code
    # regression but a coverage gap (the target run never executed the case).
    _nnr = 0
    for r in ut_rows:
        if r["cat"] == "regression" and r["base"] == "passed" and r["target"] is None:
            ut_counts["regression"] -= 1
            ut_counts["not_run"] += 1
            r["cat"] = "not_run"
            ut_target_msg.setdefault(r["key"], NOT_RUN_MSG)
            _nnr += 1
    if _nnr:
        print(f"  UT not-run (loss-to-run) reclassified: {_nnr}")
    ut_base_sum = ut_status_summary(ut_base)
    ut_target_sum = ut_status_summary(ut_target)

    acc_bvm = acc_value_map(acc_base)
    acc_tvm = acc_value_map(acc_target)
    acc_issues, acc_issue_lookup = [], {}
    acc_target_msg = parse_acc_messages(TARGET_DIR, acc_tvm)
    acc_rows, acc_counts = compare_acc(acc_bvm, acc_tvm)
    acc_base_sum = acc_status_summary(acc_bvm)
    acc_target_sum = acc_status_summary(acc_tvm)

    perf_issues, perf_issue_lookup = [], {}
    perf_base_notrun = perf_notrun_status(BASE_DIR, perf_base)
    perf_target_notrun = perf_notrun_status(TARGET_DIR, perf_target)
    perf_target_msg = parse_perf_messages(TARGET_DIR, perf_target)
    perf_rows, perf_counts = compare_perf2(perf_base, perf_target)
    perf_base_sum = perf_ver_summary(perf_base, perf_base_notrun)
    perf_target_sum = perf_ver_summary(perf_target, perf_target_notrun)
    perf_eag_gm, perf_ind_gm = perf_geomeans(perf_rows)

    ts = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %Z")
    meta = {"base": BASE_LABEL, "target": TARGET_LABEL,
            "component": COMPONENT, "target_version": TARGET_VERSION,
            "base_version": BASE_VERSION, "ts": ts}

    # Sections in scope (tested) for the gate; default = all three.
    scope = {s.strip() for s in (os.environ.get("ACC_SECTIONS") or "ut,acc,perf").split(",") if s.strip()}
    gate_reasons = compute_gate(scope, ut_rows, ut_counts, acc_rows, acc_counts,
                                acc_target_sum["Passed"], perf_rows, perf_counts)
    meta["gate_reasons"] = gate_reasons

    ut_keys = ["Test File", "Test Class", "Test Name"]
    e2e_keys = ["Suite", "Dtype", "Mode", "Name", "Scenario"]

    # For readability the HTML shows only changed rows; totals stay in the cards.
    CHANGED = {"regression", "improvement", "new", "removed", "crash", "timeout", "not_run"}
    ch = lambda rows: [r for r in rows if r["cat"] in CHANGED]
    note = (" · HTML lists changed rows only (regression / improvement / new / removed); "
            "consistent pass/fail are in the cards above and the full XLSX.")

    # ---- summary cards: one per status category (they partition every row, so
    # Total == their sum) ----
    _ut_tone = {"regression": "reg", "improvement": "imp", "new": "new",
                "removed": "rem", "crash": "crash", "timeout": "timeout", "not_run": "notrun",
                "fail": "fail", "pass": "pass", "others": "other", "unknown": "other"}
    ut_cards = [("Total", len(ut_rows), "total", "all")]
    for cat in CATEGORY_ORDER:
        n = ut_counts.get(cat, 0)
        if cat == "unknown" and n == 0:
            continue
        ut_cards.append((CATEGORY_LABEL[cat], n, _ut_tone[cat], cat))

    def e2e_cards(counts):
        return [
            ("Total", sum(counts.values()), "total"),
            ("Regressions", counts.get("regression", 0), "reg"),
            ("Improvements", counts.get("improvement", 0), "imp"),
        ]

    # accuracy cards mirror the accuracy filters (pass-based comparison)
    acc_cards = [
        ("Total", len(acc_rows), "total", "all"),
        ("Regressions", acc_counts.get("regression", 0), "reg", "regression"),
        ("Improvements", acc_counts.get("improvement", 0), "imp", "improvement"),
        ("Both Fail", acc_counts.get("fail", 0), "fail", "both_fail"),
        ("No Change", acc_counts.get("pass", 0), "pass", "no_change"),
    ]

    # performance cards mirror the perf filters (eager/inductor)
    pc = perf_counts
    perf_cards = [
        ("Total", len(perf_rows), "total", "all"),
        ("New Fail", pc.get("new_fail", 0), "reg", "new_fail"),
        ("Drops", pc.get("drop", 0), "reg", "drop"),
        ("New Pass", pc.get("new_pass", 0), "imp", "new_pass"),
        ("Improves", pc.get("improve", 0), "imp", "improve"),
        ("BS Change", pc.get("bs_change", 0), "other", "bs_change"),
        ("Stable", pc.get("stable", 0), "pass", "stable"),
        ("Both Fail", pc.get("both_fail", 0), "fail", "both_fail"),
        ("Inductor geomean (B/T)", _gm(perf_ind_gm),
         "imp" if (perf_ind_gm or 1) > 1.05 else ("reg" if (perf_ind_gm or 1) < 0.95 else "pass")),
        ("Eager geomean (B/T)", _gm(perf_eag_gm),
         "imp" if (perf_eag_gm or 1) > 1.05 else ("reg" if (perf_eag_gm or 1) < 0.95 else "pass")),
    ]

    sections = [
        {"id": "ut", "title": "Unit Tests",
         "desc": "JUnit xml results. Key = (test file, test class, test name). "
                 "Regression = passed in baseline but failed/error in target. "
                 "All cases are loaded below; the table shows regressions by default.",
         "counts": ut_counts, "cards": ut_cards,
         "badge": ut_counts.get("regression", 0),
         "table": ut_summary_table_html(ut_base_sum, ut_target_sum) + ut_detail_html(ut_rows, ut_issues, ut_issue_lookup, ut_target_msg)},
        {"id": "acc", "title": "Accuracy",
         "desc": "E2E accuracy from *accuracy.csv. Key = (suite, dtype, mode, name, scenario). "
                 "Improvement = target passes but baseline does not (or null); regression is the opposite; "
                 "both pass = No Change; both not pass = Both Fail.",
         "counts": acc_counts, "cards": acc_cards, "card_onclick": "accCardFilter",
         "badge": acc_counts.get("regression", 0),
         "table": (acc_summary_table_html(acc_base_sum, acc_target_sum)
                   + acc_suite_breakdown_html(acc_bvm, acc_tvm)
                   + acc_detail_html(acc_rows, acc_issues, acc_issue_lookup, acc_target_msg))},
        {"id": "perf", "title": "Performance",
         "desc": "E2E performance from *performance.csv. Key = (suite, dtype, mode, name, scenario). "
                 "Inductor latency = abs_latency; Eager = abs_latency × speedup; ratio = baseline/target latency; "
                 f"improve > {PERF_IMP_THRESHOLD:.1f}, drop < {PERF_REG_THRESHOLD:.1f}. Showing Inductor Drop by default.",
         "counts": perf_counts, "cards": perf_cards,
         "card_onclick": "pfCardFilter", "card_default": "drop",
         "badge": pc.get("drop", 0) + pc.get("new_fail", 0),
         "table": (perf_summary_table_html(perf_base_sum, perf_target_sum, perf_eag_gm, perf_ind_gm)
                   + perf_suite_breakdown_html(perf_base, perf_target, perf_rows, perf_base_notrun, perf_target_notrun)
                   + perf2_detail_html(perf_rows, perf_issues, perf_issue_lookup, perf_target_msg))},
    ]

    def _pct(v):
        return f"{v * 100:.2f}%"

    meta["summary"] = [
        {"id": "ut", "title": "Unit Tests", "metrics": [
            ("Pass rate", _pct(ut_target_sum["Passrate"]),
             _rate_tone(ut_target_sum["Passrate"], ut_base_sum["Passrate"])),
            ("New failures", ut_counts.get("regression", 0), "reg"),
            ("New passes", ut_counts.get("improvement", 0), "imp"),
        ]},
        {"id": "acc", "title": "Accuracy", "metrics": [
            ("Pass rate", _pct(acc_target_sum["Passrate"]),
             _rate_tone(acc_target_sum["Passrate"], acc_base_sum["Passrate"])),
            ("New failures", acc_counts.get("regression", 0), "reg"),
            ("New passes", acc_counts.get("improvement", 0), "imp"),
        ]},
        {"id": "perf", "title": "Performance", "metrics": [
            ("Inductor geomean (B/T)", _gm(perf_ind_gm), _gm_tone(perf_ind_gm)),
            ("Eager geomean (B/T)", _gm(perf_eag_gm), _gm_tone(perf_eag_gm)),
            ("New failures", perf_counts.get("new_fail", 0), "reg"),
            ("Drops", perf_counts.get("drop", 0), "reg"),
            ("New passes", perf_counts.get("new_pass", 0), "imp"),
            ("Improves", perf_counts.get("improve", 0), "imp"),
        ]},
    ]
    meta["summary_caption"] = "Accuracy & Performance cover all tested models."

    html_out = build_html(sections, meta)
    _slug = f"{COMPONENT}_{TARGET_VERSION}_vs_{BASE_VERSION}_report".replace(" ", "_").replace("/", "-")
    os.makedirs(OUT_DIR, exist_ok=True)
    html_path = os.path.join(OUT_DIR, _slug + ".html")
    with open(html_path, "w") as fh:
        fh.write(html_out)
    print(f"HTML  -> {html_path}")

    # ---- xlsx ----
    xw = XlsxWriter()
    xw.add_sheet("Summary", summary_sheet_rows(sections))
    xw.add_sheet("UT Status Summary", ut_summary_sheet_rows(ut_base_sum, ut_target_sum))
    xw.add_sheet("Unit Tests", xlsx_ut_rows(ut_keys, ut_rows, BASE_LABEL, TARGET_LABEL, ut_issues, ut_issue_lookup, ut_target_msg))
    xw.add_sheet("Accuracy Summary", acc_summary_sheet_rows(acc_base_sum, acc_target_sum, acc_bvm, acc_tvm))
    xw.add_sheet("Accuracy", xlsx_acc_rows(acc_rows, BASE_LABEL, TARGET_LABEL, acc_issues, acc_issue_lookup, acc_target_msg))
    xw.add_sheet("Performance Summary", perf_summary_sheet_rows(perf_base_sum, perf_target_sum, perf_base, perf_target, perf_rows, perf_base_notrun, perf_target_notrun, perf_eag_gm, perf_ind_gm))
    xw.add_sheet("Performance", xlsx_perf2_rows(perf_rows, perf_issues, perf_issue_lookup, perf_target_msg))
    xlsx_path = os.path.join(OUT_DIR, _slug + ".xlsx")
    xw.save(xlsx_path)
    print(f"XLSX  -> {xlsx_path}")

    # ---- console summary ----
    print("\n=== Summary (regression / improvement / new / removed / total) ===")
    for s in sections:
        c = s["counts"]
        print(f"  {s['title']:<18} reg={c.get('regression',0):<4} "
              f"imp={c.get('improvement',0):<4} new={c.get('new',0):<4} "
              f"rem={c.get('removed',0):<4} total={sum(c.values())}")

    # ---- brief GitHub step summary ----
    gh_sum = os.environ.get("GITHUB_STEP_SUMMARY")
    if gh_sum:
        lines = ["## Acceptance report", "",
                 f"**Component:** `{COMPONENT}` &nbsp;&nbsp; **Target:** `{TARGET_VERSION}` &nbsp;&nbsp; **Baseline:** `{BASE_VERSION}`", "",
                 "| Section | Regression | Improvement | New | Removed | Total |",
                 "| --- | --- | --- | --- | --- | --- |"]
        for s in sections:
            c = s["counts"]
            lines.append(f"| {s['title']} | {c.get('regression', 0)} | "
                         f"{c.get('improvement', 0)} | {c.get('new', 0)} | "
                         f"{c.get('removed', 0)} | {sum(c.values())} |")
        lines += ["", f"Full report: `{os.path.basename(html_path)}` (+ `.xlsx`) in artifacts."]
        if gate_reasons:
            lines += ["", "### ❌ Acceptance gate FAILED"] + [f"- {r}" for r in gate_reasons]
        else:
            lines += ["", "### ✅ Acceptance gate PASSED"]
        with open(gh_sum, "a") as fh:
            fh.write("\n".join(lines) + "\n")

    # ---- acceptance gate: fail on any reason (see compute_gate) ----
    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a") as fh:
            fh.write(f"gate_failed={'true' if gate_reasons else 'false'}\n")
            fh.write("gate_reasons=" + " | ".join(gate_reasons) + "\n")
    if gate_reasons:
        sys.exit("Acceptance gate FAILED:\n  - " + "\n  - ".join(gate_reasons))


if __name__ == "__main__":
    main()
