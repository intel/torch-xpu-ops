#!/usr/bin/env python3
"""The only step of the E2E accuracy auto-issue pipeline that writes to GitHub.

file --evidence E --drafts D --out C [--dry-run]
    in:  E = evidence.json after `merge`; D = drafts.json from the e2e-accuracy-issue-authoring skill
    out: C = created.json {created, updated, rejected, not_filed, deep_queue}
         <dir of C>/bodies/<draft id>.md, the rendered issues
         GITHUB_OUTPUT deep_skus, deep_models
    Rows an open auto issue already lists are updated in that issue instead of filed again.
    In a dry run deep_queue still holds the drafted issues, with `issue` null.
post --triage F [F ...] [--dry-run]
    in:  triage.json files from `e2e_accuracy_triage.py`
    does: comment the conclusion on each issue; if verify passed, put the reproducer between the
          e2e-accuracy-minimal-repro markers; add the owner's module label

--dry-run reads GitHub but writes nothing; it prints each write instead. The issue sections match
.github/ISSUE_TEMPLATE/e2e-accuracy.yml.
"""

import argparse
import json
import os
import re
import tempfile
from pathlib import Path

from e2e_accuracy_collect_evidence import AUTO_MARKER, REPO, SKUS, SUITES, gh
from get_issue import fast_extract_table_rows

SERVER = os.environ.get("GITHUB_SERVER_URL") or "https://github.com"
IMAGE = "intelgpu/ubuntu-24.04-lts2:2523.40"
NIGHTLY = "https://download.pytorch.org/whl/nightly/xpu"
REPRO_BEGIN, REPRO_END = "<!-- e2e-accuracy-minimal-repro:begin -->", "<!-- e2e-accuracy-minimal-repro:end -->"
MAX_ISSUES = int(os.environ.get("E2E_ACCURACY_AUTO_ISSUE_MAX_PER_RUN", 10))
MAX_DEEP = int(os.environ.get("E2E_ACCURACY_TRIAGE_MAX_PER_RUN", 3))
DEEP_PRIORITY = {"crash": 0, "accuracy": 0, "eager_nondeterministic": 0, "eager_fail": 1, "oom": 2, "timeout": 2}
OWNER_LABEL = {"torch-xpu-ops": "module: ops", "oneDNN": "module: ops", "inductor": "module: inductor",
               "triton-xpu": "module: inductor", "upstream-pytorch": "module: core", "model/benchmark": "module: infra"}


def write_gh(dry_run: bool, *args: str) -> str:
    if dry_run:
        print("[dry-run] gh", *args)
        return ""
    return gh(*args)


def body_file(text: str) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(text)
    return f.name


def table_row(r: dict) -> str:
    """A 6-cell row: get_issue.py drops empty cells, and check_expected.py mutes row[4] only with >= 6."""
    return f"| {r['suite']} | {r['dtype']} | {r['mode']} | {r['model']} | {r['value']} | {r['message'] or '-'} |"


def model_rows(body: str, keys: set) -> list[list[str]]:
    """Rows of `body` the gate would read as muting one of `keys`."""
    return [r for r in fast_extract_table_rows(body) if ",".join(r[:4]) in keys]


def check_draft(draft: dict, rows: dict, group_of: dict) -> str:
    """Why the draft must not be filed, or ''."""
    keys = draft.get("rows") or []
    if not keys or any(k not in rows for k in keys):
        return "a row is not in evidence.json"
    groups = [group_of[k] for k in keys]
    if len({g["category"] for g in groups}) > 1:
        return "rows of different categories"
    if any(g["confirm_skip"].startswith("deferred") for g in groups):
        return "a group is still waiting for confirmation"
    if draft.get("error_row") not in keys:
        return "error_row is not one of rows"
    if not set(draft.get("message_lines", [])) <= set(rows[draft["error_row"]]["log_excerpt"]):
        return "message_lines are not verbatim from the error_row log"
    return ""


def reproducer(r: dict, run: dict, public: dict) -> str:
    v = run["env"].get("versions", {})
    if public:
        torch = (f"uv pip install torch=={public['torch']} torchvision=={public['torchvision']} "
                 f"torchaudio=={public['torchaudio']} --pre --index-url {NIGHTLY}")
    else:
        torch = (f"# Not reproduced with (or not tried on) the public nightly: install the CI wheel of\n"
                 f"# {SERVER}/{REPO}/actions/runs/{run['run_id']}, or build pytorch {v.get('torch', '<commit>')}\n"
                 f"# with third_party/xpu.txt set to {v.get('torch_xpu_ops', '<commit>')}")
    torchbench = ("git clone https://github.com/pytorch/benchmark ../benchmark && (cd ../benchmark && git checkout "
                  "$(cat ../pytorch/.ci/docker/ci_commit_pins/torchbench.txt) && "
                  f"python install.py --continue_on_fail {r['model']})\n"
                  "export PYTHONPATH=$(realpath ../benchmark):$PYTHONPATH\n" if r["suite"] == "torchbench" else "")
    return f"""```bash
docker run -it --device=/dev/mem --device=/dev/dri --group-add video --privileged --shm-size=8g {IMAGE} bash
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/local/bin" sh
uv venv myvenv --python 3.12 && source myvenv/bin/activate
uv pip install pip wheel "setuptools<81"
{torch}
pytorch_commit="$(python -c 'import torch; print(torch.version.git_version)')"
git clone https://github.com/pytorch/pytorch && cd pytorch && git checkout ${{pytorch_commit}}
# Same as CI: torch-xpu-ops' benchmark configs override upstream's
git clone https://github.com/intel/torch-xpu-ops /tmp/xpu-ops && git -C /tmp/xpu-ops checkout {v.get('torch_xpu_ops', 'main')}
rsync -a /tmp/xpu-ops/.ci/benchmarks/ benchmarks/dynamo/
uv pip install pandas psutil scipy requests accelerate
uv pip install -r .ci/docker/ci_commit_pins/huggingface-requirements.txt
uv pip install git+https://github.com/huggingface/pytorch-image-models@$(cat .ci/docker/ci_commit_pins/timm.txt)
{torchbench}uv pip install "numpy<2"
{r['cmd']}
```"""


def render(draft: dict, rows: list[dict], groups: list[dict], ev: dict, related: set) -> tuple[str, str, bool]:
    """(title, body, random) of one issue."""
    run, cat, first = ev["run"], groups[0]["category"], next(r for r in rows if r["key"] == draft["error_row"])
    confirmed = [v for g in groups for v in g.get("confirm_result", {}).get("reps", {}).values()]
    public = next((v["public_versions"] for v in confirmed if v.get("public_wheel_reproduces")), {})
    random = any(v["conclusion"] == "random" for v in confirmed)
    sku_tag = f"[PVC {first['sku'][4:]}]" if any(v["conclusion"] == "sku_specific" for v in confirmed) else ""
    models = sorted({r["model"] for r in rows})
    title = (f"[E2E][{cat}]{sku_tag} {rows[0]['suite']} {', '.join(models[:3])}"
             f"{f' +{len(models) - 3}' if len(models) > 3 else ''} {'/'.join(sorted({r['dtype'] for r in rows}))} "
             f"{'/'.join(sorted({r['mode'] for r in rows}))} {' '.join(draft['title_text'].split())}")
    if len(title) > 140:  # cut at a word, never inside one
        title = title[:141].rsplit(" ", 1)[0]

    notes = [f"Related: {', '.join(f'#{n}' for n in sorted(related))}" if related else "",
             draft.get("sku_note", ""),
             "**Suggested reference update**: these rows pass on the other SKU while the reference CSV expects "
             "a failure; update the reference to `pass` by PR." if draft.get("suggest_reference_update") else "",
             f"Not confirmed across SKUs: {groups[0]['confirm_skip']}" if groups[0]["confirm_skip"] else ""]
    details = []
    for r in [first] + [r for r in rows if r["key"] in groups[0]["reps"] and r is not first]:
        log = "\n".join([f"$ {r['cmd']}", *r["log_excerpt"], f"Device: {r['device']}"])
        details.append(f"### {r['sku'].upper().replace('-', ' ')} (run {run['run_id']}, card {r['card']})\n"
                       f"```text\n{log}\n```")
    for v in confirmed:
        runs = "\n".join(f"{sku}: {', '.join(vals) or 'not run'}" for sku, vals in v.get("runs", {}).items())
        details.append(f"### Cross-SKU confirmation: {v['conclusion']}\n```text\n{runs}\n"
                       f"public nightly wheel: {v.get('public_wheel') or 'not run'}\n```")
    env = run["env"]
    versions = [f"{k}: {v}" for k, v in {**env.get("versions", {}), **env.get("packages", {})}.items()]
    versions += [f"image: {IMAGE}", f"devices: {', '.join(sorted({r['device'] for r in rows}))}"]
    marker = (f"{AUTO_MARKER}: run={run['run_id']} digest={ev['digest']} category={cat} "
              f"first_seen={first['first_seen']} last_good={first['last_good']} -->")
    nl = "\n"
    # Free text and logs lose their `|` so no line outside Affected Cases can parse as a table row.
    body = f"""### 🐛 Describe the bug

## Summary
{draft['summary'].replace('|', '/')}

{nl.join(n.replace('|', '/') for n in notes if n)}

## Affected Cases
| Suite | Dtype | Mode | Name | Target | Message |
|---|---|---|---|---|---|
{nl.join(table_row(r) for r in rows)}

## Reproducer
{reproducer(first, run, public)}

## Details
{nl.join(details).replace('|', '¦')}

{marker}

{REPRO_BEGIN}
## Minimal reproducer
pending triage
{REPRO_END}

### Versions
```text
{nl.join(versions).replace('|', '¦')}
```
<details><summary>collect_env</summary>

```text
{env.get('collect_env', '').replace('|', '¦')}
```
</details>
"""
    return title.replace("|", "/"), body, random


def file_issues(ev: dict, drafts: dict, bodies: Path, dry_run: bool) -> dict:
    rows = {r["key"]: r for g in ev["groups"] for r in g["rows"]}
    group_of = {r["key"]: g for g in ev["groups"] for r in g["rows"]}
    auto, manual = {}, {}
    for i in ev["issues"]:
        for t in i["table_rows"]:
            key = ",".join(t[:4])
            if i["auto"]:
                auto[key] = i["issue_number"]
            else:
                manual.setdefault(key, set()).add(i["issue_number"])
    labels_in_repo = set(gh("label", "list", "--repo", REPO, "--limit", "500", "--json", "name", "-q", ".[].name")
                         .splitlines())
    out = {"run_id": ev["run"]["run_id"], "dry_run": dry_run, "created": [], "updated": [], "rejected": [],
           "not_filed": [], "deep_queue": []}
    if drafts.get("digest") != ev["digest"]:
        out["rejected"].append({"id": "*", "reason": "drafts.json missing or its digest does not match evidence.json"})
        return out
    bodies.mkdir(parents=True, exist_ok=True)
    for d in drafts.get("drafts", []):
        if not d.get("file"):
            out["not_filed"].append({"id": d.get("id"), "reason": d.get("reason", "")})
            continue
        reason = check_draft(d, rows, group_of) or ("over E2E_ACCURACY_AUTO_ISSUE_MAX_PER_RUN"
                                                     if len(out["created"]) >= MAX_ISSUES else "")
        if reason:
            out["rejected"].append({"id": d.get("id"), "reason": reason})
            continue

        for key in [k for k in d["rows"] if k in auto]:  # value changed under an open auto issue: update it
            r, number = rows[key], str(auto[key])
            old = gh("issue", "view", number, "--repo", REPO, "--json", "body", "-q", ".body")
            line = r"^\|\s*" + r"\s*\|\s*".join(map(re.escape, key.split(","))) + r"\s*\|.*$"
            new = re.sub(line, lambda _: table_row(r), old, flags=re.M)
            write_gh(dry_run, "issue", "edit", number, "--repo", REPO, "--body-file", body_file(new))
            write_gh(dry_run, "issue", "comment", number, "--repo", REPO, "--body",
                     f"Run {ev['run']['run_id']}: `{key}` is now `{r['value']}` (`{r['message']}`); table updated.")
            out["updated"].append({"issue": int(number), "row": key, "value": r["value"]})

        new_rows = [rows[k] for k in d["rows"] if k not in auto]
        if not new_rows:
            continue
        if d["error_row"] not in [r["key"] for r in new_rows]:
            d["error_row"] = new_rows[0]["key"]
        groups = list({group_of[r["key"]]["id"]: group_of[r["key"]] for r in new_rows}.values())
        title, body, random = render(d, new_rows, groups, ev, {n for r in new_rows for n in manual.get(r["key"], ())})
        (bodies / f"{d['id']}.md").write_text(f"# {title}\n\n{body}")
        hits = model_rows(body, set(rows))
        if [h[:5] for h in hits] != [[r["suite"], r["dtype"], r["mode"], r["model"], r["value"]] for r in new_rows] \
                or any(len(h) < 6 for h in hits):
            out["rejected"].append({"id": d["id"], "reason": "rendered body fails the table self-check"})
            continue

        cat = groups[0]["category"]
        labels = {"test: e2e", "Accuracy", "hw: PVC", "ai_generated", *(f"dtype: {r['dtype']}" for r in new_rows),
                  *(r["mode"] for r in new_rows)}
        labels |= {"regression"} if any(r["gate"] != "new_pass" for r in new_rows) else set()
        labels |= ({"random"} if random else set()) | ({"module: infra"} if cat == "env" else set())
        labels = sorted(labels & labels_in_repo)
        url = write_gh(dry_run, "issue", "create", "--repo", REPO, "--title", title, "--body-file", body_file(body),
                       *[a for label in labels for a in ("--label", label)])
        number = int(url.strip().rsplit("/", 1)[-1]) if url else None
        out["created"].append({"id": d["id"], "issue": number, "title": title, "labels": labels,
                               "rows": [r["key"] for r in new_rows]})
        if (number or dry_run) and cat in DEEP_PRIORITY and not random:
            r = rows[d["error_row"]]
            out["deep_queue"].append({
                "issue": number, "category": cat, "priority": DEEP_PRIORITY[cat],
                "sku": r["sku"] if r["sku"] in SKUS.values() else "pvc",
                **{k: r[k] for k in ("key", "value", "batch_size", "message", "log_excerpt")}})
    # Issue numbers exist only now. A dry run has none, so it names the drafts instead.
    filed = {c["id"]: c["issue"] for c in out["created"]}
    for d in drafts.get("drafts", []):
        links = [f"#{filed[x]}" if filed[x] else f"<draft {x}>" for x in d.get("related") or [] if x in filed]
        if d.get("id") in filed and links:
            write_gh(dry_run, "issue", "comment", str(filed[d["id"]]), "--repo", REPO, "--body",
                     f"Related: {', '.join(links)}, same cause filed separately (e.g. its value changes category).")
    out["deep_queue"] = sorted(out["deep_queue"], key=lambda i: i["priority"])[:MAX_DEEP]
    return out


def post_triage(path: Path, dry_run: bool) -> None:
    t = json.loads(path.read_text())
    item, res, ver = t["item"], t.get("result") or {}, t.get("verify") or {}
    number, owner, confidence = str(item["issue"]), res.get("owner", "unknown"), res.get("confidence", "low")
    if owner == "torch-xpu-ops" and confidence == "high" and ver.get("passed"):
        next_step = "Comment `@torchxpubot fix` to hand this to the fix agent."
    else:
        next_step = f"Suggested destination: {owner}. The issue stays here; filing upstream is a human decision."
    facts = "\n".join(f"- {k}: {res[k]}" for k in ("ops", "dtypes", "shapes", "layouts", "slowest_stage", "memory")
                      if res.get(k))
    experiments = "\n".join(f"{e['name']}: {e.get('result', '')} ({e.get('elapsed_s', '-')}s)"
                            for e in t.get("experiments", []))
    verified = "passed" if ver.get("passed") else f"not passed: {ver.get('reason', 'not run')}"
    write_gh(dry_run, "issue", "comment", number, "--repo", REPO, "--body", f"""## E2E triage of `{item['key']}`
**Category**: {res.get('category', item['category'])} · **Owner**: {owner} ({confidence} confidence)

{res.get('conclusion', 'No conclusion was written.')}

{facts}

```text
{experiments.replace('|', '¦')}
```
Reproducer verify: {verified}. Artifacts: {SERVER}/{REPO}/actions/runs/{os.environ.get('GITHUB_RUN_ID', '')}

{next_step}""")
    if ver.get("passed"):
        script = (path.parent / res["repro_script"]).read_text()
        section = f"{REPRO_BEGIN}\n## Minimal reproducer\n```python\n{script}\n```\n{REPRO_END}"
        if not item["issue"]:  # a dry run's draft: no issue body to put it in
            print(f"[dry-run] Minimal reproducer section for `{item['key']}`:\n{section}")
        else:
            body = gh("issue", "view", number, "--repo", REPO, "--json", "body", "-q", ".body")
            new = re.sub(re.escape(REPRO_BEGIN) + ".*?" + re.escape(REPRO_END), lambda _: section, body, flags=re.S)
            suite_rows = lambda text: [r for r in fast_extract_table_rows(text) if r[0] in SUITES]  # noqa: E731
            if suite_rows(new) == suite_rows(body):  # the script must not add or change a muting row
                write_gh(dry_run, "issue", "edit", number, "--repo", REPO, "--body-file", body_file(new))
    if owner in OWNER_LABEL and confidence != "low":
        write_gh(dry_run, "issue", "edit", number, "--repo", REPO, "--add-label", OWNER_LABEL[owner])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("file")
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--drafts", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("post")
    p.add_argument("--triage", type=Path, nargs="*", default=[])
    p.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.cmd == "post":
        for path in args.triage:
            post_triage(path, args.dry_run)
        return
    drafts = json.loads(args.drafts.read_text()) if args.drafts.exists() else {}
    created = file_issues(json.loads(args.evidence.read_text()), drafts, args.out.parent / "bodies", args.dry_run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(created, indent=1))
    queue = created["deep_queue"]
    with open(os.environ.get("GITHUB_OUTPUT", os.devnull), "a") as f:
        f.write(f"deep_skus={json.dumps(sorted({i['sku'] for i in queue}))}\n"
                f"deep_models={','.join(sorted({i['key'].split(',')[3] for i in queue}))}\n")
    print(json.dumps({k: v for k, v in created.items() if k != "deep_queue"}, indent=1))


if __name__ == "__main__":
    main()
