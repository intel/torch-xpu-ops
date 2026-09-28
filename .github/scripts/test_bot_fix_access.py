#!/usr/bin/env python3
# Copyright 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0

"""Tests for the `fix` permission gate in bot.yml.

Usage:
    python -m pytest .github/scripts/test_bot_fix_access.py

bot_fix_access.js has its own self-test for the grant arithmetic. What only
this file can reach is the wiring around it: the gate is an inline
github-script block, so it is lifted out of the workflow and run under node
with `github`, `core` and `context` stubbed. A typo in that if-chain either
opens `fix` to everyone or closes it to everyone, and neither shows up until
someone comments on a live issue.

`require` is given the same resolution rule as github-script's wrapRequire
(relative paths resolve from the working directory), so the module is loaded
the way the workflow loads it.
"""

import json
import os
import re
import subprocess
import tempfile

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORKFLOW = os.path.join(REPO, ".github", "workflows", "bot.yml")

HARNESS = """
const path = require('path');
const wrapRequire = (id) => require(id.startsWith('.') ? path.resolve(id) : id);
const spec = JSON.parse(process.argv[2]);
const out = {};
const core = {
  setOutput: (k, v) => { out[k] = v; },
  setFailed: (m) => { out.failed = m; },
};
const github = {paginate: async () => spec.grants, rest: {issues: {listComments: () => {}}}};
const context = {
  repo: {owner: 'intel', repo: 'torch-xpu-ops'},
  payload: {
    comment: {id: 1, body: spec.body, user: {login: spec.login}, author_association: spec.assoc},
    issue: Object.assign({number: spec.issue, state: 'open'}, spec.isPR ? {pull_request: {}} : {}),
  },
};
(async () => {
  await new Function('core', 'github', 'context', 'require',
    'return (async () => {' + spec.script + '})()')(core, github, context, wrapRequire);
  console.log(JSON.stringify(out));
})();
"""


def gate_script():
    """The Parse Command script, with the access issue number substituted in."""
    text = open(WORKFLOW).read()
    issue = re.search(r"^  FIX_ACCESS_ISSUE: (\d+)$", text, re.M)
    assert issue, "bot.yml lost its FIX_ACCESS_ISSUE"
    step = re.search(
        r"\n      - name: Parse Command\n.*?\n          script: \|\n(.*?)\n(?=      - name: |\n  \w)",
        text,
        re.S,
    )
    assert step, "bot.yml lost its Parse Command step"
    body = "\n".join(line[12:] for line in step.group(1).split("\n"))
    return body.replace("${{ env.FIX_ACCESS_ISSUE }}", issue.group(1)), int(issue.group(1))


SCRIPT, ACCESS_ISSUE = gate_script()
APPROVER = "EikanWang"
GRANT = [{"user": {"login": APPROVER}, "body": "@torchxpubot allow @an-applicant"}]


def run(body, login, assoc="MEMBER", issue=5272, is_pr=False, grants=()):
    spec = {
        "script": SCRIPT,
        "body": body,
        "login": login,
        "assoc": assoc,
        "issue": issue,
        "isPR": is_pr,
        "grants": list(grants),
    }
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(HARNESS)
        harness = f.name
    try:
        p = subprocess.run(
            ["node", harness, json.dumps(spec)],
            cwd=REPO, capture_output=True, text=True, check=True,
        )
    finally:
        os.unlink(harness)
    return json.loads(p.stdout)


@pytest.mark.parametrize(
    ("body", "login", "assoc", "issue", "grants", "command", "authorized"),
    [
        # `fix` answers to the list, and to nothing else. An org member who is
        # not on it used to get in on MEMBER alone.
        ("@torchxpubot fix", "guangyey", "MEMBER", 5272, (), "fix", "true"),
        ("@torchxpubot fix", "an-applicant", "NONE", 5272, GRANT, "fix", "true"),
        ("@torchxpubot fix", "an-applicant", "NONE", 5272, (), "fix", "false"),
        ("@torchxpubot fix", "not-on-the-list", "MEMBER", 5272, (), "fix", "false"),
        # The queue's own account, whose author_association the REST API and
        # the webhook disagree about, so the gate must not read it.
        ("@torchxpubot fix", "torchxpubot", "CONTRIBUTOR", 5272, (), "fix", "true"),
        # Granting is for approvers, on the access issue.
        ("@torchxpubot allow @x", APPROVER, "MEMBER", ACCESS_ISSUE, (), "allow", "true"),
        ("@torchxpubot deny @x", "tye1", "MEMBER", ACCESS_ISSUE, (), "deny", "true"),
        ("@torchxpubot allow @x", "guangyey", "MEMBER", ACCESS_ISSUE, (), "allow", "false"),
        # Anywhere else it is refused outright rather than silently ignored.
        ("@torchxpubot allow @x", APPROVER, "MEMBER", 5272, (), "invalid", "false"),
    ],
)
def test_gate(body, login, assoc, issue, grants, command, authorized):
    out = run(body, login, assoc=assoc, issue=issue, grants=grants)
    assert out["command"] == command
    assert out["authorized"] == authorized


@pytest.mark.parametrize(
    ("body", "assoc", "authorized"),
    [
        ("@torchxpubot merge", "COLLABORATOR", "true"),
        ("@torchxpubot merge -f", "COLLABORATOR", "false"),
        ("@torchxpubot merge -f", "MEMBER", "true"),
        ("@torchxpubot review", "NONE", "false"),
    ],
)
def test_other_commands_unchanged(body, assoc, authorized):
    """The list is for `fix`; every other command still goes by association."""
    out = run(body, "not-on-the-list", assoc=assoc, issue=4000, is_pr=True)
    assert out["authorized"] == authorized


def test_help_is_open_to_anyone():
    assert run("@torchxpubot help", "a-stranger", assoc="NONE", issue=4000, is_pr=True)[
        "authorized"
    ] == "true"


def test_access_module_self_test():
    subprocess.run(
        ["node", ".github/scripts/bot_fix_access.js", "--self-test"],
        cwd=REPO, check=True, capture_output=True,
    )
