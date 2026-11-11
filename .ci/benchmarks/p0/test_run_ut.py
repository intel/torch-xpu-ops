# Copyright 2020-2026 Intel Corporation
# Licensed under the Apache License, Version 2.0

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml


_P0_DIR = Path(__file__).resolve().parent
_SELECTED_FILES = (
    "test/inductor/test_torchinductor.py",
    "test/xpu/test_ops_xpu.py",
    "test/xpu/distributed/test_c10d_xccl.py",
)
_PARAMETER = "value with spaces [$HOME]"


@pytest.fixture
def runner(tmp_path):
    script_dir = tmp_path / "scripts"
    script_dir.mkdir()
    shutil.copyfile(_P0_DIR / "run_ut.sh", script_dir / "run_ut.sh")
    root = tmp_path / "pytorch"
    nested = root / "third_party/torch-xpu-ops"
    nested.mkdir(parents=True)
    for index, name in enumerate(_SELECTED_FILES):
        path = (root if index == 0 else nested) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "import pytest\n"
            f"@pytest.mark.parametrize('value', [1], ids=[{_PARAMETER!r}])\n"
            "def test_skip(value):\n"
            "    assert False, 'configured deselection did not match'\n"
            "def test_keep():\n"
            "    pass\n"
        )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("python", "pytest", "xpu-smi"):
        path = bin_dir / name
        if name == "python":
            path.symlink_to(sys.executable)
        elif name == "xpu-smi":
            path.write_text("#!/bin/sh\nexit 0\n")
        else:
            path.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                "with open(os.environ['INVOCATIONS'], 'a') as f:\n"
                "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
                f"if sys.argv[1] in {_SELECTED_FILES!r}:\n"
                "    os.execv(os.environ['REAL_PYTEST'], ['pytest', *sys.argv[1:]])\n"
            )
        if name != "python":
            path.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "PYTORCH_ROOT_DIR": str(root),
        "XML_OUTPUT_DIR": str(tmp_path / "reports"),
        "INVOCATIONS": str(tmp_path / "invocations.jsonl"),
        "REAL_PYTEST": shutil.which("pytest"),
        "PYTEST_ADDOPTS": "",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    }
    return script_dir, root, env


def test_deselects_match_upstream_and_nested_tests(runner):
    script_dir, root, env = runner
    nodes = [f"{name}::test_skip[{_PARAMETER}]" for name in _SELECTED_FILES]
    (script_dir / "ut_skips.yaml").write_text(
        yaml.safe_dump({"pytest_deselects": nodes})
    )
    result = subprocess.run(
        ["bash", str(script_dir / "run_ut.sh"), "inductor", "non-inductor", "distributed"],
        cwd=script_dir.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("1 passed, 1 deselected") == 3, result.stdout
    calls = [json.loads(line) for line in Path(env["INVOCATIONS"]).read_text().splitlines()]
    for index, name in enumerate(_SELECTED_FILES):
        call = next(args for args in calls if args[0] == name)
        prefix = "" if index == 0 else "third_party/torch-xpu-ops/"
        assert f"--deselect={prefix}{nodes[index]}" in call
        assert f"--rootdir={root}" in call


@pytest.mark.parametrize(
    "content",
    [
        None,
        "",
        "pytest_deselects: [",
        "other_key: []",
        "pytest_deselects: null",
        "pytest_deselects: a-string",
        "pytest_deselects: [123]",
        "pytest_deselects: ['']",
        "pytest_deselects: ['   ']",
        'pytest_deselects: ["test/a.py::test_a\\ntest/b.py::test_b"]',
        'pytest_deselects: ["test/a.py::test_a\\0"]',
    ],
)
def test_invalid_config_stops_runner(runner, content):
    script_dir, _, env = runner
    if content is not None:
        (script_dir / "ut_skips.yaml").write_text(content)
    result = subprocess.run(
        ["bash", str(script_dir / "run_ut.sh"), "inductor"],
        cwd=script_dir.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert "ERROR: Could not load P0 skip config:" in result.stderr
    assert not Path(env["INVOCATIONS"]).exists()


def test_empty_list_loads_as_empty_array(runner):
    script_dir, _, env = runner
    (script_dir / "ut_skips.yaml").write_text("pytest_deselects: []\n")
    result = subprocess.run(
        ["bash", str(script_dir / "run_ut.sh"), "--help"],
        cwd=script_dir.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "ut_skips.yaml" in result.stdout
    assert "+ PYTEST_DESELECTS=()" in result.stderr
    assert "+ mapfile" not in result.stderr


def test_missing_yaml_dependency_stops_runner(runner):
    script_dir, _, env = runner
    (script_dir / "ut_skips.yaml").write_text("pytest_deselects: []\n")
    python = Path(env["PATH"].split(os.pathsep)[0]) / "python"
    python.unlink()
    python.write_text(f'#!/bin/sh\nexec "{sys.executable}" -S "$@"\n')
    python.chmod(0o755)
    result = subprocess.run(
        ["bash", str(script_dir / "run_ut.sh"), "inductor"],
        cwd=script_dir.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert "No module named 'yaml'" in result.stderr
    assert "ERROR: Could not load P0 skip config:" in result.stderr
    assert not Path(env["INVOCATIONS"]).exists()
