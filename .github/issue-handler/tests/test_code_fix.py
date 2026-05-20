"""Tests for code_fix verification helpers."""
import pytest
from issue_handler.fixing_steps.code_fix import _get_test_command


class TestGetTestCommand:
    def test_reproducer_bash_block(self):
        body = (
            "## Reproducer\n"
            "```bash\npytest -xvs test_ops.py -k test_fake_autocast\n```\n"
            "## Other\nstuff"
        )
        assert _get_test_command(body) == "pytest -xvs test_ops.py -k test_fake_autocast"

    def test_reproducer_plain_command(self):
        body = (
            "## Reproducer\n"
            "python test/test_ops.py TestFakeTensor\n"
            "## Other\n"
        )
        assert _get_test_command(body) == "python test/test_ops.py TestFakeTensor"

    def test_failed_tests_section(self):
        body = (
            "## Failed Tests\n"
            "- `test_ops.py::TestFakeTensorXPU::test_fake_autocast_xpu_float32`\n"
            "- `test_ops.py::TestFakeTensorXPU::test_fake_autocast_pinverse_xpu_float32`\n"
            "## Other\n"
        )
        cmd = _get_test_command(body)
        assert cmd is not None
        assert "pytest" in cmd
        assert "test_ops.py::TestFakeTensorXPU::test_fake_autocast_xpu_float32" in cmd

    def test_no_reproducer_no_tests(self):
        body = (
            "## Summary\nSome issue\n"
            "## Reproducer\n```bash\n\n```\n"
        )
        assert _get_test_command(body) is None

    def test_pending_reproducer_skipped(self):
        body = (
            "## Reproducer\n_Pending triage_\n"
            "## Failed Tests\n_None identified_\n"
        )
        assert _get_test_command(body) is None

    def test_reproducer_with_pytest_k(self):
        body = (
            "## Reproducer\n"
            "```bash\npytest -v test_ops.py -k <case>\n```\n"
        )
        cmd = _get_test_command(body)
        assert cmd == "pytest -v test_ops.py -k <case>"

    def test_multiline_reproducer_strips_cd_all_lines(self):
        body = (
            "## Reproducer\n"
            "```bash\n"
            "cd <pytorch> && pytest -v test/a.py -k test1\n"
            "cd <pytorch> && pytest -v test/b.py -k test2\n"
            "cd /home/user/pytorch && pytest -v test/c.py -k test3\n"
            "```\n"
        )
        cmd = _get_test_command(body)
        assert cmd is not None
        lines = cmd.splitlines()
        assert len(lines) == 3
        assert lines[0] == "pytest -v test/a.py -k test1"
        assert lines[1] == "pytest -v test/b.py -k test2"
        assert lines[2] == "pytest -v test/c.py -k test3"
