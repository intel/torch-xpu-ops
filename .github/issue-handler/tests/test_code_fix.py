"""Tests for code_fix target repo detection."""
from issue_handler.fixing_steps.code_fix import _detect_target_repo


def test_detect_from_metadata():
    body = "<!-- target_repo: torch-xpu-ops -->\nSome body"
    assert _detect_target_repo(body) == "torch-xpu-ops"


def test_detect_from_metadata_pytorch():
    body = "<!-- target_repo: pytorch -->\nSome body"
    assert _detect_target_repo(body) == "pytorch"


def test_detect_from_fix_strategy_xpu_path():
    body = (
        "## Proposed Fix Strategy\n"
        "Modify src/ATen/native/xpu/sycl/Atomics.h to use target.load()\n"
        "## Next Section\n"
    )
    assert _detect_target_repo(body) == "torch-xpu-ops"


def test_detect_from_fix_strategy_pytorch_path():
    body = (
        "## Proposed Fix Strategy\n"
        "Modify torch/_dynamo/variables/tensor.py to handle XPU\n"
        "## Next Section\n"
    )
    assert _detect_target_repo(body) == "pytorch"


def test_detect_default_pytorch():
    body = "No metadata or fix strategy here"
    assert _detect_target_repo(body) == "pytorch"


def test_detect_torch_xpu_ops_keyword():
    body = (
        "## Proposed Fix Strategy\n"
        "Fix the kernel in torch-xpu-ops repo\n"
        "## Next Section\n"
    )
    assert _detect_target_repo(body) == "torch-xpu-ops"
