# Owner(s): ["module: cpp-extensions"]

import os
import shutil
import subprocess
import sys
import unittest

import torch
import torch.testing._internal.common_utils as common
from torch.testing._internal.common_utils import TEST_XPU

# Resolved in setUpClass; points to pytorch/test/ in the CI layout.
_test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../test")
_cpp_extensions_dir = os.path.join(_test_dir, "cpp_extensions")


def _build_cpp_extensions():
    """Build AOT cpp extensions from pytorch/test/cpp_extensions/.

    Mirrors the build logic from pytorch/test/run_test.py's
    _test_cpp_extensions_aot helper, but only builds the parts needed
    for the SYCL extension test.
    """
    build_dir = os.path.join(_cpp_extensions_dir, "build")
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)

    shell_env = os.environ.copy()
    shell_env["USE_NINJA"] = "1"
    if sys.platform == "win32":
        shell_env["DISTUTILS_USE_SDK"] = "1"

    install_cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-build-isolation",
        ".",
        "--root",
        "./install",
    ]

    return_code = subprocess.call(install_cmd, cwd=_cpp_extensions_dir, env=shell_env)
    if return_code != 0:
        raise RuntimeError(
            f"Failed to build cpp extensions (exit code {return_code}). Build dir: {_cpp_extensions_dir}"
        )

    # Add the installed packages to sys.path so they can be imported
    install_directories = []
    for root, directories, _ in os.walk(os.path.join(_cpp_extensions_dir, "install")):
        for directory in directories:
            if "-packages" in directory:
                install_directories.append(os.path.join(root, directory))

    for d in install_directories:
        if d not in sys.path:
            sys.path.insert(0, d)


def _check_sycl_extension_available():
    try:
        import torch_test_cpp_extension.sycl  # noqa: F401

        return True
    except ModuleNotFoundError as e:
        # Only treat the extension module itself being absent as "not available".
        # Re-raise other missing-module errors (e.g. a broken dependency) so the
        # underlying failure is visible instead of being silently masked.
        if getattr(e, "name", None) in (
            "torch_test_cpp_extension",
            "torch_test_cpp_extension.sycl",
        ):
            return False
        raise


@torch.testing._internal.common_utils.markDynamoStrictTest
class TestCppExtensionAOT(common.TestCase):
    """Tests ahead-of-time SYCL cpp extensions on XPU."""

    _sycl_extension_built = False

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not TEST_XPU:
            return
        # Defer sys.path mutation until we actually need it
        if _test_dir not in sys.path:
            sys.path.insert(0, _test_dir)
        if _check_sycl_extension_available():
            cls._sycl_extension_built = True
            return
        # Build the AOT cpp extensions; let failures surface as test errors
        _build_cpp_extensions()
        cls._sycl_extension_built = _check_sycl_extension_available()

    @unittest.skipIf(not TEST_XPU, "XPU not found")
    def test_sycl_extension(self):
        if not self._sycl_extension_built:
            self.skipTest("torch_test_cpp_extension.sycl not built")

        import torch_test_cpp_extension.sycl as sycl_extension

        x = torch.zeros(100, device="xpu", dtype=torch.float32)
        y = torch.zeros(100, device="xpu", dtype=torch.float32)

        z = sycl_extension.sigmoid_add(x, y).cpu()

        # 2 * sigmoid(0) = 2 * 0.5 = 1
        self.assertEqual(z, torch.ones_like(z))


if __name__ == "__main__":
    common.run_tests()
