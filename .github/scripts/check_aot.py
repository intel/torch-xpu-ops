#!/usr/bin/env python3
"""Check if AOT (Ahead-Of-Time) compilation is effective on the current platform.

Steps:
1. Print the AOT platform list supported by the installed wheel.
2. Detect the current platform type.
3. Clean the NEO compiler cache, run a simple XPU case, then check if the
   cache folder remains empty (empty means AOT is effective).
"""

import os
import platform
import shutil
import sys


def get_aot_platforms():
    """Print AOT platforms supported by the installed torch wheel."""
    print("=" * 60)
    print("Step 1: Query AOT platforms supported by installed wheel")
    print("=" * 60)
    try:
        import torch

        # The AOT target list is embedded in the torch XPU build info
        if hasattr(torch, "xpu") and hasattr(torch.xpu, "get_arch_list"):
            arch_list = torch.xpu.get_arch_list()
            print(f"Supported AOT platforms: {arch_list}")
            return arch_list
        else:
            # Fallback: try to get from _C
            if hasattr(torch._C, "_xpu_getArchList"):
                arch_list = torch._C._xpu_getArchList()
                print(f"Supported AOT platforms: {arch_list}")
                return arch_list
            print("WARNING: Cannot query AOT platform list from this torch build.")
            print("torch.xpu.get_arch_list() is not available.")
            return []
    except Exception as e:
        print(f"ERROR: Failed to get AOT platforms: {e}")
        return []


def get_current_platform():
    """Detect and print the current platform type."""
    print("\n" + "=" * 60)
    print("Step 2: Detect current platform")
    print("=" * 60)
    system = platform.system()
    print(f"OS: {system}")
    print(f"Platform: {platform.platform()}")
    print(f"Machine: {platform.machine()}")

    # Try to get XPU device name
    try:
        import torch

        if torch.xpu.is_available():
            device_name = torch.xpu.get_device_name(0)
            print(f"XPU Device: {device_name}")
    except Exception as e:
        print(f"Cannot get XPU device info: {e}")

    return system


def get_cache_dir(system):
    """Return the NEO compiler cache directory for the given OS."""
    if system == "Linux":
        return os.path.expanduser("~/.cache/neo_compiler_cache")
    elif system == "Windows":
        return r"C:\Users\Lengda\AppData\Local\NEO\neo_compiler_cache"
    else:
        print(f"WARNING: Unsupported OS '{system}', using Linux path as fallback.")
        return os.path.expanduser("~/.cache/neo_compiler_cache")


def clean_cache(cache_dir):
    """Remove the NEO compiler cache directory."""
    print(f"\nCleaning cache directory: {cache_dir}")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print("  Cache directory removed.")
    else:
        print("  Cache directory does not exist (already clean).")


def is_cache_empty(cache_dir):
    """Check if the cache directory is empty or does not exist."""
    if not os.path.exists(cache_dir):
        return True
    # Check if directory has any files (recursively)
    for _, _, files in os.walk(cache_dir):
        if files:
            return False
    return True


def run_xpu_workload():
    """Run a simple XPU workload: check availability and add two random tensors."""
    print("\n" + "=" * 60)
    print("Step 3: Run simple XPU workload and check AOT effectiveness")
    print("=" * 60)

    import torch

    # Check XPU availability
    if not torch.xpu.is_available():
        print("ERROR: XPU is not available on this system.")
        sys.exit(1)
    print(f"XPU is available. Device count: {torch.xpu.device_count()}")

    # Run a simple computation
    print("Running: random tensor addition on XPU...")
    a = torch.randn(1024, 1024, device="xpu")
    b = torch.randn(1024, 1024, device="xpu")
    c = a + b
    torch.xpu.synchronize()
    print("  Computation completed successfully.")


def check_aot_effective(cache_dir):
    """Check if AOT is effective by inspecting the cache directory."""
    print(f"\nChecking cache directory: {cache_dir}")
    if is_cache_empty(cache_dir):
        print("  Cache directory is EMPTY.")
        print("\n>>> RESULT: AOT is EFFECTIVE. No JIT compilation occurred. <<<")
        return True
    else:
        # List some files for debugging
        file_count = 0
        for root, dirs, files in os.walk(cache_dir):
            file_count += len(files)
        print(f"  Cache directory is NOT empty ({file_count} file(s) found).")
        print("\n>>> RESULT: AOT is NOT effective. JIT compilation occurred. <<<")
        return False


def main():
    # Step 1: Print supported AOT platforms
    aot_platforms = get_aot_platforms()

    # Step 2: Detect current platform
    system = get_current_platform()

    # Step 3: Clean cache, run workload, check AOT
    cache_dir = get_cache_dir(system)
    clean_cache(cache_dir)
    run_xpu_workload()
    effective = check_aot_effective(cache_dir)

    print("\n" + "=" * 60)
    sys.exit(0 if effective else 1)


if __name__ == "__main__":
    main()
