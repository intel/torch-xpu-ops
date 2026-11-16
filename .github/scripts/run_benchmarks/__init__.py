"""benchmarks — E2E benchmark orchestration for PyTorch XPU.

Usage:
    python -m benchmarks --suite huggingface --dt float32 --scenario accuracy
    python -m benchmarks --task-file results.csv
"""

from .config import TestTask

__all__ = ["TestTask"]
