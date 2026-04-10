# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Portions of this file are derived from PyTorch
# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: BSD-3-Clause

# Owner(s): ["module: intel"]
import sys

sys.path.append("../../../../test/export")

try:
    from . import test_export_xpu
    import testing
except ImportError:
    import test_export  # @manual=fbcode//caffe2/test:test_export-library
    import testing  # @manual=fbcode//caffe2/test:test_export-library

from torch._export import config
from torch.export import export


test_classes = {}


def mocked_strict_export_v2(*args, **kwargs):
    # If user already specified strict, don't make it strict
    with config.patch(use_legacy_dynamo_graph_capture=False):
        if "strict" in kwargs:
            return export(*args, **kwargs)
        return export(*args, **kwargs, strict=True)


def make_dynamic_cls(cls):
    cls_prefix = "StrictExportV2"

    test_class = testing.make_test_cls_with_mocked_export(
        cls,
        cls_prefix,
        test_export_xpu.STRICT_EXPORT_V2_SUFFIX,
        mocked_strict_export_v2,
        xfail_prop="_expected_failure_strict_v2",
    )

    test_classes[test_class.__name__] = test_class
    # REMOVING THIS LINE WILL STOP TESTS FROM RUNNING
    globals()[test_class.__name__] = test_class
    test_class.__module__ = __name__
    return test_class


tests = [
    test_export_xpu.TestDynamismExpression,
    test_export_xpu.TestExport,
]
for test in tests:
    make_dynamic_cls(test)
del test

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
