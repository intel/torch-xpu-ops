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

import inspect
import io

import torch
from torch._export.serde import serialize as serde_serialize

try:
    import testing_xpu as testing

    from . import test_export_xpu
except ImportError:
    import test_export_xpu  # @manual=fbcode//caffe2/test:test_export-library
    import testing_xpu as testing  # @manual=fbcode//caffe2/test:test_export-library

from torch.export import export, load, save

test_classes = {}


def _module_safe_globals():
    # SerDes round-trips can contain test helper classes and torch.nn module classes.
    test_classes = [
        obj
        for _, obj in inspect.getmembers(test_export_xpu, inspect.isclass)
        if obj.__module__ == test_export_xpu.__name__
    ]
    nn_classes = [obj for _, obj in inspect.getmembers(torch.nn, inspect.isclass)]
    serde_helpers = [serde_serialize._reconstruct_fake_tensor]
    return test_classes + nn_classes + serde_helpers


def _load_with_safe_globals(buffer):
    with torch.serialization.safe_globals(_module_safe_globals()):
        return load(buffer)


def mocked_serder_export_strict(*args, **kwargs):
    if "strict" not in kwargs:
        ep = export(*args, **kwargs, strict=True)
    else:
        ep = export(*args, **kwargs)

    buffer = io.BytesIO()
    save(ep, buffer)
    buffer.seek(0)
    loaded_ep = _load_with_safe_globals(buffer)
    return loaded_ep


def mocked_serder_export_non_strict(*args, **kwargs):
    ep = export(*args, **kwargs)
    buffer = io.BytesIO()
    save(ep, buffer)
    buffer.seek(0)
    loaded_ep = _load_with_safe_globals(buffer)
    return loaded_ep


def make_dynamic_cls(cls, strict):
    if strict:
        test_class = testing.make_test_cls_with_mocked_export(
            cls,
            "SerDesExport",
            test_export_xpu.SERDES_STRICT_SUFFIX,
            mocked_serder_export_strict,
            xfail_prop="_expected_failure_serdes",
        )
    else:
        test_class = testing.make_test_cls_with_mocked_export(
            cls,
            "SerDesExportNonStrict",
            test_export_xpu.SERDES_NON_STRICT_SUFFIX,
            mocked_serder_export_non_strict,
            xfail_prop="_expected_failure_serdes_non_strict",
        )

    test_classes[test_class.__name__] = test_class
    # REMOVING THIS LINE WILL STOP TESTS FROM RUNNING
    globals()[test_class.__name__] = test_class
    test_class.__module__ = __name__


tests = [
    test_export_xpu.TestDynamismExpression,
    test_export_xpu.TestExport,
]
for test in tests:
    make_dynamic_cls(test, True)
    make_dynamic_cls(test, False)
del test

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
