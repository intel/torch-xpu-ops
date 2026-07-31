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
import torch
from torch._export.serde import serialize as serde_serialize
from torch._export.serde.serialize import deserialize, serialize

try:
    from . import test_export_xpu, testing_xpu
except ImportError:
    import test_export_xpu  # @manual=fbcode//caffe2/test:test_export-library
    import testing_xpu  # @manual=fbcode//caffe2/test:test_export-library

from torch.export import export
from torch.testing._internal import custom_tensor as custom_tensor_mod
from torch.testing._internal import two_tensor as two_tensor_mod

test_classes = {}


def _module_safe_globals():
    test_classes = [
        obj
        for _, obj in inspect.getmembers(test_export_xpu, inspect.isclass)
        if obj.__module__ == test_export_xpu.__name__
    ]
    nn_classes = [obj for _, obj in inspect.getmembers(torch.nn, inspect.isclass)]
    two_tensor_classes = [obj for _, obj in inspect.getmembers(two_tensor_mod, inspect.isclass)]
    custom_tensor_classes = [obj for _, obj in inspect.getmembers(custom_tensor_mod, inspect.isclass)]
    serde_helpers = [serde_serialize._reconstruct_fake_tensor, torch.ScriptObject]
    return test_classes + nn_classes + two_tensor_classes + custom_tensor_classes + serde_helpers


def _deserialize_with_safe_globals(payload):
    with torch.serialization.safe_globals(_module_safe_globals()):
        return deserialize(payload)


def mocked_cpp_serdes_export(*args, **kwargs):
    ep = export(*args, **kwargs)
    try:
        payload = serialize(ep)
    except Exception:
        return ep
    cpp_ep = torch._C._export.deserialize_exported_program(payload.exported_program)
    loaded_json = torch._C._export.serialize_exported_program(cpp_ep)
    payload.exported_program = loaded_json.encode()
    loaded_ep = _deserialize_with_safe_globals(payload)
    return loaded_ep


def make_dynamic_cls(cls):
    cls_prefix = "CppSerdes"

    test_class = testing_xpu.make_test_cls_with_mocked_export(
        cls,
        cls_prefix,
        "_cpp_serdes",
        mocked_cpp_serdes_export,
        xfail_prop="_expected_failure_cpp_serdes",
    )

    test_classes[test_class.__name__] = test_class
    # REMOVING THIS LINE WILL STOP TESTS FROM RUNNING
    globals()[test_class.__name__] = test_class
    test_class.__module__ = __name__


tests = [
    test_export_xpu.TestExport,
]
for test in tests:
    make_dynamic_cls(test)
del test

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
