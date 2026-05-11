# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

import torch.testing._internal.common_device_type as common_device_type_mod

_FORCE_XPU_OPT = "--force-xpu-test-collection"


def _should_force_xpu_test_collection(config) -> bool:
    return config.getoption(_FORCE_XPU_OPT)


def pytest_addoption(parser):
    parser.addoption(
        _FORCE_XPU_OPT,
        action="store_true",
        default=False,
        help="Force XPU device-test class generation for collection-only scenarios.",
    )


def pytest_configure(config):
    if not _should_force_xpu_test_collection(config):
        return

    original_test_xpu = common_device_type_mod.TEST_XPU
    original_setup_class = common_device_type_mod.XPUTestBase.setUpClass
    original_get_all_devices = common_device_type_mod.XPUTestBase.get_all_devices

    @classmethod
    def _xpu_collect_only_setup_class(cls):
        cls.primary_device = "xpu:0"

    @classmethod
    def _xpu_collect_only_get_all_devices(cls):
        return [cls.get_primary_device()]

    common_device_type_mod.TEST_XPU = True
    common_device_type_mod.XPUTestBase.setUpClass = _xpu_collect_only_setup_class
    common_device_type_mod.XPUTestBase.get_all_devices = (
        _xpu_collect_only_get_all_devices
    )

    config._force_xpu_test_collection_state = (
        original_test_xpu,
        original_setup_class,
        original_get_all_devices,
    )


def pytest_unconfigure(config):
    state = getattr(config, "_force_xpu_test_collection_state", None)
    if state is None:
        return

    original_test_xpu, original_setup_class, original_get_all_devices = state
    common_device_type_mod.TEST_XPU = original_test_xpu
    common_device_type_mod.XPUTestBase.setUpClass = original_setup_class
    common_device_type_mod.XPUTestBase.get_all_devices = original_get_all_devices
