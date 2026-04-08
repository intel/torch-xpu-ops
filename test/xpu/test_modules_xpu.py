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


import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_modules import module_db, modules
from torch.testing._internal.common_utils import freeze_rng_state, run_tests


def _gradients_helper(self, device, dtype, module_info, training, check):
    # Check gradients
    module_cls = module_info.module_cls
    module_inputs = module_info.module_inputs_func(
        module_info, device=device, dtype=dtype, requires_grad=True, training=training
    )
    # === Set nondet tol for gradcheck to user-defined value if on XPU
    gradcheck_nondet_tol = 0.0
    if torch.device(device).type == "xpu":
        gradcheck_nondet_tol = module_info.gradcheck_nondet_tol

    for module_input in module_inputs:
        if module_input.forward_input is None:
            continue

        # === Instantiate the module. ===
        args, kwargs = (
            module_input.constructor_input.args,
            module_input.constructor_input.kwargs,
        )
        m = module_cls(*args, **kwargs)
        m.to(device).to(dtype)
        m.train(training)

        params = tuple(m.parameters())

        # === Lazy modules need to see an input to initialize params before gradcheck is run. ===
        input_args, input_kwargs = (
            module_input.forward_input.args,
            module_input.forward_input.kwargs,
        )
        if issubclass(module_info.module_cls, torch.nn.modules.lazy.LazyModuleMixin):
            with torch.no_grad():
                m(*input_args, **input_kwargs)

        # === Perform gradient check on the input_args ===
        other_kwargs = {}
        kwarg_tensors = []
        for name, obj in input_kwargs.items():
            if isinstance(obj, torch.Tensor):
                kwarg_tensors.append((name, obj))
            else:
                other_kwargs[name] = obj

        def fn_to_gradcheck(*flat_input_and_params):
            input_and_params = torch.utils._pytree.tree_unflatten(
                flat_input_and_params, flat_spec
            )
            new_input_args = input_and_params[: len(input_args)]
            kwarg_args = input_and_params[-len(kwarg_tensors) :]
            new_kwargs = {
                name: obj for (name, _), obj in zip(kwarg_tensors, kwarg_args)
            }

            with freeze_rng_state():
                output = m(*new_input_args, **new_kwargs, **other_kwargs)
                output_flattened = torch.utils._pytree.tree_leaves(output)
                return output_flattened

        # check total derivative
        grad_input = input_args + params + tuple(obj for (_, obj) in kwarg_tensors)
        flat_input, flat_spec = torch.utils._pytree.tree_flatten(grad_input)

        self.assertTrue(
            check(fn_to_gradcheck, flat_input, nondet_tol=gradcheck_nondet_tol)
        )

        # check partial derivatives
        old_params_requires_grad = [p.requires_grad for p in params]
        for p in params:
            p.requires_grad = False

        old_kwargs_requires_grad = [obj.requires_grad for (_, obj) in kwarg_tensors]
        for _, obj in kwarg_tensors:
            obj.requires_grad = False

        for p, old in zip(params, old_params_requires_grad):
            p.requires_grad = old
            grad_input = input_args + params + tuple(obj for (_, obj) in kwarg_tensors)
            flat_input, flat_spec = torch.utils._pytree.tree_flatten(grad_input)
            self.assertTrue(
                check(fn_to_gradcheck, flat_input, nondet_tol=gradcheck_nondet_tol)
            )
            p.requires_grad = False

        for (_, obj), old in zip(kwarg_tensors, old_kwargs_requires_grad):
            obj.requires_grad = old
            grad_input = input_args + params + tuple(obj for (_, obj) in kwarg_tensors)
            flat_input, flat_spec = torch.utils._pytree.tree_flatten(grad_input)
            self.assertTrue(
                check(fn_to_gradcheck, flat_input, nondet_tol=gradcheck_nondet_tol)
            )
            obj.requires_grad = False


@modules(module_db)
def _test_multiple_device_transfer(self, device, dtype, module_info, training):
    module_cls = module_info.module_cls
    module_inputs_device = module_info.module_inputs_func(
        module_info, device=device, dtype=dtype, requires_grad=False, training=training
    )
    module_inputs_cpu = module_info.module_inputs_func(
        module_info, device="cpu", dtype=dtype, requires_grad=False, training=training
    )
    for module_input_device, module_input_cpu in zip(
        module_inputs_device, module_inputs_cpu
    ):
        if module_input_device.forward_input is None:
            continue

        with freeze_rng_state():
            # === Instantiate the module. ===
            args, kwargs = (
                module_input_device.constructor_input.args,
                module_input_device.constructor_input.kwargs,
            )
            m = module_cls(*args, **kwargs)
            m.to(device).to(dtype)
            m.train(training)

            # === Do forward pass on GPU ===
            input_device_args = module_input_device.forward_input.args
            input_device_kwargs = module_input_device.forward_input.kwargs
            m(*input_device_args, **input_device_kwargs)
            self._assert_module_parameters_and_buffer_are(m, device, dtype)

            # === Move to CPU ===
            input_cpu_args = module_input_cpu.forward_input.args
            input_cpu_kwargs = module_input_cpu.forward_input.kwargs
            m.cpu()
            m(*input_cpu_args, **input_cpu_kwargs)
            self._assert_module_parameters_and_buffer_are(m, "cpu", dtype)

            # === Move back to GPU and forward pass ===
            m.xpu()
            m(*input_device_args, **input_device_kwargs)
            self._assert_module_parameters_and_buffer_are(m, device, dtype)

            if torch.cuda.device_count() >= 2:
                # === test cross-GPU transfer works
                def _to_device1(objs):
                    if isinstance(objs, tuple | list):
                        return type(objs)(_to_device1(item) for item in objs)
                    elif isinstance(objs, dict):
                        return {name: _to_device1(item) for name, item in objs.items()}
                    elif isinstance(objs, torch.Tensor):
                        return objs.cuda(1)
                    else:
                        return objs

                input_device_1_args = _to_device1(input_device_args)
                input_device_1_kwargs = _to_device1(input_device_kwargs)

                m.cuda(1)
                with torch.cuda.device(1):
                    m(*input_device_1_args, **input_device_1_kwargs)
                self._assert_module_parameters_and_buffer_are(
                    m, torch.device("cuda:1"), dtype
                )


try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    from .xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_modules import TestModule

    TestModule._test_gradients_helper = _gradients_helper
    TestModule.test_multiple_device_transfer = _test_multiple_device_transfer

instantiate_device_type_tests(TestModule, globals(), only_for="xpu", allow_xpu=True)


if __name__ == "__main__":
    run_tests()
