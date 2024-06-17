# Owner(s): ["module: intel"]

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, freeze_rng_state

def _gradients_helper(self, device, dtype, module_info, training, check):
    # Check gradients
    module_cls = module_info.module_cls
    module_inputs = module_info.module_inputs_func(module_info, device=device, dtype=dtype,
                                                    requires_grad=True, training=training)
    # === Set nondet tol for gradcheck to user-defined value if on XPU
    gradcheck_nondet_tol = 0.0
    if (torch.device(device).type == 'xpu'):
        gradcheck_nondet_tol = module_info.gradcheck_nondet_tol

    for module_input in module_inputs:
        if module_input.forward_input is None:
            continue

        # === Instantiate the module. ===
        args, kwargs = module_input.constructor_input.args, module_input.constructor_input.kwargs
        m = module_cls(*args, **kwargs)
        m.to(device).to(dtype)
        m.train(training)

        params = tuple(m.parameters())

        # === Lazy modules need to see an input to initialize params before gradcheck is run. ===
        input_args, input_kwargs = module_input.forward_input.args, module_input.forward_input.kwargs
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
            input_and_params = torch.utils._pytree.tree_unflatten(flat_input_and_params, flat_spec)
            new_input_args = input_and_params[:len(input_args)]
            kwarg_args = input_and_params[-len(kwarg_tensors):]
            new_kwargs = {name: obj for (name, _), obj in zip(kwarg_tensors, kwarg_args)}

            with freeze_rng_state():
                output = m(*new_input_args, **new_kwargs, **other_kwargs)
                output_flattened = torch.utils._pytree.tree_leaves(output)
                return output_flattened

        # check total derivative
        grad_input = input_args + params + tuple(obj for (_, obj) in kwarg_tensors)
        flat_input, flat_spec = torch.utils._pytree.tree_flatten(grad_input)

        self.assertTrue(check(fn_to_gradcheck, flat_input, nondet_tol=gradcheck_nondet_tol))

        # check partial derivatives
        old_params_requires_grad = [p.requires_grad for p in params]
        for p in params:
            p.requires_grad = False

        old_kwargs_requires_grad = [obj.requires_grad for (_, obj) in kwarg_tensors]
        for (_, obj) in kwarg_tensors:
            obj.requires_grad = False

        for p, old in zip(params, old_params_requires_grad):
            p.requires_grad = old
            grad_input = input_args + params + tuple(obj for (_, obj) in kwarg_tensors)
            flat_input, flat_spec = torch.utils._pytree.tree_flatten(grad_input)
            self.assertTrue(check(fn_to_gradcheck, flat_input, nondet_tol=gradcheck_nondet_tol))
            p.requires_grad = False

        for (_, obj), old in zip(kwarg_tensors, old_kwargs_requires_grad):
            obj.requires_grad = old
            grad_input = input_args + params + tuple(obj for (_, obj) in kwarg_tensors)
            flat_input, flat_spec = torch.utils._pytree.tree_flatten(grad_input)
            self.assertTrue(check(fn_to_gradcheck, flat_input, nondet_tol=gradcheck_nondet_tol))
            obj.requires_grad = False

try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    from .xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_modules import TestModule

    TestModule._test_gradients_helper = _gradients_helper

instantiate_device_type_tests(TestModule, globals(), only_for="xpu")


if __name__ == "__main__":
    run_tests()
