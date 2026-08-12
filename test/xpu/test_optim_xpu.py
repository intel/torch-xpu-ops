# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]


from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests

try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    from .xpu_test_utils import XPUPatchForImport
with XPUPatchForImport(False):
    from test_optim import _bf16_state_init_hook, TestOptimRenewed

from copy import deepcopy

import torch
from torch.testing._internal.common_device_type import TEST_WITH_ROCM
from torch.testing._internal.common_dtype import floating_types_and
from torch.testing._internal.common_optimizers import optim_db, optims, TensorTracker
from torch.testing._internal.common_utils import parametrize, TEST_WITH_TORCHDYNAMO

for optim in optim_db:
    for c in [
        torch.optim.Adam,
        torch.optim.AdamW,
        torch.optim.SGD,
        torch.optim.Adagrad,
    ]:
        if optim.optim_cls is c:
            if (
                "cuda" in optim.supports_fused_on
                and "xpu" not in optim.supports_fused_on
            ):
                optim.supports_fused_on = ("xpu",) + optim.supports_fused_on


@optims(
    [
        optim
        for optim in optim_db
        if "cpu" in optim.supports_fused_on and "xpu" in optim.supports_fused_on
    ],
    dtypes=floating_types_and(
        torch.bfloat16,
        torch.float16,
    ),
)
def _test_fused_cpu_matches_cuda(self, device, dtype, optim_info):
    optim_cls = optim_info.optim_cls
    optim_inputs = optim_info.optim_inputs_func(device="cpu")
    for optim_input in optim_inputs:
        inpts, models, optimizers = [], [], []
        for dev in ("cpu", "xpu"):
            kwargs = optim_input.kwargs
            kwargs["fused"] = True
            inpt = torch.tensor(
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=dtype, device=dev
            ).reshape(3, 2)

            torch.manual_seed(1)
            model = torch.nn.Sequential(
                torch.nn.Linear(2, 3),
                torch.nn.Sigmoid(),
                torch.nn.Linear(3, 1),
                torch.nn.Sigmoid(),
            )
            model.to(dtype=dtype, device=dev)

            # foreach/fused optimizers should be tested with a
            # zero_size tensor as its last param.
            # ref: https://github.com/pytorch/pytorch/issues/100701
            empty_param = torch.empty((), device=dev, dtype=dtype, requires_grad=True)
            empty_param.grad = torch.rand_like(empty_param)
            params = list(model.parameters()) + [empty_param]

            optimizer = optim_cls(params, **kwargs)
            inpts.append(inpt)
            models.append(model)
            optimizers.append(optimizer)
    self._compare_between(inpts, models, optimizers)


TestOptimRenewed.test_fused_cpu_matches_cuda = _test_fused_cpu_matches_cuda


@optims(
    [optim for optim in optim_db if "foreach" in optim.supported_impls],
    dtypes=[torch.float32],
)
def _test_peak_memory_foreach(self, device, dtype, optim_info):
    nparams = 10
    optim_inputs = optim_info.optim_inputs_func(device=device)
    optim_cls = optim_info.optim_cls
    for optim_input in optim_inputs:
        kwargs = deepcopy(optim_input.kwargs)
        max_mems = []
        for flag_value in (False, True):
            kwargs["foreach"] = flag_value
            # The 16 * 8 = 128 is critical here! Our CUDACachingAllocator allocates in blocks
            # of 512, meaning any tensor that occupies <512 bytes of memory will allocate a
            # whole 512 bytes anyway. We use 128 (cuz datasize would be 4 bytes) so that param
            # is size 512 exactly, making our later calculations for intermediate_size easy.
            param = torch.rand(16, 8, device=device, dtype=dtype)
            params = [torch.rand_like(param) for _ in range(nparams)]

            optimizer = optim_cls(params, **kwargs)

            for p in params:
                p.grad = torch.rand_like(p)

            optimizer.step()
            import gc

            gc.collect()
            torch.xpu.reset_peak_memory_stats()
            optimizer.step()
            gc.collect()
            max_mems.append(torch.xpu.max_memory_allocated())

        st_max_mem, mt_max_mem = max_mems
        intermediate_size = nparams * param.nelement() * param.element_size()
        nintermediates = 1  # we expect a budget of 1 intermediate most of the time

        # Check the param group directly to handle if the compiler set capturable
        if optimizer.param_groups[0].get("capturable", False) or optim_cls.__name__ in [
            "Adadelta",
            "ASGD",
            "RAdam",
        ]:
            # with capturable in Adam(W), we have 2 extra intermediates for the bias_corrections
            # with Adadelta, we have 2 extra for (acc_delta + eps) and (square_avg + eps)
            # ASGD allocates axs, 2x mus, 2x etas, and grads at the same time
            nintermediates = 3
            if optim_cls.__name__ == "NAdam":
                # with capturable in NAdam, we have 3 extra intermediates for the
                # bias_correction, mus, and mu_nexts
                if TEST_WITH_TORCHDYNAMO:
                    # With dynamo, the eager/FX backend appears to hold memory longer than
                    # vanilla eager: https://github.com/pytorch/pytorch/issues/125511
                    nintermediates = 8
                else:
                    nintermediates = 5

            if optim_cls.__name__ == "RAdam":
                # RAdam has four intermediates with capturable
                # num, unrect_step_size, buffer, grouped_grads
                if TEST_WITH_TORCHDYNAMO:
                    # With dynamo, the eager/FX backend appears to hold memory than
                    # vanilla eager: https://github.com/pytorch/pytorch/issues/125511
                    nintermediates = 6
                else:
                    nintermediates = 4

        elif optim_cls.__name__ in ["NAdam", "Adagrad", "RMSprop", "Adafactor"]:
            # NAdam uses two intermediates at the same time (grads & exp_avg_sq_sqrt)
            # Adagrad uses std and grads at the same time
            # RMSprop uses avg and grads
            # Adafactor uses row/col var and its mean
            nintermediates = 2

            if optim_cls.__name__ == "Adafactor" and kwargs.get("maximize", False):
                # When maximize is True, Adafactor also tracks device_grad
                nintermediates = 3

        # Dynamo ST uses less mem than eager in the case of Adam/Adagrad/Nadam/RAdam
        # which makes the foreach memory check fail
        if TEST_WITH_TORCHDYNAMO:
            st_max_mem += 6000

        expected_max_mem = st_max_mem + intermediate_size * nintermediates
        # hipcc currently can't generate efficient code for the small buffer optimization
        # code path (see Note [small buffer optimization] for details), thus we always
        # dynamically allocate the tensor metadata for ROCM. Adjusting the expected max
        # memory usage to account for this.
        if TEST_WITH_ROCM:
            expected_max_mem *= 1.02
        else:
            expected_max_mem *= 1.05  # Patch for XPU testing

        self.assertLessEqual(mt_max_mem, expected_max_mem)


TestOptimRenewed.test_peak_memory_foreach = _test_peak_memory_foreach


@parametrize("amsgrad", [False, True])
@optims(
    [o for o in optim_db if o.optim_cls.__name__ in ["Adam", "AdamW"]],
    dtypes=[torch.float32],
)
def _test_fused_mixed_precision_state_init(self, device, dtype, optim_info, amsgrad):
    optim_cls = optim_info.optim_cls
    params = [torch.rand(20, 7, device=device, dtype=dtype) for _ in range(5)]
    for p in params:
        p.grad = torch.rand_like(p)

    optim = optim_cls(params, lr=1e-3, fused=True, amsgrad=amsgrad)
    optim.register_step_pre_hook(_bf16_state_init_hook)

    optim.step()

    for p in params:
        self.assertEqual(p.dtype, torch.float32)
        state = optim.state[p]
        self.assertEqual(state["step"].dtype, torch.float32)
        self.assertEqual(state["exp_avg"].dtype, torch.bfloat16)
        self.assertEqual(state["exp_avg_sq"].dtype, torch.bfloat16)
        if amsgrad:
            self.assertEqual(state["max_exp_avg_sq"].dtype, torch.bfloat16)

    # Second step: hook should be idempotent (skips already-populated state)
    for p in params:
        p.grad = torch.rand_like(p)
    optim.step()

    for p in params:
        state = optim.state[p]
        self.assertEqual(state["step"].dtype, torch.float32)
        self.assertEqual(state["exp_avg"].dtype, torch.bfloat16)
        self.assertEqual(state["exp_avg_sq"].dtype, torch.bfloat16)
        if amsgrad:
            self.assertEqual(state["max_exp_avg_sq"].dtype, torch.bfloat16)


TestOptimRenewed.test_fused_mixed_precision_state_init = (
    _test_fused_mixed_precision_state_init
)


@parametrize("amsgrad", [False, True])
@optims(
    [o for o in optim_db if o.optim_cls.__name__ in ["Adam", "AdamW"]],
    dtypes=[torch.float32],
)
def _test_fused_mixed_precision_hook_skips_existing_state(
    self, device, dtype, optim_info, amsgrad
):
    optim_cls = optim_info.optim_cls
    g1_params = [torch.rand(10, 5, device=device, dtype=dtype) for _ in range(2)]
    g2_params = [torch.rand(10, 5, device=device, dtype=dtype) for _ in range(2)]
    for p in g1_params + g2_params:
        p.grad = torch.rand_like(p)

    optim = optim_cls(
        [{"params": g1_params}, {"params": g2_params}],
        lr=1e-3,
        fused=True,
        amsgrad=amsgrad,
    )

    for p in g1_params:
        optim.state[p]["step"] = torch.zeros((), dtype=torch.float32, device=p.device)
        optim.state[p]["exp_avg"] = torch.zeros_like(p)
        optim.state[p]["exp_avg_sq"] = torch.zeros_like(p)
        if amsgrad:
            optim.state[p]["max_exp_avg_sq"] = torch.zeros_like(p)

    optim.register_step_pre_hook(_bf16_state_init_hook)
    optim.step()

    # Group 1: hook skipped (state was non-empty), dtypes stay f32.
    for p in g1_params:
        state = optim.state[p]
        self.assertEqual(state["step"].dtype, torch.float32)
        self.assertEqual(state["exp_avg"].dtype, torch.float32)
        self.assertEqual(state["exp_avg_sq"].dtype, torch.float32)
        if amsgrad:
            self.assertEqual(state["max_exp_avg_sq"].dtype, torch.float32)

    # Group 2: hook initialized state in bf16.
    for p in g2_params:
        state = optim.state[p]
        self.assertEqual(state["step"].dtype, torch.float32)
        self.assertEqual(state["exp_avg"].dtype, torch.bfloat16)
        self.assertEqual(state["exp_avg_sq"].dtype, torch.bfloat16)
        if amsgrad:
            self.assertEqual(state["max_exp_avg_sq"].dtype, torch.bfloat16)


TestOptimRenewed.test_fused_mixed_precision_hook_skips_existing_state = (
    _test_fused_mixed_precision_hook_skips_existing_state
)


@optims(
    [o for o in optim_db if o.optim_cls.__name__ in ["Adam", "AdamW"]],
    dtypes=[torch.float32],
)
def _test_fused_mixed_precision_numerics(self, device, dtype, optim_info):
    optim_inputs = optim_info.optim_inputs_func(device=device, dtype=dtype)
    optim_cls = optim_info.optim_cls
    for optim_input in optim_inputs:
        kwargs = {**optim_input.kwargs, "fused": True}

        params = [torch.rand(20, 7, device=device, dtype=dtype) for _ in range(10)]
        for p in params:
            p.grad = torch.rand_like(p)

        params_c = [p.clone() for p in params]
        for p, pc in zip(params, params_c):
            pc.grad = p.grad.clone()

        ref_optim = optim_cls(params, **kwargs)
        bf16_optim = optim_cls(params_c, **kwargs)
        bf16_optim.register_step_pre_hook(_bf16_state_init_hook)

        # Simulate bf16 storage: after each ref step, quantize states to
        # bf16 and back so the reference matches the mixed-precision kernel.
        tracker = TensorTracker()
        for i in range(7):
            ref_optim.step()
            bf16_optim.step()
            for p in params:
                tracker.add(p)
                tracker.add(p.grad)
            for d in ref_optim.state.values():
                exp_avg_bf16 = d["exp_avg"].to(torch.bfloat16)
                tracker.add(exp_avg_bf16)
                d["exp_avg"] = exp_avg_bf16.to(torch.float32)
                exp_avg_sq_bf16 = d["exp_avg_sq"].to(torch.bfloat16)
                tracker.add(exp_avg_sq_bf16)
                d["exp_avg_sq"] = exp_avg_sq_bf16.to(torch.float32)
                if "max_exp_avg_sq" in d:
                    max_exp_avg_sq_bf16 = d["max_exp_avg_sq"].to(torch.bfloat16)
                    tracker.add(max_exp_avg_sq_bf16)
                    d["max_exp_avg_sq"] = max_exp_avg_sq_bf16.to(torch.float32)

            for e, pc in enumerate(params_c):
                tracker.pop_check_set(pc, self)
                tracker.pop_check_set(pc.grad, self)

            for p, pc in zip(params, params_c):
                self.assertEqual(p, pc)

            for dc in bf16_optim.state.values():
                tracker.pop_check_set(dc["exp_avg"], self)
                tracker.pop_check_set(dc["exp_avg_sq"], self)
                if "max_exp_avg_sq" in dc:
                    tracker.pop_check_set(dc["max_exp_avg_sq"], self)
            self.assertTrue(tracker.all_popped())


TestOptimRenewed.test_fused_mixed_precision_numerics = (
    _test_fused_mixed_precision_numerics
)

instantiate_device_type_tests(
    TestOptimRenewed, globals(), only_for="xpu", allow_xpu=True
)

if __name__ == "__main__":
    run_tests()
