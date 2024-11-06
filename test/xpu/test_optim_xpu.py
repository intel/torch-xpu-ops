# Owner(s): ["module: intel"]


from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests

try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    from .xpu_test_utils import XPUPatchForImport
with XPUPatchForImport(False):
    from test_optim import (
        TestOptimRenewed
    )

import torch
from torch.nn import Parameter
from copy import deepcopy
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests
)
from torch.testing._internal.common_dtype import floating_types_and
from torch.testing._internal.common_optimizers import (
    _get_optim_inputs_including_global_cliquey_kwargs,
    optim_db,
    optims,
)
from torch.testing._internal.common_device_type import (
    TEST_WITH_ROCM,
)
from torch.testing._internal.common_utils import (
    TEST_WITH_TORCHDYNAMO,
)

for optim in optim_db:
    for c in [torch.optim.Adam, torch.optim.AdamW]:
        if optim.optim_cls is c:
            if "cuda" in optim.supports_fused_on and "xpu" not in optim.supports_fused_on:
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
            empty_param = torch.empty(
                (), device=dev, dtype=dtype, requires_grad=True
            )
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
        if optimizer.param_groups[0].get(
            "capturable", False
        ) or optim_cls.__name__ in ["Adadelta", "ASGD", "RAdam"]:
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
            expected_max_mem *= 1.05 # Patch for XPU testing

        self.assertLessEqual(mt_max_mem, expected_max_mem)

TestOptimRenewed.test_peak_memory_foreach = _test_peak_memory_foreach


@optims(optim_db, dtypes=[torch.float32])
def _test_state_dict_with_cuda_params(self, device, dtype, optim_info):
    optim_cls = optim_info.optim_cls

    # Skip differentiable testing for now, see https://github.com/pytorch/pytorch/issues/116490
    # We limit our configs to CPU only, because we will be moving them to CUDA later
    cpu_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
        "cpu", dtype, optim_info, skip=("differentiable",)
    )

    # Needed for second order optims like LBFGS
    closure_loss = torch.rand(1, device=device, dtype=dtype)

    def closure():
        return closure_loss if optim_info.step_requires_closure else None

    for optim_input in cpu_optim_inputs:
        if (
            "fused" in optim_input.kwargs
            and "xpu" not in optim_info.supports_fused_on
        ):
            self.skipTest(
                f"xpu is not supported for fused on {optim_cls.__name__}"
            )
        params = [
            Parameter(torch.randn(2, 3, device="cpu", dtype=dtype))
            for _ in range(2)
        ]
        for p in params:
            p.grad = torch.randn_like(p)
            if optim_info.only_supports_sparse_grads:
                # For this test, we naively convert the Tensor layout, which we know does
                # NOT represent the expected use case for optims like SparseAdam!
                p.grad = p.grad.to_sparse()

        optimizer = optim_cls(params, **optim_input.kwargs)

        for _ in range(3):
            optimizer.step(closure)

        with torch.no_grad():
            params_cuda = [p.to(device="xpu") for p in params]
            for i, p in enumerate(params_cuda):
                p.grad = params[i].grad.to(device="xpu")
        optimizer_cuda = optim_cls(params_cuda, **optim_input.kwargs)

        state_dict_cpu = deepcopy(optimizer.state_dict())
        state_dict_cuda = deepcopy(optimizer.state_dict())
        optimizer_cuda.load_state_dict(state_dict_cuda)

        # Make sure state_dict_cuda isn't modified by merely calling load_state_dict
        self.assertEqual(state_dict_cpu, state_dict_cuda)

        # Make sure that device of state['step'] is still CPU _unless_ torch.compile() added a capturable!
        capturable = state_dict_cpu["param_groups"][0].get("capturable", False)
        fused = state_dict_cpu["param_groups"][0].get("fused", False)
        new_state_dict = optimizer_cuda.state_dict()
        for state_cpu, state_cuda in zip(
            state_dict_cpu["state"].values(), new_state_dict["state"].values()
        ):
            if "step" in state_cpu and torch.is_tensor(state_cpu["step"]):
                self.assertEqual(
                    state_cuda["step"].device.type,
                    "xpu" if capturable or fused else "cpu",
                )

        for _ in range(5):
            optimizer.step(closure)
            optimizer_cuda.step(closure)
            self.assertEqual(params, params_cuda)
            self.assertEqual(optimizer.state_dict(), optimizer_cuda.state_dict())
TestOptimRenewed.test_state_dict_with_cuda_params = _test_state_dict_with_cuda_params

instantiate_device_type_tests(TestOptimRenewed, globals(), only_for="xpu", allow_xpu=True)

if __name__ == "__main__":
    run_tests()
