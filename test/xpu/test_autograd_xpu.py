# Owner(s): ["module: intel"]

import torch
import torch.autograd.forward_ad as fwAD
import threading
import warnings
from copy import deepcopy
from functools import partial
from torch import nn
from torch.autograd import Function
from torch.autograd.profiler import emit_nvtx
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyXPU,
)
from torch.testing._internal.common_utils import (
    gradcheck,
    gradgradcheck,
    instantiate_parametrized_tests,
    run_tests,
    slowTest,
)
from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts, CheckpointPolicy
import torch.utils.checkpoint
from torch.utils.flop_counter import FlopCounterMode


def autograd_multiple_dispatch_registrations(self, device):
    t = torch.randn(3, 3, device=device, requires_grad=True)
    # using _test_autograd_multiple_dispatch.fullcoverage which has
    # registrations in derivatives.yaml for Default, AutogradCUDA and NestedTensorAutograd
    out = torch._test_autograd_multiple_dispatch(t)
    grad = torch.randn(3, 3, device=device)
    out.backward(grad)
    if "XPU" not in device:
        # bogus default gradient registered for Autograd is grad + 1
        self.assertEqual(t.grad, grad + 1)
    else:
        # bogus gradient registered for AutogradCUDA is grad * 2
        self.assertEqual(t.grad, grad * 2)
    # test registered AutogradNestedTensor formula
    a = (
        torch.arange(6, dtype=torch.float, device=device)
        .reshape(2, 3)
        .requires_grad_(True)
    )
    b = (
        torch.arange(8, dtype=torch.float, device=device)
        .reshape(2, 4)
        .requires_grad_(True)
    )
    nt = torch.nested.as_nested_tensor([a, b], dtype=torch.float, device=device)
    nt_out = torch._test_autograd_multiple_dispatch(nt)
    c = torch.randn(2, 3, device=device)
    d = torch.randn(2, 4, device=device)
    nt_grad = torch.nested.nested_tensor([c, d], dtype=torch.float, device=device)
    nt_out.backward(nt_grad)
    # bogus gradient for AutogradNestedTensor is grad * grad
    self.assertEqual(a.grad, c * c)
    self.assertEqual(b.grad, d * d)


def foward_mode_AD(self, device):
    # check that forward mode AD is only registered for the Default
    # dispatch for _test_autograd_multiple_dispatch.fullcoverage and not AutogradCUDA
    primal = torch.randn(3, device=device)
    tangent = torch.randn(3, device=device)
    with fwAD.dual_level():
        dual_input = fwAD.make_dual(primal, tangent)
        err_msg = r"Trying to use forward AD with .* that does not support it"
        hint_msg = "Running forward AD for an OP that does not implement it should raise a NotImplementedError"
        if "XPU" in device:
            with self.assertRaisesRegex(NotImplementedError, err_msg, msg=hint_msg):
                torch._test_autograd_multiple_dispatch(dual_input)
        else:
            torch._test_autograd_multiple_dispatch(dual_input)


def view_copy(self, device):
    # tests that view_copy derivative formulas are also generated per dispatch key
    # from their respective view ops in derivatives.yaml
    t = torch.randn(2, 2, device=device, requires_grad=True)
    t_ref = t.clone().detach().requires_grad_()
    # _test_autograd_multiple_dispatch_view does a .view(-1) on the input
    t_view = torch._test_autograd_multiple_dispatch_view(t_ref)
    t_view_copy = torch._test_autograd_multiple_dispatch_view_copy(t)
    grad = torch.randn(4, device=device)
    t_view_copy.backward(grad)
    t_view.backward(grad.clone())
    # forward and backward give the same shape + result
    self.assertEqual(t_view_copy, t_view)
    self.assertEqual(t.grad, t_ref.grad)
    # backward results are per-dispatch-key in derivatives.yaml
    if "XPU" in device:
        # gradient registered to AutogradCUDA is grad.reshape_as(self) + 1
        self.assertEqual(t.grad, grad.reshape_as(t) + 1)
    else:
        # Default gradient registered is grad.reshape_as(self)
        self.assertEqual(t.grad, grad.reshape_as(t))


@onlyXPU
def backward_single_threaded(self, device):
    threads_eq = None

    class TestFn(Function):
        @staticmethod
        def forward(ctx, x, self):
            ctx.self = self
            ctx.tid = threading.get_ident()
            return x.clone()

        @staticmethod
        def backward(ctx, gO):
            nonlocal threads_eq
            threads_eq = ctx.tid == threading.get_ident()
            return gO, None
    inp = torch.rand(10, device=device, requires_grad=True)
    with torch.autograd.set_multithreading_enabled(False):
        TestFn.apply(inp, None).sum().backward()
    self.assertTrue(threads_eq)
    TestFn.apply(inp, None).sum().backward()
    self.assertFalse(threads_eq)


@onlyXPU
def backward_tls_stash(self, device):
    local = threading.local()
    local.my_obj = {}
    local.my_obj[10] = 10
    test_self = self
    torch._C._stash_obj_in_tls("my_obj", local.my_obj)

    class TestFn(Function):
        @staticmethod
        def forward(ctx, x, self):
            return x.clone()

        @staticmethod
        def backward(ctx, gO):
            test_self.assertTrue(torch._C._is_key_in_tls("my_obj"))
            test_self.assertTrue(torch._C._get_obj_in_tls("my_obj")[10] == 10)
            torch._C._get_obj_in_tls("my_obj")[10] = 5
            return gO, None
    inp = torch.rand(10, device=device, requires_grad=True)
    TestFn.apply(inp, None).sum().backward()
    self.assertEqual(local.my_obj[10], 5)


@onlyXPU
def pin_memory(self, device):
    x = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
    self.assertEqual(x, x.pin_memory(device))
    self.assertIsNot(x, x.pin_memory(device))
    self.assertTrue(x.pin_memory(device).requires_grad)
    gradcheck(lambda x: x.pin_memory(device), [x])
    gradgradcheck(lambda x: x.pin_memory(device), [x])


def checkpointing_without_reentrant_dataparallel(self):
    """
    Verifies gradient correctness when checkpoint without reentrant autograd
    is used in conjunction with DataParallel.
    """

    class LinearModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 2, bias=False)

        def forward(self, inp):
            return self.linear(inp)

    a = torch.randn(2, 2, requires_grad=True, device="xpu")

    model = LinearModule().to("xpu")

    b = deepcopy(model)(a).sum()
    b.backward()
    b_grad = a.grad

    a.grad = None

    module = torch.nn.DataParallel(deepcopy(model))
    c = checkpoint(module, a, use_reentrant=False).sum()
    c.backward()
    c_grad = a.grad

    self.assertEqual(b_grad, c_grad)


@onlyXPU
def gradcheck_input_output_different_device(self, device):
    x = torch.ones((1,), dtype=torch.double, device="xpu", requires_grad=True)
    gradcheck(lambda x: x.to("cpu"), (x,))
    x = torch.ones((1,), dtype=torch.double, device="cpu", requires_grad=True)
    gradcheck(lambda x: x.to("xpu"), (x,))

def profiler_emit_nvtx(self, device):
    # This test is not intended to ensure correctness of nvtx ranges.
    # That would require something a great deal more complex (you'd have to create a
    # profile in a subprocess, open it, and parse the sql somehow).
    # This test is merely intended to catch if emit_nvtx breaks on construction.
    a = torch.tensor([1, 2, 3], dtype=torch.float32, device=device)
    with torch.xpu.profiler.profile():
        with emit_nvtx():
            a.add(1.0)

def dataparallel_saved_tensors_hooks(self):
    def pack(x):
        warnings.warn("pack")
        return x

    _self = self

    class Model(torch.nn.Module):
        def forward(self, x):
            with warnings.catch_warnings(record=True) as w:
                y = x * x
                if torch.cuda.device_count() >= 2:
                    # DataParallel is calling the forward in different threads
                    # without progating TLS, so hooks should not be called here
                    _self.assertEqual(len(w), 0)
                else:
                    # DataParallel only uses one thread
                    # so hooks should be called here
                    _self.assertGreater(len(w), 0)

    x = torch.ones(5, 5, requires_grad=True)
    model = torch.nn.DataParallel(Model())

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
        model(x)
        with warnings.catch_warnings(record=True) as w:
            y = x * x
            # hooks should be called here
            _self.assertGreater(len(w), 0)


def callback_propagates_errors_from_device_thread(self):
    def callback():
        raise RuntimeError("blah")

    def hook_with_callback(*args):
        torch.autograd.Variable._execution_engine.queue_callback(callback)

    t = torch.tensor([1.0, 2.0], requires_grad=True, device=torch.device("xpu"))
    t.register_hook(hook_with_callback)
    output = t**2
    loss = output.sum()


def checkpointing_non_reentrant_autocast(self, device_type):
    for enabled in [True, False]:

        def foo(x, y, z):
            # torch.mm is on autocast's list of ops that should run in
            # the autocast precision
            x = torch.mm(x, y)
            y = torch.mm(x, z)
            z = torch.mm(z, z)
            expected_dtype = torch.float32 if not enabled else torch.bfloat16
            self.assertEqual(expected_dtype, z.dtype)
            return z

        x = torch.randn(3, 3, requires_grad=True)
        y = torch.randn(3, 3, requires_grad=True)
        z = torch.randn(3, 3, requires_grad=True)
        if device_type == "xpu":
            x = x.xpu()
            y = y.xpu()
            z = z.xpu()

        with torch.autocast(
            enabled=enabled, device_type=device_type, dtype=torch.bfloat16
        ):
            loss = checkpoint(foo, x, y, z, use_reentrant=False)
            loss = loss.sum()

        # Without saving + recasting the autocast type, would raise error in autograd
        # about mismatched dtypes.
        loss.backward()  # triggers recomputation to check it runs in bfloat


def checkpointing_non_reentrant_autocast_gpu(self):
    """
    Test that autocast args/kwargs such as the dtype are preserved during
    non-reentrant checkpoint recomputation on GPU.
    """
    self._test_checkpointing_non_reentrant_autocast(device_type="xpu")


@slowTest
def checkpointing_without_reentrant_memory_savings(self):
    class MyModel(nn.Module):
        def __init__(self, n, use_checkpoint, use_reentrant):
            super().__init__()
            self.n = n
            self.use_checkpoint = use_checkpoint
            self.use_reentrant = use_reentrant
            self.layers = nn.ModuleList()
            for i in range(self.n):
                layer = nn.Sequential(
                    nn.Linear(256, 256), nn.Linear(256, 256), nn.Linear(256, 256)
                )
                self.layers.append(layer)
            # pre-allocate the grad so that increased memory usage is mainly
            # due to activations.
            for layer in self.layers:
                for lin in layer:
                    lin.weight.grad = torch.ones_like(lin.weight)
                    lin.bias.grad = torch.ones_like(lin.bias)

        def forward(self, x):
            for i in range(self.n):
                if not self.use_checkpoint:
                    x = self.layers[i](x)
                else:
                    x = checkpoint(
                        self.layers[i], x, use_reentrant=self.use_reentrant
                    )
            return x
    model_no_checkpoint = MyModel(
        8, use_checkpoint=False, use_reentrant=False
    ).to(device="xpu")
    model_reentrant_checkpoint = MyModel(
        8, use_checkpoint=True, use_reentrant=True
    ).to(device="xpu")
    model_no_reentrant_checkpoint = MyModel(
        8, use_checkpoint=True, use_reentrant=False
    ).to(device="xpu")
    x = torch.randn(100, 256, requires_grad=True, device="xpu")
    torch.xpu.reset_peak_memory_stats()
    loss = model_no_checkpoint(x.clone()).sum()
    loss.backward()
    mem_no_checkpoint = torch.xpu.max_memory_allocated()
    torch.xpu.reset_peak_memory_stats()
    loss = model_reentrant_checkpoint(x.clone()).sum()
    loss.backward()
    mem_reentrant_checkpoint = torch.xpu.max_memory_allocated()
    torch.xpu.reset_peak_memory_stats()
    loss = model_no_reentrant_checkpoint(x.clone()).sum()
    loss.backward()
    mem_no_reentrant_checkpoint = torch.xpu.max_memory_allocated()
    self.assertTrue(mem_reentrant_checkpoint < mem_no_checkpoint)
    self.assertTrue(mem_no_reentrant_checkpoint < mem_no_checkpoint)


def gradcheck_default_device_placement_context(self):
    # During gradcheck with fast_mode=True, we create a random vector on the CPU device using a CPU generator.
    # This test ensures that this still works when the default device is set to something else by the user.
    with torch.device("xpu"):
        x = torch.randn(3, dtype=torch.double, requires_grad=True)

        def func(inp):
            return inp**2.0

        self.assertTrue(gradcheck(func, x, fast_mode=True))


def graph_save_on_cpu_cuda(self):
    def f(x):
        a = x + 1
        return a * a

    # with grad
    a = torch.ones(1, requires_grad=True, device="xpu")
    y = f(a)
    memory_with_grad = torch.xpu.memory_allocated()

    del a
    del y

    # without grad
    a = torch.ones(1, requires_grad=True, device="xpu")
    with torch.no_grad():
        y = f(a)
    memory_without_grad = torch.xpu.memory_allocated()

    self.assertGreater(memory_with_grad, memory_without_grad)

    del a
    del y

    # with hooks
    with torch.autograd.graph.save_on_cpu():
        a = torch.ones(1, requires_grad=True, device="xpu")
        y = f(a)
        memory_with_hooks = torch.xpu.memory_allocated()
        self.assertEqual(memory_with_hooks, memory_without_grad)

def scalar_grad_mixed_device(self):
    x = torch.tensor(1.0, requires_grad=True)
    y = torch.randn(2, 2, device="xpu")
    out = x * y
    out.sum().backward()

def custom_function_propagates_errors_from_device_thread(self):
    class MyFunc(Function):
        @staticmethod
        def forward(ctx, x):
            return x

        @staticmethod
        def backward(ctx, gO):
            raise RuntimeError("blah")
            return gO

    t = torch.tensor([1.0, 2.0], requires_grad=True, device=torch.device("xpu"))
    out = MyFunc.apply(t).sum()

    with self.assertRaisesRegex(RuntimeError, "blah"):
        out.backward()

def flops_and_mem(self):
    # From https://github.com/pytorch/pytorch/pull/126320
    def get_act_mem(f):
        out = f()
        out.backward()
        # Why do one forward and backward?
        start_mem = torch.xpu.memory_stats()["requested_bytes.all.current"]
        out = f()
        cur_mem = torch.xpu.memory_stats()["requested_bytes.all.current"]
        act_mem = (cur_mem - start_mem) / (1024 * 1024)
        out.backward()
        return act_mem

    def get_bw_flops(f):
        # Normalized so that a 512 square matmul returns 1
        f().backward()
        out = f()
        # NB: FlopCounterMode is pushed onto the mode stack before CachedMode, so
        # it will be able to observe whether an op is cached or not.
        with FlopCounterMode(display=False) as mode:
            out.backward()
        return mode.get_total_flops() / (512**3 * 2)

    x = torch.randn(512, 512, requires_grad=True, device="xpu")
    y = torch.randn(512, 512, requires_grad=True, device="xpu")

    def fn(x, y):
        return torch.mm(x.cos(), y).sin().sum()

    def fn_ac(x, y):
        return checkpoint(fn, x, y, use_reentrant=False)

    def fn_sac(x, y):
        context_fn = partial(
            create_selective_checkpoint_contexts,
            [
                torch.ops.aten.mm.default,
            ],
        )
        out = checkpoint(fn, x, y, use_reentrant=False, context_fn=context_fn)
        return out

    def policy_fn(ctx, op, *args, **kwargs):
        if op == torch.ops.aten.mm.default:
            return CheckpointPolicy.MUST_SAVE
        else:
            return CheckpointPolicy.PREFER_RECOMPUTE

    def fn_sac2(x, y):
        context_fn = partial(
            create_selective_checkpoint_contexts,
            policy_fn,
        )
        out = checkpoint(fn, x, y, use_reentrant=False, context_fn=context_fn)
        return out

    def policy_fn_bool(ctx, op, *args, **kwargs):
        return op == torch.ops.aten.mm.default

    def fn_sac3(x, y):
        context_fn = partial(
            create_selective_checkpoint_contexts,
            policy_fn_bool,
        )
        out = checkpoint(fn, x, y, use_reentrant=False, context_fn=context_fn)
        return out

    act_mem_noac = get_act_mem(lambda: fn(x, y))
    bw_flops_noac = get_bw_flops(lambda: fn(x, y))

    self.assertEqual(act_mem_noac, 2.0)
    self.assertEqual(bw_flops_noac, 2.0)

    act_mem_ac = get_act_mem(lambda: fn_ac(x, y))
    bw_flops_ac = get_bw_flops(lambda: fn_ac(x, y))

    self.assertEqual(act_mem_ac, 0.0)
    self.assertEqual(bw_flops_ac, 3.0)

    act_mem_sac = get_act_mem(lambda: fn_sac(x, y))
    bw_flops_sac = get_bw_flops(lambda: fn_sac(x, y))

    self.assertEqual(act_mem_sac, 1.0)
    self.assertEqual(bw_flops_sac, 2.0)

    act_mem_sac2 = get_act_mem(lambda: fn_sac2(x, y))
    bw_flops_sac2 = get_bw_flops(lambda: fn_sac2(x, y))

    self.assertEqual(act_mem_sac2, 1.0)
    self.assertEqual(bw_flops_sac2, 2.0)

    act_mem_sac3 = get_act_mem(lambda: fn_sac3(x, y))
    bw_flops_sac3 = get_bw_flops(lambda: fn_sac3(x, y))

    self.assertEqual(act_mem_sac3, 1.0)
    self.assertEqual(bw_flops_sac3, 2.0)

try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    from .xpu_test_utils import XPUPatchForImport

torch.utils.checkpoint.DefaultDeviceType.set_device_type("xpu")

with XPUPatchForImport(False):
    from test_autograd import (
        TestAutograd,
        TestAutogradForwardModeBatchedGrad,
        TestAutogradForwardMode,
        TestAutogradDeviceType,
        TestAllowMutationOnSaved,
        TestAutogradInferenceMode,
        TestAutogradMultipleDispatch,
        TestMultithreadAutograd,
        TestNestedCheckpoint,
        TestSelectiveActivationCheckpoint,
    )
    from autograd.test_complex import TestAutogradComplex  # noqa: F401
    from autograd.test_functional import TestAutogradFunctional, base_and_logging_tensor  # noqa: F401
    from autograd.test_logging import TestAutogradLogging  # noqa: F401

    @base_and_logging_tensor
    def construct_standard_basis_for_cuda(self, ctors):
        test_cases = [
            (ctors.randn(2), ctors.randn(3, device="xpu")),
            (ctors.randn(3, device="xpu"), ctors.randn(2)),
        ]

        for inputs in test_cases:
            self._test_construct_standard_basis_for(inputs)

    @base_and_logging_tensor
    def jacobian_vectorize_correctness_different_devices(self, ctors):
        def f(x, y):
            return x * y, (x * y).xpu()

        x = ctors.randn(3)
        y = ctors.randn(3)
        self._check_jacobian_vectorize_correctness(f, (x, y))

    TestAutograd.test_checkpointing_without_reentrant_dataparallel = checkpointing_without_reentrant_dataparallel
    TestAutograd.test_callback_propagates_errors_from_device_thread = callback_propagates_errors_from_device_thread
    TestAutograd._test_checkpointing_non_reentrant_autocast = checkpointing_non_reentrant_autocast
    TestAutograd.test_checkpointing_non_reentrant_autocast_gpu = checkpointing_non_reentrant_autocast_gpu
    TestAutograd.test_checkpointing_without_reentrant_memory_savings = checkpointing_without_reentrant_memory_savings
    TestAutograd.test_gradcheck_default_device_placement_context = gradcheck_default_device_placement_context
    TestAutograd.test_graph_save_on_cpu_cuda = graph_save_on_cpu_cuda
    TestAutograd.test_scalar_grad_mixed_device=scalar_grad_mixed_device

    TestAutogradDeviceType.test_gradcheck_input_output_different_device = gradcheck_input_output_different_device
    TestAutogradDeviceType.test_pin_memory = pin_memory
    TestAutogradDeviceType.test_profiler_emit_nvtx = profiler_emit_nvtx
    TestMultithreadAutograd.test_dataparallel_saved_tensors_hooks = dataparallel_saved_tensors_hooks
    TestMultithreadAutograd.test_custom_function_propagates_errors_from_device_thread = custom_function_propagates_errors_from_device_thread
    TestAutogradMultipleDispatch.test_autograd_multiple_dispatch_registrations = autograd_multiple_dispatch_registrations
    TestAutogradMultipleDispatch.test_foward_mode_AD = foward_mode_AD
    TestAutogradMultipleDispatch.test_view_copy = view_copy
    TestAutogradMultipleDispatch.test_backward_single_threaded = backward_single_threaded
    TestAutogradMultipleDispatch.test_backward_tls_stash = backward_tls_stash
    TestSelectiveActivationCheckpoint.test_flops_and_mem = flops_and_mem
    TestAutogradFunctional.test_construct_standard_basis_for_cuda = construct_standard_basis_for_cuda
    TestAutogradFunctional.test_jacobian_vectorize_correctness_different_devices = jacobian_vectorize_correctness_different_devices
instantiate_device_type_tests(TestAutogradDeviceType, globals(), only_for="xpu", allow_xpu=True)

instantiate_device_type_tests(
    TestAutogradMultipleDispatch, globals(), only_for="xpu", allow_xpu=True
)

instantiate_parametrized_tests(TestAutograd)
instantiate_parametrized_tests(TestNestedCheckpoint)
instantiate_parametrized_tests(TestAutogradFunctional)


if __name__ == "__main__":
    run_tests()
