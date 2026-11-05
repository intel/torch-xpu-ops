# Owner(s): ["module: dynamo"]
import functools
import re
import unittest
import weakref
from unittest.mock import patch

import torch
import torch._dynamo.test_case
from torch._dynamo.graph_bytecode_inputs import (
    reset_user_object_tracking,
    store_user_object_weakrefs,
)
from torch._dynamo.testing import extract_graph, remove_trailing_space
from torch.testing._internal.inductor_utils import GPU_TYPE
from torch.testing._internal.triton_utils import requires_gpu_and_triton

requires_multigpu = functools.partial(
    unittest.skipIf,
    torch.get_device_module(GPU_TYPE).device_count() <= 1,
    "requires multiple GPU devices",
)


def remove_file_comment(gm_str: str) -> str:
    return remove_trailing_space(re.sub(r"File.*\n", "\n", gm_str))


def print_graph(graph: torch.fx.GraphModule) -> str:
    return remove_file_comment(graph.print_readable(print_output=False))


class TestStreams(torch._dynamo.test_case.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

    @requires_gpu_and_triton
    def test_stream_weakref(self):
        s = torch.Stream()
        weakref.ref(s)

    @requires_gpu_and_triton
    def test_event_weakref(self):
        e = torch.Event()
        weakref.ref(e)

    @requires_gpu_and_triton
    def test_stream_enter_exit(self):
        def fn(x, y, s1, s2):
            with s1:
                z1 = torch.add(x, y)
            with s2:
                z = torch.add(x, y)
                y = z + 2 + z1

            return y

        inp = (torch.ones(2, 2) + 1, torch.ones(2, 2), torch.Stream(), torch.Stream())
        expected = fn(*inp)
        (
            actual,
            _,
            fw_graphs,
            _,
        ) = extract_graph(fn, *inp)
        self.assertEqual(len(fw_graphs), 1)
        self.assertEqual(expected, actual)
        self.assertExpectedInline(
            print_graph(fw_graphs[0]),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[2, 2]", arg1_1: "f32[2, 2]"):
        # Annotation: {'stream': 1}
        add: "f32[2, 2]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1)

        # Annotation: {'stream': 2}
        add_1: "f32[2, 2]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None

        # Annotation: {'stream': 2}
        add_2: "f32[2, 2]" = torch.ops.aten.add.Tensor(add_1, 2);  add_1 = None
        add_3: "f32[2, 2]" = torch.ops.aten.add.Tensor(add_2, add);  add_2 = add = None
        return (add_3,)
""",
        )

    @requires_gpu_and_triton
    @unittest.skip("Needs graph break support with annotation context")
    def test_stream_context_graph_break(self):
        def fn(x, y):
            s2 = torch.Stream()
            s1 = torch.Stream()
            with s1:
                z1 = torch.add(x, y)
            with s2:
                z = torch.add(x, y)
                y = z + 2 + z1
                torch._dynamo.graph_break()
                y = y + 1

            return y

        inp = (torch.ones(2, 2) + 1, torch.ones(2, 2))
        expected = fn(*inp)
        (
            actual,
            _,
            fw_graphs,
            _,
        ) = extract_graph(fn, *inp)
        self.assertEqual(expected, actual)
        self.assertEqual(len(fw_graphs), 2)
        self.assertExpectedInline(print_graph(fw_graphs[0]), """""")
        self.assertExpectedInline(print_graph(fw_graphs[1]), """""")

    @requires_gpu_and_triton
    def test_stream_input(self):
        def fn(x, y, s):
            z = torch.add(x, y)
            y = z + 2
            return y, s

        inp = (torch.ones(2, 2) + 1, torch.ones(2, 2), torch.Stream(device=GPU_TYPE))
        expected = fn(*inp)
        fn_opt = torch.compile(fn, fullgraph=True)
        actual = fn_opt(*inp)
        self.assertEqual(expected, actual)

    @requires_gpu_and_triton
    def test_local_stream_return(self):
        def fn(x, y):
            s = torch.Stream()
            z = torch.add(x, y)
            y = z + 2
            return y, s

        inp = (torch.ones(2, 2) + 1, torch.ones(2, 2))
        fn_opt = torch.compile(fn, fullgraph=True)
        _, s0 = fn_opt(*inp)
        _, s1 = fn_opt(*inp)
        # Streams will be different values for each invocation
        # so don't check for equality
        self.assertIsInstance(s0, torch.Stream)
        # Stream should be newly allocated on each call
        self.assertNotEqual(s0, s1)

    @requires_gpu_and_triton
    def test_get_current_stream_return(self):
        def fn(x, s):
            with s:
                s0 = torch.accelerator.current_stream()
            return x, s0

        s_inp = torch.Stream(device=GPU_TYPE)
        inp = (torch.ones(2, 2) + 1, s_inp)
        fn_opt = torch.compile(fn, fullgraph=True)
        _, s0 = fn_opt(*inp)
        _, s1 = fn_opt(*inp)
        self.assertEqual(s_inp, s0)
        self.assertEqual(s0, s1)

    @requires_gpu_and_triton
    @requires_multigpu()
    def test_get_current_stream_return_different_device(self):
        def fn(x, s0, s1):
            with s1:
                with s0:
                    s = torch.accelerator.current_stream(torch.device(f"{GPU_TYPE}:1"))
            return s

        s0 = torch.Stream(device=f"{GPU_TYPE}:0")
        s1 = torch.Stream(device=f"{GPU_TYPE}:1")
        inp = (torch.ones(2, 2) + 1, s0, s1)
        fn_opt = torch.compile(fn, fullgraph=True)
        s_act = fn_opt(*inp)
        s_exp = fn(*inp)
        self.assertEqual(s_act, s_exp)

    @requires_gpu_and_triton
    @requires_multigpu()
    def test_get_current_stream_return_no_index(self):
        def fn(x, s0, s1):
            with s1:
                with s0:
                    s = torch.accelerator.current_stream(torch.device(GPU_TYPE))
            return s

        s0 = torch.Stream(device=f"{GPU_TYPE}:0")
        s1 = torch.Stream(device=f"{GPU_TYPE}:1")
        inp = (torch.ones(2, 2) + 1, s0, s1)
        fn_opt = torch.compile(fn, fullgraph=True)
        s_act = fn_opt(*inp)
        s_exp = fn(*inp)
        self.assertEqual(s_act, s_exp)

    @requires_gpu_and_triton
    def test_nested_stream_enter_exit(self):
        def fn(x, y, s0, s1, s2):
            with s1:
                with s2:
                    z1 = torch.add(x, y)
            with s0:
                z0 = torch.add(x, y)
                with s2:
                    y = 2 + z1

            return z0, y

        inp = (
            torch.ones(2, 2) + 1,
            torch.ones(2, 2),
            torch.Stream(),
            torch.Stream(),
            torch.Stream(),
        )
        expected = fn(*inp)
        (
            actual,
            _,
            fw_graphs,
            _,
        ) = extract_graph(fn, *inp)
        self.assertEqual(len(fw_graphs), 1)
        self.assertEqual(expected, actual)
        self.assertExpectedInline(
            print_graph(fw_graphs[0]),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[2, 2]", arg1_1: "f32[2, 2]"):
        # Annotation: {'stream': 2}
        add: "f32[2, 2]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1)

        # Annotation: {'stream': 3}
        add_1: "f32[2, 2]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None

        # Annotation: {'stream': 2}
        add_2: "f32[2, 2]" = torch.ops.aten.add.Tensor(add, 2);  add = None
        return (add_1, add_2)
""",
        )

    @unittest.skip("Needs graph break support with annotation context")
    def test_stream_enter_exit_graph_break(self):
        pass

    @unittest.skip("Needs graph break support with annotation context")
    def test_nested_stream_enter_exit_graph_break(self):
        pass

    @requires_gpu_and_triton
    def test_local_stream_enter_exit(self):
        def fn(x, y):
            s2 = torch.Stream()
            s1 = torch.Stream()
            with s1:
                z1 = torch.add(x, y)
            with s2:
                z = torch.add(x, y)
                y = z + 2 + z1

            return y

        inp = (torch.ones(2, 2) + 1, torch.ones(2, 2))
        expected = fn(*inp)
        (
            actual,
            _,
            fw_graphs,
            _,
        ) = extract_graph(fn, *inp)
        self.assertEqual(len(fw_graphs), 1)
        self.assertEqual(expected, actual)
        self.assertExpectedInline(
            print_graph(fw_graphs[0]),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[2, 2]", arg1_1: "f32[2, 2]"):
        # Annotation: {'stream': 2}
        add: "f32[2, 2]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1)

        # Annotation: {'stream': 1}
        add_1: "f32[2, 2]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None

        # Annotation: {'stream': 1}
        add_2: "f32[2, 2]" = torch.ops.aten.add.Tensor(add_1, 2);  add_1 = None
        add_3: "f32[2, 2]" = torch.ops.aten.add.Tensor(add_2, add);  add_2 = add = None
        return (add_3,)
""",
        )

    @requires_gpu_and_triton
    def test_local_stream_nested_enter_exit(self):
        def fn(x, y):
            s2 = torch.Stream()
            s1 = torch.Stream()
            s0 = torch.Stream()
            with s1:
                with s2:
                    z1 = torch.add(x, y)
            with s0:
                z0 = torch.add(x, y)
                with s2:
                    y = 2 + z1

            return z0, y

        inp = (torch.ones(2, 2) + 1, torch.ones(2, 2))
        expected = fn(*inp)
        (
            actual,
            _,
            fw_graphs,
            _,
        ) = extract_graph(fn, *inp)
        self.assertEqual(len(fw_graphs), 1)
        self.assertEqual(expected, actual)
        self.assertExpectedInline(
            print_graph(fw_graphs[0]),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[2, 2]", arg1_1: "f32[2, 2]"):
        # Annotation: {'stream': 1}
        add: "f32[2, 2]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1)

        # Annotation: {'stream': 3}
        add_1: "f32[2, 2]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None

        # Annotation: {'stream': 1}
        add_2: "f32[2, 2]" = torch.ops.aten.add.Tensor(add, 2);  add = None
        return (add_1, add_2)
""",
        )

    @requires_gpu_and_triton
    @requires_multigpu()
    def test_new_event_api(self) -> None:
        from torch._dynamo.graph_bytecode_inputs import get_external_object_by_index
        from torch._dynamo.variables.streams import new_event

        def event_generation_backend(gm, *args, **kwargs):  # type: ignore[no-untyped-def]
            e0_ind = new_event()
            with torch.Stream(device=f"{GPU_TYPE}:1"):
                get_external_object_by_index(e0_ind).record()
            e1_ind = new_event()
            self.assertNotEqual(e0_ind, e1_ind)
            self.assertNotEqual(
                get_external_object_by_index(e0_ind),
                get_external_object_by_index(e1_ind),
            )
            with gm.graph.inserting_after(next(iter(gm.graph.nodes))):
                gm.graph.call_function(
                    get_external_object_by_index, args=(1,), kwargs={}
                )
            return gm

        @torch.compile(backend=event_generation_backend)
        def fn(x):
            return x + 1

        fn(torch.ones(2, 2, device=f"{GPU_TYPE}:0"))

    @requires_gpu_and_triton
    def test_new_stream_api(self) -> None:
        from torch._dynamo.graph_bytecode_inputs import get_external_object_by_index
        from torch._dynamo.variables.streams import new_stream

        def stream_generation_backend(gm, *args, **kwargs):  # type: ignore[no-untyped-def]
            s0_ind = new_stream()
            s1_ind = new_stream()
            self.assertNotEqual(s0_ind, s1_ind)
            self.assertNotEqual(
                get_external_object_by_index(s0_ind),
                get_external_object_by_index(s1_ind),
            )
            with gm.graph.inserting_after(next(iter(gm.graph.nodes))):
                gm.graph.call_function(
                    get_external_object_by_index, args=(1,), kwargs={}
                )
            return gm

        @torch.compile(backend=stream_generation_backend)
        def fn(x):
            return x + 1

        fn(torch.ones(2, 2, device=f"{GPU_TYPE}:0"))

    @requires_gpu_and_triton
    def test_current_stream_api(self) -> None:
        from torch._dynamo.graph_bytecode_inputs import get_external_object_by_index
        from torch._dynamo.variables.streams import get_current_stream

        cur_stream = torch.accelerator.current_stream()
        s0 = None

        def stream_generation_backend(gm, *args, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal s0
            s0_ind = get_current_stream(torch.device(f"{GPU_TYPE}:0"))
            self.assertEqual(get_external_object_by_index(s0_ind), cur_stream)
            with gm.graph.inserting_after(next(iter(gm.graph.nodes))):
                gm.graph.call_function(
                    get_external_object_by_index, args=(s0_ind,), kwargs={}
                )
                gm.graph.call_function(
                    lambda x: self.assertEqual(
                        cur_stream, get_external_object_by_index(x)
                    ),
                    args=(s0_ind,),
                    kwargs={},
                )
            return gm

        @torch.compile(backend=stream_generation_backend)
        def fn(x):
            return x + 1

        fn(torch.ones(2, 2, device=f"{GPU_TYPE}:0"))

    @requires_gpu_and_triton
    def test_stream_with_mutation(self):
        def fn(x, y):
            s2 = torch.Stream()
            s1 = torch.Stream()
            s0 = torch.Stream()
            with s1:
                with s2:
                    x.add_(y)
            with s0:
                z1 = torch.add(y, y)
                z0 = torch.add(z1, y)
                with s2:
                    y = 2 + z1

            return z0, y

        inp = (torch.ones(2, 2) + 1, torch.ones(2, 2))
        expected = fn(*inp)
        (
            actual,
            _,
            fw_graphs,
            _,
        ) = extract_graph(fn, *inp)
        self.assertEqual(len(fw_graphs), 1)
        self.assertEqual(expected, actual)
        self.assertExpectedInline(
            print_graph(fw_graphs[0]),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[2, 2]", arg1_1: "f32[2, 2]"):
        # Annotation: {'stream': 1}
        add: "f32[2, 2]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1)

        # Annotation: {'stream': 3}
        add_1: "f32[2, 2]" = torch.ops.aten.add.Tensor(arg1_1, arg1_1)

        # Annotation: {'stream': 1}
        add_2: "f32[2, 2]" = torch.ops.aten.add.Tensor(add_1, 2)

        # Annotation: {'stream': 3}
        add_3: "f32[2, 2]" = torch.ops.aten.add.Tensor(add_1, arg1_1);  add_1 = arg1_1 = None

        # Annotation: {'stream': 1}
        copy_: "f32[2, 2]" = torch.ops.aten.copy_.default(arg0_1, add);  arg0_1 = add = copy_ = None
        return (add_3, add_2)
""",
        )

    @requires_gpu_and_triton
    def test_stream_backward_simple(self) -> None:
        def fn(x, y):
            s2 = torch.Stream()
            s0 = torch.Stream()
            with s0:
                y0 = 2 * x + y
            with s2:
                z = 2 * x + y

            return y0, z

        inp = (
            torch.ones(2, 2, requires_grad=True) + 1,
            torch.ones(2, 2, requires_grad=True),
        )
        expected = fn(*inp)
        (
            actual,
            _,
            fw_graphs,
            bw_graphs,
        ) = extract_graph(fn, *inp)
        self.assertEqual(len(fw_graphs), 1)
        self.assertEqual(expected, actual)
        self.assertExpectedInline(
            print_graph(fw_graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[2, 2]", primals_2: "f32[2, 2]"):
        # Annotation: {'stream': 2}
        mul: "f32[2, 2]" = torch.ops.aten.mul.Tensor(primals_1, 2)
        add: "f32[2, 2]" = torch.ops.aten.add.Tensor(mul, primals_2);  mul = None

        # Annotation: {'stream': 1}
        mul_1: "f32[2, 2]" = torch.ops.aten.mul.Tensor(primals_1, 2);  primals_1 = None
        add_1: "f32[2, 2]" = torch.ops.aten.add.Tensor(mul_1, primals_2);  primals_2 = None
        return (add, add_1, mul_1, add_1)
""",
        )

        actual[1].sum().backward()
        self.assertExpectedInline(
            print_graph(bw_graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, mul_1: "f32[2, 2]", add_1: "f32[2, 2]", tangents_1: "f32[2, 2]", tangents_2: "f32[2, 2]"):
        # Annotation: {'stream': 1} Backward of forward node:
        mul_2: "f32[2, 2]" = torch.ops.aten.mul.Tensor(tangents_2, 2)

        # Backward of forward node:
        add_2: "f32[2, 2]" = torch.ops.aten.add.Tensor(tangents_2, tangents_1);  tangents_2 = None

        # Annotation: {'stream': 2} Backward of forward node:
        mul_3: "f32[2, 2]" = torch.ops.aten.mul.Tensor(tangents_1, 2);  tangents_1 = None

        # Annotation: {'stream': 1} Backward of forward node:
        add_3: "f32[2, 2]" = torch.ops.aten.add.Tensor(mul_2, mul_3)

        # No stacktrace found for following nodes
        subgraph_record_event_default = self.subgraph_record_event_default
        control_deps = torch.ops.higher_order.control_deps((mul_1, add_1, mul_2, add_3, add_2), subgraph_record_event_default, add_3, add_2);  mul_1 = add_1 = mul_2 = add_3 = add_2 = subgraph_record_event_default = None

        # Backward of forward node:
        getitem_1: "f32[2, 2]" = control_deps[2]

        # Annotation: {'stream': 1} Backward of forward node:
        getitem: "f32[2, 2]" = control_deps[1];  control_deps = None

        # No stacktrace found for following nodes
        sync_dealloc_default = torch.ops.streams.sync_dealloc.default(3, 2, mul_3);  mul_3 = sync_dealloc_default = None
        return (getitem, getitem_1)

    class subgraph_record_event_default(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]", dep_1: "f32[2, 2]"):
            # No stacktrace found for following nodes
            record_event_default = torch.ops.streams.record_event.default(3, 1)
            return (record_event_default, dep_0, dep_1)
""",
        )

    @requires_gpu_and_triton
    def test_stream_backward_sync(self) -> None:
        def fn(x, y):
            s2 = torch.Stream()
            s0 = torch.Stream()
            with s0:
                y0 = 2 * x + y
            with s2:
                z = 2 * x + y

            return y0, z

        inp = (
            torch.ones(2, 2, device=f"{GPU_TYPE}:0", requires_grad=True) + 1,
            torch.ones(2, 2, device=f"{GPU_TYPE}:0", requires_grad=True),
        )
        expected = fn(*inp)
        (
            actual,
            _,
            fw_graphs,
            bw_graphs,
        ) = extract_graph(fn, *inp)
        self.assertEqual(len(fw_graphs), 1)
        self.assertEqual(expected, actual)
        self.assertExpectedInline(
            print_graph(fw_graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[2, 2]", primals_2: "f32[2, 2]"):
        # Annotation: {'stream': 2}
        mul: "f32[2, 2]" = torch.ops.aten.mul.Tensor(primals_1, 2)
        add: "f32[2, 2]" = torch.ops.aten.add.Tensor(mul, primals_2)

        # Annotation: {'stream': 1}
        mul_1: "f32[2, 2]" = torch.ops.aten.mul.Tensor(primals_1, 2);  primals_1 = None
        add_1: "f32[2, 2]" = torch.ops.aten.add.Tensor(mul_1, primals_2);  primals_2 = None
        return (add, add_1, mul, add, mul_1, add_1)
""",
        )

        actual[1].sum().backward()
        self.assertExpectedInline(
            print_graph(bw_graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, mul: "f32[2, 2]", add: "f32[2, 2]", mul_1: "f32[2, 2]", add_1: "f32[2, 2]", tangents_1: "f32[2, 2]", tangents_2: "f32[2, 2]"):
        # Annotation: {'stream': 1} Backward of forward node:
        mul_2: "f32[2, 2]" = torch.ops.aten.mul.Tensor(tangents_2, 2)

        # Backward of forward node:
        add_2: "f32[2, 2]" = torch.ops.aten.add.Tensor(tangents_2, tangents_1);  tangents_2 = None

        # Annotation: {'stream': 2} Backward of forward node:
        mul_3: "f32[2, 2]" = torch.ops.aten.mul.Tensor(tangents_1, 2);  tangents_1 = None

        # No stacktrace found for following nodes
        subgraph_record_event_default = self.subgraph_record_event_default
        control_deps = torch.ops.higher_order.control_deps((mul, add, mul_3, add_2), subgraph_record_event_default, mul_3, add_2);  mul = add = mul_3 = add_2 = subgraph_record_event_default = None

        # Backward of forward node:
        getitem_1: "f32[2, 2]" = control_deps[2]

        # Annotation: {'stream': 2} Backward of forward node:
        getitem: "f32[2, 2]" = control_deps[1]

        # No stacktrace found for following nodes
        subgraph_wait_event_default = self.subgraph_wait_event_default
        control_deps_1 = torch.ops.higher_order.control_deps((control_deps, mul_1, add_1, mul_2, getitem, getitem_1), subgraph_wait_event_default, mul_2, getitem, getitem_1);  control_deps = mul_1 = add_1 = mul_2 = getitem = getitem_1 = subgraph_wait_event_default = None

        # Backward of forward node:
        getitem_4: "f32[2, 2]" = control_deps_1[3]

        # Annotation: {'stream': 2} Backward of forward node:
        getitem_3: "f32[2, 2]" = control_deps_1[2]

        # Annotation: {'stream': 1} Backward of forward node:
        getitem_2: "f32[2, 2]" = control_deps_1[1]

        # Annotation: {'stream': 1} Backward of forward node:
        add_3: "f32[2, 2]" = torch.ops.aten.add.Tensor(getitem_2, getitem_3)

        # No stacktrace found for following nodes
        subgraph_record_event_default_1 = self.subgraph_record_event_default_1
        control_deps_2 = torch.ops.higher_order.control_deps((add_3, control_deps_1, getitem_2, getitem_3, getitem_4), subgraph_record_event_default_1, add_3, getitem_3, getitem_4);  add_3 = control_deps_1 = getitem_2 = getitem_3 = getitem_4 = subgraph_record_event_default_1 = None

        # Backward of forward node:
        getitem_7: "f32[2, 2]" = control_deps_2[3]

        # Annotation: {'stream': 2} Backward of forward node:
        getitem_6: "f32[2, 2]" = control_deps_2[2]

        # Annotation: {'stream': 1} Backward of forward node:
        getitem_5: "f32[2, 2]" = control_deps_2[1];  control_deps_2 = None

        # No stacktrace found for following nodes
        sync_dealloc_default = torch.ops.streams.sync_dealloc.default(4, 2, getitem_6);  getitem_6 = sync_dealloc_default = None
        return (getitem_5, getitem_7)

    class subgraph_record_event_default(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]", dep_1: "f32[2, 2]"):
            # No stacktrace found for following nodes
            record_event_default = torch.ops.streams.record_event.default(3, 2)
            return (record_event_default, dep_0, dep_1)

    class subgraph_wait_event_default(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]", dep_1: "f32[2, 2]", dep_2: "f32[2, 2]"):
            # No stacktrace found for following nodes
            wait_event_default = torch.ops.streams.wait_event.default(3, 1)
            return (wait_event_default, dep_0, dep_1, dep_2)

    class subgraph_record_event_default_1(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]", dep_1: "f32[2, 2]", dep_2: "f32[2, 2]"):
            # No stacktrace found for following nodes
            record_event_default = torch.ops.streams.record_event.default(4, 1)
            return (record_event_default, dep_0, dep_1, dep_2)
""",
        )

    @requires_gpu_and_triton
    def test_event_tracing(self):
        def fn(x) -> None:
            e = torch.Event()
            e.record()
            x.add_(1)
            return x

        inp = (torch.ones(2, 2, device=GPU_TYPE),)
        (
            _,
            _,
            fw_graphs,
            _,
        ) = extract_graph(fn, *inp)

        self.assertExpectedInline(
            print_graph(fw_graphs[0]),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[2, 2]"):
        #
        record_event = torch.ops.streams.record_event.default(1, 0);  record_event = None

        #
        add: "f32[2, 2]" = torch.ops.aten.add.Tensor(arg0_1, 1)
        copy_: "f32[2, 2]" = torch.ops.aten.copy_.default(arg0_1, add);  arg0_1 = add = None
        return (copy_,)
""",
        )

    @requires_gpu_and_triton
    def test_run_opcheck_fork_join(self):
        from torch._dynamo.variables.streams import fork_stream, join_stream
        from torch.library import opcheck

        original_stream = torch.accelerator.current_stream()
        try:
            s0 = torch.Stream()
            s1 = torch.Stream()
            store_user_object_weakrefs(s0, s1)

            sample_inputs = [
                (0, 1),
                (1, 0),
            ]
            for args in sample_inputs:
                opcheck(fork_stream, args)
                opcheck(join_stream, args)
        finally:
            torch.accelerator.set_stream(original_stream)
            reset_user_object_tracking()

    @requires_gpu_and_triton
    def test_run_opcheck_wait_record(self):
        from torch._dynamo.variables.streams import record_event, wait_event
        from torch.library import opcheck

        original_stream = torch.accelerator.current_stream()
        try:
            s0 = torch.Stream()
            s1 = torch.Stream()
            e0 = torch.Event()
            e1 = torch.Event()
            store_user_object_weakrefs(s0, s1, e0, e1)

            sample_inputs = [
                (2, 0),
                (3, 1),
            ]
            for args in sample_inputs:
                opcheck(wait_event, args)
                opcheck(record_event, args)
        finally:
            torch.accelerator.set_stream(original_stream)
            reset_user_object_tracking()

    @requires_gpu_and_triton
    def test_run_opcheck_wait_record_stream(self):
        from torch._dynamo.variables.streams import wait_stream
        from torch.library import opcheck

        try:
            s0 = torch.Stream()
            s1 = torch.Stream()
            s2 = torch.Stream()
            store_user_object_weakrefs(s0, s1, s2)

            sample_inputs = [
                (0, 1),
                (2, 0),
            ]
            for args in sample_inputs:
                opcheck(wait_stream, args)
        finally:
            reset_user_object_tracking()

    @requires_gpu_and_triton
    def test_record_stream_problem_basic(self):
        # see https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html#torch.Tensor.record_stream
        # for what this tests/solves for
        # We expect there to be a sync_dealloc op added to the graph for y
        # synchronizing the first stream w/ the second stream after the second stream is finished
        def fn(x):
            e = torch.Event()
            with torch.Stream(device=f"{GPU_TYPE}:0"):
                y = torch.ones(2, 2, device=f"{GPU_TYPE}:0")
                e.record()
                z = y * x

            with torch.Stream(device=f"{GPU_TYPE}:0"):
                e.wait()
                z0 = y * 2 * x

            return z0, z

        inp = (torch.ones(2, 2, device=GPU_TYPE, requires_grad=True),)
        (
            actual,
            _,
            fw_graphs,
            bw_graphs,
        ) = extract_graph(fn, *inp)

        actual[1].sum().backward()

        self.assertExpectedInline(
            print_graph(bw_graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, mul: "f32[2, 2]", getitem_1: "f32[2, 2]", mul_1: "f32[2, 2]", mul_2: "f32[2, 2]", tangents_1: "f32[2, 2]", tangents_2: "f32[2, 2]"):
        # Annotation: {'stream': 3} Backward of forward node:
        mul_3: "f32[2, 2]" = torch.ops.aten.mul.Tensor(tangents_1, mul_1);  tangents_1 = None

        # Annotation: {'stream': 2} Backward of forward node:
        mul_4: "f32[2, 2]" = torch.ops.aten.mul.Tensor(tangents_2, getitem_1);  tangents_2 = getitem_1 = None

        # No stacktrace found for following nodes
        subgraph_record_event_default = self.subgraph_record_event_default
        control_deps_2 = torch.ops.higher_order.control_deps((mul, mul_4), subgraph_record_event_default, mul_4);  mul = mul_4 = subgraph_record_event_default = None

        # Annotation: {'stream': 2} Backward of forward node:
        getitem_2: "f32[2, 2]" = control_deps_2[1]

        # No stacktrace found for following nodes
        subgraph_wait_event_default = self.subgraph_wait_event_default
        control_deps_3 = torch.ops.higher_order.control_deps((control_deps_2, mul_1, mul_2, mul_3, getitem_2), subgraph_wait_event_default, mul_3, getitem_2);  control_deps_2 = mul_1 = mul_2 = mul_3 = getitem_2 = subgraph_wait_event_default = None

        # Annotation: {'stream': 2} Backward of forward node:
        getitem_4: "f32[2, 2]" = control_deps_3[2]

        # Annotation: {'stream': 3} Backward of forward node:
        getitem_3: "f32[2, 2]" = control_deps_3[1]

        # Annotation: {'stream': 3} Backward of forward node:
        add: "f32[2, 2]" = torch.ops.aten.add.Tensor(getitem_3, getitem_4)

        # No stacktrace found for following nodes
        subgraph_record_event_default_1 = self.subgraph_record_event_default_1
        control_deps_4 = torch.ops.higher_order.control_deps((add, control_deps_3, getitem_3, getitem_4), subgraph_record_event_default_1, add, getitem_4);  add = control_deps_3 = getitem_3 = getitem_4 = subgraph_record_event_default_1 = None

        # Annotation: {'stream': 2} Backward of forward node:
        getitem_6: "f32[2, 2]" = control_deps_4[2]

        # Annotation: {'stream': 3} Backward of forward node:
        getitem_5: "f32[2, 2]" = control_deps_4[1];  control_deps_4 = None

        # No stacktrace found for following nodes
        sync_dealloc_default = torch.ops.streams.sync_dealloc.default(5, 2, getitem_6);  getitem_6 = sync_dealloc_default = None
        return (getitem_5,)

    class subgraph_record_event_default(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]"):
            # No stacktrace found for following nodes
            record_event_default = torch.ops.streams.record_event.default(4, 2)
            return (record_event_default, dep_0)

    class subgraph_wait_event_default(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]", dep_1: "f32[2, 2]"):
            # No stacktrace found for following nodes
            wait_event_default = torch.ops.streams.wait_event.default(4, 3)
            return (wait_event_default, dep_0, dep_1)

    class subgraph_record_event_default_1(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]", dep_1: "f32[2, 2]"):
            # No stacktrace found for following nodes
            record_event_default = torch.ops.streams.record_event.default(5, 3)
            return (record_event_default, dep_0, dep_1)
""",
        )

    @requires_gpu_and_triton
    def test_record_stream_problem_interleaved(self):
        # see https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html#torch.Tensor.record_stream
        # for what this tests/solves for
        # This will have interleaved computation where y is
        # first allocated on the first stream used on the second stream
        # used on the first stream again then finally used on the last stream
        def fn(x):
            e = torch.Event()
            with torch.Stream(device=f"{GPU_TYPE}:0"):
                y = torch.ones(2, 2, device=f"{GPU_TYPE}:0")
                z = y * x
                e.record()

            with torch.Stream(device=f"{GPU_TYPE}:0"):
                e.wait()
                z0 = y * 2 * z
                e.record()

            with torch.Stream(device=f"{GPU_TYPE}:0"):
                e.wait()
                z1 = y * x * z0
                e.record()

            with torch.Stream(device=f"{GPU_TYPE}:0"):
                e.wait()
                z2 = y * 4 * z1
                e.record()

            e.wait()
            return z, z1, z2

        inp = (torch.ones(2, 2, device=GPU_TYPE, requires_grad=True),)
        (
            actual,
            _,
            fw_graphs,
            bw_graphs,
        ) = extract_graph(fn, *inp)

        actual[1].sum().backward()

        self.assertExpectedInline(
            print_graph(bw_graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, getitem_29: "f32[2, 2]", getitem_28: "f32[2, 2]", getitem_27: "f32[2, 2]", getitem_26: "f32[2, 2]", getitem_25: "f32[2, 2]", tangents_1: "f32[2, 2]", tangents_2: "f32[2, 2]", tangents_3: "f32[2, 2]"):
        # Annotation: {'stream': 5} Backward of forward node:
        mul_7: "f32[2, 2]" = torch.ops.aten.mul.Tensor(tangents_3, getitem_25);  tangents_3 = getitem_25 = None

        # No stacktrace found for following nodes
        subgraph_record_event_default = self.subgraph_record_event_default
        control_deps_8 = torch.ops.higher_order.control_deps((mul_7,), subgraph_record_event_default, mul_7);  mul_7 = subgraph_record_event_default = None

        # Annotation: {'stream': 5} Backward of forward node:
        getitem_30: "f32[2, 2]" = control_deps_8[1]

        # No stacktrace found for following nodes
        subgraph_wait_event_default = self.subgraph_wait_event_default
        control_deps_9 = torch.ops.higher_order.control_deps((control_deps_8, getitem_30), subgraph_wait_event_default, getitem_30);  control_deps_8 = getitem_30 = subgraph_wait_event_default = None

        # Annotation: {'stream': 5} Backward of forward node:
        getitem_31: "f32[2, 2]" = control_deps_9[1]

        # Annotation: {'stream': 4} Backward of forward node:
        add: "f32[2, 2]" = torch.ops.aten.add.Tensor(tangents_2, getitem_31);  tangents_2 = None

        # No stacktrace found for following nodes
        subgraph_record_event_default_4 = self.subgraph_record_event_default_4
        control_deps_10 = torch.ops.higher_order.control_deps((add, control_deps_9, getitem_31), subgraph_record_event_default_4, add, getitem_31);  add = control_deps_9 = getitem_31 = subgraph_record_event_default_4 = None

        # Annotation: {'stream': 5} Backward of forward node:
        getitem_33: "f32[2, 2]" = control_deps_10[2]

        # Annotation: {'stream': 4} Backward of forward node:
        getitem_32: "f32[2, 2]" = control_deps_10[1]

        # No stacktrace found for following nodes
        sync_dealloc_default = torch.ops.streams.sync_dealloc.default(10, 5, getitem_33);  sync_dealloc_default = None

        # Annotation: {'stream': 4} Backward of forward node:
        mul_8: "f32[2, 2]" = torch.ops.aten.mul.Tensor(getitem_32, getitem_26);  getitem_26 = None

        # No stacktrace found for following nodes
        subgraph_record_event_default_1 = self.subgraph_record_event_default_1
        control_deps_11 = torch.ops.higher_order.control_deps((mul_8, control_deps_10, getitem_32, getitem_33), subgraph_record_event_default_1, mul_8, getitem_32);  mul_8 = control_deps_10 = getitem_32 = getitem_33 = subgraph_record_event_default_1 = None

        # Annotation: {'stream': 4} Backward of forward node:
        getitem_35: "f32[2, 2]" = control_deps_11[2]

        # Annotation: {'stream': 4} Backward of forward node:
        getitem_34: "f32[2, 2]" = control_deps_11[1]
        mul_9: "f32[2, 2]" = torch.ops.aten.mul.Tensor(getitem_35, getitem_28);  getitem_28 = None
        mul_10: "f32[2, 2]" = torch.ops.aten.mul.Tensor(mul_9, getitem_29)

        # No stacktrace found for following nodes
        subgraph_wait_event_default_1 = self.subgraph_wait_event_default_1
        control_deps_12 = torch.ops.higher_order.control_deps((control_deps_11, getitem_34, getitem_35), subgraph_wait_event_default_1, getitem_34);  subgraph_wait_event_default_1 = None

        # Annotation: {'stream': 4} Backward of forward node:
        getitem_36: "f32[2, 2]" = control_deps_12[1]

        # Annotation: {'stream': 3} Backward of forward node:
        mul_11: "f32[2, 2]" = torch.ops.aten.mul.Tensor(getitem_36, getitem_27);  getitem_27 = None

        # No stacktrace found for following nodes
        subgraph_record_event_default_5 = self.subgraph_record_event_default_5
        control_deps_13 = torch.ops.higher_order.control_deps((mul_11, control_deps_12, getitem_36), subgraph_record_event_default_5, mul_11, getitem_36);  mul_11 = control_deps_12 = getitem_36 = subgraph_record_event_default_5 = None

        # Annotation: {'stream': 4} Backward of forward node:
        getitem_38: "f32[2, 2]" = control_deps_13[2]

        # Annotation: {'stream': 3} Backward of forward node:
        getitem_37: "f32[2, 2]" = control_deps_13[1]

        # No stacktrace found for following nodes
        sync_dealloc_default_1 = torch.ops.streams.sync_dealloc.default(11, 4, getitem_38);  sync_dealloc_default_1 = None
        subgraph_record_event_default_2 = self.subgraph_record_event_default_2
        control_deps_14 = torch.ops.higher_order.control_deps((control_deps_13, getitem_37, getitem_38), subgraph_record_event_default_2, getitem_37);  control_deps_13 = getitem_37 = getitem_38 = subgraph_record_event_default_2 = None

        # Annotation: {'stream': 3} Backward of forward node:
        getitem_39: "f32[2, 2]" = control_deps_14[1]

        # No stacktrace found for following nodes
        subgraph_wait_event_default_2 = self.subgraph_wait_event_default_2
        control_deps_15 = torch.ops.higher_order.control_deps((control_deps_14, getitem_39), subgraph_wait_event_default_2, getitem_39);  control_deps_14 = getitem_39 = subgraph_wait_event_default_2 = None

        # Annotation: {'stream': 3} Backward of forward node:
        getitem_40: "f32[2, 2]" = control_deps_15[1]

        # Annotation: {'stream': 2} Backward of forward node:
        add_1: "f32[2, 2]" = torch.ops.aten.add.Tensor(tangents_1, getitem_40);  tangents_1 = None

        # No stacktrace found for following nodes
        subgraph_record_event_default_6 = self.subgraph_record_event_default_6
        control_deps_16 = torch.ops.higher_order.control_deps((add_1, control_deps_15, getitem_40), subgraph_record_event_default_6, add_1, getitem_40);  add_1 = control_deps_15 = getitem_40 = subgraph_record_event_default_6 = None

        # Annotation: {'stream': 3} Backward of forward node:
        getitem_42: "f32[2, 2]" = control_deps_16[2]

        # Annotation: {'stream': 2} Backward of forward node:
        getitem_41: "f32[2, 2]" = control_deps_16[1]

        # No stacktrace found for following nodes
        sync_dealloc_default_2 = torch.ops.streams.sync_dealloc.default(12, 3, getitem_42);  sync_dealloc_default_2 = None

        # Annotation: {'stream': 2} Backward of forward node:
        mul_12: "f32[2, 2]" = torch.ops.aten.mul.Tensor(getitem_41, getitem_29);  getitem_29 = None

        # No stacktrace found for following nodes
        subgraph_record_event_default_3 = self.subgraph_record_event_default_3
        control_deps_17 = torch.ops.higher_order.control_deps((mul_12, control_deps_16, getitem_41, getitem_42), subgraph_record_event_default_3, mul_12);  mul_12 = control_deps_16 = getitem_41 = getitem_42 = subgraph_record_event_default_3 = None

        # Annotation: {'stream': 2} Backward of forward node:
        getitem_43: "f32[2, 2]" = control_deps_17[1]

        # No stacktrace found for following nodes
        subgraph_wait_event_default_3 = self.subgraph_wait_event_default_3
        control_deps_18 = torch.ops.higher_order.control_deps((control_deps_17, mul_9, mul_10, control_deps_11, getitem_34, getitem_35, getitem_43), subgraph_wait_event_default_3, mul_10, getitem_43);  control_deps_17 = mul_9 = mul_10 = control_deps_11 = getitem_34 = getitem_35 = getitem_43 = subgraph_wait_event_default_3 = None

        # Annotation: {'stream': 2} Backward of forward node:
        getitem_45: "f32[2, 2]" = control_deps_18[2]

        # Annotation: {'stream': 4} Backward of forward node:
        getitem_44: "f32[2, 2]" = control_deps_18[1]

        # Annotation: {'stream': 4} Backward of forward node:
        add_2: "f32[2, 2]" = torch.ops.aten.add.Tensor(getitem_44, getitem_45)

        # No stacktrace found for following nodes
        subgraph_record_event_default_7 = self.subgraph_record_event_default_7
        control_deps_19 = torch.ops.higher_order.control_deps((add_2, control_deps_18, getitem_44, getitem_45), subgraph_record_event_default_7, add_2, getitem_45);  add_2 = control_deps_18 = getitem_44 = getitem_45 = subgraph_record_event_default_7 = None

        # Annotation: {'stream': 2} Backward of forward node:
        getitem_47: "f32[2, 2]" = control_deps_19[2]

        # Annotation: {'stream': 4} Backward of forward node:
        getitem_46: "f32[2, 2]" = control_deps_19[1];  control_deps_19 = None

        # No stacktrace found for following nodes
        sync_dealloc_default_3 = torch.ops.streams.sync_dealloc.default(13, 2, getitem_47);  getitem_47 = sync_dealloc_default_3 = None
        return (getitem_46,)

    class subgraph_record_event_default(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]"):
            # No stacktrace found for following nodes
            record_event_default = torch.ops.streams.record_event.default(6, 5)
            return (record_event_default, dep_0)

    class subgraph_wait_event_default(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]"):
            # No stacktrace found for following nodes
            wait_event_default = torch.ops.streams.wait_event.default(6, 4)
            return (wait_event_default, dep_0)

    class subgraph_record_event_default_4(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]", dep_1: "f32[2, 2]"):
            # No stacktrace found for following nodes
            record_event_default = torch.ops.streams.record_event.default(10, 4)
            return (record_event_default, dep_0, dep_1)

    class subgraph_record_event_default_1(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]", dep_1: "f32[2, 2]"):
            # No stacktrace found for following nodes
            record_event_default = torch.ops.streams.record_event.default(7, 4)
            return (record_event_default, dep_0, dep_1)

    class subgraph_wait_event_default_1(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]"):
            # No stacktrace found for following nodes
            wait_event_default = torch.ops.streams.wait_event.default(7, 3)
            return (wait_event_default, dep_0)

    class subgraph_record_event_default_5(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]", dep_1: "f32[2, 2]"):
            # No stacktrace found for following nodes
            record_event_default = torch.ops.streams.record_event.default(11, 3)
            return (record_event_default, dep_0, dep_1)

    class subgraph_record_event_default_2(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]"):
            # No stacktrace found for following nodes
            record_event_default = torch.ops.streams.record_event.default(8, 3)
            return (record_event_default, dep_0)

    class subgraph_wait_event_default_2(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]"):
            # No stacktrace found for following nodes
            wait_event_default = torch.ops.streams.wait_event.default(8, 2)
            return (wait_event_default, dep_0)

    class subgraph_record_event_default_6(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]", dep_1: "f32[2, 2]"):
            # No stacktrace found for following nodes
            record_event_default = torch.ops.streams.record_event.default(12, 2)
            return (record_event_default, dep_0, dep_1)

    class subgraph_record_event_default_3(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]"):
            # No stacktrace found for following nodes
            record_event_default = torch.ops.streams.record_event.default(9, 2)
            return (record_event_default, dep_0)

    class subgraph_wait_event_default_3(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]", dep_1: "f32[2, 2]"):
            # No stacktrace found for following nodes
            wait_event_default = torch.ops.streams.wait_event.default(9, 4)
            return (wait_event_default, dep_0, dep_1)

    class subgraph_record_event_default_7(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]", dep_1: "f32[2, 2]"):
            # No stacktrace found for following nodes
            record_event_default = torch.ops.streams.record_event.default(13, 4)
            return (record_event_default, dep_0, dep_1)
""",
        )

    @requires_gpu_and_triton
    def test_epilogue_copy_streams_inference(self):
        def fn(x):
            with torch.Stream(device=f"{GPU_TYPE}:0"):
                with torch.no_grad():
                    x.add_(2)

            return x

        x = torch.ones(2, 2, requires_grad=True, device=f"{GPU_TYPE}:0")

        inp = (x,)
        (
            actual,
            _,
            fw_graphs,
            bw_graphs,
        ) = extract_graph(fn, *inp)

        actual.sum().backward()
        self.assertExpectedInline(
            print_graph(fw_graphs[0]),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[2, 2]"):
        # Annotation: {'stream': 1}
        add: "f32[2, 2]" = torch.ops.aten.add.Tensor(arg0_1, 2)
        copy_: "f32[2, 2]" = torch.ops.aten.copy_.default(arg0_1, add);  arg0_1 = add = None
        return (copy_,)
""",
        )

    @requires_gpu_and_triton
    def test_epilogue_copy_streams_external(self):
        @torch.compile(backend="eager")
        def fn(x):
            with torch.Stream(device=f"{GPU_TYPE}:0"):
                x.mul_(3)
            return x.sin()

        x = torch.ones(2, 2, requires_grad=True, device=f"{GPU_TYPE}:0")
        inp = (x.clone(),)
        with self.assertRaisesRegex(
            RuntimeError,
            "Mutations on inputs with user-specified streams are not yet supported",
        ):
            extract_graph(fn, *inp)

    @requires_gpu_and_triton
    def test_epilogue_copy_stream_tracking(self):
        """
        Test that epilogue copies for mutated inputs use the correct stream.
        This verifies that ViewAndMutationMeta.mutated_inp_stream_indices is
        properly populated and used at runtime.
        Uses a custom autograd.Function where the backward mutates a saved
        tensor on a specific stream.
        """

        class BwMutationWithStream(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, y):
                ctx.save_for_backward(x)
                ctx.s1 = torch.Stream(device=f"{GPU_TYPE}:0")
                ctx.s2 = torch.Stream(device=f"{GPU_TYPE}:0")
                # Do computation on stream s2
                with ctx.s2:
                    result = x * 2 + y
                return result

            @staticmethod
            def backward(ctx, grad_output):
                (x,) = ctx.saved_tensors
                # Mutate saved tensor x on stream s1 in backward
                with ctx.s1:
                    x.mul_(2)
                # Compute gradients on stream s2
                with ctx.s2:
                    grad_x = grad_output * 2
                    grad_y = grad_output.clone()
                return grad_x, grad_y, None, None

        def fn(x, y):
            result = BwMutationWithStream.apply(x, y)
            return result

        x = torch.ones(2, 2, requires_grad=True, device=f"{GPU_TYPE}:0")
        y = torch.ones(2, 2, requires_grad=True, device=f"{GPU_TYPE}:0")
        (
            actual,
            _,
            fw_graphs,
            bw_graphs,
        ) = extract_graph(fn, x.clone(), y.clone())
        self.assertEqual(len(fw_graphs), 1)
        # Forward graph should show computation on stream 1 (s2)
        self.assertExpectedInline(
            print_graph(fw_graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[2, 2]", primals_2: "f32[2, 2]"):
        # Annotation: {'stream': 2}
        mul: "f32[2, 2]" = torch.ops.aten.mul.Tensor(primals_1, 2)
        add: "f32[2, 2]" = torch.ops.aten.add.Tensor(mul, primals_2);  mul = primals_2 = None
        return (add, primals_1)
""",
        )
        # Run backward and check that the epilogue copy uses stream 0 (s1)
        actual.sum().backward()
        # The backward graph should show:
        # 1. Mutation happening on stream 0 (s1)
        # 2. Gradient computation on stream 1 (s2)
        # 3. Epilogue copy for the mutated tensor on stream 0 (s1)
        self.assertExpectedInline(
            print_graph(bw_graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[2, 2]", tangents_1: "f32[2, 2]"):
        # Annotation: {'stream': 2} Backward of forward node:
        mul_2: "f32[2, 2]" = torch.ops.aten.mul.Tensor(tangents_1, 2)

        # Annotation: {'stream': 2} Backward of forward node:
        clone: "f32[2, 2]" = torch.ops.aten.clone.default(tangents_1);  tangents_1 = None

        # Annotation: {'stream': 1} Backward of forward node:
        mul_1: "f32[2, 2]" = torch.ops.aten.mul.Tensor(primals_1, 2)

        # Annotation: {'stream': 1} No stacktrace found for following nodes
        copy_: "f32[2, 2]" = torch.ops.aten.copy_.default(primals_1, mul_1);  primals_1 = mul_1 = copy_ = None
        return (mul_2, clone)
""",
        )

    @requires_gpu_and_triton
    def test_inductor_lowering(self):
        with patch("torch._inductor.config.implicit_fallbacks", False):

            @torch.compile()
            def fn(x):
                e = torch.Event()
                x += x + 1
                e.record()
                return x

            inp = (torch.ones(2, 2, device=GPU_TYPE),)
            fn(*inp)

    def test_is_marked_side_effectful(self):
        self.assertIn(
            torch.ops.streams.fork.default, torch.fx.node._side_effectful_functions
        )
        self.assertIn(
            torch.ops.streams.join.default, torch.fx.node._side_effectful_functions
        )
        self.assertIn(
            torch.ops.streams.wait_event.default,
            torch.fx.node._side_effectful_functions,
        )
        self.assertIn(
            torch.ops.streams.record_event.default,
            torch.fx.node._side_effectful_functions,
        )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
