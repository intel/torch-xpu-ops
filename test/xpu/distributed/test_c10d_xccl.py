# Owner(s): ["oncall: distributed"]

import math
import os
import random
import signal
import sys
import time
from datetime import timedelta
from enum import auto, Enum
from unittest import mock

import torch
import torch.distributed as c10d
import torch.distributed._functional_collectives as _functional_collectives

if not c10d.is_available() or not c10d.is_xccl_available():
    print("c10d XCCL not available, skipping tests", file=sys.stderr)
    sys.exit(0)

import torch.distributed as dist
import torch.testing._internal.common_utils as common
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_SANDCASTLE,
    parametrize,
    retry_on_connect_failures,
    run_tests,
    skip_but_pass_in_sandcastle_if,
    TEST_XPU,
    TestCase,
)


def skip_if_lt_x_gpu(x):
    return skip_but_pass_in_sandcastle_if(
        not torch.xpu.device_count() >= x,
        f"atleast {x} GPUs needed",
    )


def requires_xccl():
    return skip_but_pass_in_sandcastle_if(
        not c10d.is_xccl_available(),
        "c10d was not compiled with the XCCL backend",
    )


def init_multigpu_helper(world_size: int, backend: str):
    """Multigpu tests are designed to simulate the multi nodes with multi
    GPUs on each node. Nccl backend requires equal #GPUs in each process.
    On a single node, all visible GPUs are evenly
    divided to subsets, each process only uses a subset.
    """
    nGPUs = torch.xpu.device_count()
    visible_devices = range(nGPUs)

    # If rank is less than or equal to number of available GPU's
    # then each rank can be mapped to corresponding GPU.
    nGPUs_per_process = 1
    if world_size > nGPUs:
        nGPUs_per_process = nGPUs // world_size
    rank_to_GPU = {
        i: list(visible_devices[i * nGPUs_per_process : (i + 1) * nGPUs_per_process])
        for i in range(world_size)
    }
    return rank_to_GPU


def simple_reduce_tests(rank, world_size):
    tests = [
        (
            c10d.ReduceOp.SUM,
            torch.tensor([rank + 1.0]),
            torch.tensor([float(world_size * (world_size + 1) / 2)]),
        ),
        (
            c10d.ReduceOp.PRODUCT,
            torch.tensor([rank + 1.0]),
            torch.tensor([float(math.factorial(world_size))]),
        ),
        (
            c10d.ReduceOp.MIN,
            torch.tensor([rank + 1.0]),
            torch.tensor([1.0]),
        ),
        (
            c10d.ReduceOp.MAX,
            torch.tensor([rank + 1.0]),
            torch.tensor([world_size]),
        ),
    ]

    return tests


TEST_MULTIXPU = torch.xpu.device_count() > 1


class RendezvousEnvTest(TestCase):
    @retry_on_connect_failures
    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(not TEST_XPU, "No GPUs available, skipping test")
    def test_common_errors(self):
        vars = {
            "WORLD_SIZE": "1",
            "RANK": "0",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(common.find_free_port()),
        }

        class Env:
            def __init__(self, vars):
                self.env_patcher = mock.patch.dict(os.environ, vars, clear=True)

            def __enter__(self):
                self.env_patcher.start()

            def __exit__(self, type, value, traceback):
                self.env_patcher.stop()

        def without(d, key):
            d = d.copy()
            d.pop(key)
            return d

        def withouts(d, keys):
            d = d.copy()
            for key in keys:
                d.pop(key)
            return d

        with Env(without(vars, "WORLD_SIZE")):
            self.assertEqual(None, os.environ.get("WORLD_SIZE"))
            with self.assertRaisesRegex(ValueError, "WORLD_SIZE expected"):
                gen = c10d.rendezvous("env://")
                next(gen)
            c10d.init_process_group(backend="xccl", world_size=1)
            self.assertEqual(c10d.get_rank(), 0)
            self.assertEqual(c10d.get_world_size(), 1)
            c10d.destroy_process_group()

        with Env(without(vars, "RANK")):
            self.assertEqual(None, os.environ.get("RANK"))
            with self.assertRaisesRegex(ValueError, "RANK expected"):
                gen = c10d.rendezvous("env://")
                next(gen)
            c10d.init_process_group(backend="xccl", rank=0)
            self.assertEqual(c10d.get_rank(), 0)
            self.assertEqual(c10d.get_world_size(), 1)
            c10d.destroy_process_group()

        with Env(withouts(vars, ["RANK", "WORLD_SIZE"])):
            self.assertEqual(None, os.environ.get("RANK"))
            self.assertEqual(None, os.environ.get("WORLD_SIZE"))
            c10d.init_process_group(backend="xccl", rank=0, world_size=1)
            self.assertEqual(c10d.get_rank(), 0)
            self.assertEqual(c10d.get_world_size(), 1)
            c10d.destroy_process_group()

        with Env(vars):
            c10d.init_process_group(backend="xccl")
            self.assertEqual(c10d.get_rank(), 0)
            self.assertEqual(c10d.get_world_size(), 1)
            c10d.destroy_process_group()

        with Env(without(vars, "MASTER_ADDR")):
            self.assertEqual(None, os.environ.get("MASTER_ADDR"))
            with self.assertRaisesRegex(ValueError, "MASTER_ADDR expected"):
                gen = c10d.rendezvous("env://")
                next(gen)

        with Env(without(vars, "MASTER_PORT")):
            self.assertEqual(None, os.environ.get("MASTER_PORT"))
            with self.assertRaisesRegex(ValueError, "MASTER_PORT expected"):
                gen = c10d.rendezvous("env://")
                next(gen)

        with Env(without(vars, "WORLD_SIZE")):
            self.assertEqual(None, os.environ.get("WORLD_SIZE"))
            gen = c10d.rendezvous(f"env://?world_size={1}")
            _, _, size = next(gen)
            self.assertEqual(size, 1)

        with Env(without(vars, "RANK")):
            self.assertEqual(None, os.environ.get("RANK"))
            gen = c10d.rendezvous(f"env://?rank={0}")
            _, rank, _ = next(gen)
            self.assertEqual(rank, 0)

        with Env(withouts(vars, ["RANK", "WORLD_SIZE"])):
            self.assertEqual(None, os.environ.get("RANK"))
            self.assertEqual(None, os.environ.get("WORLD_SIZE"))
            gen = c10d.rendezvous(f"env://?rank={0}&world_size={1}")
            _, rank, size = next(gen)
            self.assertEqual(rank, 0)
            self.assertEqual(size, 1)


class ProcessGroupXCCLTest(MultiProcessTestCase):
    def _create_process_group_xccl(
        self, timeout=timedelta(seconds=600), device_id=None
    ):
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            "xccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            timeout=timeout,
            device_id=device_id,
        )
        pg = c10d.distributed_c10d._get_default_group()
        return pg

    def setUp(self):
        super().setUp()
        TEST_NAN_ASSERT_RETURN = 0 if IS_SANDCASTLE else -signal.SIGABRT
        self.special_return_code_checks = {
            self.test_nan_assert_float16.__wrapped__: TEST_NAN_ASSERT_RETURN,
            self.test_nan_assert_float32.__wrapped__: TEST_NAN_ASSERT_RETURN,
            self.test_nan_assert_float64.__wrapped__: TEST_NAN_ASSERT_RETURN,
            self.test_nan_assert_bfloat16.__wrapped__: TEST_NAN_ASSERT_RETURN,
            self.test_nan_assert_float8_e4m3fn.__wrapped__: TEST_NAN_ASSERT_RETURN,
            self.test_nan_assert_float8_e5m2.__wrapped__: TEST_NAN_ASSERT_RETURN,
        }
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @property
    def world_size(self):
        return 2

    @property
    def rank_to_GPU(self):
        # return rank to GPU map
        return init_multigpu_helper(self.world_size, "xccl")

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(
        torch.xpu.device_count() < 2, "XCCL test requires 2+ GPUs"
    )
    def test_close_multi_pg_unordered(self):
        pg = self._create_process_group_xccl()
        device = self.rank_to_GPU[self.rank][0]
        t = torch.rand(10, 10, device=device)
        # First allreduce to initialize default PG's communicator.
        pg.allreduce(t).wait()
        new_pg1 = c10d.new_group([0, 1])
        new_pg2 = c10d.new_group([0, 1])
        if self.rank == 0 or self.rank == 1:
            t1 = torch.rand(10, 10, device=device)
            t2 = torch.rand(10, 10, device=device)
            new_pg1.allreduce(t1).wait()
            new_pg2.allreduce(t2).wait()
        if self.rank == 0:
            dist.destroy_process_group(new_pg2)
            # force destruction of pg2 first
            del new_pg2
            dist.destroy_process_group(new_pg1)
            del new_pg1
        if self.rank == 1:
            c10d.destroy_process_group(new_pg1)
            # force destruction of pg1 first
            del new_pg1
            dist.destroy_process_group(new_pg2)
            del new_pg2
        dist.destroy_process_group()

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(
        torch.xpu.device_count() < 2, "XCCL test requires 2+ GPUs"
    )
    def test_file_store_check(self):
        # self.file_name is created using "delete=False"
        # e.g., self.file_name = tempfile.NamedTemporaryFile(delete=False).name
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="xccl", rank=self.rank, world_size=self.world_size, store=store
        )
        pg = dist.distributed_c10d._get_default_group()
        self.assertEqual(pg.rank(), self.rank)
        self.assertEqual(pg.size(), self.world_size)
        # give enough time for check() to be executed multiple times
        time.sleep(2)
        dist.destroy_process_group()

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIXPU, "XCCL test requires 2+ GPUs")
    def test_set_process_group_desc(self):
        device = torch.device(f"xpu:{self.rank}")
        pg_default = self._create_process_group_xccl(device_id=device)
        self.assertEqual(pg_default.group_desc, "default_pg")
        pg_1 = c10d.new_group([0, 1], group_desc="test_purpose")
        self.assertEqual(pg_1.group_desc, "test_purpose")
        pg_2 = c10d.new_group([0, 1])
        self.assertEqual(pg_2.group_desc, "undefined")

    @requires_xccl()
    @parametrize(
        "type",
        [
            torch.float16,
            torch.float32,
            torch.float64,
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ],
    )
    def test_nan_assert(self, type):
        # Expecting a device-side error when NaN is detected
        os.environ["TORCH_XCCL_NAN_CHECK"] = "1"
        pg = self._create_process_group_xccl()
        device = self.rank_to_GPU[self.rank][0]
        # Cover different buffer sizes
        if type == torch.float64:
            size = (1024,)  # 1K elements
        elif type == torch.float32:
            size = (1024, 1024)  # 1M elements
        elif type == torch.float16:
            size = (1024, 1024, 1024)  # 1G elements
        else:
            size = (1,)  # 1 element

        # Note: currently we cannot fill values into a FP8 tensor, thus we
        # create the NaN tensor in float32 type and cast it to FP8
        if type == torch.float8_e4m3fn or type == torch.float8_e5m2:
            init_type = torch.float32
        else:
            init_type = type

        nan_tensor = torch.zeros(*size, dtype=init_type, device=device)
        # randomly pick an nan element
        index = tuple([random.randrange(size[i]) for i in range(len(size))])
        nan_tensor[index] = float("nan")
        if init_type != type:
            # Now cast to the targeted dtype
            nan_tensor = nan_tensor.to(type)

        output = torch.empty(self.world_size, *size, dtype=type, device=device)

        # # confirm enable/disable flag works
        # backend._set_enable_nan_check(False)
        # # Note: using all-gather here bc some NCCL/SM version does not support
        # # FP8 reduction
        # # temporarily skip due to https://github.com/pytorch/pytorch/issues/153479
        # # pg._allgather_base(output, nan_tensor)

        # backend._set_enable_nan_check(True)
        try:
            pg._allgather_base(output, nan_tensor)
        except Exception:
            sys.exit(signal.SIGABRT)

        dist.destroy_process_group()

        # reset env
        os.environ["TORCH_XCCL_NAN_CHECK"] = "0"


class CommTest(MultiProcessTestCase):
    @property
    def device(self):
        return f"xpu:{self.rank}"

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @property
    def world_size(self) -> int:
        return 2

    def _test_broadcast_coalesced(self, process_group, device, root_rank):
        half = torch.float16

        # No support for float16 for CPU tensors
        if device == torch.device("cpu"):
            half = torch.float32

        target = torch.arange(60, dtype=half, device=device).chunk(5)
        target += torch.arange(60, dtype=torch.float32, device=device).chunk(5)
        target += torch.arange(60, dtype=half, device=device).chunk(5)
        target += torch.arange(60, dtype=torch.float64, device=device).chunk(5)
        target += torch.arange(60, dtype=half, device=device).chunk(5)
        target += torch.arange(60, dtype=torch.float32, device=device).chunk(5)

        # The tensors to pass to broadcast are identical to the target
        # only on the process that is the root of the broadcast.
        if self.rank == root_rank:
            tensors = [tensor.clone() for tensor in target]
        else:
            tensors = [torch.zeros_like(tensor) for tensor in target]

        if self.rank != root_rank:
            self.assertNotEqual(tensors, target)

        c10d._broadcast_coalesced(
            process_group, tensors, buffer_size=256, src=root_rank
        )

        if self.rank != root_rank:
            self.assertEqual(tensors, target)

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_broadcast_coalesced_xccl(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="xccl", store=store, rank=self.rank, world_size=self.world_size
        )
        process_group = c10d.distributed_c10d._get_default_group()
        device = torch.device("xpu:%d" % self.rank)
        ranks = [0, 1]
        for root_rank in ranks:
            self._test_broadcast_coalesced(process_group, device, root_rank)

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_all_reduce_coalesced_xccl(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="xccl", store=store, rank=self.rank, world_size=self.world_size
        )
        process_group = c10d.distributed_c10d._get_default_group()
        device = torch.device("xpu:%d" % self.rank)
        tensors = [
            torch.full((60 + i,), self.rank + 1 + i, device=device, dtype=torch.float)
            for i in range(5)
        ]
        torch.distributed.all_reduce_coalesced(tensors, group=process_group)
        for i, t in enumerate(tensors):
            self.assertEqual(
                t,
                torch.full_like(
                    t, self.world_size * (i + (self.world_size + 1.0) / 2.0)
                ),
            )

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_all_reduce_coalesced_manager_xccl(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="xccl", store=store, rank=self.rank, world_size=self.world_size
        )
        process_group = c10d.distributed_c10d._get_default_group()
        device = torch.device("xpu:%d" % self.rank)
        tensors = [
            torch.full((60 + i,), self.rank + 1 + i, device=device, dtype=torch.float)
            for i in range(5)
        ]
        with torch.distributed._coalescing_manager(
            group=process_group, device=device, async_ops=True
        ) as cm:
            for tensor in tensors:
                torch.distributed.all_reduce(tensor)
        self.assertEqual(len(cm.works), 1)
        cm.wait()
        for i, t in enumerate(tensors):
            self.assertEqual(
                t,
                torch.full_like(
                    t, self.world_size * (i + (self.world_size + 1.0) / 2.0)
                ),
            )

    @requires_xccl()
    @skip_if_lt_x_gpu(4)
    def test_xccl_barrier(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="xccl", rank=self.rank, world_size=self.world_size, store=store
        )

        t = torch.tensor([self.rank + 1] * 10).xpu(2 * self.rank)
        c10d.all_reduce(t)
        expected_tensor = torch.tensor([3] * 10).xpu(2 * self.rank)
        self.assertEqual(expected_tensor, t)

        # Test with new_group
        pg = c10d.new_group([0, 1])
        t = torch.tensor([self.rank + 1] * 10).xpu(2 * self.rank)
        pg.allreduce(t).wait()
        self.assertEqual(expected_tensor, t)

        pg = c10d.new_group([0])
        if self.rank == 0:
            t = torch.tensor([self.rank + 1] * 10).xpu(2 * self.rank)
            expected_tensor = torch.tensor([self.rank + 1] * 10).xpu(2 * self.rank)
            pg.allreduce(t).wait()
            self.assertEqual(expected_tensor, t)

        pg = c10d.new_group([1])
        if self.rank == 1:
            t = torch.tensor([self.rank + 1] * 10).xpu(2 * self.rank)
            expected_tensor = torch.tensor([self.rank + 1] * 10).xpu(2 * self.rank)
            pg.allreduce(t).wait()
            self.assertEqual(expected_tensor, t)

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_xccl_barrier_device_ids(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="xccl", rank=self.rank, world_size=self.world_size, store=store
        )

        c10d.barrier(device_ids=[self.rank])

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_reduce_scatter_base_k(self):
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            "xccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        output_tensor = torch.zeros(2, dtype=torch.int64).to(self.rank)
        input_tensors = torch.arange(self.world_size * 2, dtype=torch.int64).to(
            self.rank
        )
        input_tensors = torch.reshape(input_tensors, (self.world_size, 2))
        dist.reduce_scatter_tensor(output_tensor, input_tensors)
        self.assertEqual(output_tensor, input_tensors[self.rank] * self.world_size)

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_reduce_scatter_tensor_coalesced(self):
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            "xccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        output_tensors = torch.zeros(2, 2).to(self.rank)
        input_tensors = [torch.ones(2, 2).to(self.rank) for _ in range(self.world_size)]
        with dist._coalescing_manager():
            for i in range(self.world_size):
                dist.reduce_scatter_tensor(output_tensors[i], input_tensors[i])
        self.assertEqual(output_tensors, input_tensors[self.rank] * self.world_size)

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    # The difference between this case and `test_send_recv` is that `test_send_recv` uses a previously created process group,
    # whereas this case performs point-to-point operations immediately after creating the process group.
    def test_single_p2p(self):
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            "xccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        torch.manual_seed(0)
        send_tensor = torch.rand(10, 10).to(self.rank)
        if self.rank == 0:
            dist.send(send_tensor, 1)
        if self.rank == 1:
            recv_tensor = torch.rand(10, 10).to(self.rank)
            dist.recv(recv_tensor, 0)
            self.assertEqual(send_tensor, recv_tensor)

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_tensor_dtype_complex(self):
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            "xccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        tensor = torch.rand(2, device=self.device)
        tensor_c = torch.view_as_complex(tensor)
        tensor_list = [
            torch.rand(2, device=self.device) for _ in range(self.world_size)
        ]
        tensor_list_c = list(tensor_list)
        tensor_list_c[1] = torch.view_as_complex(tensor_list_c[1])

        dist.all_gather(tensor_list, tensor)
        dist.all_gather(tensor_list, tensor_c)
        dist.all_gather(tensor_list_c, tensor)
        dist.all_gather(tensor_list_c, tensor_c)

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_all_gather_into_tensor(self):
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            "xccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        device = "xpu"
        for dtype in [torch.float32, torch.float8_e4m3fn, torch.float8_e5m2]:
            tensor = torch.randn(12, 12, device=torch.device(device)).to(dtype)
            output_tensor = torch.zeros(
                self.world_size * 12, 12, device=torch.device(device)
            ).to(dtype)
            dist.all_gather_into_tensor(output_tensor, tensor)
            for i in range(self.world_size):
                start = i * 12
                end = (i + 1) * 12
                self.assertEqual(
                    output_tensor[start:end].view(torch.float32),
                    tensor.view(torch.float32),
                )

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_unwaited(self) -> None:
        # Verify that the process can terminate gracefully
        # even with unwaited tensors
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="xccl", rank=self.rank, world_size=self.world_size, store=store
        )

        # Case 1: Run collectives under context manager, and don't call wait on them.
        with _functional_collectives.allow_inflight_collective_as_graph_input_ctx():
            self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)
            input = torch.full(
                (10240, 10240), float(self.rank), device=f"xpu:{self.rank}"
            )
            dist.all_reduce(input, op=dist.ReduceOp.SUM, async_op=True)
            # Non-functional collectives run under the context manager is registered in the work registry.
            self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 1)
            # Running another collective on the same tensor should still work
            dist.all_reduce(input, op=dist.ReduceOp.SUM, async_op=True)
            self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 2)

        # Case 2: Run collectives not under context manager, and don't call wait on them.
        # NOTE: Here we intentionally test memory-stressed case.
        self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 2)
        for _ in range(50000):
            input = torch.full(
                (1024, 1024), float(self.rank), device=f"xpu:{self.rank}"
            )
            dist.all_reduce(input, op=dist.ReduceOp.SUM, async_op=True)
        # Work registry size is unchanged, since non-functional collectives not run under
        # the context manager is not registered in the work registry.
        self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 2)

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_wait_tensor(self) -> None:
        # Verify that c10d_functional.wait_tensor() can be invoked on
        # output tensor of non-functional collective
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="xccl", rank=self.rank, world_size=self.world_size, store=store
        )

        # Case 1: under context manager (i.e. work is registered in registry)
        with _functional_collectives.allow_inflight_collective_as_graph_input_ctx():
            input1 = torch.full((10, 10), float(self.rank), device=f"xpu:{self.rank}")
            self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)
            dist.all_reduce(input1, op=dist.ReduceOp.SUM, async_op=True)
            self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 1)
            torch.ops.c10d_functional.wait_tensor(input1)
            self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)

            input2 = torch.full((10, 10), float(self.rank), device=f"xpu:{self.rank}")
            self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)
            work = dist.all_reduce(input2, op=dist.ReduceOp.SUM, async_op=True)
            self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 1)
            work.wait()
            self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)
            self.assertEqual(input1, input2)

        # Case 2: not under context manager (i.e. work is not registered in registry)
        input1 = torch.full((10, 10), float(self.rank), device=f"xpu:{self.rank}")
        self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)
        dist.all_reduce(input1, op=dist.ReduceOp.SUM, async_op=True)
        self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)
        # this does not take effect, since the underlying wait_tensor() logic would not
        # be able to find the corresponding work object (because it's not registered in registry)
        torch.ops.c10d_functional.wait_tensor(input1)
        self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)

        input2 = torch.full((10, 10), float(self.rank), device=f"xpu:{self.rank}")
        self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)
        work = dist.all_reduce(input2, op=dist.ReduceOp.SUM, async_op=True)
        self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)
        work.wait()
        self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)
        self.assertEqual(input1, input2)


instantiate_parametrized_tests(ProcessGroupXCCLTest)


class SetDeviceMethod(Enum):
    TORCH_XPU_SET = auto()  # torch.xpu.set_device
    COLLECTIVE_ARGUMENT = auto()  # broadcast_object_list(device=)


if __name__ == "__main__":
    run_tests()
