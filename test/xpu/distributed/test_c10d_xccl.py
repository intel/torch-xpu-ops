# Owner(s): ["oncall: distributed"]

import json
import math
import os
import pickle
import sys
import tempfile
import threading
import time
from datetime import datetime, timedelta
from enum import auto, Enum
from unittest import mock

import torch
import torch._C._distributed_c10d
import torch.distributed as c10d

if not c10d.is_available() or not c10d.is_xccl_available():
    print("c10d XCCL not available, skipping tests", file=sys.stderr)
    sys.exit(0)

import torch.distributed as dist
import torch.testing._internal.common_utils as common
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
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


class XCCLTraceTestBase(MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        os.environ["TORCH_FR_BUFFER_SIZE"] = "1000"
        self.tempdir = tempfile.TemporaryDirectory()
        os.environ["TORCH_FR_DUMP_TEMP_FILE"] = self._trace_basename()
        os.environ["TORCH_FR_DEBUG_INFO_PIPE_FILE"] = self._trace_basename()
        self._spawn_processes()

    @classmethod
    def _run(
        cls,
        parent_conn,
        rank: int,
        test_name: str,
        file_name: str,
        parent_pipe,
        **kwargs,
    ) -> None:
        cls.parent = parent_conn
        super()._run(rank, test_name, file_name, parent_pipe)

    @property
    def local_device(self):
        return torch.device("xpu", self.rank_to_GPU[self.rank][0])

    def _join_processes(self, fn):
        # We need to patch sys.exit() as skip_if will use sys.exit() and
        # the exit code from the this process will not be caught.
        with mock.patch("sys.exit"):
            fn()
        super()._join_processes(fn)

    def _spawn_processes(self) -> None:
        proc = torch.multiprocessing.get_context("spawn").Process
        self.children_pipes = []
        parent_pipes = []
        for _ in range(self.world_size):
            parent_conn, child_conn = torch.multiprocessing.Pipe()
            self.children_pipes.append(child_conn)
            parent_pipes.append(parent_conn)
        piter = iter(parent_pipes)

        def wrap(*positional, args, **kwargs):
            args = (next(piter), *args)
            return proc(*positional, args=args, **kwargs)

        self._start_processes(wrap)

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

    def _trace_basename(self):
        # we pass the base to the env, and the dump util will append rank
        return os.path.join(self.tempdir.name, "trace_")

    def _trace_name(self, rank):
        return self._trace_basename() + str(rank)

    def started_or_scheduled(self, timing_enabled=False):
        return "started" if timing_enabled else "scheduled"


class XCCLTraceTest(XCCLTraceTestBase):
    def _verify_trace(self, t, include_collectives, is_json, timing_enabled=False):
        ver = t["version"]
        self.assertEqual(ver, "2.9")
        xccl_version = t["xccl_version"]
        torch_xccl_version = torch._C._distributed_c10d.get_xccl_version()
        self.assertEqual(xccl_version, torch_xccl_version)
        pg_config = t["pg_config"]
        self.assertEqual(len(pg_config), 1)
        default_pg_info = pg_config["0"]
        self.assertIn("name", default_pg_info)
        self.assertIn("desc", default_pg_info)
        self.assertIn("ranks", default_pg_info)
        pg_status = t["pg_status"]
        self.assertEqual(len(pg_status), 1)
        self.assertEqual(str(pg_status["0"]["last_enqueued_collective"]), "2")
        self.assertEqual(str(pg_status["0"]["last_completed_collective"]), "2")
        self.assertEqual(
            str(pg_status["0"]["last_started_collective"]),
            "2" if timing_enabled else "-1",
        )
        global_ranks = pg_config["0"]["ranks"]
        self.assertEqual(len(json.loads(global_ranks)), self.world_size)
        if include_collectives:
            self.assertEqual(len(t["entries"]), 2)
            t = t["entries"]
            last = t[-1]
            self.assertEqual(last["thread_id"], str(threading.current_thread().ident))
            self.assertEqual(last["thread_name"], "fr_test_thread")
            self.assertEqual(last["process_group"], ("0", "default_pg"))
            self.assertEqual(last["state"], "completed")
            s = last["time_discovered_started_ns"]
            f = last["time_discovered_completed_ns"]
            self.assertEqual(last["record_id"], 1)
            self.assertIsNotNone(f)
            if timing_enabled:
                self.assertIsNotNone(s)
                self.assertTrue(s <= f)
            # we don't collect stack traces in JSON at the moment
            if not is_json:
                self.assertIn("test_c10d_xccl.py", str(last["frames"]))
            self.assertEqual(last["input_sizes"], ((3, 4),))
            self.assertEqual(last["input_dtypes"], ["Float"])
            self.assertEqual(last["output_sizes"], ((3, 4),))
            self.assertEqual(last["output_dtypes"], ["Float"])
            self.assertEqual(last["collective_seq_id"], 2)
            self.assertEqual(last["timeout_ms"], 600000)
            now = datetime.now()
            event_created_time = datetime.fromtimestamp(
                last["time_created_ns"] / 1000000000
            )
            before_test = now - timedelta(minutes=1)
            self.assertTrue(before_test < event_created_time < now)
            if timing_enabled:
                # very loose bounds, measured 0.036 ms on devgpu
                self.assertTrue(0 < last["duration_ms"] < 100)
            else:
                self.assertTrue("duration_ms" not in last)
        else:
            self.assertTrue("entries" not in t)

    def load_libpthread_or_libc(self):
        import ctypes.util

        for base in ("pthread", "c"):
            path = ctypes.util.find_library(base)
            if path:
                try:
                    return ctypes.CDLL(path)
                except OSError:
                    continue
        raise RuntimeError("Could not load pthread or libc")

    # Directly set thread name using threading.current_thread().name does not work
    # because we use pthread_getname_np to get the thread’s OS-level name in C++
    def set_thread_name(self, name):
        import ctypes

        lib = self.load_libpthread_or_libc()
        pthread_self = lib.pthread_self
        pthread_self.restype = ctypes.c_void_p
        pthread_setname_np = lib.pthread_setname_np
        pthread_setname_np.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

        # Get current pthread handle
        tid = pthread_self()

        # Set name
        pthread_setname_np(tid, name.encode())

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    @parametrize("include_collectives", [True, False])
    def test_short_pickle(self, include_collectives, timing_enabled=False):
        if self.rank == self.MAIN_PROCESS_RANK:
            return
        pg = self._create_process_group_xccl()
        if timing_enabled:
            pg._enable_collectives_timing()
        device = self.local_device
        self.set_thread_name("fr_test_thread")
        a = torch.full((3, 4), float(self.rank), device=device)
        for _ in range(2):
            f = pg.allreduce(a)
        f.wait()
        torch.xpu.synchronize(device=device)
        # gah ok so now the duration_ms is populated best-effort since it can only happen outside "dump()" api
        time.sleep(1)
        t = pickle.loads(
            torch._C._distributed_c10d._dump_xccl_trace(
                includeCollectives=include_collectives
            )
        )
        self._verify_trace(
            t,
            include_collectives=include_collectives,
            is_json=True,
            timing_enabled=timing_enabled,
        )
        dist.destroy_process_group()

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_dump_pipe(self):
        def open_file_with_timeout(file_path, mode, timeout=1.0):
            start_time = time.time()
            while time.time() - start_time < timeout:
                if os.path.exists(file_path):
                    return open(file_path, mode)
                time.sleep(0.1)
            raise FileNotFoundError

        if self.rank == self.MAIN_PROCESS_RANK:
            for c in self.children_pipes:
                self.assertEqual(c.recv(), "next")

            dump_file = self._trace_name(rank=0)
            pipe_file = dump_file + ".pipe"
            with open_file_with_timeout(pipe_file, "w") as f:
                f.write("1\n")
            with open_file_with_timeout(dump_file, "rb", timeout=10.0) as f:
                self.assertTrue("all_reduce" in str(pickle.load(f)))

            for c in self.children_pipes:
                c.send("next")
            return

        pg = self._create_process_group_xccl()
        device = self.local_device
        a = torch.full((3, 4), float(self.rank), device=device)
        for _ in range(2):
            f = pg.allreduce(a)
        f.wait()
        torch.xpu.synchronize(device=device)
        self.parent.send("next")
        self.parent.recv()

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_long(self):
        os.environ["TORCH_FR_BUFFER_SIZE"] = "10"
        if self.rank == self.MAIN_PROCESS_RANK:
            return
        pg = self._create_process_group_xccl()
        device = self.local_device
        a = torch.full((3, 4), float(self.rank), device=device)
        for _ in range(2):
            # test some other primitives to make sure
            # their strings are valid
            xs = [torch.ones(3, 4, device=device)]
            pg.broadcast(xs).wait()
            pg.allreduce(xs).wait()
            pg.reduce(xs).wait()
            ys = [[torch.empty(3, 4, device=device) for _ in range(self.world_size)]]
            pg.allgather(ys, xs).wait()
            pg.reduce_scatter(xs, ys).wait()
            f = pg.allreduce(a)
        f.wait()
        torch.xpu.synchronize(device=device)
        t = pickle.loads(torch._C._distributed_c10d._dump_xccl_trace())
        t = t["entries"]
        self.assertEqual(len(t), 10)
        first = t[0]
        last = t[-1]
        self.assertEqual(last["profiling_name"], "xccl:all_reduce")
        self.assertEqual(last["state"], "completed")
        self.assertIn("test_c10d_xccl.py", str(last["frames"]))
        self.assertEqual(last["input_sizes"], ((3, 4),))
        self.assertEqual(last["input_dtypes"], ["Float"])
        self.assertEqual(last["output_sizes"], ((3, 4),))
        self.assertEqual(last["output_dtypes"], ["Float"])
        self.assertEqual(last["timeout_ms"], 600000)
        self.assertEqual(last["collective_seq_id"] - first["collective_seq_id"], 9)
        dist.destroy_process_group()

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_barrier_profiling(self):
        os.environ["TORCH_FR_BUFFER_SIZE"] = "10"
        if self.rank == self.MAIN_PROCESS_RANK:
            return
        pg = self._create_process_group_xccl()
        device = self.local_device
        a = torch.full((3, 4), float(self.rank), device=device)
        f = pg.barrier()
        f = pg.allreduce(a)
        f.wait()
        torch.xpu.synchronize(device=device)
        t = pickle.loads(torch._C._distributed_c10d._dump_xccl_trace())
        t = t["entries"]
        self.assertEqual(len(t), 2)
        first = t[0]
        last = t[-1]
        self.assertEqual(first["profiling_name"], "xccl:all_reduce_barrier")
        self.assertEqual(last["profiling_name"], "xccl:all_reduce")
        dist.destroy_process_group()

instantiate_parametrized_tests(XCCLTraceTest)

class SetDeviceMethod(Enum):
    TORCH_XPU_SET = auto()  # torch.xpu.set_device
    COLLECTIVE_ARGUMENT = auto()  # broadcast_object_list(device=)


if __name__ == "__main__":
    run_tests()
