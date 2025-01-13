# Owner(s): ["module: intel"]

from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.common_distributed import (
    requires_nccl,
    requires_nccl_version,
    init_multigpu_helper,
    skip_if_lt_x_gpu,
    skip_if_rocm_multiprocess,
    MultiProcessTestCase,
)
import torch
import torch.distributed as c10d
from unittest import mock
import os
import torch.testing._internal.common_utils as common

from torch.testing._internal.common_utils import (
    retry_on_connect_failures,
    run_tests,
    skip_but_pass_in_sandcastle_if,
    parametrize,
    instantiate_parametrized_tests,
)
from torch.testing._internal.common_cuda import TEST_MULTIGPU
import torch.distributed as dist
import threading
from torch.nn.parallel import DistributedDataParallel
import datetime
import pickle
import math

try:
    from .xpu_test_utils import XPUPatchForImport, requires_xccl
except Exception as e:
    from ..xpu_test_utils import XPUPatchForImport, requires_xccl
with XPUPatchForImport(False):
    TEST_CUDA = torch.testing._internal.common_utils.TEST_CUDA
    from test_c10d_nccl import RendezvousEnvTest, TimeoutTest, ProcessGroupNCCLNoGPUTest, ProcessGroupNCCLInitTest
    

    if torch.xpu.is_available:
        ccl_backend = "xccl"
    else:
        ccl_backend = "nccl"

    
    @retry_on_connect_failures
    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_CUDA, "No GPUs available, skipping test")
    def _test_common_errors(self):
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
        
            c10d.init_process_group(backend=ccl_backend, world_size=1)
            self.assertEqual(c10d.get_rank(), 0)
            self.assertEqual(c10d.get_world_size(), 1)
            c10d.destroy_process_group()

        with Env(without(vars, "RANK")):
            self.assertEqual(None, os.environ.get("RANK"))
            with self.assertRaisesRegex(ValueError, "RANK expected"):
                gen = c10d.rendezvous("env://")
                next(gen)
            c10d.init_process_group(backend=ccl_backend, rank=0)
            self.assertEqual(c10d.get_rank(), 0)
            self.assertEqual(c10d.get_world_size(), 1)
            c10d.destroy_process_group()

        with Env(withouts(vars, ["RANK", "WORLD_SIZE"])):
            self.assertEqual(None, os.environ.get("RANK"))
            self.assertEqual(None, os.environ.get("WORLD_SIZE"))
            c10d.init_process_group(backend=ccl_backend, rank=0, world_size=1)
            self.assertEqual(c10d.get_rank(), 0)
            self.assertEqual(c10d.get_world_size(), 1)
            c10d.destroy_process_group()

        with Env(vars):
            c10d.init_process_group(backend=ccl_backend)
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
    RendezvousEnvTest.test_common_errors = _test_common_errors

    def __test_default_store_timeout_nccl(self):
        self._test_default_store_timeout(ccl_backend)
    TimeoutTest.test_default_store_timeout_nccl = __test_default_store_timeout_nccl

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(TEST_CUDA, "GPUs are available, skipping test")
    def _test_init_no_gpus(self):
        store = c10d.FileStore(self.file.name, self.world_size)
        with self.assertRaisesRegex(
            ValueError, "ProcessGroupXCCL is only supported with GPUs, no GPUs found!"
        ):
            c10d.ProcessGroupXCCL(store, self.rank, self.world_size)
    
    ProcessGroupNCCLNoGPUTest.test_init_no_gpus = _test_init_no_gpus

    def _ProcessGroupNCCLInitTest_setUp(self):
        #super().setUp()
        super(ProcessGroupNCCLInitTest, self).setUp()
        self._spawn_processes()

    ProcessGroupNCCLInitTest.device_type = 'xpu'
    ProcessGroupNCCLInitTest.setUp = _ProcessGroupNCCLInitTest_setUp

    #####################################################################
    # def __create_process_group_nccl(self, store, opts, device_id=None):
    #     # create nccl processgroup with opts
    #     c10d.init_process_group(
    #         "xccl",
    #         world_size=self.world_size,
    #         rank=self.rank,
    #         store=store,
    #         pg_options=opts,
    #         device_id=device_id,
    #     )
    #     pg = c10d.distributed_c10d._get_default_group()
    #     return pg

    # def _opts(self, high_priority_stream=False):
    #     opts = c10d.ProcessGroupXCCL.Options()
    #     opts.is_high_priority_stream = high_priority_stream
    #     return opts

    # @property
    # def _rank_to_GPU(self):
    #     # return rank to GPU map
    #     return init_multigpu_helper(self.world_size, "xccl")

    # @requires_nccl()
    # @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 1 GPU")
    # @skip_if_lt_x_gpu(1)
    # def _test_nccl_dist_backend_error(self):
    #     store = c10d.FileStore(self.file_name, self.world_size)
    #     self._create_process_group_xccl(store, self.opts())

    #     # Both rank 0 and 1 will use the same CUDA device resulting in ncclInvalidUsage
    #     with self.assertRaises(dist.DistBackendError) as cm:
    #         dist.broadcast(torch.tensor([1, 2, 3]).xpu(), 0)
    #     self.assertTrue(isinstance(cm.exception, dist.DistError))

    #     self.assertIsInstance(cm.exception, RuntimeError)


    # def _setUp(self):
    #     super(ProcessGroupNCCLGroupTest, self).setUp()
    #     self._spawn_processes()

    # ProcessGroupNCCLGroupTest.device_type = "xpu"
    # ProcessGroupNCCLGroupTest._test_default_store_timeout_nccl = __test_default_store_timeout_nccl
    # ProcessGroupNCCLGroupTest._create_process_group_nccl = __create_process_group_nccl
    # ProcessGroupNCCLGroupTest.opts = _opts
    # ProcessGroupNCCLGroupTest.rank_to_GPU = _rank_to_GPU
    # ProcessGroupNCCLGroupTest.test_nccl_dist_backend_error = _test_nccl_dist_backend_error
    # ProcessGroupNCCLGroupTest.setUp = _setUp

    # def __verify_trace(self, t, include_collectives, timing_enabled, is_json):
    #     ver = t["version"]
    #     self.assertEqual(ver, "2.4")
    #     pg_config = t["pg_config"]
    #     self.assertEqual(len(pg_config), 1)
    #     default_pg_info = pg_config["0"]
    #     self.assertIn("name", default_pg_info)
    #     self.assertIn("desc", default_pg_info)
    #     self.assertIn("ranks", default_pg_info)
    #     pg_status = t["pg_status"]
    #     self.assertEqual(len(pg_status), 1)
    #     self.assertEqual(str(pg_status["0"]["last_enqueued_collective"]), "2")
    #     self.assertEqual(str(pg_status["0"]["last_completed_collective"]), "2")
    #     self.assertEqual(
    #         str(pg_status["0"]["last_started_collective"]),
    #         "2" if timing_enabled else "-1",
    #     )
    #     global_ranks = pg_config["0"]["ranks"]
    #     self.assertEqual(len(json.loads(global_ranks)), self.world_size)
    #     if include_collectives:
    #         self.assertEqual(len(t["entries"]), 2)
    #         t = t["entries"]
    #         last = t[-1]
    #         self.assertEqual(last["process_group"], ("0", "default_pg"))
    #         self.assertEqual(last["state"], "completed")
    #         s = last["time_discovered_started_ns"]
    #         f = last["time_discovered_completed_ns"]
    #         self.assertEqual(last["record_id"], 1)
    #         self.assertIsNotNone(f)
    #         if timing_enabled:
    #             self.assertIsNotNone(s)
    #             self.assertTrue(s <= f)
    #         # we don't collect stack traces in JSON at the moment
    #         if not is_json:
    #             self.assertIn("test_c10d_nccl_xpu.py", str(last["frames"]))
    #         self.assertEqual(last["input_sizes"], ((3, 4),))
    #         self.assertEqual(last["input_dtypes"], ["Float"])
    #         self.assertEqual(last["output_sizes"], ((3, 4),))
    #         self.assertEqual(last["output_dtypes"], ["Float"])
    #         self.assertEqual(last["collective_seq_id"], 2)
    #         self.assertEqual(last["timeout_ms"], 600000)
    #         now = datetime.now()
    #         event_created_time = datetime.fromtimestamp(
    #             last["time_created_ns"] / 1000000000
    #         )
    #         before_test = now - timedelta(minutes=1)
    #         self.assertTrue(before_test < event_created_time < now)
    #         if timing_enabled:
    #             # very loose bounds, measured 0.036 ms on devgpu
    #             self.assertTrue(0 < last["duration_ms"] < 100)
    #         else:
    #             self.assertTrue("duration_ms" not in last)
    #     else:
    #         self.assertTrue("entries" not in t)
    
    # NCCLTraceTest._verify_trace = __verify_trace

    # @requires_nccl()
    # @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    # def _test_long(self):
    #     os.environ["TORCH_NCCL_TRACE_BUFFER_SIZE"] = "10"
    #     if self.rank == self.MAIN_PROCESS_RANK:
    #         return
    #     pg = self._create_process_group_nccl()
    #     device = self.local_device
    #     a = torch.full((3, 4), float(self.rank), device=device)
    #     for _ in range(2):
    #         # test some other primitives to make sure
    #         # their strings are valid
    #         xs = [torch.ones(3, 4, device=device)]
    #         pg.broadcast(xs).wait()
    #         pg.allreduce(xs).wait()
    #         pg.reduce(xs).wait()
    #         ys = [[torch.empty(3, 4, device=device) for _ in range(self.world_size)]]
    #         pg.allgather(ys, xs).wait()
    #         pg.reduce_scatter(xs, ys).wait()
    #         f = pg.allreduce(a)
    #     f.wait()
    #     torch.cuda.synchronize(device=device)
    #     t = pickle.loads(torch._C._distributed_c10d._dump_nccl_trace())
    #     t = t["entries"]
    #     self.assertEqual(len(t), 10)
    #     first = t[0]
    #     last = t[-1]
    #     self.assertEqual(last["profiling_name"], "nccl:all_reduce")
    #     self.assertEqual(last["state"], "completed")
    #     self.assertIn("test_c10d_nccl_xpu.py", str(last["frames"]))
    #     self.assertEqual(last["input_sizes"], ((3, 4),))
    #     self.assertEqual(last["input_dtypes"], ["Float"])
    #     self.assertEqual(last["output_sizes"], ((3, 4),))
    #     self.assertEqual(last["output_dtypes"], ["Float"])
    #     self.assertEqual(last["timeout_ms"], 600000)
    #     self.assertEqual(last["collective_seq_id"] - first["collective_seq_id"], 9)
    #     dist.destroy_process_group()
    # NCCLTraceTest.test_long = _test_long


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

from datetime import timedelta
import time
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
        nGPUs = torch.xpu.device_count()
        visible_devices = range(nGPUs)
        nGPUs_per_process = 1
        if self.world_size > nGPUs:
            nGPUs_per_process = nGPUs // self.world_size
        GPUs = {
            i: list(visible_devices[i * nGPUs_per_process : (i + 1) * nGPUs_per_process])
            for i in range(self.world_size)
        }
        return GPUs

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

    # todo: https://github.com/pytorch/pytorch/blob/c06b5048ba866e2dd39e5da5399fe8261322c7ca/torch/distributed/distributed_c10d.py#L1862 device agnostic
    # @requires_xccl()
    # @skip_but_pass_in_sandcastle_if(not TEST_MULTIXPU, "XCCL test requires 2+ GPUs")
    # def test_set_process_group_desc(self):
    #     device = torch.device(f"xpu:{self.rank}")
    #     pg_default = self._create_process_group_xccl(device_id=device)
    #     self.assertEqual(pg_default.group_desc, "default_pg")
    #     pg_1 = c10d.new_group([0, 1], group_desc="test_purpose")
    #     self.assertEqual(pg_1.group_desc, "test_purpose")
    #     pg_2 = c10d.new_group([0, 1])
    #     self.assertEqual(pg_2.group_desc, "undefined")

    def _test_allreduce_basics(self, fn):
        pg = self._create_process_group_xccl()
        device = torch.device("xpu:" + str(self.rank))
        # Single input tests
        tests = simple_reduce_tests(self.rank, self.world_size)
        for op, input, expected in tests:
            opts = c10d.AllreduceOptions()
            opts.reduceOp = op
            tensor = fn(input.to(device))
            fut = pg.allreduce([tensor], opts).get_future()
            fut.wait()
            result = fut.value()
            self.assertEqual(expected, result[0], exact_dtype=False)

        x = fn(torch.tensor([self.rank + 1.0], device=device))
        fut = pg.allreduce(x).get_future()
        fut.wait()
        result = fut.value()
        self.assertEqual(
            torch.tensor([float(self.world_size * (self.world_size + 1) / 2)]),
            result[0],
        )

    @requires_xccl()
    def test_allreduce_basics(self):
        self._test_allreduce_basics(lambda t: t.clone())


#instantiate_parametrized_tests(ProcessGroupNCCLGroupTest)

if __name__ == "__main__":
    assert (
        not torch.xpu._initialized
    ), "test_distributed must not have initialized XPU context on main process"
    
    run_tests()
