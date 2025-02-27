# Owner(s): ["oncall: distributed"]
# This test file contains positive tests for c10d with XCCL backend.
# During the test, it is expected that ProcessGroup will not be aborted, destroyed or incur fatal error.
# Please be mindful of this when adding tests here.
# If you need to add tests for group creation, abort or destroy, please add tests in test_c10d_xccl.py.

# There are two ways to launch tests in this file:
# 1. Run this file directly with `python test_c10d_ops_xccl.py`
# 2. Use multi-process launcher, e.g. `torchrun --standalone --nproc-per-node 2 test_c10d_ops_xccl.py`

import math
import os
import sys
import tempfile

import torch
import torch.distributed as c10d

if not c10d.is_available() or not c10d.is_xccl_available():
    print("c10d XCCL not available, skipping tests", file=sys.stderr)
    sys.exit(0)

import torch.distributed as dist

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from test_c10d_xccl import init_multigpu_helper, requires_xccl
from torch.testing._internal.common_distributed import MultiProcContinousTest
from torch.testing._internal.common_utils import (
    skip_but_pass_in_sandcastle_if,
    TEST_WITH_DEV_DBG_ASAN,
    TEST_XPU,
)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip ASAN as torch + multiprocessing spawn have known issues", file=sys.stderr
    )
    sys.exit(0)

TEST_MULTIGPU = TEST_XPU and torch.xpu.device_count() >= 2


class ProcessGroupXCCLOpTest(MultiProcContinousTest):
    @classmethod
    def backend_str(cls) -> str:
        return "xccl"

    # @classmethod
    # def opts(cls):
    #     opts = c10d.ProcessGroupXCCL.Options()
    #     return opts

    @property
    def rank_to_GPU(self):
        # return rank to GPU map
        return init_multigpu_helper(self.world_size, "xccl")

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "XCCL test requires 2+ GPUs")
    def test_empty_tensors(self):
        pg = self.pg
        local_device_idx = self.rank_to_GPU[self.rank][0]

        xs = [torch.FloatTensor([]).xpu(local_device_idx)]
        pg.broadcast(xs).wait()
        self.assertEqual(0, xs[0].numel())

        pg.allreduce(xs).wait()
        self.assertEqual(0, xs[0].numel())

        pg.reduce(xs).wait()
        self.assertEqual(0, xs[0].numel())

        ys = [
            [
                torch.FloatTensor([]).xpu(local_device_idx)
                for _ in range(self.world_size)
            ]
        ]
        pg.allgather(ys, xs).wait()
        for y in ys[0]:
            self.assertEqual(0, y.numel())

        ys = [torch.FloatTensor([]).xpu(local_device_idx)]
        xs = [
            [
                torch.FloatTensor([]).xpu(local_device_idx)
                for _ in range(self.world_size)
            ]
        ]
        pg.reduce_scatter(ys, xs).wait()
        self.assertEqual(0, ys[0].numel())

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "XCCL test requires 2+ GPUs")
    def test_broadcast_ops(self):
        pg = self.pg

        def broadcast(xs, rootRank, rootTensor):
            opts = c10d.BroadcastOptions()
            opts.rootRank = rootRank
            opts.rootTensor = rootTensor
            work = pg.broadcast(xs, opts)
            work.wait()
            return xs

        # Every rank is root once
        for i in range(self.world_size):
            # Run with 1 input tensor
            x = torch.tensor([self.rank]).xpu(self.rank_to_GPU[self.rank][0])
            output = broadcast([x], i, 0)
            self.assertEqual(torch.tensor([i]), output[0])

            expected_tensor = torch.empty([i + 1, i + 1]).fill_(i + 1)
            xs = [
                torch.empty([i + 1, i + 1]).fill_(-1).xpu(device=device_idx)
                for device_idx in self.rank_to_GPU[self.rank]
            ]

            # test with multiple input tensors (multiple gpu in one rank)
            for j in range(len(xs)):
                if self.rank == i:
                    xs[j] = expected_tensor.xpu(device=self.rank_to_GPU[self.rank][j])

                broadcast(xs, i, j)

                for tensor in xs:
                    self.assertEqual(tensor, expected_tensor)

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "XCCL test requires 2+ GPUs")
    def test_allreduce_ops(self):
        device_count = torch.xpu.device_count()
        pg = self.pg
        local_device_id = self.rank_to_GPU[self.rank][0]

        def allreduce(tensors, op):
            opts = c10d.AllreduceOptions()
            opts.reduceOp = op
            work = pg.allreduce(tensors, opts)
            work.wait()

        # Sum
        tensors = [torch.tensor([self.rank + 1]).xpu(local_device_id)]

        allreduce(tensors, c10d.ReduceOp.SUM)

        ndev = self.world_size
        self.assertEqual(
            torch.tensor([ndev * (ndev + 1) // 2]),
            tensors[0],
        )

        # Avg
        tensors = [torch.tensor([self.rank + 1.0]).xpu(local_device_id)]

        allreduce(tensors, c10d.ReduceOp.AVG)
        ndev = self.world_size
        self.assertEqual(
            torch.tensor([ndev * (ndev + 1.0) / (2.0 * ndev)]),
            tensors[0],
        )

        # Product
        tensors = [torch.tensor([self.rank + 1]).xpu(local_device_id)]

        allreduce(tensors, c10d.ReduceOp.PRODUCT)
        self.assertEqual(torch.tensor([math.factorial(self.world_size)]), tensors[0])

        # Min
        tensors = [torch.tensor([self.rank + 1]).xpu(local_device_id)]

        allreduce(tensors, c10d.ReduceOp.MIN)
        self.assertEqual(torch.tensor([1]), tensors[0])

        # Max
        tensors = [torch.tensor([self.rank + 1]).xpu(local_device_id)]

        allreduce(tensors, c10d.ReduceOp.MAX)
        self.assertEqual(torch.tensor([self.world_size]), tensors[0])

        for op, err in zip(
            (c10d.ReduceOp.BAND, c10d.ReduceOp.BOR, c10d.ReduceOp.BXOR),
            ("ReduceOp.BAND", "ReduceOp.BOR", "ReduceOp.BXOR"),
        ):
            with self.assertRaisesRegex(ValueError, "Cannot use " + err + " with XCCL"):
                allreduce(tensors, op)

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "XCCL test requires 2+ GPUs")
    def test_alltoall_ops_with_xpufree_race(self):
        pg = self.pg
        opts = c10d.AllToAllOptions()
        local_device = f"xpu:{self.rank_to_GPU[self.rank][0]}"
        torch.xpu.set_device(local_device)
        input = torch.rand(1000, 1000, device=local_device)
        output = torch.rand(1000, 1000, device=local_device)
        race_tensors = []
        # create some tensors to race with alltoall collective
        for _ in range(10):
            tmp = []
            for i in range(5):
                tmp.append(torch.rand(10 ** (3 + i), device=local_device))
            race_tensors.append(tmp)

        for i in range(10):
            race_tensors.pop()
            work = pg.alltoall_base(output, input, [], [], opts)
            # this triggers xpuFree
            torch.xpu.empty_cache()
            work.wait()
        torch.xpu.synchronize(device=local_device)

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "XCCL test requires 2+ GPUs")
    def test_reduce_ops(self):
        pg = self.pg
        local_device_id = self.rank_to_GPU[self.rank][0]

        def reduce(xs, rootRank, rootTensor, op=None):
            opts = c10d.ReduceOptions()
            opts.rootRank = rootRank
            opts.rootTensor = rootTensor
            if op:
                opts.reduceOp = op
            work = pg.reduce(xs, opts)
            work.wait()

        # for every root tensor
        for rt in range(self.world_size):
            tensors = [torch.tensor([self.rank + 1]).xpu(local_device_id)]

            reduce(tensors, rt, 0)

            if self.rank == rt:
                self.assertEqual(
                    torch.tensor([self.world_size * (self.world_size + 1) // 2]),
                    tensors[0],
                )
            else:
                self.assertEqual(
                    torch.tensor([self.rank + 1]),
                    tensors[0],
                )

            for op, err in zip(
                (c10d.ReduceOp.BAND, c10d.ReduceOp.BOR, c10d.ReduceOp.BXOR),
                ("ReduceOp.BAND", "ReduceOp.BOR", "ReduceOp.BXOR"),
            ):
                with self.assertRaisesRegex(
                    ValueError, "Cannot use " + err + " with XCCL"
                ):
                    reduce(tensors, self.rank, rt, op)

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "XCCL test requires 2+ GPUs")
    def test_allgather_ops(self):
        pg = self.pg
        local_device_ids = self.rank_to_GPU[self.rank]

        def allgather(output_ts, input_ts):
            work = pg.allgather(output_ts, input_ts)
            return work.wait()

        tensors = [torch.empty(2, 2).fill_(2).xpu(device=i) for i in local_device_ids]
        output_tensors = []
        expected_output = []

        output_per_gpu = (
            [torch.empty(2, 2).fill_(-1)] * len(local_device_ids) * self.world_size
        )
        expected_per_gpu = (
            [torch.empty(2, 2).fill_(2)] * len(local_device_ids) * self.world_size
        )

        for gpu in local_device_ids:
            output_tensors.append([t.xpu(device=gpu) for t in output_per_gpu])
            expected_output.append([t.xpu(device=gpu) for t in expected_per_gpu])

        result = allgather(output_tensors, tensors)

        # Verification
        self.assertEqual(output_tensors, expected_output)

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "XCCL test requires 2+ GPUs")
    def test_allgather_base_ops(self):
        pg = self.pg
        local_device_id = self.rank_to_GPU[self.rank][0]

        def allgather_base(output_t, input_t):
            work = pg._allgather_base(output_t, input_t)
            work.wait()

        # allgather_base is GPU number agnostic.
        # Each rank contribute one tensor regardless of GPU counts
        tensor = torch.tensor([self.rank]).xpu(local_device_id)
        output_t = torch.empty((self.world_size), dtype=tensor.dtype).xpu(
            local_device_id
        )

        allgather_base(output_t, tensor)

        # Verification
        self.assertEqual(torch.arange(self.world_size), output_t)

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "XCCL test requires 2+ GPUs")
    def test_allgather_base_basics(self):
        pg = self.pg
        local_device_id = self.rank_to_GPU[self.rank][0]

        def allgather_base(output_t, input_t):
            work = pg._allgather_base(output_t, input_t)
            work.wait()

        # anticipate an error
        with self.assertRaisesRegex(
            ValueError,
            "output tensor size must be equal to world_size times input tensor size",
        ):
            tensor = torch.tensor([self.rank]).xpu(local_device_id)
            output_t = torch.empty((self.world_size + 1), dtype=tensor.dtype).xpu(
                local_device_id
            )
            # fails the check because output_t is not correctly sized
            allgather_base(output_t, tensor)

        # anticipate an error
        with self.assertRaisesRegex(
            TypeError, "output tensor must have the same type as input tensor"
        ):
            tensor = torch.tensor([self.rank], dtype=torch.float).xpu(local_device_id)
            output_t = torch.empty((self.world_size + 1), dtype=torch.long).xpu(
                local_device_id
            )
            # fails the check because the dtype is different
            allgather_base(output_t, tensor)

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "XCCL test requires 2+ GPUs")
    def test_gather_ops(self):
        pg = self.pg
        local_device_ids = self.rank_to_GPU[self.rank]
        num_gpus = len(local_device_ids)

        def gather(output_t, input_t, rootRank):
            opts = c10d.GatherOptions()
            opts.rootRank = rootRank
            if rootRank == self.rank:
                work = pg.gather(output_t, input_t, opts)
            else:
                work = pg.gather([], input_t, opts)
            work.wait()

        # init input
        tensors = []
        for device_id in local_device_ids:
            tensors.append(torch.tensor([self.rank]).xpu(device_id))

        # init output
        output_ts = []
        for idx in range(num_gpus):
            gpu_idx = local_device_ids[idx]
            output_ts.append([])
            for rank in range(self.world_size):
                output_ts[idx].append(torch.tensor([-1]).xpu(gpu_idx))

        expected = [[torch.tensor([rank]) for rank in range(self.world_size)]]
        for rank in range(self.world_size):
            gather(output_ts, tensors, rank)
            if rank == self.rank:
                self.assertEqual(expected, output_ts)

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "XCCL test requires 2+ GPUs")
    def test_gather_stress(self):
        pg = self.pg
        local_device_ids = self.rank_to_GPU[self.rank]
        num_gpus = len(local_device_ids)

        def gather(output_t, input_t, rootRank):
            opts = c10d.GatherOptions()
            opts.rootRank = rootRank
            if rootRank == self.rank:
                work = pg.gather(output_t, input_t, opts)
            else:
                work = pg.gather([], input_t, opts)
            work.wait()

        stress_length = 1000

        # init input
        tensors = []
        for i in range(stress_length):
            tensors.append([])
            for device_id in local_device_ids:
                tensors[i].append(torch.tensor([self.rank]).xpu(device_id))

        # init output
        output_ts = []
        for i in range(stress_length):
            output_ts.append([[] for _ in range(num_gpus)])
            for idx, ls in enumerate(output_ts[i]):
                gpu_idx = local_device_ids[idx]
                for _ in range(self.world_size):
                    ls.append(torch.tensor([-1]).xpu(gpu_idx))

        expected = [[torch.tensor([rank]) for rank in range(self.world_size)]]
        for i in range(stress_length):
            for rank in range(self.world_size):
                gather(output_ts[i], tensors[i], rank)
                # Verification
                if rank == self.rank:
                    self.assertEqual(output_ts[i], expected)

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "XCCL test requires 2+ GPUs")
    def test_gather_checks(self):
        pg = self.pg
        device_id = self.rank_to_GPU[self.rank][0]

        # init input
        tensor = torch.tensor([self.rank]).xpu(device_id)

        # init output
        output_ts = []
        for rank in range(self.world_size):
            output_ts.append(torch.tensor([-1]).xpu(device_id))

        with self.assertRaisesRegex(ValueError, "invalid root rank"):
            opts = c10d.GatherOptions()
            opts.rootRank = -1
            pg.gather([output_ts], [tensor], opts)

        with self.assertRaisesRegex(TypeError, "incompatible function arguments"):
            pg.gather([output_ts], [tensor], 0)

        with self.assertRaisesRegex(ValueError, "invalid root rank"):
            opts = c10d.GatherOptions()
            opts.rootRank = self.world_size
            pg.gather([output_ts], [tensor], opts)

        with self.assertRaisesRegex(
            # throws error message from dispatcher
            RuntimeError,
            "There were no tensor arguments to this function",
        ):
            opts = c10d.GatherOptions()
            opts.rootRank = 0
            pg.gather([output_ts], [], opts)

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "XCCL test requires 2+ GPUs")
    def test_scatter_ops(self):
        pg = self.pg
        local_device_ids = self.rank_to_GPU[self.rank]
        num_gpus = len(local_device_ids)

        def scatter(output_t, input_t, rootRank):
            opts = c10d.ScatterOptions()
            opts.rootRank = rootRank
            if rootRank == self.rank:
                work = pg.scatter(output_t, input_t, opts)
            else:
                work = pg.scatter(output_t, [], opts)
            work.wait()

        # init output
        tensors = []
        for device_id in local_device_ids:
            tensors.append(torch.tensor([-1]).xpu(device_id))

        # init input
        scatter_list = []
        for idx in range(num_gpus):
            gpu_idx = local_device_ids[idx]
            scatter_list.append([])
            for rank in range(self.world_size):
                scatter_list[idx].append(torch.tensor([rank]).xpu(gpu_idx))

        # test each rank to scatter
        expected = [torch.tensor([self.rank])]
        for rank in range(self.world_size):
            scatter(tensors, scatter_list, rank)
            self.assertEqual(expected, tensors)

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "XCCL test requires 2+ GPUs")
    def test_scatter_stress(self):
        pg = self.pg
        local_device_ids = self.rank_to_GPU[self.rank]
        num_gpus = len(local_device_ids)

        def scatter(output_t, input_t, rootRank):
            opts = c10d.ScatterOptions()
            opts.rootRank = rootRank
            if rootRank == self.rank:
                work = pg.scatter(output_t, input_t, opts)
            else:
                work = pg.scatter(output_t, [], opts)
            work.wait()

        stress_length = 1000

        # init output
        tensors = []
        for i in range(stress_length):
            tensors.append([])
            for device_id in local_device_ids:
                tensors[i].append(torch.tensor([-1]).xpu(device_id))

        # init input
        scatter_list = []
        for i in range(stress_length):
            scatter_list.append([[] for _ in range(num_gpus)])
            for idx, ls in enumerate(scatter_list[i]):
                gpu_idx = local_device_ids[idx]
                for rank in range(self.world_size):
                    ls.append(torch.tensor([rank]).xpu(gpu_idx))

        # test each rank to scatter
        expected = [torch.tensor([self.rank])]
        for i in range(stress_length):
            for rank in range(self.world_size):
                scatter(tensors[i], scatter_list[i], rank)
                # Verification
                self.assertEqual(tensors[i], expected)

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "XCCL test requires 2+ GPUs")
    def test_scatter_checks(self):
        pg = self.pg
        local_device_ids = self.rank_to_GPU[self.rank]
        num_gpus = len(local_device_ids)

        # init output
        tensors = []
        for device_id in local_device_ids:
            tensors.append(torch.tensor([-1]).xpu(device_id))

        # init input
        scatter_list = []
        for idx in range(num_gpus):
            gpu_idx = local_device_ids[idx]
            scatter_list.append([])
            for rank in range(self.world_size):
                scatter_list[idx].append(torch.tensor([rank]).xpu(gpu_idx))

        with self.assertRaisesRegex(ValueError, "invalid root rank"):
            opts = c10d.ScatterOptions()
            opts.rootRank = -1
            pg.scatter(tensors, scatter_list, opts)

        with self.assertRaisesRegex(TypeError, "incompatible function arguments"):
            pg.scatter(tensors, scatter_list, 0)

        with self.assertRaisesRegex(ValueError, "invalid root rank"):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.world_size
            pg.scatter(tensors, scatter_list, opts)

        with self.assertRaisesRegex(
            # throws error message from dispatcher
            RuntimeError,
            "There were no tensor arguments to this function",
        ):
            opts = c10d.ScatterOptions()
            opts.rootRank = 0
            pg.scatter([], scatter_list, opts)

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "XCCL test requires 2+ GPUs")
    def test_reduce_scatter_base_basics(self):
        pg = self.pg
        local_device_id = self.rank_to_GPU[self.rank][0]

        def reduce_scatter_base(output_t, input_t):
            work = pg._reduce_scatter_base(output_t, input_t)
            work.wait()

        # anticipate an error
        with self.assertRaisesRegex(
            ValueError,
            "input tensor must be the same size as output size times world size",
        ):
            input_t = torch.tensor([self.rank]).xpu(local_device_id)
            output_t = torch.empty((self.world_size + 1), dtype=input_t.dtype).xpu(
                local_device_id
            )
            # fails the check because output_t is not correctly sized
            reduce_scatter_base(output_t, input_t)

        # anticipate an error
        with self.assertRaisesRegex(
            TypeError, "input tensor must be the same type as the output tensor."
        ):
            tensor = torch.tensor([self.rank], dtype=torch.float).xpu(local_device_id)
            output_t = torch.empty((self.world_size + 1), dtype=torch.long).xpu(
                local_device_id
            )
            # fails the check because the dtype is different
            reduce_scatter_base(output_t, tensor)

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "XCCL test requires 2+ GPUs")
    def test_reduce_scatter_ops(self):
        pg = self.pg
        local_device_ids = self.rank_to_GPU[self.rank]
        num_gpus = len(local_device_ids)

        def reduce_scatter(outputs, input_lists, op):
            opts = c10d.ReduceScatterOptions()
            opts.reduceOp = op
            work = pg.reduce_scatter(outputs, input_lists, opts)
            work.wait()

        output = [torch.tensor([0]).xpu(i) for i in local_device_ids]

        #  GPU/rank
        #   0         [1], [2], [3], [4]
        #   1         [2], [3], [4], [5]
        #   2         [3], [4], [5], [6]
        #   3         [4], [5], [6], [7]

        # Sum
        tensor_lists = []
        input_per_gpu = []

        for i in range(self.world_size):
            input_per_gpu.append(torch.tensor([self.rank + i + 1]))

        for gpu in local_device_ids:
            tensor_lists.append([t.xpu(device=gpu) for t in input_per_gpu])

        reduce_scatter(output, tensor_lists, c10d.ReduceOp.SUM)

        for i in range(num_gpus):
            expected = torch.tensor(
                [
                    (1 + self.world_size) * self.world_size // 2
                    + self.world_size * self.rank
                ]
            )

            self.assertEqual(expected, output[i])

        # Min
        reduce_scatter(output, tensor_lists, c10d.ReduceOp.MIN)

        for i in range(num_gpus):
            expected = torch.tensor([self.rank + 1 + i])
            self.assertEqual(expected, output[i])

        # Max
        reduce_scatter(output, tensor_lists, c10d.ReduceOp.MAX)

        for i in range(num_gpus):
            expected = torch.tensor([self.rank + self.world_size + i])
            self.assertEqual(expected, output[i])

        # Product
        reduce_scatter(output, tensor_lists, c10d.ReduceOp.PRODUCT)

        # math package don't have math.perm until python 3.8, so
        # we implement a naive version here.
        def perm(n, k):
            prod_val = n
            for val in range(n - k + 1, n):
                prod_val *= val
            return prod_val

        for i in range(num_gpus):
            prod_val = perm(self.rank + self.world_size, self.world_size)

            expected = torch.tensor([prod_val])
            self.assertEqual(expected, output[i])

        # Test the input params overridden scenarios, aka, when the input is
        # a list and output is just one tensor.
        # Sum
        output_tensor = torch.empty_like(input_per_gpu[0][0]).xpu(self.rank)
        input_list = [tensor[0].xpu(self.rank) for tensor in input_per_gpu]
        pg.reduce_scatter(output_tensor, input_list, c10d.ReduceOp.SUM).wait()
        expected = torch.tensor(
            (1 + self.world_size) * self.world_size // 2 + self.world_size * self.rank
        )
        self.assertEqual(expected, output_tensor)

        # Min
        pg.reduce_scatter(output_tensor, input_list, c10d.ReduceOp.MIN).wait()
        expected = torch.tensor(self.rank + 1)
        self.assertEqual(expected, output_tensor)

        # Max
        pg.reduce_scatter(output_tensor, input_list, c10d.ReduceOp.MAX).wait()
        expected = torch.tensor(self.rank + self.world_size)
        self.assertEqual(expected, output_tensor)

        # Product
        pg.reduce_scatter(output_tensor, input_list, c10d.ReduceOp.PRODUCT).wait()
        prod_val = self.rank + 1
        for k in range(1, self.world_size):
            prod_val = prod_val * (self.rank + 1 + k)
        expected = torch.tensor(prod_val)
        self.assertEqual(expected, output_tensor)

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "XCCL test requires 2+ GPUs")
    def test_reduce_scatter_base_ops(self):
        pg = self.pg
        local_device_id = self.rank_to_GPU[self.rank][0]

        def reduce_scatter_base(output_t, input_t):
            work = pg._reduce_scatter_base(output_t, input_t)
            work.wait()

        # reduce_scatter_base is GPU number agnostic.
        # Each rank contribute one tensor regardless of GPU counts
        output_t = torch.empty([1]).xpu(local_device_id)
        tensor = torch.arange(self.world_size, dtype=output_t.dtype).xpu(
            local_device_id
        )

        reduce_scatter_base(output_t, tensor)

        # Verification
        self.assertEqual(output_t[0], self.rank * self.world_size)

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "XCCL test requires 2+ GPUs")
    def test_barrier(self):
        pg = self.pg
        local_device_ids = self.rank_to_GPU[self.rank]

        def allreduce(tensors):
            opts = c10d.AllreduceOptions()
            work = pg.allreduce(tensors, opts)
            return work

        # Making the collective to operate on
        # 1, 2, 3, 4, .... len(local_device_ids) GPUs
        tensors_list = [[] for _ in range(len(local_device_ids))]

        for i in range(1, len(local_device_ids) + 1):
            for j in range(i):
                tensors_list[i - 1].append(
                    torch.tensor([j + 1]).xpu(local_device_ids[j])
                )

        works = []
        for tensors in tensors_list:
            work = allreduce(tensors)
            works.append(work)

        # Barrier will ensure that all previous work is completed
        pg.barrier().wait()

        for i in range(1, len(local_device_ids) + 1):
            for j in range(i):
                self.assertEqual(
                    torch.tensor([(j + 1) * self.world_size]), tensors_list[i - 1][j]
                )

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "XCCL test requires 2+ GPUs")
    def test_send_recv(self):
        pg = self.pg
        device = self.rank_to_GPU[self.rank][0]

        # Generate the same random tensor
        torch.manual_seed(0)
        send_tensor = torch.rand(10, 10, device=device)
        if self.rank == 0:
            dist.send(send_tensor, 1)
        if self.rank == 1:
            recv_tensor = torch.rand(10, 10, device=device)
            dist.recv(recv_tensor, 0)
            self.assertEqual(send_tensor, recv_tensor)

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "XCCL test requires 2+ GPUs")
    def test_send_recv_complex(self):
        pg = self.pg
        device = self.rank_to_GPU[self.rank][0]

        # Generate the same random tensor
        torch.manual_seed(0)
        send_tensor = torch.rand(10, 10, dtype=torch.cfloat, device=device)
        if self.rank == 0:
            dist.send(send_tensor, 1)
        if self.rank == 1:
            recv_tensor = torch.rand(10, 10, dtype=torch.cfloat, device=device)
            dist.recv(recv_tensor, 0)
            self.assertEqual(send_tensor, recv_tensor)

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "XCCL test requires 2+ GPUs")
    def test_send_recv_object_list(self):
        device = self.rank_to_GPU[self.rank][0]

        val = 99 if self.rank == 0 else None
        object_list = [val] * self.world_size
        if self.rank == 0:
            dist.send_object_list(object_list, 1, device=device)
        if self.rank == 1:
            dist.recv_object_list(object_list, 0, device=device)
            self.assertEqual(object_list[0], 99)


if __name__ == "__main__":
    rank = int(os.getenv("RANK", -1))
    world_size = int(os.getenv("WORLD_SIZE", 2))

    if rank != -1:
        # Launched with torchrun or other multi-proc launchers. Directly run the test.
        ProcessGroupXCCLOpTest.run_rank(rank, world_size)
    else:
        # Launched as a single process. Spawn subprocess to run the tests.
        # Also need a rendezvous file for `init_process_group` purpose.
        rdvz_file = tempfile.NamedTemporaryFile(delete=False).name
        torch.multiprocessing.spawn(
            ProcessGroupXCCLOpTest.run_rank,
            nprocs=world_size,
            args=(world_size, rdvz_file),
        )
