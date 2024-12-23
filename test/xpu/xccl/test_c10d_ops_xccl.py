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
from torch.testing._internal.common_distributed import (
    init_multigpu_helper,
    MultiProcContinousTest,
)
from torch.testing._internal.common_utils import (
    skip_but_pass_in_sandcastle_if,
    skipIfRocm,
    TEST_WITH_DEV_DBG_ASAN,
    TEST_XPU,
)

def requires_xccl():
    return skip_but_pass_in_sandcastle_if(
        not c10d.is_xccl_available(),
        "c10d was not compiled with the XCCL backend",
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


