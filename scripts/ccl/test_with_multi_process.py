import torch
import torch.distributed as dist
import os
import torch.multiprocessing as mp
# import oneccl_bindings_for_pytorch 

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29503'
    dist.init_process_group(backend='xccl', rank=rank, world_size=world_size)

def test_collectives(rank, world_size):
    setup(rank, world_size)

    torch.accelerator.set_device_index(rank)
    device = torch.device('xpu', rank)

    tensor = torch.ones(171392, dtype=torch.float32, device=device) * (rank + 1)
    output_tensor = torch.ones(27262976, dtype=torch.float32).to(rank) * rank
    if rank == 0 or rank == 1:
        input_tensor = torch.zeros(33554432, dtype=torch.float32).to(rank)
    else:  
        input_tensor = torch.zeros(25165824, dtype=torch.float32).to(rank)
    outputSplitSizes = [4194304, 4194304, 3145728, 3145728, 3145728, 3145728, 3145728, 3145728]

    b_tensor = torch.tensor([rank], dtype=torch.int32, device=device)

    for _ in range(10):
        for _ in range(2):
            dist.all_reduce(tensor)

        dist.all_to_all_single(output=output_tensor, input=input_tensor, output_split_sizes=outputSplitSizes)

        dist.all_to_all_single(output=input_tensor, input=output_tensor, input_split_sizes=outputSplitSizes)

        dist.broadcast(b_tensor, src=0)

        torch.xpu.synchronize()

    dist.destroy_process_group()

if __name__ == '__main__':
    world_size = 8
    mp.spawn(test_collectives, args=(world_size,), nprocs=world_size, join=True)
