import torch
import intel_extension_for_pytorch
import oneccl_bindings_for_pytorch
import torch.distributed as dist
import os

os.environ['RANK'] = str(os.environ.get('PMI_RANK', 0))
os.environ['WORLD_SIZE'] = str(os.environ.get('PMI_SIZE', 1))
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29800'

dist.init_process_group("ccl")
rank = dist.get_rank()

tensor_size = 2
device = torch.device(f'xpu:{rank}')
output_tensor = torch.zeros(tensor_size, 0, device=device)
if dist.get_rank() == 0:
    t_ones = torch.ones(tensor_size, 0, device=device)
    t_fives = torch.ones(tensor_size, 0,  device=device) * 5
    t_twos = torch.ones(tensor_size, 0, device=device) * 2
    t_three = torch.ones(tensor_size, 0, device=device) * 3
    scatter_list = [t_ones, t_fives, t_twos, t_three]
else:
    scatter_list = None

dist.scatter(output_tensor, scatter_list, src=0)

