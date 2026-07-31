import os

import torch
import torch.distributed as dist


dist.init_process_group("nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
torch.cuda.set_device(rank)
value = torch.tensor([rank + 1.0], device="cuda")
dist.all_reduce(value)
expected = world_size * (world_size + 1) / 2
assert value.item() == expected, (value.item(), expected)
if rank == 0:
    print(f"NCCL all-reduce across {world_size} GPUs: OK")
dist.destroy_process_group()
