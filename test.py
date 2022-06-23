import os

import torch
from torch.optim import Adam
import torch.distributed as dist
import torch.multiprocessing as mp

from model.model import ParallelGPT2
from utils import ParallelCrossEntropyLoss


def run(rank, size):

    model = ParallelGPT2(rank=rank,
                         vocab_size=4,
                         max_len=10,
                         d_h=4,
                         head=size,
                         d_ff=16,
                         n_layer=2,
                         pad=0,
                         p=0.1).to(f'cpu')

    loss_fn = ParallelCrossEntropyLoss(rank, size, 4)

    optimizer = Adam(params=model.parameters(),
                     lr=2.5e-4,
                     weight_decay=0.01,
                     eps=1e-8)

    torch.random.manual_seed(40)
    input = torch.randint(0, 4, (2, 10))

    logits = model(input)
    print(f'rank {rank} start compute loss \n')
    loss = loss_fn(logits.view(-1, logits.size(-1)), input.view(-1))
    print(f'rank {rank} finish compute loss \n')
    print(f'rank {rank} start compute gradient \n')
    loss.backward()
    print(f'rank {rank} end compute gradient \n')
    print(f'rank {rank} start optimize weight \n')
    optimizer.step()
    print(f'rank {rank} end optimize weight \n')

def init_process(rank, size, fn):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group('gloo', rank=rank, world_size=size)
    fn(rank, size)

if __name__=="__main__":
    size = 2
    processes = []
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()