import torch
from torch import nn
import torch.distributed as dist


class ParallelCrossEntropyLoss(nn.Module):

    def __init__(self, rank: int, world_size: int, num_class: int):
        """parallel cross entropy loss
        
        Args:
            rank (int): rank
            world_size (int): world size
            num_class (int): number of class
        """
        super().__init__()

        assert num_class % world_size == 0
        sub_class = int(num_class / world_size)
        self.start_idx = int(rank * sub_class)
        self.end_idx = self.start_idx + sub_class - 1

        self.rank = rank

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        """forward propagation
        
        Args:
            logits (torch.Tensor(len, sub_class): sub_logits
            target (torch.Tensor(len): targets
        """

        assert logits.dim() == 2 and target.dim() == 1

        # 1. compute index
        idx = (target >= self.start_idx) & (target <= self.end_idx)
        sub_target = target[idx].clone() - self.start_idx

        # 2. compute predict logits
        predict_logits = logits[idx, sub_target]

        # 3. compute sum of logits
        sum_exp_logits = logits.exp().sum(dim=-1)
        dist.all_reduce(sum_exp_logits)

        loss = (sum_exp_logits[idx].log() - predict_logits).sum(dim=-1)
        dist.all_reduce(loss)

        return loss / target.size(dim=-1)

