import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from model.sub_layer import ParallelFeedForward, ParallelMultiHeadAttention


class ParallelEncoderLayer(nn.Module):

    def __init__(self, 
                 d_h: int,
                 head: int,
                 d_ff: int,
                 p: float):
        """parallel encoder layer
        
        Args:
            d_h (int): attn hidden dim
            head (int): number of attn layers(=world size)
            d_ff (int): FFN hidden dim
            p (float): dropout rate
        """
        super().__init__()

        self.p, self.d_h = p, d_h

        self.attn = ParallelMultiHeadAttention(d_h, head, p)
        self.ffn = ParallelFeedForward(d_h, d_ff, head)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor):
        """forward propagation
        
        Args:
            x (torch.Tensor(bz, len, d_h): input
            pad_mask (torch.Tensor(bz, 1, len): pad mask

        Returns:
            output (torch.Tensor(bz, len, d_h)): output
        """

        # 1. Multi-Head Attention
        attn = F.layer_norm(x, self.d_h)
        attn = self.attn(attn, attn, attn, pad_mask)
        # no need to make backward function
        dist.all_reduce(attn, dist.ReduceOp.SUM)
        attn = F.dropout(attn, self.p)
        attn = attn + x

        # 2. Feed Forward
        ffn = F.layer_norm(attn, self.d_h)
        ffn = self.ffn(ffn)
        # no need to make backward function
        dist.all_reduce(ffn, dist.ReduceOp.SUM)
        ffn = F.dropout(ffn, self.p)
        output = ffn + attn

        return output


class ParallelDecoderLayer(nn.Module):

    def __init__(self, 
                 d_h: int,
                 head: int,
                 d_ff: int,
                 p: float):
        """parallel decoder layer
        
        Args:
            d_h (int): attn hidden dim
            head (int): number of attn layers(=world size)
            d_ff (int): FFN hidden dim
            p (float): dropout rate
        """
        super().__init__()

        self.p, self.d_h = p, d_h

        self.attn = ParallelMultiHeadAttention(d_h, head, p)
        self.ffn = ParallelFeedForward(d_h, d_ff, head)

    def forward(self, 
                x: torch.Tensor,
                tri_mask: torch.Tensor):
        """forward propagation
        
        Args:
            x (torch.Tensor(bz, len, d_h)): input
            tri_mask (torch.Tensor(bz, len, len)): look ahead mask

        Returns:
            output (torch.Tensor(bz, len, d_h)): output
        """

        # 1. Multi-Head Attention
        attn = F.layer_norm(x, [self.d_h])
        attn = self.attn(attn, attn, attn, tri_mask)
        # no need to make backward function
        dist.all_reduce(attn, dist.ReduceOp.SUM)
        attn = F.dropout(attn, self.p)
        attn = attn + x

        # 2. Feed Forward
        ffn = F.layer_norm(attn, [self.d_h])
        ffn = self.ffn(ffn)
        # no need to make backward function
        dist.all_reduce(ffn, dist.ReduceOp.SUM)
        ffn = F.dropout(ffn, self.p)
        output = ffn + attn

        return output


class ParallelCrossEntropyLoss(nn.Module):

    def __init__(self, rank, world_size, num_class):
        super().__init__()

        assert num_class % world_size == 0
        sub_class = int(num_class / world_size)
        self.start_idx = int(rank * sub_class)
        self.end_idx = self.start_idx + sub_class - 1

        self.rank = rank

    def forward(self, logits, target):
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

