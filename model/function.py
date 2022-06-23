import torch
from torch.autograd import Function
import torch.distributed as dist


class all_gather(Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, rank: int, world_size: int):
        """all gather forward in output embedding

        Args:
            x (torch.Tensor(bz, len, sub_vocab_size)): sub_logits
            rank (int): rank
            world_size (int): world_size

        Return:
            x_list (torch.Tensor(bz, len, vocab_size)): logits
        """

        ctx.save_for_backward(torch.tensor(rank), torch.tensor(world_size))
        x_list = [torch.zeros_like(x) for _ in range(world_size)]
        dist.all_gather(x_list, x)
        return torch.cat(x_list, dim=-1)

    @staticmethod
    def backward(ctx, d_y: torch.Tensor):
        """all gather backward in output embedding
        
        Args: 
            d_y (torch.Tensor(bz, len, vocab_size)): output derivative

        Returns:
            d_x (torch.Tensor(bz, len, sub_vocab_size)): input derivative
        """

        rank, world_size = ctx.saved_tensors
        return d_y.chunk(world_size, dim=-1)[rank]