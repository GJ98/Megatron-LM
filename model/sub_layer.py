import math
import torch
from torch import nn
import torch.nn.functional as F


class ParallelMultiHeadAttention(nn.Module):

    def __init__(self, d_h: int, head: int, p: float):
        """parallel multi head attention
        
        Args:
            d_h (int): hidden dim
            head (int): number of attn layers(=world size)
            p (float): dropout rate
        """
        super().__init__()

        assert d_h % head == 0

        self.d_h, self.head, self.p = d_h, head, p
        self.d_attn = d_h // head

        self.w_v = nn.Linear(d_h, self.d_attn, bias=False)
        self.w_k = nn.Linear(d_h, self.d_attn, bias=False)
        self.w_q = nn.Linear(d_h, self.d_attn, bias=False)

        self.w_o = nn.Linear(self.d_attn, d_h, bias=False)

    def forward(self, 
                v: torch.Tensor, 
                k: torch.Tensor, 
                q: torch.Tensor, 
                mask: torch.Tensor):
        """forward propagation
        
        Args:
            v (torch.Tensor(bz, len_k, d_h)): value
            k (torch.Tensor(bz, len_k, d_h)): key 
            q (torch.Tensor(bz, len_q, d_h)): query
            mask (torch.Tensor(bz, :, len_k)): mask

        Returns:
            output (torch.Tensor(bz, len_q, d_h)): output
        """

        v, k, q = self.w_v(v), self.w_k(k), self.w_q(q)

        attn = self.attention(q, k, v, mask)

        output = self.w_o(attn)

        return output

    def attention(self, 
                  q: torch.Tensor,
                  k: torch.Tensor,
                  v: torch.Tensor,
                  mask: torch.Tensor):
        """scale dot product attention
        
        Args:
            q (torch.Tensor(bz, len_q, d_attn)): query
            k (torch.Tensor(bz, len_k, d_attn)): key
            v (torch.Tensor(bz, len_k, d_attn)): value
            mask (torch.Tensor(bz, :, len_k)): mask

        Returns:
            output (torch.Tensor(bz, len_q, d_attn)): output
        """

        weight = (q @ k.transpose(-1, -2)) / math.sqrt(self.d_attn)

        if mask is not None:
            weight.masked_fill(mask==True, -1e9)

        scale_weight = F.dropout(F.softmax(weight, dim=-1), self.p)

        output = scale_weight @ v
        
        return output


class ParallelFeedForward(nn.Module):

    def __init__(self, d_h: int, d_ff: int, world_size: int):
        """parallel position wise feed forward
        
        Args:
            d_h (int): attn hidden dim
            d_ff (int): FFN hidden dim
            world_size (int): world size
        """
        super().__init__()

        assert d_ff % world_size == 0

        d_ff = int(d_ff / world_size)

        self.w_1 = nn.Linear(d_h, d_ff)
        self.w_2 = nn.Linear(d_ff, d_h)

    def forward(self, x: torch.Tensor):
        """forward propagation
        
        Args:
            x (torch.Tensor(bz, len_q, d_h)): input

        Returns:
            output (torch.Tensor(bz, len_q, d_h)): output
        """

        x = F.gelu(self.w_1(x))

        output = self.w_2(x)

        return output