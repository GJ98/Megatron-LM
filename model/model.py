import torch
from torch import nn
import torch.nn.functional as F

from model.embed import positional_encoding, ParallelEmbedding, ParallelProjLM
from model.layer import ParallelDecoderLayer


class ParallelGPT2(nn.Module):

    def __init__(self, 
                 rank: int,
                 vocab_size: int,
                 max_len: int,
                 d_h: int,
                 head: int,
                 d_ff: int,
                 n_layer: int,
                 pad: int,
                 p : float):
        """Parallel GPT2
        
        Args:
            rank (int): rank
            vocab_size(int): vocabulary size
            max_len (int): max length
            d_h (int): attn hidden dim(=embedding dim)
            head (int): number of attn layers(=world size)
            d_ff (int): FFN hidden dim
            n_layer (int): number of encoder layers
            pad (int): pad index
            p (float): dropout rate
        """
        super().__init__()

        self.p, self.pad = p, pad
        
        embed_parallel = True
        if embed_parallel == True:
            self.token = ParallelEmbedding(rank, head, vocab_size, d_h)
            self.proj_lm = ParallelProjLM(rank, head, self.token.embed.weight)
        else:
            self.token = nn.Embedding(vocab_size, head)
            self.proj_lm = nn.Linear(d_h, vocab_size)
            self.proj_lm.weight = self.token.weight

        self.token = ParallelEmbedding(rank, head, vocab_size, d_h)

        self.pos = nn.Parameter(positional_encoding(max_len, d_h), False)

        self.layers = [ParallelDecoderLayer(d_h, head, d_ff, p) \
                        for _ in range(n_layer)]

    def forward(self, seq: torch.Tensor):
        """forward propagation
        
        Args:
            seq (torch.Tensor(bz, len)): sequences

        Returns:
            output (torch.Tensor(bz, len, vocab_size) or
                    torch.Tensor(bz, len, sub_voab_size)): output
        """

        seq_len = seq.size(1)

        x = self.token(seq) + \
            self.pos[:, :seq_len, :]
        x = F.dropout(x, self.p)

        pad_mask = seq.eq(self.pad)[:, None, None, :].repeat(1, 1, seq_len, 1)
        tri_mask = torch.triu(torch.ones_like(pad_mask), diagonal=1)
        mask = (pad_mask + tri_mask).type(torch.BoolTensor)

        for layer in self.layers:
            x = layer(x, mask)

        output = self.proj_lm(x)

        return output