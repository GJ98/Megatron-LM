import torch
from torch import nn
import torch.distributed as dist

from model.function import all_gather


def positional_encoding(max_len: int, d_emb: int):
    """positional encoding
    
    Args:
        max_len (int): max length
        d_emb (int): embedding dim

    Returns:
        pos_enc (torch.Tensor(1, max_len, d_emb)): positional encoding
    """
    pos_enc = torch.zeros(max_len, d_emb)
    pos_enc.requires_grad = False

    pos = torch.arange(start=0, end=max_len).unsqueeze(1)

    _2i = torch.arange(start=0, end=d_emb, step=2)

    pos_enc[:, 0::2] = torch.sin(pos / 10000 ** (_2i / d_emb))
    pos_enc[:, 1::2] = torch.cos(pos / 10000 ** (_2i / d_emb))

    return pos_enc[None, :, :]


class ParallelEmbedding(nn.Module):

    def __init__(self, 
                 rank: int,
                 world_size: int, 
                 num_embeddings: int, 
                 embedding_dim: int):
        """parallel embedding 

        Args:
            rank (int): rank
            world_size (int): world size
            vocab_size (int): vocab size
            d_emb (int): embedding dim
        """
        super().__init__()

        assert num_embeddings % world_size == 0
        sub_vocab_size = int(num_embeddings / world_size)
        self.start_idx = int(rank * sub_vocab_size)
        self.end_idx = self.start_idx + sub_vocab_size - 1
        self.embed = nn.Embedding(sub_vocab_size, embedding_dim, None)

    def forward(self, x: torch.Tensor):
        """forward propagation
        
        Args:
            x (torch.Tensor(bz, len)): input

        Returns:
            token (torch.Tensor(bz, len, d_emb)): token embedding
        """

        mask = (x < self.start_idx) | (x > self.end_idx)
        masked_input = x.clone() - self.start_idx
        masked_input[mask] = 0

        token = self.embed(masked_input)
        token[mask, :] = 0.0
        # no need to make backward function
        dist.all_reduce(token, dist.ReduceOp.SUM)

        return token


class ParallelProjLM(nn.Module):

    def __init__(self, 
                 rank: int,
                 world_size: int, 
                 weight: torch.Tensor):
        """parallel projection layer
        
        Args:
            rank (int): rank
            world_size (int): world size
            weight (torch.Tensor(sub_vocab_size, d_emb)): embedding weight
        """
        super().__init__()

        self.rank, self.world_size = rank, world_size
        self.proj_lm = nn.Parameter(weight.T)

    def forward(self, x: torch.Tensor):
        """forward propagation
        
        Args:
            x (torch.Tensor(bz, len, d_emb)): output of model

        Returns:
            output (torch.Tensor(bz, len, vocab_size)): logits
        """

        sub_logits = x @ self.proj_lm
        # gather all vocabulary 
        # logits = all_gather.apply(sub_logits, self.rank, self.world_size)
        return sub_logits