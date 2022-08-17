# Megatron-LM 

Megatron-LM implemented by `PyTorch`

## Implementation details

`torch.distributed` module is used to communicate between processes. (e.g., `dist.all_reduce`)

`torch.mutliprocessing` module is used to spawn processes. (e.g., `mp.Process`)

#### 1. `ParallelEncoderLayer`
```python
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
```
- In `ParallelEncoderLayer`, we use `dist.all_reduce`, which is not a `torch.autograd.Function`. In other word, we do not implement `dist.all_reduce` backward. Because all processes receive same differential value $\frac{dL}{dy}$.

<!--
- In `ParallelEncoderLayer`, we use `dist.all_reduce`. It is not a `torch.autograd.Function`, so it is not part of `torch.autograd`. Because all processes receive same derivative. 

 which is not a `torch.autograd.Function`.

all processes receive same derivative. So we do not define `torch.autograd.Function` about **all-reduce** operation.`dist.all_reduce` backward does not change output derivative $(\frac{dL}{dx} = \frac{dL}{dy})$. Because all processes receive same output derivative.
-->

- Unlike paper, $f$ and $g$ are not used.

#### 2. `ParallelEmbedding`
```Python
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
```

- Each process contains portion of the embedding table, denoted as $E_i$. So if input embedding vector is not in $E_i$, mask input token by $0$.

- And gather all input token by `dist.all_reduce`.

#### 3. `ParallelCrossEntropyLoss`
```python
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
        #dist.all_reduce(sum_exp_logits)
        all_reduce.apply(sum_exp_logits)

        loss = (sum_exp_logits[idx].log() - predict_logits).sum(dim=-1)
        dist.all_reduce(loss)

        return loss / target.size(dim=-1)
```

- Similar to `ParallelEmbedding`, each process computes fraction of *CrossEntropyLoss*. And sum by `dist.all_reduce`. By doing so, communication cost is reduced.

- `ParallelCrossEntropyLoss` use `all_reduce`, which is `torch.autograd.Function`. In other word, we implement `all_reduce` backward. Because all processes receive different differential value $\frac{dL}{dy}$. (e.g., *process 1* receives $\frac{dL}{dy} = \left[\frac{dL}{dy_{1}}, \frac{dL}{dy_{2}}, 0 \right]$ and *process 2* receives $\frac{dL}{dy} = \left[0, 0, \frac{dL}{dy_{3}}\right]$). So to make them same, we use `dist.all_reduce` operation. 

### testing...