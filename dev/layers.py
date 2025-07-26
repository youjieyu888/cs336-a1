import math
from typing import Mapping, Any

from jaxtyping import Float, Int
import sys
import numpy.typing as npt
import torch
from torch import Tensor, nn
from einops import rearrange, einsum
from torch.nn import init


class Linear(torch.nn.Module):

    def __init__(self, in_features: int, out_features:int, device=None, dtype=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features).to(device=device, dtype=dtype))
        std = math.sqrt(2.0/(in_features+out_features))
        init.trunc_normal_(self.weight,0,std, -3*std, 3*std)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, " ... d_in, d_out d_in ->... d_out")


class Embedding(torch.nn.Module):

    def __init__(self, vocab_size: int, d_model: int, device=None, dtype=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model, device=device, dtype=dtype))
        init.trunc_normal_(self.weight,0, -3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x]

class RMS(torch.nn.Module):

    def __init__(self, d_model: int, eps: float, device=None, dtype=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        x=x.float()
        rms = (x.pow(2).mean(dim=-1, keepdim=True) +self.eps).sqrt()
        result = x/rms * self.weight
        return result.to(in_type)

class SwiGlu(torch.nn.Module):
    def __init__(self, d_model: int, d_ff:int | None, device=None, dtype=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model=d_model
        if d_ff is None:
            raw_ff_dim = d_model *8 / 3.0
            self.d_ff = int(64 * math.ceil(raw_ff_dim/64))
        else:
            self.d_ff=d_ff
        self.w1 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.w2 = Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)
        self.w3 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)
        nn.init.xavier_uniform_(self.w3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.w1(x)
        gated =  a * torch.sigmoid(a)
        h = gated* self.w3(x)
        y = self.w2(h)
        return y

class RoPE(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = float(theta)
        self.d_k = int(d_k)
        self.max_seq_len = int(max_seq_len)
        k_arr= torch.arange(d_k//2, device = device, dtype=torch.float32)
        freq = self.theta**(-2*k_arr/d_k)
        positions = torch.arange(max_seq_len, device=device,dtype=torch.float32)
        angles: Float[Tensor, "S half"] = torch.outer(positions, freq)
        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)

    def forward(self,
                x: Float[Tensor, " ... sequence_length d_k"],
                token_positions: Int[Tensor, " ... sequence_length"]
    ) -> torch.Tensor:
        cos_sel: Float[Tensor, "S half"] = self.cos_cached.to(dtype=x.dtype, device=x.device)[token_positions]
        sin_sel: Float[Tensor, "S half"] = self.sin_cached.to(dtype=x.dtype, device=x.device)[token_positions]
        *prefix, D = x.shape
        half = D//2
        x_pairs = x.reshape(*prefix, half, 2)
        x_even = x_pairs[..., 0]
        x_odd = x_pairs[..., 1]
        x_rot_even = x_even * cos_sel - x_odd * sin_sel
        x_rot_odd  = x_even * sin_sel + x_odd * cos_sel
        out = torch.stack([x_rot_even, x_rot_odd], dim=-1).reshape(*prefix, D)
        return out

def softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    shifted = in_features - in_features.max(dim=dim, keepdim = True).values
    exp_shifted = shifted.exp()
    return exp_shifted / exp_shifted.sum(dim=dim, keepdim=True)

class Attn(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int,
            max_seq_len: int, theta: float = 0,
            device=None, dtype=None
        ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.W_qkv = Linear(d_model, 3*d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        mask = ~torch.triu(torch.ones((max_seq_len, max_seq_len), device=device, dtype=torch.bool), diagonal=1)
        self.register_buffer('mask', mask, persistent=True)
        if theta:
            self.rope = RoPE(theta=theta, d_k = d_model//num_heads, max_seq_len=max_seq_len, device=device)
        else:
            self.rope=None

    def forward(self,
        in_features: Float[Tensor, " ... sequence_length d_in"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ) -> torch.Tensor:
        head_dim = self.d_model // self.num_heads
        *batch_dims, seq_len, d_in = in_features.shape
        qkv_proj = self.W_qkv(in_features)
        q, k, v = qkv_proj.chunk(3, dim=-1)
        q_head = rearrange(q, "... S (h d_h) -> ... h S d_h", h=self.num_heads)
        k_head = rearrange(k, "... S (h d_h) -> ... h S d_h", h=self.num_heads)
        v_head = rearrange(v, "... S (h d_h) -> ... h S d_h", h=self.num_heads)
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len)
            q_head = self.rope(q_head, token_positions)
            k_head = self.rope(k_head, token_positions)
        scores = einsum(q_head, k_head, "... S1 d_h, ... S2 d_h-> ... S1 S2") * head_dim ** -0.5
        scores = torch.where(self.mask[:seq_len,:seq_len], scores, float("-inf"))
        attn = softmax(scores, -1)
        ctx = einsum(attn, v_head, "... S1 S2, ... S2 d_h->... S1 d_h")
        ctx = rearrange(ctx, "... h S d_h -> ... S (h d_h)")
        out = self.output_proj(ctx)
        return out.view(*batch_dims, seq_len, self.d_model)

class Block(torch.nn.Module):
    def __init__(self, d_model: int,num_heads: int,d_ff: int,max_seq_len: int,theta: float, device=None, dtype=None):
        super().__init__()
        self.ln1 = RMS(d_model=d_model, eps=1e-5, device=device, dtype=dtype)
        self.ln2 = RMS(d_model=d_model, eps=1e-5, device=device, dtype=dtype)
        self.attn = Attn(d_model=d_model, num_heads=num_heads, max_seq_len=max_seq_len, theta=theta, device=device, dtype=dtype)
        self.ffn = SwiGlu(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self,
        in_features: Float[Tensor, " ... sequence_length d_in"],
    ) -> torch.Tensor:
        in_features = in_features + self.attn(self.ln1(in_features))
        in_features = in_features + self.ffn(self.ln2(in_features))
        return in_features


class TransformerLm(torch.nn.Module):
    def __init__(self, vocab_size:int, d_model: int,num_heads: int,d_ff: int,max_seq_len: int,theta: float, num_layers:int, device=None, dtype=None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size=vocab_size, d_model=d_model, device=device)
        self.layers = torch.nn.ModuleList([
            Block(d_model=d_model,num_heads=num_heads,d_ff=d_ff,max_seq_len=max_seq_len,theta=theta, device=device)  for _ in range(num_layers)
        ])
        self.ln_final = RMS(d_model=d_model, eps=1e-5, device=device, dtype=torch.float32)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size,  device=device, dtype=torch.float32)

    def forward(self,
        in_indices:  Int[Tensor, " batch_size sequence_length"],
    ) -> torch.Tensor:
        embeddings = self.token_embeddings(in_indices)
        for layer in self.layers:
            embeddings = layer(embeddings)
        with torch.autocast(device_type='cuda', enabled=False):
            return self.lm_head(self.ln_final(embeddings.to(torch.float32)))