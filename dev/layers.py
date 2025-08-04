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
        std = math.sqrt(2.0/(in_features+out_features))
        self.weight = nn.Parameter(init.trunc_normal_(torch.empty(out_features, in_features),mean=0, std=std, a=-3*std, b=3*std).to(device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, " ... d_in, d_out d_in ->... d_out")


class Embedding(torch.nn.Module):

    def __init__(self, vocab_size: int, d_model: int, device=None, dtype=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.weight = nn.Parameter(init.trunc_normal_(torch.empty(vocab_size, d_model, device=device, dtype=dtype),mean=0, std=1, a=-3, b=3))

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
        k_arr= torch.arange(d_k//2, device = device, dtype=torch.float32)
        freq = theta**(-2*k_arr/d_k)
        positions = torch.arange(max_seq_len, device=device,dtype=torch.float32)
        angles: Float[Tensor, "S half"] = torch.outer(positions, freq)
        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)

    def forward(self,
                x: Float[Tensor, "batch_size sequence_length d_k"],
                token_positions: Int[Tensor, "batch_size sequence_length"]
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
        self.max_seq_len=max_seq_len
        self.W_qkv = Linear(d_model, 3*d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        mask = ~torch.triu(torch.ones((max_seq_len, max_seq_len), device=device, dtype=torch.bool), diagonal=1)
        self.register_buffer('mask', mask, persistent=True)
        self.rope = RoPE(theta=theta, d_k = d_model//num_heads, max_seq_len=max_seq_len, device=device)

    def forward(self,
        in_features: Float[Tensor, " ... sequence_length d_in"],
        past_kv: tuple[Tensor, Tensor],
        attention_mask: Int[Tensor, "batch_size sequence_length"],
    ) -> torch.Tensor:
        """
        in_features: the whole sequence during prefilling, the new token during decoding
        past_kv: KV cache
        attention_mask: each row consists of 1s followed by 0s, where 1s indicate the true token in the sequence including the newly decoded token (not padding).
        the shape of past_kv and attention_mask are both padded to max_len.
        """

        head_dim = self.d_model // self.num_heads
        batch_size, input_len, d_in = in_features.shape
        seq_lens = attention_mask.sum(dim=-1)
        qkv_proj = self.W_qkv(in_features)
        q, k, v = qkv_proj.chunk(3, dim=-1)

        q_head = rearrange(q, "b S (h d_h) -> b h S d_h", h=self.num_heads)
        k_head = rearrange(k, "b S (h d_h) -> b h S d_h", h=self.num_heads)
        v_head = rearrange(v, "b S (h d_h) -> b h S d_h", h=self.num_heads)

        # get the true position of the input tokens, no need to modify past_kv because they already have positional information
        if input_len > 1: # prefilling
            token_positions = torch.arange(input_len).unsqueeze(0) # 1,T
        else:
            token_positions = (seq_lens-1).unsqueeze(1) #B,1
        token_positions = token_positions.to(torch.long)
        q_head = self.rope(q_head, token_positions)
        k_head = self.rope(k_head, token_positions)
        past_k, past_v = past_kv

        b = torch.arange(batch_size, device=in_features.device)
        # breakpoint()
        if input_len > 1: # Prefilling
            past_k[:, :, :input_len, :] = k_head
            past_v[:, :, :input_len, :] = v_head
        else: # decoding
            past_k[b, :, token_positions.squeeze(-1), :] = k_head.squeeze(-2)
            past_v[b, :, token_positions.squeeze(-1), :] = v_head.squeeze(-2)


        scores = einsum(q_head, past_k, "... S_q d_h, ... S_k d_h-> ... S_q S_k") * head_dim ** -0.5
        # mask the input's attention to the padding,
        ar_full = torch.arange(self.max_seq_len, device=in_features.device) # T
        valid_k = ar_full.unsqueeze(0) < seq_lens.unsqueeze(1)  # B,T
        if input_len > 1: # prefill
            allowed = self.mask.unsqueeze(0).unsqueeze(1) & valid_k.unsqueeze(1).unsqueeze(2)  # B,H,T,T
        else:
            allowed = torch.broadcast_to(valid_k, (valid_k.shape[0], self.num_heads, 1, valid_k.shape[1]))
        scores.masked_fill_(~allowed, float("-inf"))
        attn = softmax(scores, -1)
        ctx = einsum(attn, past_v, "... S1 S2, ... S2 d_h->... S1 d_h")
        ctx = rearrange(ctx, "... h S d_h -> ... S (h d_h)")
        out = self.output_proj(ctx)
        return out.view(batch_size, input_len, self.d_model)


class Block(torch.nn.Module):
    def __init__(self, d_model: int,num_heads: int,d_ff: int,max_seq_len: int,theta: float, device=None, dtype=None):
        super().__init__()
        self.ln1 = RMS(d_model=d_model, eps=1e-5, device=device, dtype=dtype)
        self.ln2 = RMS(d_model=d_model, eps=1e-5, device=device, dtype=dtype)
        self.attn = Attn(d_model=d_model, num_heads=num_heads, max_seq_len=max_seq_len, theta=theta, device=device, dtype=dtype)
        self.ffn = SwiGlu(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self,
        in_features: Float[Tensor, " ... sequence_length d_in"],
        layer_past_kv: tuple[Tensor, Tensor],
        attention_mask: Int[Tensor, "batch_size sequence_length"],
    ) -> torch.Tensor:
        a_out = self.attn(self.ln1(in_features), past_kv=layer_past_kv, attention_mask=attention_mask)
        in_features = in_features + a_out
        in_features = in_features + self.ffn(self.ln2(in_features))
        return in_features


class TransformerLm(torch.nn.Module):
    def __init__(self, vocab_size:int, d_model: int,num_heads: int,d_ff: int,max_seq_len: int,theta: float, num_layers:int, device=None, dtype=None):
        super().__init__()
        self.num_heads=num_heads
        self.token_embeddings = Embedding(vocab_size=vocab_size, d_model=d_model, device=device)
        self.layers = torch.nn.ModuleList([
            Block(d_model=d_model,num_heads=num_heads,d_ff=d_ff,max_seq_len=max_seq_len,theta=theta, device=device)  for _ in range(num_layers)
        ])
        self.ln_final = RMS(d_model=d_model, eps=1e-5, device=device, dtype=torch.float32)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size,  device=device, dtype=torch.float32)

    def forward(self,
        in_indices:  Int[Tensor, " batch_size sequence_length"],
        past_kv: list[tuple[Tensor, Tensor]],
        attention_mask: Int[Tensor, "batch_size sequence_length"],
    ) -> torch.Tensor:
        embeddings = self.token_embeddings(in_indices)
        for i, block in enumerate(self.layers):
            block_past_kv = past_kv[i]
            embeddings = block(embeddings, layer_past_kv = block_past_kv, attention_mask=attention_mask)
        with torch.autocast(device_type='cuda', enabled=False):
            return self.lm_head(self.ln_final(embeddings.to(torch.float32)))