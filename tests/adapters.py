from __future__ import annotations

import concurrent.futures
import heapq
import itertools
import multiprocessing
import os
from collections import defaultdict
import time
from typing import Generator, Iterator, TypeVar, Generic, Any, BinaryIO, IO
from collections.abc import Iterable
import numpy.typing as npt
import numpy as np

from dev.dataloader import get_batch
from dev.serving_model import Linear, Embedding, RMS, SwiGlu, RoPE, softmax, Attn, Block, TransformerLm
from jaxtyping import Float, Int
from torch import Tensor
import torch
import regex as re
import pickle
from einops import rearrange, einsum

from dev.tokenizer import Tokenizer
from dev.training import cross_entropy_loss, AdamW, lr_cosine_schedule, gradient_clip, save_checkpoint, load_checkpoint


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    lin = Linear(d_in, d_out, device=weights.device, dtype=weights.dtype)
    with torch.no_grad():
        lin.weight.copy_(weights)
    return lin(in_features)



def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """

    embedding = Embedding(vocab_size, d_model, device=weights.device, dtype=weights.dtype)
    with torch.no_grad():
        embedding.weight.copy_(weights)
    return embedding(token_ids)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    swiglu = SwiGlu(d_model, d_ff, device=w1_weight.device, dtype=w1_weight.dtype)
    with torch.no_grad():
        swiglu.w1.weight.copy_(w1_weight)
        swiglu.w2.weight.copy_(w2_weight)
        swiglu.w3.weight.copy_(w3_weight)
    return swiglu(in_features)


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    raise NotImplementedError


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    attn = Attn(d_model=d_model, num_heads=num_heads, max_seq_len=in_features.shape[-2])
    with torch.no_grad():
        attn.W_qkv.weight.copy_(torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=0))
        attn.o_proj_weight.weight.copy_(o_proj_weight)
    return attn(in_features)

def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    attn = Attn(d_model=d_model, num_heads=num_heads, max_seq_len=max_seq_len, theta=theta)
    with torch.no_grad():
        attn.W_qkv.weight.copy_(torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=0))
        attn.output_proj.weight.copy_(o_proj_weight)
    return attn(in_features, token_positions=token_positions)


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    rope = RoPE(theta=theta, d_k=d_k, max_seq_len=max_seq_len, device=in_query_or_key.device)
    return rope(in_query_or_key, token_positions)

def _merge_attention_weights(weights: dict[str, Tensor]) -> dict[str, Tensor]:
    if "attn.q_proj.weight" in weights:
        weights["attn.W_qkv.weight"] = torch.cat(
            [weights["attn.q_proj.weight"], weights["attn.k_proj.weight"], weights["attn.v_proj.weight"]], dim=0
        )

        del weights["attn.q_proj.weight"]
        del weights["attn.k_proj.weight"]
        del weights["attn.v_proj.weight"]

    layer_prefixes = set()
    for key in list(weights.keys()):
        if key.startswith("layers.") and key.endswith(".attn.q_proj.weight"):
            layer_prefix = key.rsplit(".attn.q_proj.weight", 1)[0]
            layer_prefixes.add(layer_prefix)

    for prefix in layer_prefixes:
        q_key = f"{prefix}.attn.q_proj.weight"
        k_key = f"{prefix}.attn.k_proj.weight"
        v_key = f"{prefix}.attn.v_proj.weight"

        if q_key in weights and k_key in weights and v_key in weights:
            weights[f"{prefix}.attn.W_qkv.weight"] = torch.cat([weights[q_key], weights[k_key], weights[v_key]], dim=0)

            del weights[q_key]
            del weights[k_key]
            del weights[v_key]

    return weights

def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    block = Block(d_model=d_model, num_heads=num_heads, d_ff=d_ff, max_seq_len=max_seq_len, theta=theta)
    block.load_state_dict(_merge_attention_weights(weights))
    return block(in_features)

def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    lm = TransformerLm(vocab_size=vocab_size, d_model=d_model,num_heads=num_heads,
                  d_ff=d_ff,max_seq_len=context_length,theta=rope_theta, num_layers=num_layers)
    lm.load_state_dict(_merge_attention_weights(weights))
    return lm(in_indices)


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    rms = RMS(d_model, eps, device=weights.device, dtype=weights.dtype)
    with torch.no_grad():
        rms.weight.copy_(weights)
    return rms(in_features)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    raise NotImplementedError


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    return get_batch(dataset, batch_size, context_length, device)


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
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
    return softmax(in_features, dim)


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    return cross_entropy_loss(inputs, targets)


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    return gradient_clip(parameters, max_l2_norm)


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    return AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    return lr_cosine_schedule(
        it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters
    )


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    return save_checkpoint(model, optimizer, iteration, out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    return load_checkpoint(src, model, optimizer)

T = TypeVar('T')
class MaxHeapNode(Generic[T]):
    def __init__(self, key: T):
        self.key = key

    def __lt__(self, other: MaxHeapNode[T]):
        # Invert the comparison logic to simulate a max-heap
        return self.key > other.key

PAT_regex = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
UTF8 = 'utf-8'


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Tokenizer:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return Tokenizer(vocab, merges, special_tokens if special_tokens else [])

def count_words_in_chunk(lines: list[str]) -> dict[tuple[bytes], int]:
    # 2. each process remove special token and start counting.
    # 3. then each process hash its key and return [{key hash =0}, {key hash =1}] to master
    st_time = time.time()
    print('MP start time', st_time)
    counter = defaultdict(int)
    for line in lines:
        for word in PAT_regex.findall(line):
        # for word in line.split(' '):
            counter[tuple([bytes([b]) for b in word.encode(UTF8)])]+=1
    print('pretoken time: ', time.time()-st_time)
    print('processing size: ', sum([len(k)*v for k,v in counter.items()]))
    return counter

def compute_pair_count(word_count: dict[tuple[bytes], int]) -> \
        (dict[tuple[bytes, bytes], int], dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]):
    bpe_count = defaultdict(int)
    pair_to_word = defaultdict(lambda:set())
    for word, count in word_count.items():
        for b1, b2 in zip(word[:-1], word[1:]):
            bpe_count[(b1, b2)]+=count
            pair_to_word[(b1, b2)].add(word)
    return bpe_count, pair_to_word

def compute_top_pair(bpe_count: dict[tuple[bytes, bytes], int], pq: list[MaxHeapNode[tuple[int, tuple[bytes, bytes]]]])-> tuple[bytes, bytes] | None:
    top_count, top_pair = pq[0].key
    while bpe_count[top_pair] != top_count:
        heapq.heappop(pq)
        heapq.heappush(pq, MaxHeapNode((bpe_count[top_pair], top_pair)))
        top_count, top_pair = pq[0].key
    return top_pair

def merge_bpe(word_count: dict[tuple[bytes, ...], int],
              pair_count: dict[tuple[bytes, bytes], int],
              pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes,...]]],
              b1:bytes, b2:bytes,
              pq: list[MaxHeapNode[tuple[int, tuple[bytes, bytes]]]])-> dict[tuple[bytes, ...], int]:
    assert (b1, b2) == pq[0].key[1]
    del pair_count[(b1,b2)]
    heapq.heappop(pq)
    new_pairs = defaultdict(int)
    word_update = {}
    for word in pair_to_words[(b1,b2)]:
        count  = word_count[word]
        # print('merging:', word, count, b1, b2)
        i=0
        # merge the word
        merge_indices = []
        while i<len(word)-1:
            if word[i]==b1 and word[i+1] == b2:
                merge_indices.append(i)
                i+=2
            else:
                i+=1
        if not merge_indices:
            continue
        new_word: list[bytes] = list(word[:merge_indices[0]])
        for merge_idx, next_merge_idx in zip(merge_indices[:-1], merge_indices[1:]):
            new_word.append(word[merge_idx]+word[merge_idx+1])
            new_word.extend(word[merge_idx+2:next_merge_idx])
        new_word.append(word[merge_indices[-1]] + word[merge_indices[-1] + 1])
        new_word.extend(word[merge_indices[-1]+2:])
        new_word: tuple[bytes, ...] = tuple(new_word)
        word_update[word] = new_word
        del word_count[word]
        word_count[new_word] = count
        for merge_idx in merge_indices:
            # x b1 (cur) b2 y
            if merge_idx>0:
                pair_count[(word[merge_idx-1], b1)]-=count
            if merge_idx<len(word)-2:
                pair_count[(b2, word[merge_idx+2])]-=count
        new_word_merge_indices = [a-b for a,b in zip(merge_indices, range(len(merge_indices)))]
        for new_merge_index in new_word_merge_indices:
            if new_merge_index>0:
                pair_count[(new_word[new_merge_index-1], b1+b2)] +=count
                new_pairs[(new_word[new_merge_index-1], b1+b2)] +=count
            if new_merge_index<len(new_word)-1:
                pair_count[(b1+b2, new_word[new_merge_index+1])] +=count
                new_pairs[(b1+b2, new_word[new_merge_index+1])] +=count
    for word, new_word in word_update.items():
        for p1, p2 in zip(word[:-1], word[1:]):
            pair_to_words[(p1, p2)].discard(word)
        for p1, p2 in zip(new_word[:-1], new_word[1:]):
            pair_to_words[(p1, p2)].add(new_word)
    for new_pair, new_pair_count in new_pairs.items():
        heapq.heappush(pq, MaxHeapNode((new_pair_count, new_pair)))
    return word_count

def find_first(chunk:str, special_tokens:list[str], offset:int)->tuple[int, int]:
    idx = len(chunk)
    first_token = ''
    for token in special_tokens:
        first = chunk.find(token, offset)
        if first!=-1 and first < idx:
            idx = first
            first_token = token
    return idx, idx+len(first_token)

def split_by_special_tokens(chunk:str, special_tokens:list[str])->list[str]:
    ans = []
    cur = 0
    while cur<len(chunk):
        st, ed = find_first(chunk, special_tokens, cur)
        ans.append(chunk[cur:st])
        cur = ed
    return ans

# read file, generate lines seperated by special tokens
def read_file_in_chunk(input_path: str | os.PathLike, special_tokens:list[str])-> Generator[list[str]]:
    with open(input_path, 'r', encoding=UTF8) as f:
        chunk_size = 1_000_000
        chunk = f.read(chunk_size)
        read_size = len(chunk)
        while chunk:
            lines = split_by_special_tokens(chunk, special_tokens)
            if read_size == 0: # last chunk
                chunk=''
                yield lines
            else:
                yield lines[:-1]
                chunk = f'{lines[-1]}{f.read(chunk_size)}'
                read_size = len(chunk) - len(lines[-1])


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # master: chunk the doc and shuffle
    num_processes = multiprocessing.cpu_count() -1
    # num_processes=1
    file_size = os.path.getsize(input_path)
    if file_size<1_000_000:
        st_time = time.time()
        text = []
        for lines in read_file_in_chunk(input_path, special_tokens):
            text.extend(lines)
        word_count = count_words_in_chunk(text)
    else:
        chunk_size = min(file_size // num_processes + 1, 100*1_000_000)
        word_count = defaultdict(int)
        # 1. each process handles a chunk in a streaming manner
        st_time = time.time()
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
            results_iterator = executor.map(count_words_in_chunk, read_file_in_chunk(input_path, special_tokens))
            for result in results_iterator:
                if len(word_count)==0:
                    word_count=result
                else:
                    for k,v in result.items():
                        word_count[k]+=v
        print('pretoken time:', time.time()-st_time)
        print('total size from word count: ', sum([len(k)*v for k,v in word_count.items()]))
    # print(word_count)
    # 2. child process get pair frequency (note: key is of the same hash, but its bytes are not)
    vocab = [token.encode(UTF8) for token in special_tokens] + [bytes([b]) for b in (range(256))]
    merges = []
    pair_count, pair_to_word = compute_pair_count(word_count)

    pq = [MaxHeapNode((v,k)) for k,v in pair_count.items()]
    heapq.heapify(pq)
    while len(vocab)<vocab_size:
        st_time = time.time()
        top_pair = compute_top_pair(pair_count, pq)
        if top_pair is None:
            break
        b1, b2 = top_pair
        word_count = merge_bpe(word_count, pair_count, pair_to_word, b1, b2, pq)
        vocab.append(b1+b2)
        merges.append((b1, b2))
    return {i:item for i,item in enumerate(vocab)}, merges

def iter_file_chunks(path: str, chunk_size: int = 10 * 1024 * 1024) -> Iterator[str]:
    """Yield fixed-size text chunks with Windows line endings normalized to '\n'."""
    with open(path, 'r', encoding='utf-8', newline='') as f:
        for chunk in iter(lambda: f.read(chunk_size), ""):  # sentinel is empty string now
            yield chunk

def tokenize_to_file(tokenizer_path:str, corpus_path:str, output_file:str):
    print(f'using {tokenizer_path} to tokenize {corpus_path}')
    st_time = time.time()
    with open(tokenizer_path, 'rb') as f:
        vocab, merges = pickle.load(f)
        tokenizer = get_tokenizer(vocab, merges)
        tokens = tokenizer.encode_iterable(iter_file_chunks(corpus_path))
        with open(output_file, 'wb') as output:
            chunk = list(itertools.islice(tokens, 1_000_000))
            while chunk:
                np.asarray(chunk, dtype=np.uint16).tofile(output)
                print('writing 1MB: ', chunk[:10])
                chunk = list(itertools.islice(tokens, 1_000_000))
    print(f'using time {(time.time()-st_time)/60}')

# poetry env activate
# source /mnt/f/CS336/assignment1-basics/.venv/bin/activate
if __name__=="__main__":
    tokenize_to_file('owt_tokenizer.pkl', 'data/TinyStoriesV2-GPT4-train.txt', 'ts_train_owt_tokenized')
    tokenize_to_file('owt_tokenizer.pkl', 'data/TinyStoriesV2-GPT4-valid.txt', 'ts_valid_owt_tokenized')