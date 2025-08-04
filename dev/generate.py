import pickle
import time

import torch

from dev.launch import Config, load_config
from dev.layers import TransformerLm
from dev.tokenizer import Tokenizer
from dev.training import load_checkpoint
from tests.adapters import get_tokenizer

MODEL_CKPT = "data/runs/run-1756624406/checkpoints/latest.pt"
TOKENIZER = 'owt_tokenizer.pkl'

def tokenizer_from_file(path: str)->Tokenizer:
    with open(path, 'rb') as f:
        vocab, merges = pickle.load(f)
        tokenizer = get_tokenizer(vocab, merges, special_tokens=["<|endoftext|>", "<|padding|>"])
        return tokenizer

def decode(
    model: TransformerLm,
    config: Config,
    tokenizer: Tokenizer,
    prompts: list[str],
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> list[str]:
    temperature = max(temperature, 1e-6)
    window_len = 256
    pad_token_id  = len(tokenizer._vocab)-1
    assert tokenizer.decode([pad_token_id ])=="<|padding|>"
    eos_id = tokenizer.encode("<|endoftext|>")[0]
    device="cuda"
    batch_size = len(prompts)
    batch_tokens = [tokenizer.encode(p) for p in prompts]
    input_ids = torch.full((batch_size, window_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, window_len), dtype=torch.bool, device=device)
    for i,tokens in enumerate(batch_tokens):
        input_ids[i, :len(tokens)] = torch.tensor(tokens)
        attention_mask[i, :len(tokens)] = 1
    t=time.time()
    kv_cache = [
        (torch.empty(batch_size, config.model.num_heads, config.model.max_seq_len,
                     config.model.d_model // config.model.num_heads, device=device, dtype=torch.bfloat16),
         torch.empty(batch_size, config.model.num_heads, config.model.max_seq_len,
                     config.model.d_model // config.model.num_heads, device=device, dtype=torch.bfloat16))
        for i in range(config.model.num_layers)
    ]
    with torch.no_grad() and torch.autocast(device_type=device, dtype=torch.bfloat16):
        for i in range(max_new_tokens):
            seq_lens = attention_mask.sum(dim=-1)
            if i==0:
                current_input_ids = input_ids
            else:
                current_input_ids = input_ids[torch.arange(batch_size), seq_lens-1].unsqueeze(1)
            logits= model(
                in_indices=current_input_ids,
                past_kv = kv_cache,
                attention_mask = attention_mask,
            )
            print(f'generating token in {(time.time()-t)*1000:1f}ms')
            t=time.time()
            if i==0:
                next_logits = logits[torch.arange(batch_size), seq_lens-1, :]
            else:
                next_logits = logits[torch.arange(batch_size), -1, :]
            probs = torch.softmax(next_logits/temperature, dim=-1) # (B,V)
            sorted_probs, sorted_idxs=torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            mask = cumsum<=top_p
            mask[..., 0] = True
            trimmed_probs = sorted_probs * mask
            trimmed_probs /= trimmed_probs.sum(dim=-1, keepdim=True)
            next_sorted_idx = torch.multinomial(trimmed_probs, num_samples=1)
            # input_ids is (B,T), next_sorted_idx is (B, 1), sorted_idxs is (B,V)
            next_token = torch.gather(sorted_idxs, dim=-1, index=next_sorted_idx) # B, 1
            input_ids[torch.arange(batch_size),seq_lens] = next_token.squeeze(-1)
            attention_mask[torch.arange(batch_size),seq_lens] = 1
            print(f'sampling takes {(time.time()-t)*1000:1f}ms')
            t=time.time()
            if (next_token == eos_id).any():
                if (input_ids == eos_id).sum(dim=1).bool().all():
                    break
    print('decoding complete')
    output_texts = []
    for seq_tensor in input_ids:
        seq_list = seq_tensor.tolist()
        try:
            eos_index = seq_list.index(eos_id)
        except ValueError:
            eos_index=  len(seq_list)
        output_texts.append(tokenizer.decode(seq_list[:eos_index]))
    return output_texts


if __name__ == "__main__":
    tokenizer = tokenizer_from_file(TOKENIZER)
    config = Config(load_config())
    model = TransformerLm(**config.model, device="cuda", dtype=torch.bfloat16)
    load_checkpoint(MODEL_CKPT, model)
    breakpoint()
    print(decode(model, config, tokenizer, ["The"])[0])