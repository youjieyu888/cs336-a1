import pickle
import time

import torch

from dev.launch import Config, load_config
from dev.layers import TransformerLm
from dev.tokenizer import Tokenizer
from dev.training import load_checkpoint
from tests.adapters import get_tokenizer

MODEL_CKPT = "data/runs/run-1757468131/checkpoints/latest.pt"
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
    pad_token_id  = len(tokenizer._vocab)-1
    assert tokenizer.decode([pad_token_id ])=="<|padding|>"
    eos_id = tokenizer.encode("<|endoftext|>")[0]
    device="cuda"
    outputs= []
    for prompt in prompts:
        tokens_list = tokenizer.encode(prompt)
        tokens = torch.tensor(tokens_list, device = device, dtype=torch.long)
        with torch.autocast(device_type=device, dtype=torch.bfloat16) and torch.no_grad():
            while tokens.shape[0]<max_new_tokens:
                t = time.time()
                output_logits = model(tokens)
                next_logits = output_logits[-1]
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
                tokens = torch.concat((tokens, next_token), dim=-1)
                print(f'sampling takes {(time.time() - t) * 1000:1f}ms')
            outputs.append(tokenizer.decode(tokens.tolist()))
    return outputs


if __name__ == "__main__":
    tokenizer = tokenizer_from_file(TOKENIZER)
    config = Config(load_config())
    model = TransformerLm(**config.model, device="cuda", dtype=torch.bfloat16)
    load_checkpoint(MODEL_CKPT, model)
    breakpoint()
    print(decode(model, config, tokenizer, ["The"])[0])