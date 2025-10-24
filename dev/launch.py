import argparse
import json
import math
import os
import time
import yaml

# Third-party imports
import numpy as np
import torch
import wandb
import wandb.wandb_run

from dev.dataloader import get_batch
from dev.layers import TransformerLm
from dev.training import AdamW, load_checkpoint, lr_cosine_schedule, lr_linear_schedule, lr_double_schedule, \
    cross_entropy_loss, gradient_clip, save_checkpoint


class Logger:

    def __init__(self, log_file: str | None = None, wandb_run: wandb.wandb_run.Run | None = None, resume: bool = False):
        self.log_file = log_file
        self.wandb_run = wandb_run
        if self.log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            if not resume:
                with open(self.log_file, 'w') as f: # clear file
                    pass

    def log_info(self, message: str | dict, console: bool = True):
        """Logs a general message to the console and/or a local file."""
        if isinstance(message, dict):
            message = self.format_metrics(message)

        if console:
            print(message)

        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(message + "\n")


    def log_metrics(self, metrics: dict):
        """Logs a dictionary of metrics to Weights & Biases for visualization."""
        if self.wandb_run:
            self.wandb_run.log(metrics)

    @staticmethod
    def format_metrics(metrics_dict: dict) -> str:
        """Formats a dictionary of metrics into a clean, readable string for console logging."""
        return " | ".join(f"{key}: {value}" for key, value in metrics_dict.items())

class Config(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = Config(value)

def load_config_from_file(config_path: str) -> dict:
    """Loads a configuration from a YAML or JSON file."""
    ext = os.path.splitext(config_path)[1].lower()
    with open(config_path) as f:
        if ext == ".json":
            return json.load(f)
        elif ext in [".yaml", ".yml"]:
            return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {ext}")

def load_config(config_path: str | None = None, base_config: dict | None = None) -> Config:
    if base_config is None:
        # Load the default configuration as the foundation.
        default_config_path = os.path.join(os.path.dirname(__file__), "default.yml")
        config = load_config_from_file(default_config_path)
    else:
        # If resuming, the previous run's config is the base.
        config = base_config

    if config_path:
        user_config = load_config_from_file(config_path)
        for section, section_config in user_config.items():
            if section in config and isinstance(config[section], dict):
                config[section].update(section_config)
            else:
                config[section] = section_config
    config["run"]["run_id"] = config["run"]["run_id"].replace("<timestamp>", f"{int(time.time())}")
    config["device"] = "cuda"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    config["dtype"] = str(dtype)
    return Config(config)


def get_peak_memory(device: str) -> float:
    """Gets the peak GPU memory usage in Megabytes (MB) for the current run."""
    if device != "cuda":
        return 0.0
    peak_mem_bytes = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats() # Reset stats for the next measurement interval.
    return peak_mem_bytes / (1024 * 1024)

def get_perplexity(loss: float) -> float:
    return math.exp(min(loss, 20))

def train(config: Config):
    resuming = config.training.get("resume", False)
    run_dir = os.path.join(config.run.out_dir, config.run.run_id)
    log_file = os.path.join(run_dir, "log.txt")
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    latest_symlink = os.path.join(config.run.out_dir, "latest")
    if os.path.lexists(latest_symlink):
        os.remove(latest_symlink)
    os.symlink(os.path.abspath(run_dir), latest_symlink, target_is_directory=True)
    wandb_run = None
    if config.run.wandb_project:
        wandb_run = wandb.init(
            project=config.run.wandb_project,
            id=config.run.run_id,
            resume="must" if resuming else None, # Ensures we connect to the same W&B run.
            name=config.run.run_id,
            config=config,
            dir=run_dir,
            tags=config.run.wandb_tags,
        )

    logger = Logger(log_file=log_file, wandb_run=wandb_run, resume=resuming)

    if not resuming:
        config_outfile = os.path.join(run_dir, "config.json")
        with open(config_outfile, "w") as f:
            json.dump(config, f, indent=2)
        logger.log_info(f"Saved config to: {config_outfile}")

    device = config.device
    dtype = getattr(torch, config.dtype.split(".")[-1])

    train_data = np.memmap(config.data.train_data_path, dtype=np.uint16, mode="r")
    valid_data = np.memmap(config.data.valid_data_path, dtype=np.uint16, mode="r")
    model = TransformerLm(**config.model, device=device, dtype=dtype).to(device)

    logger.log_info(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": config.optimizer.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = AdamW(optim_groups, lr=config.training.lr_max, betas=config.optimizer.betas)

    start_step = 1
    if resuming:
        checkpoint_path = config.training.resume_checkpoint
        logger.log_info(f"Loading checkpoint from: {checkpoint_path}")
        # `load_checkpoint` restores model weights, optimizer state, and the step number.
        start_step = load_checkpoint(checkpoint_path, model, optimizer) + 1
        logger.log_info(f"Resuming training from step {start_step}")
    if config.training.use_compile:
        model = torch.compile(model, backend="cudagraphs")
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
    cfg_train = config.training
    max_steps = cfg_train.max_steps
    if cfg_train.get("warmup_iters") is None:
        warmup_iters = int(cfg_train.warmup_ratio * max_steps)
    else:
        warmup_iters = cfg_train.warmup_iters

    def get_lr(step: int) -> float:
        """Calculates the learning rate for the current step based on the configured schedule."""
        schedule_type = cfg_train.lr_schedule
        if schedule_type == "cosine":
            cycle_iters = cfg_train.get("cosine_cycle_iters") or max_steps
            return lr_cosine_schedule(step, cfg_train.lr_max, cfg_train.lr_min, warmup_iters, cycle_iters)
        elif schedule_type == "linear":
            cycle_iters = cfg_train.get("linear_cycle_iters") or max_steps
            return lr_linear_schedule(step, cfg_train.lr_max, cfg_train.lr_min, warmup_iters, cycle_iters)
        elif schedule_type == "double":
            # This is a more complex schedule with two phases.
            return lr_double_schedule(
                step, cfg_train.lr_max, cfg_train.lr_inter, cfg_train.lr_min, warmup_iters,
                cfg_train.phase_one_iters, cfg_train.get("phase_two_iters") or max_steps, cfg_train.phase_two_type
            )
        else:
            raise ValueError(f"Unknown LR schedule: {schedule_type}")

    @torch.no_grad()
    def evaluate(current_step: int):
        model.eval()
        val_loss = 0.0
        for _ in range(cfg_train.eval_steps):
            x, y = get_batch(valid_data, cfg_train.eval_batch_size, config.model.max_seq_len, device)
            with torch.autocast(device_type = device, dtype=dtype):
                logits = model(x)
            loss = cross_entropy_loss(logits, y)
            val_loss += loss.item()
        val_loss /= cfg_train.eval_steps
        progress_str = f"step {current_step}/{max_steps} ({current_step / max_steps * 100:.2f}%)"
        logger.log_info({
            "step": progress_str,
            "v_loss": f"{val_loss:.4f}",
            "v_ppl": f"{get_perplexity(val_loss):.2f}",
        })
        logger.log_metrics({
            "eval/loss": val_loss,
            "eval/perplexity": get_perplexity(val_loss),
            "step": current_step,
        })
        model.train()

    if cfg_train.eval_before_training and not resuming:
        evaluate(0)

    # Pre-fetch the first batch to start the loop.
    x, y = get_batch(train_data, cfg_train.batch_size, config.model.max_seq_len, device)
    model.train()

    for step in range(start_step, max_steps + 1):
        t0 = time.time()
        loss_accum = 0.0

        # --- Gradient Accumulation ---
        # This inner loop processes `grad_accum_steps` micro-batches.
        # Gradients from each micro-batch are accumulated, and the optimizer step is
        # performed only once at the end. This simulates a larger effective batch size.
        for _ in range(cfg_train.grad_accum_steps):
            with torch.autocast(device_type=device, dtype=dtype):
                logits = model(x)
                # Scale the loss to average it over the accumulation steps. This is crucial.
                loss = cross_entropy_loss(logits, y) / cfg_train.grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()

            # Immediately fetch the next batch to overlap data loading with computation.
            x, y = get_batch(train_data, cfg_train.batch_size, config.model.max_seq_len, device)

        # Clip gradients to prevent them from exploding, a common cause of instability.
        norm = gradient_clip(model.parameters(), cfg_train.max_l2_norm)

        # Update learning rate for the current step.
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Perform the optimizer step and clear gradients for the next accumulation cycle.
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)  # `set_to_none=True` is a small memory optimization.

        # --- Logging and Checkpointing ---
        if device in ("cuda", "mps"):
            torch.mps.synchronize() if device == "mps" else torch.cuda.synchronize()

        dt_ms = (time.time() - t0) * 1000
        tokens_per_sec = (cfg_train.batch_size * cfg_train.grad_accum_steps * config.model.max_seq_len) / (
                    dt_ms / 1000)

        # Log metrics for the current training step.
        progress_str = f"step {step}/{max_steps} ({step / max_steps * 100:.2f}%)"
        train_loss = loss_accum.item()
        logger.log_info({
            "step": progress_str,
            "t_loss": f"{train_loss:.4f}",
            "t_ppl": f"{get_perplexity(train_loss):.2f}",
            "lr": f"{lr:.2e}",
            "norm": f"{norm:.2f}",
            "tok/s": f"{int(tokens_per_sec):,}",
            "dt": f"{dt_ms:.2f}ms",
            "memory": f"{get_peak_memory(device)}MB",
        })
        logger.log_metrics({
            "train/loss": train_loss,
            "train/perplexity": get_perplexity(train_loss),
            "train/lr": lr,
            "train/grad_norm": norm,
            "train/tokens_per_sec": tokens_per_sec,
            "train/peak_memory_MB": get_peak_memory(device),
            "step": step,
        })

        is_last_step = step == max_steps
        if step % cfg_train.eval_interval == 0 or is_last_step:
            evaluate(step)

        if step % cfg_train.checkpoint_interval == 0 or is_last_step:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}.pt")
            save_checkpoint(model, optimizer, step, checkpoint_path)
            # Update the `latest.pt` symlink to point to this new checkpoint.
            latest_ckpt_symlink = os.path.join(checkpoint_dir, "latest.pt")
            if os.path.lexists(latest_ckpt_symlink):
                os.remove(latest_ckpt_symlink)
            os.symlink(os.path.abspath(checkpoint_path), latest_ckpt_symlink)
            logger.log_info(f"Saved checkpoint to: {checkpoint_path}")

    if wandb_run:
        wandb.finish()



def parse_value(value_str: str):
    """Parses a string from the command line into a Python type (list, int, float, bool, or str)."""
    if value_str.startswith("[") and value_str.endswith("]"):
        content = value_str[1:-1].strip()
        return [parse_value(v.strip()) for v in content.split(",")] if content else []
    try: return int(value_str)
    except ValueError: pass
    try: return float(value_str)
    except ValueError: pass
    lower = value_str.lower()
    if lower in ("true", "false"): return lower == "true"
    return value_str


def deep_set(config_dict: dict, key_path: str, value):
    """Sets a value in a nested dictionary using a dot-separated key path (e.g., 'model.d_model')."""
    keys = key_path.split(".")
    d = config_dict
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value

# export WANDB_MODE=disabled
# python dev/launch.py --override-param training.use_compile=true
if __name__ == "__main__":
    print('parsing arguments')
    parser = argparse.ArgumentParser(description="Train a transformer model.")

    parser = argparse.ArgumentParser(description="Train a transformer model.")
    parser.add_argument("--config", type=str, help="Path to a custom config file (e.g., configs/small.yml).")
    parser.add_argument("--resume-from", type=str, help="Path to a run directory to resume training from.")
    parser.add_argument("--override-param", action="append", default=[],
                        help="Override a config param, e.g., --override-param training.batch_size=32 (can be repeated).")
    args = parser.parse_args()

    base_config = None
    # --- Logic for Resuming a Run ---
    if args.resume_from:
        resume_config_path = os.path.join(args.resume_from, "config.json")
        if not os.path.exists(resume_config_path):
            raise FileNotFoundError(f"Resume config not found at {resume_config_path}")

        base_config = load_config_from_file(resume_config_path)
        base_config["training"]["resume"] = True
        base_config["training"]["resume_checkpoint"] = os.path.join(args.resume_from, "checkpoints/latest.pt")
        print(f"Resuming from run directory: {args.resume_from}")
    # Load config, starting with default, then resume config, then user file.
    config = load_config(args.config, base_config=base_config)
    for override_str in args.override_param:
        if "=" not in override_str:
            raise ValueError(f"Invalid override format: {override_str}. Use 'key=value'.")
        key, raw_value = override_str.split("=", 1)
        value = parse_value(raw_value)
        deep_set(config, key, value)
        print(f"Overriding config: {key} = {value}")

    train(config)