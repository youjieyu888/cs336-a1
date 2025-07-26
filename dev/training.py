import math
import os
from typing import Iterable, BinaryIO, IO

import torch

@torch.compile
def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    max_logits = logits.max(dim=-1, keepdim=True).values
    shifted_logits = logits - max_logits
    log_sum_exp = shifted_logits.exp().sum(dim=-1).log()
    target_logits = shifted_logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return - (target_logits - log_sum_exp).mean()


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        **kwargs,
    ):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)


    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            b1, b2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                t = state.get("t", 1)

                grad = p.grad

                state["m"] = b1 * m + (1 - b1) * grad
                state["v"] = b2 * v + (1 - b2) * grad.pow(2)

                step_size = lr * (math.sqrt(1 - b2**t) / (1 - b1**t))

                p.addcdiv_(state["m"], torch.sqrt(state["v"]) + eps, value=-step_size)

                if wd != 0:
                    p.add_(p.data, alpha=-lr * wd)

                state["t"] = t + 1

        return loss

def lr_cosine_schedule(it: int, lr_max: float, lr_min: float, warmup_iters: int, cosine_cycle_iters: int):
    if it < warmup_iters:
        return (it / warmup_iters) * lr_max

    if it <= cosine_cycle_iters:
        decay_step = it - warmup_iters
        decay_steps = cosine_cycle_iters - warmup_iters
        cos = math.cos((decay_step / decay_steps) * math.pi)
        return lr_min + 1 / 2 * (1 + cos) * (lr_max - lr_min)

    return lr_min



def lr_linear_schedule(it: int, lr_max: float, lr_min: float, warmup_iters: int, linear_cycle_iters: int):
    if it < warmup_iters:
        return (it / warmup_iters) * lr_max

    if it <= linear_cycle_iters:
        decay_step = it - warmup_iters
        decay_steps = linear_cycle_iters - warmup_iters
        return lr_max - (decay_step / decay_steps) * (lr_max - lr_min)

    return lr_min


def lr_double_schedule(
    it: int,
    lr_max: float,
    lr_inter: int,
    lr_min: float,
    warmup_iters: int,
    phase_one_iters: int,
    phase_two_iters: int,
    phase_two_type: str,
):
    """
    A double-decay learning rate schedule.

    Args:
        it: The current iteration.
        lr_max: Max. LR, to which we warm up linearly.
        lr_inter: LR to which we decay exponentially from lr_max.
        lr_min: Min. LR, to which we decay from lr_inter, linearly or cosine.
        warmup_iters: The number of iters for linear warmup from zero to lr_max
        exp_decay_iters: The iter at which the exponential decay phase should end
        phase_two_iters: The iter at which the second decay phase (linear or cosine) should end
        phase_two_type: The type of decay to use for the second phase (linear or cosine)

    Note:
        - exp_decay_iters is NOT the number of iterations for the exponential decay phase.
          It is the iter at which the exponential decay should end.
        - phase_two_iters is NOT the number of iterations for the second decay phase.
          It is the iter at which the second decay should end.
    Example:
        - Want: warmup for 1000 iters, exp decay for 1000 iters, linear decay for 1000 iters
        - Set:
            warmup_iters = 1000
            exp_decay_iters = 2000
            phase_two_iters = 3000
            phase_two_type = "linear"
    """
    if it < warmup_iters:
        # We're in the warmup phase
        return (it / warmup_iters) * lr_max

    if it <= phase_one_iters:
        # We're in the exponential decay phase
        decay_step = it - warmup_iters
        decay_steps = phase_one_iters - warmup_iters
        return lr_max * (lr_inter / lr_max) ** (decay_step / decay_steps)

    # We're in phase two (linear or cosine decay from lr_inter to lr_min)
    it2 = it - phase_one_iters
    phase_two_decay_steps = phase_two_iters - phase_one_iters

    if phase_two_type == "linear":
        # The second decay phase of the schedule is linear
        return lr_linear_schedule(
            it2, lr_max=lr_inter, lr_min=lr_min, warmup_iters=0, linear_cycle_iters=phase_two_decay_steps
        )

    if phase_two_type == "cosine":
        # The second decay phase of the schedule is cosine
        return lr_cosine_schedule(
            it2, lr_max=lr_inter, lr_min=lr_min, warmup_iters=0, cosine_cycle_iters=phase_two_decay_steps
        )

    return lr_min

@torch.compile
def gradient_clip(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    device = grads[0].device
    total_sq = torch.zeros((), device=device)
    for g in grads:
        total_sq += g.pow(2).sum()
    total_norm = total_sq.sqrt()
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + 1e-6)
        with torch.no_grad():
            for g in grads:
                g.mul_(scale)
    return total_norm

def save_checkpoint(
    model: torch.nn.Module,
    optimizers: list[torch.optim.Optimizer] | torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    # Extract the original model from a compiled module if present
    orig_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    if isinstance(optimizers, torch.optim.Optimizer):
        optimizers = [optimizers]

    torch.save(
        {
            "model": orig_model.state_dict(),
            "optimizer": [optimizer.state_dict() for optimizer in optimizers],
            "iteration": iteration,
        },
        out,
    )


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module | None = None,
    optimizers: list[torch.optim.Optimizer] | torch.optim.Optimizer | None = None,
):
    checkpoint = torch.load(src)

    if model is not None:
        model.load_state_dict(checkpoint["model"])

    if optimizers is not None:
        if isinstance(optimizers, torch.optim.Optimizer):
            optimizers = [optimizers]

        for optimizer, state_dict in zip(optimizers, checkpoint["optimizer"]):
            optimizer.load_state_dict(state_dict)

    return checkpoint["iteration"]



