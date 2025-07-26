import numpy.typing as npt
import numpy as np
import torch

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str, rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(dataset) - context_length -1
    rng = rng or np.random.default_rng()
    starts = rng.integers(0, max_start+1, size =batch_size, dtype=np.int64)
    offsets = np.arange(context_length +1,dtype=np.int64)
    indices = starts[:, None] + offsets[None, :]
    seq = dataset[indices]
    x_np = seq[:, :-1]
    y_np = seq[:, 1:]
    x_batch = torch.as_tensor(x_np, dtype = torch.long)
    y_batch = torch.as_tensor(y_np, dtype = torch.long)
    if device.startswith('cuda'):
        x_batch = x_batch.pin_memory().to(device, non_blocking=True)
        y_batch = y_batch.pin_memory().to(device, non_blocking=True)
    else:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

    return x_batch, y_batch