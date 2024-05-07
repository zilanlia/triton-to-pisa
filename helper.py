import torch

def rand_strided(
    size,
    stride,
    dtype,
    device,
    extra_size: int = 0,
):
    needed_size = (
        sum((shape - 1) * stride for shape, stride in zip(size, stride))
        + 1
        + extra_size
    )
    if dtype.is_floating_point:
        buffer = torch.randn(needed_size, dtype=dtype, device=device)
    else:
        buffer = torch.zeros(size=[needed_size], dtype=dtype, device=device)
    return torch.as_strided(buffer, size, stride)

def size(shape):
    size = 1
    for dim in shape:
        size = size * dim
    return size