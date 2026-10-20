import numpy as np
import pywt
import torch


def generate_wavelet_kernel(wavelet_name, kernel_size, scale, shift, device=None, eps=1e-8):
    """Return normalized initialization coefficients shaped [C_out, C_in, K].

    Morlet and Mexican Hat use the public implementation's sampling grid,
    ``linspace(-K // 2, K // 2, K)``. Python's floor division gives -4..3 for K=7;
    this explicit convention preserves the provided template rather than silently
    introducing a new waveform. Scales and shifts only define initialization.
    """
    if not isinstance(kernel_size, int) or kernel_size < 1:
        raise ValueError("kernel_size must be a positive integer")
    if eps <= 0:
        raise ValueError("eps must be positive")
    if not isinstance(scale, torch.Tensor) or not isinstance(shift, torch.Tensor):
        raise TypeError("scale and shift must be tensors shaped [C_out, C_in, 1]")
    if scale.ndim != 3 or scale.shape[-1] != 1 or shift.shape != scale.shape:
        raise ValueError("scale and shift must have matching [C_out, C_in, 1] shapes")
    if not scale.is_floating_point():
        raise TypeError("scale must have a floating point dtype")
    device = scale.device if device is None else device
    scale = scale.to(device=device)
    shift = shift.to(device=device, dtype=scale.dtype)
    if not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
        raise ValueError("scale must contain finite, positive values")
    if not bool(torch.isfinite(shift).all()):
        raise ValueError("shift must contain finite values")
    t = torch.as_tensor(
        np.linspace(-kernel_size // 2, kernel_size // 2, kernel_size),
        dtype=scale.dtype, device=device,
    ).view(1, 1, -1)
    if wavelet_name in ("morl", "mexh"):
        normalized_t = (t - shift) / scale
        envelope = torch.exp(-(normalized_t ** 2) / 2)
        if wavelet_name == "morl":
            coefficients = torch.cos(1.75 * normalized_t) * envelope
        else:
            coefficients = (1 - normalized_t ** 2) * envelope
    elif wavelet_name in ("db4", "sym4", "coif4", "haar"):
        phi = pywt.Wavelet(wavelet_name).wavefun(level=kernel_size // 2)[0]
        if len(phi) < kernel_size:
            raise ValueError("The requested scaling-function sample contains fewer than K values")
        coefficients = torch.as_tensor(phi[:kernel_size], dtype=scale.dtype, device=device)
        coefficients = coefficients.view(1, 1, -1).repeat(scale.shape[0], scale.shape[1], 1)
    else:
        raise ValueError(f"Unsupported wavelet family: {wavelet_name}")
    return coefficients / (torch.linalg.vector_norm(coefficients, dim=-1, keepdim=True) + eps)
