"""Archived kernel generator required by the main network."""

import numpy as np
import pywt
import torch


def generate_wavelet_kernel(wavelet_name, kernel_size, scale, shift, device="cuda"):
    """Generate archived initial kernel coefficients. Forward convolution trains the discrete weight tensors directly."""
    t = np.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)
    t = torch.tensor(t, dtype=torch.float32).view(1, 1, -1).to(device)
    if wavelet_name == "morl":
        t_shifted = t - shift
        wavelet = torch.cos(1.75 * t_shifted / scale) * torch.exp(
            -(t_shifted**2) / (2 * scale**2)
        )
    elif wavelet_name == "db4":
        wavelet = pywt.Wavelet("db4").wavefun(level=kernel_size // 2)[0]
        wavelet = (
            torch.tensor(wavelet[:kernel_size], dtype=torch.float32)
            .view(1, 1, -1)
            .to(device)
        )
        wavelet = wavelet.repeat(scale.shape[0], scale.shape[1], 1)
    elif wavelet_name == "sym4":
        wavelet = pywt.Wavelet("sym4").wavefun(level=kernel_size // 2)[0]
        wavelet = (
            torch.tensor(wavelet[:kernel_size], dtype=torch.float32)
            .view(1, 1, -1)
            .to(device)
        )
        wavelet = wavelet.repeat(scale.shape[0], scale.shape[1], 1)
    elif wavelet_name == "coif4":
        wavelet = pywt.Wavelet("coif4").wavefun(level=kernel_size // 2)[0]
        wavelet = (
            torch.tensor(wavelet[:kernel_size], dtype=torch.float32)
            .view(1, 1, -1)
            .to(device)
        )
        wavelet = wavelet.repeat(scale.shape[0], scale.shape[1], 1)
    elif wavelet_name == "haar":
        wavelet = pywt.Wavelet("haar").wavefun(level=kernel_size // 2)[0]
        wavelet = (
            torch.tensor(wavelet[:kernel_size], dtype=torch.float32)
            .view(1, 1, -1)
            .to(device)
        )
        wavelet = wavelet.repeat(scale.shape[0], scale.shape[1], 1)
    elif wavelet_name == "mexh":
        t_shifted = t - shift
        wavelet = (1 - (t_shifted / scale) ** 2) * torch.exp(
            -(t_shifted**2) / (2 * scale**2)
        )
    else:
        raise ValueError(f"不支持的小波基：{wavelet_name}")
    wavelet = wavelet / torch.norm(wavelet, dim=-1, keepdim=True)
    return wavelet
