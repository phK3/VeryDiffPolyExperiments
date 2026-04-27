"""
json_to_pytorch.py
Convert Julia/Flux-serialized neural network JSON to a PyTorch nn.Sequential module.
"""

import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reshape_fortran(flat, shape, dtype):
    """Reshape flat list to shape using Fortran (column-major) order."""
    arr = np.array(flat, dtype=np.float64).reshape(shape, order='F')
    return torch.tensor(arr, dtype=dtype)


# ---------------------------------------------------------------------------
# Custom modules
# ---------------------------------------------------------------------------

class FrozenBatchNorm(nn.Module):
    """
    Applies batch normalization using stored running statistics.
    Always in eval mode; no gradients on parameters.
    Works for both 1-D (linear) and 2-D (conv) feature maps.
    """

    def __init__(self, gamma, beta, mu, var, eps, dtype):
        super().__init__()
        self.eps = eps
        self.register_buffer('weight',       torch.tensor(gamma, dtype=dtype))
        self.register_buffer('bias',         torch.tensor(beta,  dtype=dtype))
        self.register_buffer('running_mean', torch.tensor(mu,    dtype=dtype))
        self.register_buffer('running_var',  torch.tensor(var,   dtype=dtype))

    def forward(self, x):
        # Reshape for broadcasting: (1, C, 1, 1) for 4-D, (1, C) for 2-D
        shape = [1, -1] + [1] * (x.dim() - 2)
        w   = self.weight.view(shape)
        b   = self.bias.view(shape)
        mu  = self.running_mean.view(shape)
        var = self.running_var.view(shape)
        return (x - mu) / torch.sqrt(var + self.eps) * w + b


class ChebyshevPoly(nn.Module):
    """
    Element-wise Chebyshev polynomial via Clenshaw recurrence.
    coeffs: (n_neurons, degree+1)
    l, u:   (n_neurons,)  — per-neuron input bounds for normalisation.
    """

    def __init__(self, coeffs, l, u, dtype):
        super().__init__()
        self.register_buffer('coeffs', torch.tensor(coeffs, dtype=dtype))
        self.register_buffer('l',      torch.tensor(l,      dtype=dtype))
        self.register_buffer('u',      torch.tensor(u,      dtype=dtype))

    def forward(self, x):
        original_shape = x.shape
        # Flatten to (batch, n_neurons)
        x_flat = x.reshape(original_shape[0], -1)

        l = self.l
        u = self.u
        coeffs = self.coeffs          # (n_neurons, degree+1)
        degree = coeffs.shape[1] - 1

        # Normalise to [-1, 1]
        x_norm = (x_flat - 0.5 * (u + l)) / (0.5 * (u - l))  # (batch, n_neurons)

        # Clenshaw recurrence
        b_k1 = torch.zeros_like(x_norm)
        b_k2 = torch.zeros_like(x_norm)
        for k in range(degree, 0, -1):
            b_k = 2.0 * x_norm * b_k1 - b_k2 + coeffs[:, k]
            b_k2 = b_k1
            b_k1 = b_k

        result = x_norm * b_k1 - b_k2 + coeffs[:, 0]
        return result.reshape(original_shape)


class Reshape(nn.Module):
    """
    Reshape to a fixed target shape (excluding batch dim).
    shape stored in PyTorch NCHW order; batch dim is always inferred.
    """

    def __init__(self, torch_shape):
        super().__init__()
        # torch_shape includes N at index 0; we ignore it at runtime
        self.shape = torch_shape  # e.g. (N, C, H, W)

    def forward(self, x):
        return x.reshape(x.shape[0], *self.shape[1:])


class AsymmetricPaddedConv(nn.Module):
    """Wraps nn.Conv2d with asymmetric zero-padding applied before the conv."""

    def __init__(self, conv, pad):
        super().__init__()
        self.conv = conv
        # pad order for F.pad (last dim first): (left, right, top, bottom)
        # Flux pad: (padW0, padW1, padH0, padH1)
        padW0, padW1, padH0, padH1 = pad
        self.pad = (padW0, padW1, padH0, padH1)

    def forward(self, x):
        x = F.pad(x, self.pad)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Layer builders
# ---------------------------------------------------------------------------

def _build_linear(layer, dtype):
    flat_w, flat_b = layer[1], layer[2]
    out_f = len(flat_b)
    in_f  = len(flat_w) // out_f
    mod = nn.Linear(in_f, out_f)
    w = _reshape_fortran(flat_w, (out_f, in_f), dtype)
    b = torch.tensor(flat_b, dtype=dtype)
    with torch.no_grad():
        mod.weight.copy_(w)
        mod.bias.copy_(b)
    return mod


def _build_conv(layer, dtype):
    _, (kW, kH), in_ch, out_ch, flat_w, flat_b, (sW, sH), pad = layer
    padW0, padW1, padH0, padH1 = pad

    symmetric = (padW0 == padW1) and (padH0 == padH1)
    padding   = (padH0, padW0) if symmetric else 0

    conv = nn.Conv2d(
        in_channels=in_ch,
        out_channels=out_ch,
        kernel_size=(kH, kW),
        stride=(sH, sW),
        padding=padding,
        bias=True,
    )

    # Weight: reshape to (kW, kH, in_ch, out_ch) F-order → transpose to (out_ch, in_ch, kH, kW)
    w = _reshape_fortran(flat_w, (kW, kH, in_ch, out_ch), dtype)
    w = w.permute(3, 2, 1, 0)  # (out_ch, in_ch, kH, kW)
    w = w.flip([2, 3])          # flip spatial dims to match Flux's true convolution
    b = torch.tensor(flat_b, dtype=dtype)
    with torch.no_grad():
        conv.weight.copy_(w)
        conv.bias.copy_(b)

    if not symmetric:
        return AsymmetricPaddedConv(conv, (padW0, padW1, padH0, padH1))
    return conv


def _build_batchnorm(layer, dtype):
    _, flat_gamma, flat_beta, flat_mu, flat_var, eps, num_channels = layer
    return FrozenBatchNorm(flat_gamma, flat_beta, flat_mu, flat_var, eps, dtype)


def _build_chebyshev(layer, dtype):
    _, flat_coeffs, flat_l, flat_u, _ = layer
    n_neurons = len(flat_l)
    degree_plus_1 = len(flat_coeffs) // n_neurons
    coeffs = np.array(flat_coeffs, dtype=np.float64).reshape(
        (n_neurons, degree_plus_1), order='F'
    )
    l = np.array(flat_l, dtype=np.float64)
    u = np.array(flat_u, dtype=np.float64)
    return ChebyshevPoly(coeffs, l, u, dtype)


def _build_reshape(layer, dtype):
    flux_shape = layer[1]  # e.g. [W, H, C, N] in Flux WHCN
    torch_shape = list(reversed(flux_shape))  # [N, C, H, W]
    return Reshape(torch_shape)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def json_to_pytorch(path: str, double_precision: bool = False) -> nn.Module:
    """
    Load a JSON-serialized neural network and return an nn.Sequential module.

    Args:
        path: Path to JSON file produced by Julia's `convert_to_list_of_tuples`.
        double_precision: If True, use float64 tensors; otherwise float32.

    Returns:
        nn.Sequential ready for eval or training.
    """
    dtype = torch.float64 if double_precision else torch.float32

    with open(path, 'r') as f:
        layer_list = json.load(f)

    modules = []
    for layer in layer_list:
        if isinstance(layer, str):
            tag = layer
        else:
            tag = layer[0]
        if tag == 'linear':
            modules.append(_build_linear(layer, dtype))
        elif tag == 'conv':
            modules.append(_build_conv(layer, dtype))
        elif tag == 'batchnorm':
            modules.append(_build_batchnorm(layer, dtype))
        elif tag == 'relu':
            modules.append(nn.ReLU())
        elif tag == 'gelu':
            modules.append(nn.GELU())
        elif tag == 'flatten':
            modules.append(nn.Flatten(start_dim=1))
        elif tag == 'reshape':
            modules.append(_build_reshape(layer, dtype))
        elif tag == 'chebyshev':
            modules.append(_build_chebyshev(layer, dtype))
        else:
            raise ValueError(f"Unknown layer type: {tag!r}")

    model = nn.Sequential(*modules)

    if double_precision:
        model = model.double()

    return model
