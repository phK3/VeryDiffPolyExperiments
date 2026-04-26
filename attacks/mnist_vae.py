"""Simple MNIST VAE library (Kingma-style, MLP, Bernoulli likelihood).

Usage:
    from mnist_vae import train_vae, encode, decode, sample
    encoder, decoder = train_vae(train_loader, epochs=10)
    z = encode(encoder, x)          # inference: x -> mu
    x_hat = decode(decoder, z)      # inference: z -> x (flattened, in [0,1])
    imgs = sample(decoder, n=16)    # prior samples
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


class Encoder(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dim: int = 400, latent_dim: int = 20):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.view(x.size(0), -1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 20, hidden_dim: int = 400, output_dim: int = 784):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        return torch.sigmoid(self.fc4(h))  # flat (B, 784) in [0,1]

def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    return mu + std * torch.randn_like(std)


def _vae_loss(recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float) -> torch.Tensor:
    x_flat = x.view(x.size(0), -1)
    # Summed over pixels and batch (standard Kingma convention).
    bce = F.binary_cross_entropy(recon, x_flat, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + beta * kld


def train_vae(
    dataloader,
    latent_dim: int = 20,
    hidden_dim: int = 400,
    epochs: int = 10,
    lr: float = 1e-3,
    beta: float = 1.0,
    device: str | torch.device | None = None,
    verbose: bool = True,
) -> tuple[Encoder, Decoder]:
    """Train a VAE on an MNIST-like dataloader yielding (image, label) or image tensors.

    Inputs are expected to be in [0,1] (use torchvision.transforms.ToTensor()).
    Returns (encoder, decoder), both in eval mode.
    """
    device = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    encoder = Encoder(hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
    decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    opt = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)

    encoder.train(); decoder.train()
    for epoch in range(1, epochs + 1):
        total, n = 0.0, 0
        for batch in dataloader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            mu, logvar = encoder(x)
            z = _reparameterize(mu, logvar)
            recon = decoder(z)
            loss = _vae_loss(recon, x, mu, logvar, beta=beta)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            n += x.size(0)
        if verbose:
            print(f"Epoch {epoch}/{epochs}  avg loss/example: {total / n:.4f}")

    encoder.eval(); decoder.eval()
    return encoder, decoder


@torch.no_grad()
def encode(encoder: Encoder, x: torch.Tensor, sample: bool = False) -> torch.Tensor:
    """x -> z. Returns mu by default; set sample=True for reparameterized draw."""
    mu, logvar = encoder(x)
    return _reparameterize(mu, logvar) if sample else mu


@torch.no_grad()
def decode(decoder: Decoder, z: torch.Tensor, as_image: bool = False) -> torch.Tensor:
    """z -> reconstruction in [0,1]. Flat (B, 784) by default, (B,1,28,28) if as_image."""
    out = decoder(z)
    return out.view(-1, 1, 28, 28) if as_image else out


@torch.no_grad()
def sample(decoder: Decoder, n: int, latent_dim: int = 20, device: str | torch.device = "cpu",
           as_image: bool = True) -> torch.Tensor:
    """Draw n samples from the prior N(0, I) and decode them."""
    device = torch.device(device)
    z = torch.randn(n, latent_dim, device=device)
    return decode(decoder, z, as_image=as_image)