from pathlib import Path
import json
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvSAE(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc = nn.Sequential(
            nn.Conv2d(1,16,3,2,1), nn.ReLU(inplace=True),   # 224->112
            nn.Conv2d(16,32,3,2,1), nn.ReLU(inplace=True),  # 112->56
            nn.Conv2d(32,64,3,2,1), nn.ReLU(inplace=True),  # 56->28
        )
        self.enc_lin = nn.Linear(64*28*28, latent_dim)
        self.dec_lin = nn.Linear(latent_dim, 64*28*28)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64,32,4,2,1), nn.ReLU(inplace=True),  # 28->56
            nn.ConvTranspose2d(32,16,4,2,1), nn.ReLU(inplace=True),  # 56->112
            nn.ConvTranspose2d(16,1, 4,2,1), nn.Sigmoid(),           # 112->224
        )
        for m in self.modules():
            if isinstance(m,(nn.Conv2d,nn.ConvTranspose2d,nn.Linear)):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def encode(self, x):
        h = self.enc(x).flatten(1)
        return F.relu(self.enc_lin(h))  # non-negative

    def decode(self, z):
        h = self.dec_lin(z).view(-1,64,28,28)
        return self.dec(h)

    def forward(self, x):
        z = self.encode(x); xhat = self.decode(z)
        return xhat, z


def load_image(path, size=224):
    img = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    arr = (np.asarray(img, dtype=np.float32) / 255.0)[None, None, ...]  # (1,1,H,W)
    return torch.from_numpy(arr)

def _normalize01(t: torch.Tensor):
    mn, mx = t.min(), t.max()
    return (t - mn) / (mx - mn + 1e-8)

def threshold_mask(delta, thr=0.30, use_percentile=False, p=90):
    """
    delta: (H,W) in [0,1] (torch or np)
    returns boolean np.ndarray (H,W)
    """
    d = delta.numpy() if hasattr(delta, "numpy") else np.asarray(delta)
    if use_percentile:
        v = np.percentile(d.ravel(), p)
        return (d >= v)
    return (d >= thr)

def topk_latents_by_activation(z: torch.Tensor, k=3):
    vals, idx = torch.topk(z.squeeze(0), k)
    return idx.tolist(), vals.tolist()

class SAEWrapper:
    def __init__(self, run_dir: str):
        self.run_dir = Path(run_dir)
        self.cfg = json.loads((self.run_dir / "config.json").read_text())
        self.device = torch.device("cpu")
        self.model = ConvSAE(latent_dim=self.cfg["latent_dim"]).to(self.device)
        state = torch.load(self.run_dir / "model.pt", map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor):
        x = x.to(self.device)
        xhat, z = self.model(x)
        return xhat.cpu(), z.cpu()

    @torch.no_grad()
    def ablation_heatmap(self, x: torch.Tensor, j: int):
        """
        Returns:
          delta: (H,W) float32 in [0,1]  -- |xhat - xhat(j=0)|
          xhat:  (H,W) float32 in [0,1]  -- baseline reconstruction
        """
        x = x.to(self.device)
        xhat, z = self.model(x)                # (1,1,H,W), (1,m)
        z0 = z.clone(); z0[:, j] = 0.0         # ablate latent j
        xhat_j = self.model.decode(z0)
        delta = (xhat - xhat_j).abs().cpu()
        delta = _normalize01(delta)[0,0]
        return delta, xhat.cpu()[0,0]

    @torch.no_grad()
    def ablation_heatmap_with_base(self, x, j):
        """
        Same heatmap as ablation_heatmap, but returns the ORIGINAL image as base.
        """
        x_cpu = x.clone().cpu()[0,0]               # original [0,1]
        x = x.to(self.device)
        xhat, z = self.model(x)
        z0 = z.clone(); z0[:, j] = 0.0
        xhat_j = self.model.decode(z0)
        delta = (xhat - xhat_j).abs().cpu()
        delta = _normalize01(delta)[0,0]           # (H,W) in [0,1]
        return delta, x_cpu

