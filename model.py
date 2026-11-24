from __future__ import annotations

import math
import os
import pickle
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    if timesteps.ndim != 1:
        raise RuntimeError(f"timesteps must be 1D, got {tuple(timesteps.shape)}")
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(0, half, device=timesteps.device).float()
        / max(half - 1, 1)
    )
    args = timesteps.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class TimeMLP(nn.Module):
    def __init__(self, time_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(time_dim, out_dim * 4),
            nn.SiLU(),
            nn.Linear(out_dim * 4, out_dim),
        )

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        return self.net(t_emb)


def _valid_groups(ch: int, max_groups: int = 8) -> int:
    g = min(max_groups, ch)
    while g > 1 and (ch % g) != 0:
        g -= 1
    return g


class ResConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_ch: int, downsample: bool = False, eps: float = 1e-3):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = None
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1, stride=stride)

        self.norm1 = nn.GroupNorm(_valid_groups(out_ch), out_ch, eps=eps)
        self.norm2 = nn.GroupNorm(_valid_groups(out_ch), out_ch, eps=eps)

        self.time_proj = nn.Linear(time_ch, out_ch)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        tb = self.time_proj(t_emb).view(b, -1, 1, 1)

        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h + tb)

        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h + tb)

        s = x if self.skip is None else self.skip(x)
        return s + h


class MSA2DBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, time_ch: int, mlp_ratio: float = 2.0, eps: float = 1e-3):
        super().__init__()
        self.dim = dim
        self.ln1 = nn.LayerNorm(dim, eps=eps)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim, eps=eps)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )
        self.time_to_shift = nn.Linear(time_ch, dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if c != self.dim:
            raise RuntimeError(f"attention dim mismatch: got {c}, expected {self.dim}")
        tokens = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        shift = self.time_to_shift(t_emb).unsqueeze(1)

        y = self.ln1(tokens + shift)
        attn_out, _ = self.attn(y, y, y, need_weights=False)
        tokens = tokens + attn_out

        y = self.ln2(tokens + shift)
        tokens = tokens + self.mlp(y)

        return tokens.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()


class FuseBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_ch: int, eps: float = 1e-3):
        super().__init__()
        self.block1 = ResConvBlock(in_ch, out_ch, time_ch=time_ch, downsample=False, eps=eps)
        self.block2 = ResConvBlock(out_ch, out_ch, time_ch=time_ch, downsample=False, eps=eps)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.block1(x, t_emb)
        x = self.block2(x, t_emb)
        return x


@dataclass
class AlterNetUNetTinyConfig:
    image_size: Tuple[int, int] = (32, 32)
    in_ch: int = 1
    time_dim: int = 64
    c1: int = 32
    c2: int = 64
    c3: int = 128
    c4: int = 128
    h2: int = 4
    h3: int = 8
    h4: int = 8
    r2: float = 2.0
    r3: float = 2.0
    r4: float = 2.0
    norm_eps: float = 1e-3


class DiffusionForImageChange(nn.Module):
    def __init__(self, size: tuple = (32, 32), config: Optional[AlterNetUNetTinyConfig] = None):
        super().__init__()
        if config is None:
            config = AlterNetUNetTinyConfig(image_size=size)
        else:
            config.image_size = size

        self.cfg = config
        self.size = config.image_size
        self.input_size = (config.in_ch, self.size[0], self.size[1])
        self.diffusion_config = None

        self.time_mlp = TimeMLP(config.time_dim, config.time_dim)

        self.stem = nn.Sequential(
            nn.Conv2d(config.in_ch, config.c1, 3, padding=1),
            nn.GroupNorm(_valid_groups(config.c1), config.c1, eps=config.norm_eps),
            nn.SiLU(),
        )

        self.s1_b1 = ResConvBlock(config.c1, config.c1, time_ch=config.time_dim, downsample=False, eps=config.norm_eps)
        self.s1_b2 = ResConvBlock(config.c1, config.c1, time_ch=config.time_dim, downsample=False, eps=config.norm_eps)

        self.s2_down = ResConvBlock(config.c1, config.c2, time_ch=config.time_dim, downsample=True, eps=config.norm_eps)
        self.s2_conv = ResConvBlock(config.c2, config.c2, time_ch=config.time_dim, downsample=False, eps=config.norm_eps)
        self.s2_attn = MSA2DBlock(config.c2, config.h2, time_ch=config.time_dim, mlp_ratio=config.r2, eps=config.norm_eps)

        self.s3_down = ResConvBlock(config.c2, config.c3, time_ch=config.time_dim, downsample=True, eps=config.norm_eps)
        self.s3_conv = ResConvBlock(config.c3, config.c3, time_ch=config.time_dim, downsample=False, eps=config.norm_eps)
        self.s3_attn = MSA2DBlock(config.c3, config.h3, time_ch=config.time_dim, mlp_ratio=config.r3, eps=config.norm_eps)

        self.s4_down = ResConvBlock(config.c3, config.c4, time_ch=config.time_dim, downsample=True, eps=config.norm_eps)
        self.s4_conv = ResConvBlock(config.c4, config.c4, time_ch=config.time_dim, downsample=False, eps=config.norm_eps)
        self.s4_attn = MSA2DBlock(config.c4, config.h4, time_ch=config.time_dim, mlp_ratio=config.r4, eps=config.norm_eps)

        self.up3 = nn.ConvTranspose2d(config.c4, config.c3, kernel_size=2, stride=2)
        self.dec3 = FuseBlock(config.c3 + config.c3, config.c3, time_ch=config.time_dim, eps=config.norm_eps)
        self.dec3_attn = MSA2DBlock(config.c3, config.h3, time_ch=config.time_dim, mlp_ratio=config.r3, eps=config.norm_eps)

        self.up2 = nn.ConvTranspose2d(config.c3, config.c2, kernel_size=2, stride=2)
        self.dec2 = FuseBlock(config.c2 + config.c2, config.c2, time_ch=config.time_dim, eps=config.norm_eps)
        self.dec2_attn = MSA2DBlock(config.c2, config.h2, time_ch=config.time_dim, mlp_ratio=config.r2, eps=config.norm_eps)

        self.up1 = nn.ConvTranspose2d(config.c2, config.c1, kernel_size=2, stride=2)
        self.dec1 = FuseBlock(config.c1 + config.c1, config.c1, time_ch=config.time_dim, eps=config.norm_eps)

        self.out = nn.Sequential(
            nn.Conv2d(config.c1, config.c1, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(config.c1, config.in_ch, 3, padding=1),
        )

        self.reset_parameters()
        self.float()

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in", nonlinearity="relu")
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                if getattr(m, "weight", None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise RuntimeError(f"x must be BCHW, got {tuple(x.shape)}")
        if x.shape[1] != self.cfg.in_ch:
            raise RuntimeError(f"channel mismatch: got {x.shape[1]}, expected {self.cfg.in_ch}")
        if t.ndim != 1 or t.shape[0] != x.shape[0]:
            raise RuntimeError(f"t must be (B,), got {tuple(t.shape)}")
        t = t.to(dtype=torch.long)

        t_sin = sinusoidal_timestep_embedding(t, self.cfg.time_dim)
        t_emb = self.time_mlp(t_sin)

        h = self.stem(x)

        h = self.s1_b1(h, t_emb)
        h = self.s1_b2(h, t_emb)
        skip1 = h

        h = self.s2_down(h, t_emb)
        h = self.s2_conv(h, t_emb)
        h = self.s2_attn(h, t_emb)
        skip2 = h

        h = self.s3_down(h, t_emb)
        h = self.s3_conv(h, t_emb)
        h = self.s3_attn(h, t_emb)
        skip3 = h

        h = self.s4_down(h, t_emb)
        h = self.s4_conv(h, t_emb)
        h = self.s4_attn(h, t_emb)

        h = self.up3(h)
        if h.shape[-2:] != skip3.shape[-2:]:
            h = F.interpolate(h, size=skip3.shape[-2:], mode="nearest")
        h = torch.cat([h, skip3], dim=1)
        h = self.dec3(h, t_emb)
        h = self.dec3_attn(h, t_emb)

        h = self.up2(h)
        if h.shape[-2:] != skip2.shape[-2:]:
            h = F.interpolate(h, size=skip2.shape[-2:], mode="nearest")
        h = torch.cat([h, skip2], dim=1)
        h = self.dec2(h, t_emb)
        h = self.dec2_attn(h, t_emb)

        h = self.up1(h)
        if h.shape[-2:] != skip1.shape[-2:]:
            h = F.interpolate(h, size=skip1.shape[-2:], mode="nearest")
        h = torch.cat([h, skip1], dim=1)
        h = self.dec1(h, t_emb)

        return self.out(h)

    def compute_loss(self, x0: torch.Tensor, t: torch.Tensor, schedule) -> torch.Tensor:
        xt, noise = schedule.q_sample(x0, t)
        pred = self.forward(xt, t)
        return F.mse_loss(pred, noise)

    def train_step(self, x0: torch.Tensor, t: torch.Tensor, schedule, optimizer) -> float:
        optimizer.zero_grad(set_to_none=True)
        loss = self.compute_loss(x0, t, schedule)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        optimizer.step()
        return float(loss.item())

    def train_model(self, train_loader, schedule, optimizer, epochs: int = 10):
        self.train()
        device = next(self.parameters()).device
        for epoch in range(epochs):
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
            running = 0.0
            n = 0
            for x0, _ in pbar:
                x0 = x0.to(device, dtype=torch.float32)
                t = schedule.sample_timestep(x0.shape[0]).to(device, dtype=torch.long)
                loss = self.train_step(x0, t, schedule, optimizer)
                running += loss
                n += 1
                pbar.set_postfix(loss=f"{loss:.4f}")
            _ = running / max(1, n)

    @torch.no_grad()
    def denoise_data(self, x_t, t, model, alphas, alphas_cumprod, beta, generator=None):
        device = x_t.device
        b = x_t.shape[0]
        t = t.to(device, dtype=torch.long)

        a_t = alphas.to(device).index_select(0, t).view(b, 1, 1, 1)
        ab_t = alphas_cumprod.to(device).index_select(0, t).view(b, 1, 1, 1)
        b_t = beta.to(device).index_select(0, t).view(b, 1, 1, 1)

        ab_t = torch.clamp(ab_t, 1e-5, 1.0 - 1e-5)
        a_t = torch.clamp(a_t, 1e-5, 1.0)
        one_minus = torch.clamp(1.0 - ab_t, 1e-5, 1.0)

        eps_theta = model(x_t, t)

        mean = (1.0 / torch.sqrt(a_t)) * (x_t - ((1.0 - a_t) / torch.sqrt(one_minus)) * eps_theta)

        noise = torch.randn(x_t.shape, device=device, dtype=x_t.dtype, generator=generator)
        nonzero = (t != 0).float().view(b, 1, 1, 1)
        sigma = torch.sqrt(torch.clamp(b_t, 1e-6, 1.0))

        return mean + nonzero * sigma * noise
    
    @torch.no_grad()
    def sample(self, schedule, n: int, *, generator=None):
        """Run the reverse schedule; this does not assert sample quality."""
        if n < 1:
            raise ValueError("n must be positive")
        parameter = next(self.parameters())
        was_training = self.training
        self.eval()
        try:
            x = torch.randn((n, *self.input_size), device=parameter.device,
                            dtype=parameter.dtype, generator=generator)
            for step in reversed(range(schedule.T)):
                t = torch.full((n,), step, device=parameter.device, dtype=torch.long)
                x = self.denoise_data(x, t, self, schedule.alphas,
                                      schedule.alpha_bars, schedule.betas, generator)
            return x
        finally:
            self.train(was_training)

    def save_model(self, optimizer, path: str, epoch: int | None = None, *, diffusion_config=None):
        """Atomically save tensors/primitive metadata; AdamW is the supported optimizer."""
        if optimizer is not None and not isinstance(optimizer, torch.optim.AdamW):
            raise ValueError("Checkpoint optimizer must be AdamW or None")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "format_version": 1,
            "model_state": self.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "optimizer_name": "AdamW" if optimizer is not None else None,
            "epoch": epoch,
            "config": asdict(self.cfg),
            "diffusion_config": diffusion_config or self.diffusion_config,
        }
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
        try:
            torch.save(state, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def load_model(path: str, device: str = "cpu", *, restore_optimizer: bool = True):
        """Restore model and AdamW bound to its parameters using safe tensor loading.

        Legacy checkpoints containing pickled dataclasses must be regenerated.
        Unrestricted pickle loading is deliberately unsupported.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        try:
            ckpt = torch.load(path, map_location=device, weights_only=True)
        except pickle.UnpicklingError as error:
            raise ValueError("Unsafe or legacy checkpoint format; regenerate it with this version. "
                             "Unrestricted pickle loading is not supported.") from error
        if not isinstance(ckpt, dict) or ckpt.get("format_version") != 1 or not isinstance(ckpt.get("config"), dict):
            raise ValueError("Unsupported checkpoint format; expected version 1 with a configuration dictionary")
        config = AlterNetUNetTinyConfig(**ckpt["config"])
        model = DiffusionForImageChange(size=config.image_size, config=config).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.diffusion_config = ckpt.get("diffusion_config")
        model.eval()
        optimizer = None
        if restore_optimizer and ckpt.get("optimizer_state") is not None:
            if ckpt.get("optimizer_name") != "AdamW":
                raise ValueError("Unsupported checkpoint optimizer")
            optimizer = torch.optim.AdamW(model.parameters())
            optimizer.load_state_dict(ckpt["optimizer_state"])
        epoch = ckpt.get("epoch", None)
        return model, optimizer, epoch
