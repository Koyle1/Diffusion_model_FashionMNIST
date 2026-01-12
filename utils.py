import torch

class DiffusionSchedule:
    def __init__(
        self,
        T: int = 200,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device="mps",
    ):
        self.T = T
        self.device = device

        betas = torch.linspace(beta_start, beta_end, T, device=device)
        betas = torch.clamp(betas, 1e-6, 0.02)

        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        # 🔑 CRITICAL FOR MPS
        alpha_bars = torch.clamp(alpha_bars, 1e-5, 1.0 - 1e-5)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars

        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

    def sample_timestep(self, batch_size: int) -> torch.Tensor:
        return torch.randint(
            0, self.T, (batch_size,),
            device=self.device,
            dtype=torch.long
        )

    @staticmethod
    def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size):
        """
        MPS-safe replacement for gather
        """
        B = t.shape[0]
        out = a.index_select(0, t)
        return out.view(B, *([1] * (len(x_shape) - 1)))

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        if noise is None:
            noise = torch.randn_like(x_0)

        device = x_0.device
        t = t.to(device, dtype=torch.long)

        sqrt_alpha_bar_t = self._extract(
            self.sqrt_alpha_bars.to(device), t, x_0.shape
        )
        sqrt_one_minus_alpha_bar_t = self._extract(
            self.sqrt_one_minus_alpha_bars.to(device), t, x_0.shape
        )

        x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise
        return x_t, noise
