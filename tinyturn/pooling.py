"""
Endpoint-aware pooling (Section 3 "Pooling"): combine (1) the final valid temporal state, (2)
masked statistics over the final window, (3) a small attention pool over the final window --
explicitly *not* plain mean pooling across the full input, per D6's own finding (signal
concentrates near the boundary) and the external reference solution's ablation (mean pooling was
their worst option).
"""
import torch
import torch.nn as nn


class EndpointPooling(nn.Module):
    def __init__(self, dim: int, window_frames: int = 100):
        super().__init__()
        self.window_frames = window_frames
        self.attn = nn.Linear(dim, 1)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """h: (B, T, C) per-frame hidden states. mask: (B, T) bool, True = valid frame."""
        B, T, C = h.shape
        device = h.device
        idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)

        masked_idx = torch.where(mask, idx, torch.full_like(idx, -1))
        last_valid_idx = masked_idx.max(dim=1).values.clamp(min=0)  # (B,)
        last_state = h[torch.arange(B, device=device), last_valid_idx]  # (B, C)

        window_start = (last_valid_idx - self.window_frames + 1).clamp(min=0)
        win_mask = (idx >= window_start.unsqueeze(1)) & (idx <= last_valid_idx.unsqueeze(1)) & mask
        win_mask_f = win_mask.unsqueeze(-1).float()
        count = win_mask_f.sum(dim=1).clamp(min=1.0)
        mean = (h * win_mask_f).sum(dim=1) / count
        var = ((h - mean.unsqueeze(1)) ** 2 * win_mask_f).sum(dim=1) / count
        std = torch.sqrt(var + 1e-6)

        scores = self.attn(h).squeeze(-1)  # (B, T)
        scores = scores.masked_fill(~win_mask, float("-1e9"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        attn_pool = (h * weights).sum(dim=1)

        return torch.cat([last_state, mean, std, attn_pool], dim=-1)  # (B, 4C)

    @property
    def out_multiplier(self):
        return 4
