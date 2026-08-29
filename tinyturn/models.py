"""
TinyTurn-Fusion (Section 3): log-mel branch (DS-CNN + TCN), optional trajectory branch, endpoint-
focused fusion, small fusion MLP. `TinyTurnModel(use_trajectory=False)` is B0 (mel-only); the same
class with `use_trajectory=True` is B1 (or B1-f0, depending on `trajectory_dim`) -- one model class
across Steps 3-4, so the only thing that changes between experiments is config, per the brief's
ground rule against implementing multiple ideas at once.
"""
import torch
import torch.nn as nn

from tinyturn.pooling import EndpointPooling


class DSConvBlock(nn.Module):
    """Depthwise-separable 1D conv block over time, operating on (B, C, T)."""

    def __init__(self, in_ch, out_ch, kernel_size=5, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.depthwise = nn.Conv1d(in_ch, in_ch, kernel_size, padding=padding,
                                    dilation=dilation, groups=in_ch)
        self.pointwise = nn.Conv1d(in_ch, out_ch, 1)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()
        self.residual = (in_ch == out_ch)

    def forward(self, x):
        y = self.pointwise(self.depthwise(x))
        y = self.act(self.bn(y))
        return y + x if self.residual else y


class TCNBlock(nn.Module):
    """Dilated depthwise-separable conv, residual -- the "small temporal TCN" on top of the DS-CNN
    frontend."""

    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        self.block = DSConvBlock(channels, channels, kernel_size=kernel_size, dilation=dilation)

    def forward(self, x):
        return self.block(x)


class BranchEncoder(nn.Module):
    """Shared shape for the mel branch and the trajectory branch: a couple of DS-CNN blocks,
    a small dilated TCN stack, then endpoint-aware pooling. Input (B, T, in_dim) -> pooled (B, out_dim)."""

    def __init__(self, in_dim, channels=64, n_cnn_blocks=2, n_tcn_blocks=3, pool_window_frames=100):
        super().__init__()
        self.stem = nn.Conv1d(in_dim, channels, 1)
        self.cnn_blocks = nn.ModuleList([
            DSConvBlock(channels, channels, kernel_size=5) for _ in range(n_cnn_blocks)
        ])
        self.tcn_blocks = nn.ModuleList([
            TCNBlock(channels, kernel_size=3, dilation=2 ** i) for i in range(n_tcn_blocks)
        ])
        self.pool = EndpointPooling(channels, window_frames=pool_window_frames)
        self.out_dim = channels * self.pool.out_multiplier

    def forward(self, x, mask):
        """x: (B, T, in_dim), mask: (B, T) bool."""
        h = x.transpose(1, 2)  # (B, in_dim, T)
        h = self.stem(h)
        for blk in self.cnn_blocks:
            h = blk(h)
        for blk in self.tcn_blocks:
            h = blk(h)
        h = h.transpose(1, 2)  # (B, T, C)
        return self.pool(h, mask)


class TinyTurnModel(nn.Module):
    def __init__(self, n_mels=40, trajectory_dim=0, mel_channels=112, traj_channels=24,
                 fusion_hidden=64, pool_window_frames=100):
        super().__init__()
        self.use_trajectory = trajectory_dim > 0
        self.mel_encoder = BranchEncoder(n_mels, channels=mel_channels, n_cnn_blocks=2,
                                          n_tcn_blocks=3, pool_window_frames=pool_window_frames)
        fusion_in = self.mel_encoder.out_dim
        if self.use_trajectory:
            self.traj_encoder = BranchEncoder(trajectory_dim, channels=traj_channels,
                                               n_cnn_blocks=2, n_tcn_blocks=2,
                                               pool_window_frames=pool_window_frames)
            fusion_in += self.traj_encoder.out_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden),
            nn.ReLU(),
            nn.Linear(fusion_hidden, 1),
        )

    def forward(self, log_mel, valid_frame_mask, trajectory=None):
        pooled = [self.mel_encoder(log_mel, valid_frame_mask)]
        if self.use_trajectory:
            assert trajectory is not None, "trajectory tensor required when use_trajectory=True"
            pooled.append(self.traj_encoder(trajectory, valid_frame_mask))
        fused = torch.cat(pooled, dim=-1)
        logit = self.fusion(fused).squeeze(-1)
        return logit  # BCEWithLogitsLoss expects raw logits

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
