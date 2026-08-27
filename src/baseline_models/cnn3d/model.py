"""3D U-Net CNN baseline for ESP prediction.

Architecture:
  Encoder:  5 channels → 32 → 64 → 128  (stride-2 conv blocks)
  Decoder:  trilinear upsample + skip connections → 1-channel dense ESP grid
  Query:    F.grid_sample trilinear interpolation at query node xyz positions

Input:  (1, 5, D, H, W) element-occupancy voxel grid — channels are H/C/N/O/S
        counts (see voxelizer.py), no partial charges (same epistemological
        constraint as the GNN baseline)
Output: (Q,) predicted ESP at Q query node positions
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.GELU(),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Vox3DCNN(nn.Module):
    """3D U-Net for dense ESP prediction from voxel grid.

    Predicts at query positions via trilinear interpolation of the output grid.
    """

    def __init__(self) -> None:
        super().__init__()

        # Encoder
        self.enc1 = _ConvBlock(5, 32, stride=1)   # full res
        self.enc2 = _ConvBlock(32, 64, stride=2)  # /2
        self.enc3 = _ConvBlock(64, 128, stride=2) # /4

        # Bottleneck
        self.bottleneck = _ConvBlock(128, 128, stride=1)

        # Decoder
        self.dec2 = _ConvBlock(128 + 64, 64, stride=1)
        self.dec1 = _ConvBlock(64 + 32, 32, stride=1)

        # Output head: 1-channel dense ESP grid
        self.head = nn.Conv3d(32, 1, kernel_size=1)

    def forward(
        self,
        grid: Tensor,           # (1, 5, D, H, W)
        query_coords: Tensor,   # (1, Q, 1, 1, 3) normalised [-1,1]
    ) -> Tensor:
        """Return predicted ESP at query positions, shape (Q,)."""
        e1 = self.enc1(grid)          # (1, 32, D, H, W)
        e2 = self.enc2(e1)            # (1, 64, D/2, H/2, W/2)
        e3 = self.enc3(e2)            # (1, 128, D/4, H/4, W/4)

        b = self.bottleneck(e3)       # (1, 128, D/4, H/4, W/4)

        # Decoder with skip connections
        up2 = F.interpolate(b, size=e2.shape[2:], mode="trilinear", align_corners=True)
        d2 = self.dec2(torch.cat([up2, e2], dim=1))   # (1, 64, D/2, ...)

        up1 = F.interpolate(d2, size=e1.shape[2:], mode="trilinear", align_corners=True)
        d1 = self.dec1(torch.cat([up1, e1], dim=1))   # (1, 32, D, H, W)

        esp_grid = self.head(d1)  # (1, 1, D, H, W)

        # Trilinear interpolation at query positions
        # grid_sample: input (N, C, D, H, W), grid (N, D_out, H_out, W_out, 3)
        sampled = F.grid_sample(
            esp_grid,
            query_coords,            # (1, Q, 1, 1, 3)
            mode="bilinear",         # uses trilinear for 5D input
            padding_mode="border",
            align_corners=True,
        )  # (1, 1, Q, 1, 1)

        return sampled.view(-1)  # (Q,)
