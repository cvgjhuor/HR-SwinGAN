import torch
import torch.nn as nn


# Channel attention module for sequence feature maps [Batch, Length, Channels]
class LRCAB(nn.Module):
    def __init__(self, channels: int, reduction: int = 4, use_residual: bool = False):
        """Initialize channel attention module with adaptive pooling and channel excitation."""
        super().__init__()
        self.use_residual = use_residual
        reduced_channels = max(1, channels // reduction)

        # Global average pooling across tokens and two-layer MLP for channel excitation
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute sequence-level channel attention weights and scale input features."""
        b, l, c = x.shape
        # Pool across sequence length L to obtain channel descriptor [B, C]
        y = x.transpose(1, 2)
        y = self.avg_pool(y).view(b, c)
        # Compute channel attention weights [B, 1, C]
        y = self.fc(y).view(b, 1, c)

        # Apply channel scaling (with optional residual addition)
        if self.use_residual:
            return x + x * y
        return x * y


SEBlock = LRCAB
StandardSEBlock = LRCAB
