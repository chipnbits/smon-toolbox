"""Parts of the U-Net model"""

# Modified from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2 with optional time embedding."""

    def __init__(self, in_channels, out_channels, mid_channels=None, time_emb_dim=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.time_mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, mid_channels * 2))
            if time_emb_dim is not None
            else None
        )
        self.double_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.double_conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, time_emb=None):
        scale_shift = None
        if self.time_mlp is not None and time_emb is not None:
            time_emb = self.time_mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        # Apply the first convolution block
        x = self.double_conv1(x)
        if scale_shift:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        # Apply the second convolution block
        x = self.double_conv2(x)
        return x


class Down(nn.Module):
    """Downscaling with maxpool then double conv, with optional time embedding."""

    def __init__(self, in_channels, out_channels, time_emb_dim=None):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, time_emb_dim=time_emb_dim),
        )

    def forward(self, x, time_emb=None):
        return self.maxpool_conv[1](self.maxpool_conv[0](x), time_emb)


class Up(nn.Module):
    """Upscaling then double conv, with optional time embedding."""

    def __init__(self, in_channels, out_channels, bilinear=True, time_emb_dim=None):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # After upsampling, channels are reduced by 2 for concatenation
        self.conv = DoubleConv(
            in_channels // 2 + out_channels,  # Concatenation of skip connection
            out_channels,
            time_emb_dim=time_emb_dim,
        )

    def forward(self, x1, x2, time_emb=None):
        x1 = self.up(x1)
        # Match the size of x2 using padding
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # Concatenate along channel dimension and apply convolution with time embedding
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x, time_emb)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class RandomFourierEmbedding(nn.Module):
    """
    Time/Positional Embedding with random Fourier features.

    Parameters:
    num_channels: Dimension of the embedding vector
    bandwidth: Bandwidth of the frequencies, use higher frequencies for narrow time windows

    Forward:
    t: time/pos embedding vector with length = batch_size (one scalar per batch element)

    Basis Functions:
    This basis uses frequencies 'f' and phases 'phi' that are drawn from N(0,1) and U(0,1) respectively
    """

    def __init__(self, num_channels, bandwidth=100.0):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(num_channels) * bandwidth, requires_grad=False)
        self.phases = nn.Parameter(torch.rand(num_channels), requires_grad=False)

    def forward(self, t: torch.Tensor):
        # Outer product, with phases added then cosine
        y = t.ger(self.freqs)
        y = y + self.phases
        y = y.cos() * math.sqrt(2)  # scale for unit-variance
        return y


class LearnedFourierEmbedding(RandomFourierEmbedding):
    """A variation with learned frequencies and phases."""

    def __init__(self, num_channels, bandwidth=100.0):
        super().__init__(num_channels, bandwidth)
        # Learnable frequencies and phases
        self.freqs = nn.Parameter(torch.randn(num_channels) * bandwidth)
        self.phases = nn.Parameter(torch.rand(num_channels))


class UNet(nn.Module):
    def __init__(
        self,
        n_channels,
        out_channels,
        dim=32,
        mults=[1, 2, 4, 8],
        time_resolution=64,
        time_bandwidth=1000.0,
        bilinear=False,
    ):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.out_channels = out_channels
        self.layer_dims = [dim * m for m in mults]
        self.bilinear = bilinear

        time_embed = RandomFourierEmbedding(time_resolution, bandwidth=time_bandwidth)
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            time_embed,
            nn.Linear(time_resolution, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Initial layer
        self.inc = DoubleConv(n_channels, self.layer_dims[0])

        # Down layers
        self.downblocks = nn.ModuleList()
        for i in range(len(mults) - 1):
            self.downblocks.append(Down(self.layer_dims[i], self.layer_dims[i + 1]))

        # Bottleneck
        self.bottleneck = DoubleConv(self.layer_dims[-1], self.layer_dims[-1])

        # Up layers
        self.upblocks = nn.ModuleList()
        for i in range(len(mults) - 1, 0, -1):
            self.upblocks.append(
                Up(
                    self.layer_dims[i],  # Concatenated channels
                    self.layer_dims[i - 1],
                    bilinear=False,
                    time_emb_dim=time_dim,
                )
            )

        # Final layer
        self.outc = nn.Conv2d(self.layer_dims[0], out_channels, kernel_size=1)

    def forward(self, x, t):
        t = self.time_mlp(t)
        # Down path
        skips = []
        x = self.inc(x)
        for down in self.downblocks:
            skips.append(x)
            x = down(x, t)

        # Bottleneck
        x = self.bottleneck(x, t)

        # Up path
        for i, up in enumerate(self.upblocks):
            x = up(x, skips[-(i + 1)], t)

        # Final layer
        x = self.outc(x)
        return x


if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize U-Net
    model = UNet(
        n_channels=1,  # Input channels (e.g., grayscale images)
        out_channels=1,  # Output channels (e.g., single channel for segmentation mask)
        dim=32,  # Base dimension for the feature maps
        mults=[1, 2, 4],  # Multipliers for the feature dimensions
        time_resolution=64,  # Resolution of time embedding
        time_bandwidth=1000.0,  # Bandwidth for random Fourier features
        bilinear=True,  # Use bilinear upsampling
    ).to(device)

    # Dummy input and time embeddings
    x = torch.randn(4, 1, 28, 28).to(device)  # Batch of 4 grayscale images
    t = torch.randn(4).to(device)  # Batch of 4 time embeddings

    # Forward pass
    output = model(x, t)

    # Check the output
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
