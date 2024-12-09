"""
This module contains the code for Feature Pyramid U-Net architecture.
"""

# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic convolutional block: Conv -> BatchNorm -> ReLU."""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class FeaturePyramidBlock(nn.Module):
    """Feature Pyramid Block for multi-scale feature extraction."""
    def __init__(self, in_channels, out_channels):
        super(FeaturePyramidBlock, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.pyramid_scales = nn.ModuleList([
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.AdaptiveAvgPool2d(output_size=(2, 2)),
            nn.AdaptiveAvgPool2d(output_size=(4, 4))
        ])
        self.conv_blocks = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1) for _ in range(3)
        ])

    def forward(self, x):
        size = x.size()[2:]  # Original spatial dimensions
        pyramid_features = [self.conv1x1(x)]
        for pool, conv in zip(self.pyramid_scales, self.conv_blocks):
            pooled = pool(x)
            upsampled = F.interpolate(conv(pooled), size=size, mode='bilinear', align_corners=False)
            pyramid_features.append(upsampled)
        return torch.cat(pyramid_features, dim=1)

class FeaturePyramidUNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=4):
        super(FeaturePyramidUNet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = num_classes
        self.bilinear = False

        # Encoder
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)

        # Feature Pyramid Block
        self.feature_pyramid = FeaturePyramidBlock(512, 128)  # Outputs 128 * 4 = 512 channels

        # Decoder
        self.up3 = nn.ConvTranspose2d(1024, 256, kernel_size=2, stride=2)  # 512 (e4) + 512 (fp output)
        self.dec3 = ConvBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128)  # 128 (up2 output) + 128 (e2)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)  # 64 (up1 output) + 64 (e1)

        # Output layer
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)  # (B, 64, 224, 224)
        e2 = self.enc2(F.max_pool2d(e1, 2))  # (B, 128, 112, 112)
        e3 = self.enc3(F.max_pool2d(e2, 2))  # (B, 256, 56, 56)
        e4 = self.enc4(F.max_pool2d(e3, 2))  # (B, 512, 28, 28)

        # Feature Pyramid Block applied to e4
        fp = self.feature_pyramid(e4)  # (B, 512, 28, 28)

        # Decoder
        d3 = self.up3(torch.cat([fp, e4], dim=1))  # (B, 1024, 28, 28) -> upsample to (B, 256, 56, 56)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))  # (B, 512, 56, 56)

        d2 = self.up2(d3)  # (B, 128, 112, 112)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))  # (B, 256, 112, 112)

        d1 = self.up1(d2)  # (B, 64, 224, 224)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))  # (B, 128, 224, 224)

        # Output
        out = self.out_conv(d1)  # (B, num_classes, 224, 224)
        return out