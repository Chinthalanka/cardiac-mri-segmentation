"""
This module contains the code for Residual U-Net (ResU-Net) architecture.
"""

# Import libraries
import torch
import torch.nn as nn


# Define a residual block
class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.batchnorm1 = nn.GroupNorm(num_groups=8, num_channels=out_channels) # nn.BatchNorm2d(out_channels)
        self.batchnorm2 = nn.GroupNorm(num_groups=8, num_channels=out_channels) # nn.BatchNorm2d(out_channels)

        # Residual connection (1x1 convolution to match dimensions if in_channels != out_channels)
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x  # Store the input for the residual connection
        x = self.relu(self.batchnorm1(self.conv1(x)))
        x = self.batchnorm2(self.conv2(x))

        # Apply residual connection
        if self.residual:
            residual = self.residual(residual)

        x += residual  # Add the residual connection
        x = self.relu(x)  # Apply ReLU after adding residual
        return x

# Define the Residual UNet
class ResidualUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=4):
        super(ResidualUNet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels
        self.bilinear = False

        # Encoder
        self.encoder1 = ResidualConvBlock(in_channels, 64)
        self.encoder2 = ResidualConvBlock(64, 128)
        self.encoder3 = ResidualConvBlock(128, 256)
        self.encoder4 = ResidualConvBlock(256, 512)

        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = ResidualConvBlock(512, 1024)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = ResidualConvBlock(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = ResidualConvBlock(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = ResidualConvBlock(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = ResidualConvBlock(128, 64)

        # Output layer
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with residual connections
        d4 = self.upconv4(b)
        d4 = torch.cat((d4, e4), dim=1)  # Concatenate skip connection
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e3), dim=1)  # Concatenate skip connection
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)  # Concatenate skip connection
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)  # Concatenate skip connection
        d1 = self.decoder1(d1)

        # Output layer
        output = self.output(d1)
        return output