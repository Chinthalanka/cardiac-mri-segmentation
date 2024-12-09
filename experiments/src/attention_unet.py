"""
This module contains the code for Attention U-Net architecture.
"""

# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F


# Define an attention block
class AttentionBlockV3(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlockV3, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# Define a basic UNet block
class UNetConvBlockV3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetConvBlockV3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.relu(self.batchnorm1(self.conv1(x)))
        x = self.relu(self.batchnorm2(self.conv2(x)))
        return x


# Define the Attention UNet with bottleneck attention
class AttentionUNetV3(nn.Module):
    def __init__(self, in_channels=1, out_channels=4):
        self.n_channels = in_channels
        self.n_classes = out_channels
        self.bilinear = False

        super(AttentionUNetV3, self).__init__()

        # Encoder
        self.encoder1 = UNetConvBlockV3(in_channels, 64)
        self.encoder2 = UNetConvBlockV3(64, 128)
        self.encoder3 = UNetConvBlockV3(128, 256)
        self.encoder4 = UNetConvBlockV3(256, 512)

        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck with attention
        self.bottleneck = UNetConvBlockV3(512, 1024)
        self.bottleneck_att = AttentionBlockV3(F_g=1024, F_l=1024, F_int=512)

        # Decoder with attention
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att4 = AttentionBlockV3(F_g=512, F_l=512, F_int=256)
        self.decoder4 = UNetConvBlockV3(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att3 = AttentionBlockV3(F_g=256, F_l=256, F_int=128)
        self.decoder3 = UNetConvBlockV3(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = AttentionBlockV3(F_g=128, F_l=128, F_int=64)
        self.decoder2 = UNetConvBlockV3(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att1 = AttentionBlockV3(F_g=64, F_l=64, F_int=32)
        self.decoder1 = UNetConvBlockV3(128, 64)

        # Output layer
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))

        # Bottleneck with Attention
        b = self.bottleneck(self.pool(e4))
        b = self.bottleneck_att(g=b, x=b)  # Apply attention to the bottleneck features

        # Decoder with Attention
        d4 = self.upconv4(b)
        e4 = self.att4(g=d4, x=e4)
        d4 = self.decoder4(torch.cat((d4, e4), dim=1))

        d3 = self.upconv3(d4)
        e3 = self.att3(g=d3, x=e3)
        d3 = self.decoder3(torch.cat((d3, e3), dim=1))

        d2 = self.upconv2(d3)
        e2 = self.att2(g=d2, x=e2)
        d2 = self.decoder2(torch.cat((d2, e2), dim=1))

        d1 = self.upconv1(d2)
        e1 = self.att1(g=d1, x=e1)
        d1 = self.decoder1(torch.cat((d1, e1), dim=1))

        # Output layer
        output = self.output(d1)
        return output
