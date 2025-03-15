"""
This module contains the code for Feedback ResU-Net architecture.
"""

# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # self.bn1 = nn.BatchNorm2d(out_channels)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm2d(out_channels)
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        else:
            self.residual_conv = None

    def forward(self, x):
        residual = x
        if self.residual_conv:
            residual = self.residual_conv(x)
        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.gn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        # out = self.bn2(out)
        out = self.gn2(out)
        out += residual
        out = self.relu(out)
        return out


class FeedbackResUNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=4):
        super(FeedbackResUNet, self).__init__()
        self.n_channels = input_channels
        self.n_classes = num_classes
        self.bilinear = False

        # Encoder
        self.resblock1 = ResidualBlock(input_channels, 64)
        self.resblock2 = ResidualBlock(64, 128)
        self.resblock3 = ResidualBlock(128, 256)
        self.resblock4 = ResidualBlock(256, 512)

        # Bottleneck
        self.resblock_bottleneck = ResidualBlock(512, 1024)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)  # Upsampling layer
        self.resblock5 = ResidualBlock(1024, 512)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # Second upsampling layer
        self.resblock6 = ResidualBlock(512, 256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # Third upsampling layer
        self.resblock7 = ResidualBlock(256, 128)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # Fourth upsampling layer
        self.resblock8 = ResidualBlock(128, 64)

        # Final output
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)  # 1x1 convolution to produce output masks

        # Adjust channels for feedback
        self.adjust_channels = nn.Conv2d(num_classes, input_channels, kernel_size=1)

    def forward(self, x):
        batch_size, _, height, width = x.size()

        # First round - Encoder
        x1 = self.resblock1(x)
        x1_pooled = F.max_pool2d(x1, kernel_size=2)  # Downsample with max pooling

        x2 = self.resblock2(x1_pooled)
        x2_pooled = F.max_pool2d(x2, kernel_size=2)  # Downsample with max pooling

        x3 = self.resblock3(x2_pooled)
        x3_pooled = F.max_pool2d(x3, kernel_size=2)  # Downsample with max pooling

        x4 = self.resblock4(x3_pooled)
        x4_pooled = F.max_pool2d(x4, kernel_size=2)  # Downsample with max pooling

        # Bottleneck
        x_b = self.resblock_bottleneck(x4_pooled)

        # Decoder
        x5 = self.upconv1(x_b)  # Upsample the bottleneck output
        x5 = F.pad(x5, (
        0, x4.size(3) - x5.size(3), 0, x4.size(2) - x5.size(2)))  # Pad to match the encoder feature map size
        x5 = torch.cat([x5, x4], dim=1)  # Concatenate with corresponding encoder output
        x5 = self.resblock5(x5)

        x6 = self.upconv2(x5)  # Upsample
        x6 = F.pad(x6, (
        0, x3.size(3) - x6.size(3), 0, x3.size(2) - x6.size(2)))  # Pad to match the encoder feature map size
        x6 = torch.cat([x6, x3], dim=1)  # Concatenate with corresponding encoder output
        x6 = self.resblock6(x6)

        x7 = self.upconv3(x6)  # Upsample
        x7 = F.pad(x7, (
        0, x2.size(3) - x7.size(3), 0, x2.size(2) - x7.size(2)))  # Pad to match the encoder feature map size
        x7 = torch.cat([x7, x2], dim=1)  # Concatenate with corresponding encoder output
        x7 = self.resblock7(x7)

        x8 = self.upconv4(x7)  # Upsample
        x8 = F.pad(x8, (
        0, x1.size(3) - x8.size(3), 0, x1.size(2) - x8.size(2)))  # Pad to match the encoder feature map size
        x8 = torch.cat([x8, x1], dim=1)  # Concatenate with corresponding encoder output
        x8 = self.resblock8(x8)

        # Final output of the first round
        out1 = self.final_conv(x8)  # Apply final 1x1 convolution to produce output masks

        # Second round - Feedback the output of the first round into the encoder
        adjusted_out1 = self.adjust_channels(out1)  # Adjust channels of out1 to match input channels
        feedback_input = x + adjusted_out1  # Add input and adjusted output of the first round

        # Second round - Feedback the output of the first round into the encoder
        x1 = self.resblock1(feedback_input)
        x1_pooled = F.max_pool2d(x1, kernel_size=2)  # Downsample with max pooling

        x2 = self.resblock2(x1_pooled)
        x2_pooled = F.max_pool2d(x2, kernel_size=2)  # Downsample with max pooling

        x3 = self.resblock3(x2_pooled)
        x3_pooled = F.max_pool2d(x3, kernel_size=2)  # Downsample with max pooling

        x4 = self.resblock4(x3_pooled)
        x4_pooled = F.max_pool2d(x4, kernel_size=2)  # Downsample with max pooling

        # Bottleneck
        x_b = self.resblock_bottleneck(x4_pooled)

        # Decoder
        x5 = self.upconv1(x_b)  # Upsample the bottleneck output
        x5 = F.pad(x5, (
        0, x4.size(3) - x5.size(3), 0, x4.size(2) - x5.size(2)))  # Pad to match the encoder feature map size
        x5 = torch.cat([x5, x4], dim=1)  # Concatenate with corresponding encoder output
        x5 = self.resblock5(x5)

        x6 = self.upconv2(x5)  # Upsample
        x6 = F.pad(x6, (
        0, x3.size(3) - x6.size(3), 0, x3.size(2) - x6.size(2)))  # Pad to match the encoder feature map size
        x6 = torch.cat([x6, x3], dim=1)  # Concatenate with corresponding encoder output
        x6 = self.resblock6(x6)

        x7 = self.upconv3(x6)  # Upsample
        x7 = F.pad(x7, (
        0, x2.size(3) - x7.size(3), 0, x2.size(2) - x7.size(2)))  # Pad to match the encoder feature map size
        x7 = torch.cat([x7, x2], dim=1)  # Concatenate with corresponding encoder output
        x7 = self.resblock7(x7)

        x8 = self.upconv4(x7)  # Upsample
        x8 = F.pad(x8, (
        0, x1.size(3) - x8.size(3), 0, x1.size(2) - x8.size(2)))  # Pad to match the encoder feature map size
        x8 = torch.cat([x8, x1], dim=1)  # Concatenate with corresponding encoder output
        x8 = self.resblock8(x8)

        # Final output of the second round
        out2 = self.final_conv(x8)  # Apply final 1x1 convolution to produce output masks

        return out2