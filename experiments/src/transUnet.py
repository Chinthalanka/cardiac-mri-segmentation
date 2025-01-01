"""
This module contains the code for Transformer-based U-Net architecture.
"""

# Import libraries
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=1, embed_dim=512):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, mlp_ratio=4., dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers=12, embed_dim=512, num_heads=8, mlp_ratio=4., dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x)


class TransformerUNet(nn.Module):
    def __init__(self, img_size=224, in_channels=1, num_classes=4, embed_dim=512, num_heads=8):
        super(TransformerUNet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = num_classes
        self.bilinear = False

        self.embed_dim = embed_dim

        # Patch embedding and transformer encoder
        self.patch_embed = PatchEmbedding(img_size=img_size, in_channels=in_channels, embed_dim=embed_dim)
        self.transformer = TransformerEncoder(embed_dim=embed_dim, num_heads=num_heads)

        # Decoder (UNet style)
        self.dec1 = ConvBlock(embed_dim, 512)
        self.up1 = UpConv(512, 512)
        self.dec2 = ConvBlock(512, 256)
        self.up2 = UpConv(256, 256)
        self.dec3 = ConvBlock(256, 128)
        self.up3 = UpConv(128, 128)
        self.dec4 = ConvBlock(128, 64)

        # Final output layer
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

        # Upsampling to match input size
        self.up_final = nn.Upsample(size=(img_size, img_size), mode='bilinear', align_corners=True)

    def forward(self, x):
        B, C, H, W = x.shape

        # Transformer Encoder
        x = self.patch_embed(x)
        x = self.transformer(x)

        # Reshape transformer output for UNet decoding
        x = x[:, 1:].transpose(1, 2).reshape(B, self.embed_dim, H // 16, W // 16)

        # Decoder path
        d1 = self.dec1(x)  # Output: [B, 512, H//16, W//16]
        d2 = self.up1(d1)  # Output: [B, 512, H//8, W//8]
        d2 = self.dec2(d2) # Output: [B, 256, H//8, W//8]
        d3 = self.up2(d2)  # Output: [B, 256, H//4, W//4]
        d3 = self.dec3(d3) # Output: [B, 128, H//4, W//4]
        d4 = self.up3(d3)  # Output: [B, 128, H//2, W//2]
        d4 = self.dec4(d4) # Output: [B, 64, H//2, W//2]

        # Final segmentation output
        out = self.final(d4)  # Output: [B, num_classes, H//2, W//2]

        # Upsample to match input size
        out = self.up_final(out)  # Output: [B, num_classes, H, W]

        return out
