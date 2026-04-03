import torch
import torch.nn as nn
from src.layers.conv_blocks import deconv_block

class Generator(nn.Module):
    def __init__(self, latent_dim, img_channels, feature_map_g=64):
        super().__init__()
        self.net = nn.Sequential(
            deconv_block(latent_dim, feature_map_g*8, 4, 1, 0, batch_norm=True),
            deconv_block(feature_map_g*8, feature_map_g*4, 4, 2, 1, batch_norm=True),
            deconv_block(feature_map_g*4, feature_map_g*2, 4, 2, 1, batch_norm=True),
            deconv_block(feature_map_g*2, feature_map_g, 4, 2, 1, batch_norm=True),
            nn.ConvTranspose2d(feature_map_g, img_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)
