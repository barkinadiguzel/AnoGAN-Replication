import torch
import torch.nn as nn
from src.layers.conv_blocks import conv_block

class Discriminator(nn.Module):
    def __init__(self, img_channels, feature_map_d=64):
        super().__init__()
        self.layer1 = conv_block(img_channels, feature_map_d, batch_norm=False)
        self.layer2 = conv_block(feature_map_d, feature_map_d*2)
        self.layer3 = conv_block(feature_map_d*2, feature_map_d*4)
        self.layer4 = conv_block(feature_map_d*4, feature_map_d*8)
        self.final = nn.Conv2d(feature_map_d*8, 1, 4, 1, 0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feature_map = self.layer4(x)
        out = self.final(feature_map)
        out = torch.sigmoid(out).view(-1)
        return out, feature_map
