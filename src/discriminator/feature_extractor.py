import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self, discriminator):
        super().__init__()
        self.features = nn.Sequential(
            discriminator.layer1,
            discriminator.layer2,
            discriminator.layer3,
            discriminator.layer4
        )

    def forward(self, x):
        return self.features(x)
