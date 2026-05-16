import torch.nn as nn


class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator (Isola et al. / VQ-GAN). Outputs per-patch logits."""

    def __init__(self, cfg):
        super().__init__()
        in_channels = cfg.model.in_out_channels
        base_channels = cfg.discriminator.base_channels
        num_layers = cfg.discriminator.num_layers
        max_channels = cfg.discriminator.max_channels

        layers = [
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        channels = base_channels
        for _ in range(1, num_layers):
            next_channels = min(channels * 2, max_channels)
            layers += [
                nn.Conv2d(channels, next_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(next_channels),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            channels = next_channels

        next_channels = min(channels * 2, max_channels)
        layers += [
            nn.Conv2d(channels, next_channels, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(next_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(next_channels, 1, kernel_size=4, stride=1, padding=1),
        ]
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)
