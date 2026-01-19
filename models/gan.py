import torch
import torch.nn as nn

latent_dim = 256
img_channels = 3
img_size = 128  # 128x128 images


class ConvGenerator(nn.Module):
    def __init__(self, latent_dim=latent_dim, img_channels=img_channels):
        super(ConvGenerator, self).__init__()
        self.fc = nn.Linear(latent_dim, 256*8*8)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 8x8 -> 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 16x16 -> 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 32x32 -> 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, img_channels, 4, 2, 1),  # 64x64 -> 128x128
            nn.Tanh()  # outputs in [-1,1] (normalize your dataset!)
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 256, 8, 8)
        return self.net(h)

class ConvDiscriminator(nn.Module):
    def __init__(self, img_channels=img_channels):
        super(ConvDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, 32, 4, 2, 1),  # 128->64
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, 4, 2, 1),            # 64->32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),           # 32->16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),          # 16->8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Linear(256*8*8, 1)

    def forward(self, x):
        h = self.net(x)
        h = h.view(x.size(0), -1)
        out = torch.sigmoid(self.fc(h))
        return out


