import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, bias=False, padding_mode='reflect'),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2)
        )
    def forward(self, x):
        return self.conv(x)

class Descriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels*2, 64, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2)
        )
        self.model = nn.Sequential(
            CNNBlock(64, 128, stride=2),
            CNNBlock(128, 256, stride=2),
            CNNBlock(256, 512, stride=1),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, padding_mode='reflect')
        )
    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        return self.model(x)


class GenBlock(nn.Module):
    def __init__(self, in_features, out_features, stride=2, down=True, activation='relu', dropout=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_features, out_features, 4, stride, 1, bias=False, padding_mode='reflect')
            if down else nn.ConvTranspose2d(in_features, out_features, 4, stride, 1, bias=False),
            nn.InstanceNorm2d(out_features, affine=True),
            nn.ReLU() if activation == 'relu' else nn.LeakyReLU(0.2)
        )
        self.use_dropout = dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        return x if not self.use_dropout else self.dropout(x)


class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, bias=False, padding_mode='reflect'),
            nn.LeakyReLU(0.2)
        )  # 128

        self.down2 = GenBlock(features, 2 * features, down=True, activation='leaky', dropout=False)  # 64
        self.down3 = GenBlock(2 * features, 2 * features, down=True, activation='leaky', dropout=False)  # 32
        self.down4 = GenBlock(2 * features, 4 * features, down=True, activation='leaky', dropout=False)  # 16
        self.down5 = GenBlock(4 * features, 4 * features, down=True, activation='leaky', dropout=False)  # 8
        self.down6 = GenBlock(4 * features, 8 * features, down=True, activation='leaky', dropout=False)  # 4
        self.down7 = GenBlock(8 * features, 8 * features, down=True, activation='leaky', dropout=False)  # 2

        self.bottleneck = nn.Sequential(
            nn.Conv2d(8 * features, 16 * features, kernel_size=1, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16 * features, 8 * features, kernel_size=1, stride=2, padding=1),
            nn.ReLU()
        )

        self.up1 = GenBlock(2 * 8 * features, 8 * features, down=False, activation='relu', dropout=True)  # 4
        self.up2 = GenBlock(2 * 8 * features, 4 * features, down=False, activation='relu', dropout=True)  # 8
        self.up3 = GenBlock(2 * 4 * features, 4 * features, down=False, activation='relu', dropout=True)  # 16
        self.up4 = GenBlock(2 * 4 * features, 2 * features, down=False, activation='relu', dropout=True)  # 32
        self.up5 = GenBlock(2 * 2 * features, 2 * features, down=False, activation='relu', dropout=True)  # 64
        self.up6 = GenBlock(2 * 2 * features, features, down=False, activation='relu', dropout=True)  # 128
        self.up7 = nn.Sequential(
            nn.ConvTranspose2d(2 * features, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )  # 256

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)

        bn = self.bottleneck(d7)

        u1 = self.up1(torch.cat([bn, d7], dim=1))
        u2 = self.up2(torch.cat([u1, d6], dim=1))
        u3 = self.up3(torch.cat([u2, d5], dim=1))
        u4 = self.up4(torch.cat([u3, d4], dim=1))
        u5 = self.up5(torch.cat([u4, d3], dim=1))
        u6 = self.up6(torch.cat([u5, d2], dim=1))
        u7 = self.up7(torch.cat([u6, d1], dim=1))
        return u7



def init_gen(name='gen.pt', pretrained=True):
    gen = Generator(in_channels=3).to(device)
    if pretrained:
        gen.load_state_dict(torch.load(name, map_location=device))
    return gen