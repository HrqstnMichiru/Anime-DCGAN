from torch import nn
from torchkeras import summary
import torch


class Generator(nn.Module):
    """生成器定义"""

    def __init__(self, config):
        super().__init__()
        # 噪声维度
        nz = config.noise_dim
        # feature_dim: 隐藏特征尺寸
        ngf = config.gen_feature_map
        self.model = nn.Sequential(
            # input size. (nz) x 1 x 1
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            # nn.Upsample(scale_factor=2),
            # nn.Conv2d(ngf * 8, ngf * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            # nn.Upsample(scale_factor=2),
            # nn.Conv2d(ngf * 4, ngf * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            # nn.Upsample(scale_factor=3),
            # nn.Conv2d(ngf * 2, ngf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, 3, 5, 3, 1, bias=False),
            # nn.Upsample(scale_factor=2),
            # nn.Conv2d(ngf, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            # output size. 3 x 96 x 96
        )
        self._init_weights()

    def _init_weights(self):
        for layer in self.model:
            if isinstance(layer, nn.ConvTranspose2d):
                nn.init.normal_(layer.weight, 0.0, 0.02)
            if isinstance(layer, nn.BatchNorm2d):
                nn.init.normal_(layer.weight, 1.0, 0.02)
                nn.init.constant_(layer.bias, 0)

    def forward(self, input):
        output = self.model(input)
        return output


class Discriminator(nn.Module):
    """判别器定义"""

    def __init__(self, config):
        super().__init__()
        nc = 3
        # feature_dim: 隐藏特征尺寸
        ndf = config.gen_feature_map  # ?64
        self.model = nn.Sequential(
            # input is (nc) x 96 x 96
            nn.Conv2d(nc, ndf, 5, 3, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
            # output size. 1 x 1 x 1
        )
        self._init_weights()

    def _init_weights(self):
        for layer in self.model:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, 0.0, 0.02)
            if isinstance(layer, nn.BatchNorm2d):
                nn.init.normal_(layer.weight, 1.0, 0.02)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, input):
        output = self.model(input)
        return output.view(-1, 1).squeeze(1)


if __name__ == "__main__":
    model1 = Generator()
    summary(model1, input_shape=(100, 1, 1))
    model2 = Discriminator(64)
    summary(model2, input_shape=(3, 96, 96))
    a = torch.randn(8, 3, 96, 96)
    print(model2(a).shape)
