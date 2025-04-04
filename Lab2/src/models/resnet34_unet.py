# Implement your ResNet34_UNet model here
import torch

from torch.nn import functional as F
from torch import nn
from models.unet import DoubleConv

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode="replicate", stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode="replicate", bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Identity()
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        output = self.block(x)
        return F.relu(output + residual)

class ResNet34Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputSequence = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, padding_mode="replicate", bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.res64 = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64)
        )

        self.res128 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128)
        )

        self.res256 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256)
        )

        self.res512 = nn.Sequential(
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512)
        )

        self.bottleneck = nn.Sequential(
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512)
        )

        self.decodeL1 = nn.Sequential(
            DoubleConv(1024, 512),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        )

        self.decodeL2 = nn.Sequential(
            DoubleConv(512, 256),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        )

        self.decodeL3 = nn.Sequential(
            DoubleConv(256, 128),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        )

        self.decodeL4 = nn.Sequential(
            DoubleConv(128, 64),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
        )

        self.decodeL5 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.Conv2d(16, 1, kernel_size=1)
        )

    def forward(self, x):
        x = self.inputSequence(x)
        
        res_64_out = self.res64(x)
        x = res_64_out
        
        res_128_out = self.res128(x)
        x = res_128_out
        res_256_out = self.res256(x)
        x = res_256_out
        res_512_out = self.res512(x)
        x = res_512_out
        
        x = self.bottleneck(x)
        x = self.decodeL1(torch.cat((res_512_out, x), dim=1))
        x = self.decodeL2(torch.cat((res_256_out, x), dim=1))
        x = self.decodeL3(torch.cat((res_128_out, x), dim=1))

        x = self.decodeL4(torch.cat((res_64_out, x), dim=1))

        return self.decodeL5(x)
        
if __name__ == "__main__":
    model = ResNet34Unet()
    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params}")
    input_tensor = torch.randn(1, 3, 256, 256)
    output_tensor = model(input_tensor)

    print(output_tensor.shape)