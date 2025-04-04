# Implement your UNet model here
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms.functional import center_crop

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='replicate', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='replicate', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encodeL1 = DoubleConv(3, 64)   # 3 channel image input
        self.encodeL2 = DoubleConv(64, 128)
        self.encodeL3 = DoubleConv(128, 256)
        self.encodeL4 = DoubleConv(256, 512)
        
        self.decodeL1 = nn.Sequential(
            DoubleConv(512, 1024),
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        )
        self.decodeL2 = nn.Sequential(
            DoubleConv(1024, 512),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        )
        self.decodeL3 = nn.Sequential(
            DoubleConv(512, 256),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        )
        self.decodeL4 = nn.Sequential(
            DoubleConv(256, 128),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        )
        self.decodeL5 = nn.Sequential(
            DoubleConv(128, 64),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        
    def forward(self, x):
        l1_pre_down = self.encodeL1(x)
        x = F.max_pool2d(l1_pre_down, 2, 2)
        
        l2_pre_down = self.encodeL2(x)
        x = F.max_pool2d(l2_pre_down, 2, 2)
        
        l3_pre_down = self.encodeL3(x)
        x = F.max_pool2d(l3_pre_down, 2, 2)
        
        l4_pre_down = self.encodeL4(x)
        x = F.max_pool2d(l4_pre_down, 2, 2)

        x = self.decodeL1(x)
        x = self.decodeL2(torch.cat((center_crop(l4_pre_down, x.shape[2:]), x), dim=1))
        x = self.decodeL3(torch.cat((center_crop(l3_pre_down, x.shape[2:]), x), dim=1))
        x = self.decodeL4(torch.cat((center_crop(l2_pre_down, x.shape[2:]), x), dim=1))
        
        return self.decodeL5(torch.cat((center_crop(l1_pre_down, x.shape[2:]), x), dim=1))


if __name__ == "__main__":
    model = UNet()
    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params}")
    input_tensor = torch.randn(1, 3, 256, 256)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)