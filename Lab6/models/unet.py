import torch
import torch.nn.functional as F
import math

from torch import nn
from numpy import pi
from functools import partial

class Embedder(nn.Module):
    def __init__(self, num_embeddings=24, embedding_dim=128):
        super().__init__()
        # last index + 1 as padding index
        self.padding_idx = num_embeddings
        self.embedding = nn.Embedding(
            num_embeddings + 1, embedding_dim, padding_idx=num_embeddings
        )
        
    def forward(self, x):
        sum_embeds = self.embedding(x).sum(dim=1)
        num_valid = (x != self.padding_idx).sum(dim=1)
        return sum_embeds / num_valid.view(-1, 1)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, t_embedding_dim):
        super().__init__()
        self.dim = t_embedding_dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

        
class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        t_embedding_dim,
        drop_out=0.1,
        up_sample=False,
        down_sample=False,
    ):
        assert not (down_sample and up_sample)
        super().__init__()

        self.up_sample = up_sample
        self.down_sample = down_sample

        if self.down_sample:
            #self.resample = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
            self.resample = partial(F.avg_pool2d, kernel_size=2, stride=2)
        elif self.up_sample:
            #self.resample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
            self.resample = partial(F.interpolate, scale_factor=2.0, mode="nearest")
        else:
            self.resample = nn.Identity()

        self.prep1 = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6), nn.SiLU()
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.t_embedding_mlp = nn.Sequential(
            nn.SiLU(), nn.Linear(t_embedding_dim, out_channels)
        )

        self.prep2 = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6), nn.SiLU()
        )
        
        self.dropout = nn.Dropout(drop_out)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.skip_connection = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, t_embedding):
        out = x
        out = self.prep1(out)

        x = self.resample(x)
        out = self.resample(out)

        out = self.conv1(out)

        t_embedding = self.t_embedding_mlp(t_embedding)[:, :, None, None]
        out = out + t_embedding

        out = self.prep2(out)
        out = self.dropout(out)
        out = self.conv2(out)

        return out + self.skip_connection(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, batch_first=True
        )
        self.proj_out = nn.Linear(channels, channels)

    def forward(self, x):
        B, C, H, W = x.shape
        x_ = self.norm(x)

        x_ = x_.view(B, C, H * W).transpose(1, 2)

        attn_out, _ = self.attn(x_, x_, x_)

        attn_out = self.proj_out(attn_out)
        attn_out = attn_out.transpose(1, 2).view(B, C, H, W)

        return x + attn_out


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_embedding_dim, add_down=True):
        super().__init__()

        self.res = ResidualBlock(in_channels, out_channels, t_embedding_dim)
        self.down_res = (
            ResidualBlock(out_channels, out_channels, t_embedding_dim, down_sample=True)
            if add_down
            else None
        )

    def forward(self, x, t_embedding):
        x = self.res(x, t_embedding)

        if self.down_res is not None:
            down_sampled = self.down_res(x, t_embedding)
            return down_sampled, x

        return x, x


class MidBlock(nn.Module):
    def __init__(self, channels, t_embedding_dim):
        super().__init__()
        self.resblock1 = ResidualBlock(channels, channels, t_embedding_dim)
        self.attn = AttentionBlock(channels)
        self.resblock2 = ResidualBlock(channels, channels, t_embedding_dim)

    def forward(self, x, t_embedding):
        x = self.resblock1(x, t_embedding)
        x = self.attn(x)
        x = self.resblock2(x, t_embedding)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_embedding_dim, add_up=True):
        super().__init__()
        skip_in_channels = in_channels
        self.res = ResidualBlock(
            in_channels + skip_in_channels, out_channels, t_embedding_dim
        )
        self.up_res = (
            ResidualBlock(out_channels, out_channels, t_embedding_dim, up_sample=True)
            if add_up
            else None
        )

    def forward(self, x, skip_in, t_embedding):
        x = torch.cat([x, skip_in], dim=1)

        x = self.res(x, t_embedding)

        if self.up_res is not None:
            return self.up_res(x, t_embedding)

        return x


class UNetDiffuser(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim):
        super().__init__()
        #block_out_channels = (192, 384, 768, 768)
        block_out_channels = (128, 256, 512, 512)

        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            #nn.Linear(embed_dim, embed_dim),
        )

        self.class_embedding = Embedder(num_embeddings=24, embedding_dim=embed_dim)

        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=3, padding=1
        )

        self.down1 = DownBlock(
            block_out_channels[0], block_out_channels[0], t_embedding_dim=embed_dim
        )
        self.down2 = DownBlock(
            block_out_channels[0], block_out_channels[1], t_embedding_dim=embed_dim
        )
        self.down3 = DownBlock(
            block_out_channels[1], block_out_channels[2], t_embedding_dim=embed_dim
        )
        self.down4 = DownBlock(
            block_out_channels[2],
            block_out_channels[3],
            t_embedding_dim=embed_dim,
            add_down=False,
        )

        self.mid = MidBlock(block_out_channels[3], t_embedding_dim=embed_dim)

        self.up4 = UpBlock(
            block_out_channels[3], block_out_channels[2], t_embedding_dim=embed_dim
        )
        self.up3 = UpBlock(
            block_out_channels[2], block_out_channels[1], t_embedding_dim=embed_dim
        )
        self.up2 = UpBlock(
            block_out_channels[1], block_out_channels[0], t_embedding_dim=embed_dim
        )
        self.up1 = UpBlock(
            block_out_channels[0],
            block_out_channels[0],
            t_embedding_dim=embed_dim,
            add_up=False,
        )

        self.post_process = nn.Sequential(
            nn.GroupNorm(num_channels=block_out_channels[0], num_groups=32, eps=1e-6),
            nn.SiLU(),
            nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, t_step, labels):
        t_embed = self.time_embedding(t_step)

        class_embed = self.class_embedding(labels)

        embed = t_embed + class_embed

        x = self.conv_in(x)

        hidden1, skip1 = self.down1(x, embed)
        hidden2, skip2 = self.down2(hidden1, embed)
        hidden3, skip3 = self.down3(hidden2, embed)
        down_out, skip4 = self.down4(hidden3, embed)

        down_out = self.mid(down_out, embed)

        x_up = self.up4(down_out, skip4, embed)
        x_up = self.up3(x_up, skip3, embed)
        x_up = self.up2(x_up, skip2, embed)
        x_up = self.up1(x_up, skip1, embed)

        return self.post_process(x_up)


if __name__ == "__main__":
    # r = ResidualBlock(32, 256, down_sample=True).to("cuda")
    # x = torch.randn(64, 32, 64, 64).to("cuda")
    # temb = torch.randn(64, 128).to("cuda")
    # out = r(x, temb)
    # print(out.shape)
    device = "cuda"

    model = UNetDiffuser(3, 3, embed_dim=128).to(device)
    images = torch.randn(64, 3, 64, 64, device=device)
    temb = torch.randint(0, 20, (64,), device=device)
    labels = torch.tensor([[1, 4, 6] + [24] * 21] * 64, device=device)

    out = model(images, temb, labels)
    param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(out.shape)
    print(param)
