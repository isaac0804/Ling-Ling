import torch
from torch.functional import norm
import torch.nn as nn
import torch.nn.functional as F
from encoder import *


class EncoderBlock(nn.Module):
    """Encoder Block"""

    def __init__(
        self,
        dim,
        window_size=8,
        num_heads=8,
        attn_drop=0.0,
        proj_drop=0.0,
        mlp_drop=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.norm_layer1 = norm_layer([window_size, dim])
        self.norm_layer2 = norm_layer([window_size, dim])
        self.norm_layer3 = norm_layer([window_size, dim])
        self.norm_layer4 = norm_layer([window_size, dim])
        self.attention = Attention(dim, num_heads, attn_drop, proj_drop)
        self.shifted_attention = Attention(dim, num_heads, attn_drop, proj_drop)
        self.mlp1 = MLP([dim, dim], act_layer, mlp_drop)
        self.mlp2 = MLP([dim, dim], act_layer, mlp_drop)

    def shift(self, x, N, reverse=False):
        x = x.view(-1, N, self.dim)
        if reverse:
            torch.roll(x, shifts=self.window_size // 2, dims=1)
        else:
            torch.roll(x, shifts=-self.window_size // 2, dims=1)
        x = x.view(-1, self.window_size, self.dim)
        return x

    def forward(self, x):
        B, N, C = x.shape
        x = x.view(B * N // self.window_size, self.window_size, C)
        x = self.attention(self.norm_layer1(x))[0] + x
        x = self.mlp1(self.norm_layer2(x)) + x
        x = self.shift(x, N)
        x = self.shifted_attention(self.norm_layer3(x))[0] + x
        x = self.mlp2(self.norm_layer4(x)) + x
        x = self.shift(x, N, reverse=True)
        x = x.view(B, N, C)
        return x

    def debug(self, x):
        B, N, C = x.shape
        x = x.view(B * N // self.window_size, self.window_size, C)
        y, attn = self.attention(self.norm_layer1(x))
        y = y + x
        x = self.mlp1(self.norm_layer2(x)) + x
        x = self.shift(x, N)
        y, attn_2 = self.shifted_attention(self.norm_layer3(x))
        y = y + x
        x = self.mlp2(self.norm_layer4(x)) + x
        x = self.shift(x, N, reverse=True)
        x = x.view(B, N, C)
        return x, attn, attn_2


class PatchMerge(nn.Module):
    def __init__(
        self,
        dim,
        downsample_res=2,
        mlp_drop=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.downsample_res = downsample_res
        self.reduction = MLP(
            [self.downsample_res * dim, self.downsample_res // 2 * dim],
            act_layer,
            mlp_drop,
        )
        self.norm_layer = norm_layer(self.downsample_res * dim)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, N // self.downsample_res, self.downsample_res * C)
        x = self.reduction(self.norm_layer(x))
        return x


class BasicBlock(nn.Module):
    def __init__(
        self,
        dim,
        window_size=16,
        downsample_res=None,
        num_blocks=2,
        num_heads=8,
        attn_drop=0.0,
        proj_drop=0.0,
        mlp_drop=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                EncoderBlock(
                    dim,
                    window_size,
                    num_heads,
                    attn_drop,
                    proj_drop,
                    mlp_drop,
                    act_layer,
                    norm_layer,
                )
                for i in range(num_blocks)
            ]
        )
        if downsample_res:
            self.merge = PatchMerge(
                dim//2, downsample_res, mlp_drop, act_layer, norm_layer
            )
        else:
            self.merge = None

    def forward(self, x):
        if self.merge:
            x = self.merge(x)
        res = x
        for block in self.blocks:
            x = block(x) + res
        return x

    def debug(self, x):
        if self.merge:
            x = self.merge(x)
        for i, block in enumerate(self.blocks):
            x, attn, attn_2 = block.debug(x)
            if i == 0:
                return x, attn, attn_2
        return x


class SwinEncoder(nn.Module):
    """Encoder with Swin Transformer Architecture"""

    def __init__(
        self,
        embed_dim=64,
        window_size=16,
        num_heads=8,
        num_classes=None,
        downsample_res=2,
        depth=[4, 4, 2, 2],
        seq_length=1024,
        attn_drop=0.0,
        proj_drop=0.0,
        mlp_drop=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert seq_length % (downsample_res ** (len(depth)-1)) == 0
        assert seq_length / downsample_res**(len(depth)-1) >= window_size
        self.window_size = window_size
        self.note_embed = NoteEmbed(embed_dim=embed_dim)
        self.embs_dim = [embed_dim, embed_dim, embed_dim*2, embed_dim*4]
        self.blocks = nn.ModuleList(
            [
                BasicBlock(
                    embed_dim * (downsample_res//2) ** i,
                    window_size,
                    downsample_res if i > 0 else None,
                    num_attention,
                    num_heads,
                    attn_drop,
                    proj_drop,
                    mlp_drop,
                    act_layer,
                    norm_layer,
                )
                for i, num_attention in enumerate(depth)
            ]
        )
        self.bottleneck_dim = embed_dim * (downsample_res // 2) ** (len(depth)-1)
        self.head_dim = num_classes if num_classes else self.bottleneck_dim
        self.norm = nn.LayerNorm(self.bottleneck_dim)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.head = nn.Linear(self.bottleneck_dim, self.head_dim)
        self.head = MLP([self.bottleneck_dim, self.head_dim, self.head_dim], act_layer, mlp_drop)

    def forward(self, x):
        x = self.note_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.avgpool(x.transpose(-1, -2))
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x

    def debug(self, x):
        x = self.note_embed(x)
        for idx, block in enumerate(self.blocks):
            x, attn, attn_2 = block.debug(x)
            if idx == 3:
                return x
                return attn
        x = self.norm(x)
        x = self.avgpool(x.transpose(-1, -2))
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x

    def return_attention(self, x):
        x = self.note_embed(x)
        x, attn, attn_2 = self.blocks[0].debug(x)
        return attn