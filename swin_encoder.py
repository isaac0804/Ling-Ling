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
        # self.mlp1 = MLP([dim, dim, dim], act_layer, mlp_drop)
        # self.mlp2 = MLP([dim, dim, dim], act_layer, mlp_drop)

    def shift(self, x, reverse=False):
        if reverse:
            torch.roll(x, shifts=self.window_size // 2, dims=1)
        else:
            torch.roll(x, shifts=-self.window_size // 2, dims=1)
        return x

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B * N // self.window_size, self.window_size, C)
        x = self.attention(self.norm_layer1(x))[0]
        # x = self.mlp1(self.norm_layer2(y)) + x
        x = x.reshape(B, N, C)
        x = self.shift(x)
        x = x.reshape(B * N // self.window_size, self.window_size, C)
        x = self.shifted_attention(self.norm_layer3(x))[0] 
        # x = self.mlp2(self.norm_layer4(y)) + x
        x = x.reshape(B, N, C)
        x = self.shift(x, reverse=True)
        return x

    def debug(self, x):
        B, N, C = x.shape
        x = x.reshape(B * N // self.window_size, self.window_size, C)
        y, attn = self.attention(self.norm_layer1(x))
        # x = self.mlp1(self.norm_layer2(y)) + x
        x = x.reshape(B, N, C)
        x = self.shift(x)
        x = x.reshape(B * N // self.window_size, self.window_size, C)
        y, attn_2 = self.shifted_attention(self.norm_layer3(x))
        # x = self.mlp2(self.norm_layer4(y)) + x
        x = x.reshape(B, N, C)
        x = self.shift(x, reverse=True)
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
            [self.downsample_res * dim, self.downsample_res // 2 * dim, self.downsample_res // 2 * dim],
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
        downsample_res=4,
        seq_length=1024,
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
                    norm_layer,
                )
                for i in range(num_blocks)
            ]
        )
        self.norm_layer_blocks = norm_layer([seq_length, dim])
        if downsample_res:
            self.merge = PatchMerge(
                dim//2, downsample_res, mlp_drop, act_layer, norm_layer
            )
            self.norm_layer_merge = norm_layer([seq_length, dim])
        else:
            self.merge = None

    def forward(self, x):
        if self.merge:
            x = self.merge(x)
            x = self.norm_layer_merge(x)
        for block in self.blocks:
            x = block(x) + x
        return self.norm_layer_blocks(x)

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
        downsample_res=4,
        depth=[4, 4, 2, 2],
        seq_length=1024,
        attn_drop=0.0,
        proj_drop=0.0,
        mlp_drop=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        num_classes=None
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
                    seq_length // (downsample_res) ** i,
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
        self.head = MLP([self.bottleneck_dim, self.head_dim, self.head_dim], act_layer, mlp_drop)

    def forward(self, x):
        x = self.note_embed(x)
        a = [x]
        for block in self.blocks:
            x = block(x)
            a.append(x)
        return a

    def debug(self, x):
        x = self.note_embed(x)
        for idx, block in enumerate(self.blocks):
            x, attn, attn_2 = block.debug(x)
            if idx == 2:
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