import torch
import torch.nn as nn
from modules import MLP, Attention, NoteEmbed


class EncoderBlock(nn.Module):
    """Encoder Block using Shifted window self attention

    Parameters
    ----------
    dim : int
        Embedding dimension

    window_size : int
        Window size of Swin Encoder

    num_heads : int
        Number of heads in each Attention Block

    attn_drop : float
        Dropout value for Attention

    proj_drop : float
        Dropout value for Projection layer or MLP

    norm_layer : nn.Module
        Normalization layer used in model
    """

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
        self.attention = Attention(dim, num_heads, attn_drop, proj_drop)
        self.shifted_attention = Attention(dim, num_heads, attn_drop, proj_drop)

    def shift(self, x, reverse=False):
        if reverse:
            torch.roll(x, shifts=self.window_size // 2, dims=1)
        else:
            torch.roll(x, shifts=-self.window_size // 2, dims=1)
        return x

    def forward(self, x):
        """Run the Forward Pass.

        Parameters
        ----------
        x : torch.Tensor
            Input Tensor of shape (batch_size, num_notes, dim)

        Returns
        -------
        x : torch.Tensor
            Output Tensor of shape (batch_size, num_notes, dim)
        """
        B, N, C = x.shape
        x = x.reshape(B * N // self.window_size, self.window_size, C)
        x = self.attention(self.norm_layer1(x))[0]
        x = x.reshape(B, N, C)
        x = self.shift(x)
        x = x.reshape(B * N // self.window_size, self.window_size, C)
        x = self.shifted_attention(self.norm_layer2(x))[0]
        x = x.reshape(B, N, C)
        x = self.shift(x, reverse=True)
        return x

    def debug(self, x):
        B, N, C = x.shape
        x = x.reshape(B * N // self.window_size, self.window_size, C)
        x, attn = self.attention(self.norm_layer1(x))
        x = x.reshape(B, N, C)
        x = self.shift(x)
        x = x.reshape(B * N // self.window_size, self.window_size, C)
        x, attn_2 = self.shifted_attention(self.norm_layer2(x))
        x = x.reshape(B, N, C)
        x = self.shift(x, reverse=True)
        return x, attn, attn_2


class PatchMerge(nn.Module):
    """Patch Merge, merge the consecutive notes to a higher dimension features

    Parameters
    ----------
    dim : int
        Embedding Dimension

    downsample_ratio : int
        Ratio of downsampling, default 4

    mlp_drop : float
        Dropout value for MLP

    act_layer : nn.Module
        Activation Layer

    norm_layer : nn.Module
        Normalization Layer
    """

    def __init__(
        self,
        dim,
        downsample_ratio=4,
        mlp_drop=0.5,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.downsample_ratio = downsample_ratio
        self.reduction = MLP(
            [
                self.downsample_ratio * dim,
                2 * dim,
                2 * dim,
            ],
            act_layer,
            mlp_drop,
        )
        self.norm_layer = norm_layer(self.downsample_ratio * dim)

    def forward(self, x):
        """Run the forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input Tensor of shape (batch_size, num_notes, dim)
            
        Returns
        -------
        x : torch.Tensor
            Output Tensor of shape (batch_size, num_notes // downsample_ratio, 2 * dim)
        """
        B, N, C = x.shape
        x = x.reshape(B, N // self.downsample_ratio, self.downsample_ratio * C)
        x = self.reduction(self.norm_layer(x))
        return x


class BasicBlock(nn.Module):
    """Basic Block consists of Encoder Block and Patch Merge"""

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
                dim // 2, downsample_res, mlp_drop, act_layer, norm_layer
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
        num_classes=None,
    ):
        super().__init__()
        assert seq_length % (downsample_res ** (len(depth) - 1)) == 0
        assert seq_length / downsample_res ** (len(depth) - 1) >= window_size
        self.window_size = window_size
        self.note_embed = NoteEmbed(embed_dim=embed_dim)
        self.embs_dim = [embed_dim, embed_dim, embed_dim * 2, embed_dim * 4]
        self.blocks = nn.ModuleList(
            [
                BasicBlock(
                    embed_dim * (downsample_res // 2) ** i,
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
        self.bottleneck_dim = embed_dim * (downsample_res // 2) ** (len(depth) - 1)
        self.head_dim = num_classes if num_classes else self.bottleneck_dim
        self.norm = nn.LayerNorm(self.bottleneck_dim)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = MLP(
            [self.bottleneck_dim, self.head_dim, self.head_dim], act_layer, mlp_drop
        )

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


if __name__ == "__main__":
    encoder = SwinEncoder(
        embed_dim=16,
        window_size=16,
        num_heads=4,
        downsample_res=4,
        depth=[4, 4, 2, 2],
        seq_length=1024,
        attn_drop=0.5,
        proj_drop=0.5,
        mlp_drop=0.5,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    )
    data = torch.randint(0, 8, [1, 1024, 8])
    outputs = encoder(data)
    print([output.shape for output in outputs])
