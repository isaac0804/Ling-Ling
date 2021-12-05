from typing import ForwardRef
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, features, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        self.act_layer = act_layer()
        self.drop = nn.Dropout(drop)
        self.linear_layers = nn.ModuleList(
            [nn.Linear(features[i], features[i + 1]) for i in range(len(features) - 1)]
        )

    def forward(self, x):
        for idx, linear in enumerate(self.linear_layers):
            x = linear(x)
            x = self.act_layer(x)
            if idx != len(self.linear_layers) - 1:
                x = self.drop(x)
        return x


class Attention(nn.Module):
    """Attention Implementation

    Parameters
    ----------
    num_heads : int
        Number of heads

    head_dim : int
        Dimension of data in each head
    """

    def __init__(self, dim, num_heads=8, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        assert dim % self.num_heads == 0
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """Run the Forward Pass.

        Parameters
        ----------
        x : torch.Tensor
            Input Tensor of shape (batch_size, num_notes, dim)

        Returns
        -------
        x : torch.FloatTensor
            of shape (batch_size, num_notes, dim)

        dim : torch.Tensor
            of shape ()
        """
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class Transformer(nn.Module):
    def __init__(self, dim, num_attn, num_heads=8, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.attns = nn.ModuleList([
            Attention(dim, num_heads, attn_drop, proj_drop)
            for _ in range(num_attn)
        ])

    def forward(self, x):
        for attn in self.attns:
            x = attn(x)[0]
        return x

class LocalBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        mlp_drop=0.0,
        attn_drop=0.0,
        proj_drop=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm_layer1 = norm_layer(dim)
        self.norm_layer2 = norm_layer(dim)
        self.attention = Attention(dim, num_heads, attn_drop, proj_drop)
        self.mlp = MLP([dim, dim], act_layer, drop=mlp_drop)

    def forward(self, x, return_attention=False):
        B, N, P, C = x.shape
        x = x.reshape(B * N, P, C)
        y, attn = self.attention(self.norm_layer1(x))
        x = self.norm_layer2(x + y)
        x = self.mlp(x)
        x = x.reshape(B, N, P, C)
        return x if not return_attention else (x, attn)


class PatchBlock(nn.Module):
    def __init__(
        self,
        dim,
        patch_size,
        num_patch=16,
        mlp_drop=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm_layer = norm_layer([num_patch, dim])
        self.emb = MLP(
            [patch_size, patch_size // 2, 1], act_layer=act_layer, drop=mlp_drop
        )
        primes = [1, 2, 3, 5]  # yes, i know 1 is not a prime
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    1,
                    1,
                    kernel_size=(4, dim),
                    dilation=(prime, 1),
                    groups=1,
                    padding="same",
                    padding_mode="circular",
                )
                for prime in primes
            ]
        )
        self.mlp = MLP([dim, dim], act_layer, drop=mlp_drop)

    def forward(self, x):
        x = x.transpose(-1, -2)
        x = self.emb(x)
        x = torch.unsqueeze(torch.squeeze(x), dim=1)
        y = []
        for conv in self.convs:
            temp = conv(x)
            y.append(torch.squeeze(temp))
        x = torch.stack(y, dim=-2)
        x = torch.sum(x, dim=-2)
        x = self.norm_layer(self.mlp(x))
        return x


class GlobalBlock(nn.Module):
    def __init__(self, embed_dim, num_features, mlp_drop=0.0, act_layer=nn.GELU):
        super().__init__()
        self.emb_t = MLP([num_features, num_features // 8], act_layer, drop=mlp_drop)
        self.emb = MLP([embed_dim, 2 * embed_dim], act_layer, drop=mlp_drop)
        self.mlp_t = MLP([num_features // 8, 1], act_layer)
        self.mlp = MLP([2 * embed_dim, 4 * embed_dim], act_layer)

    def forward(self, x):
        x = x.transpose(-1, -2)
        x = self.emb_t(x).transpose(-1, -2)
        # emb = torch.clone(x)
        x = self.emb(x).transpose(-1, -2)
        # emb = torch.clone(x)
        x = self.mlp_t(x).transpose(-1, -2)
        # emb = torch.clone(x)
        x = self.mlp(x).transpose(-1, -2)
        return torch.squeeze(x)


class NoteEmbed(nn.Module):
    """Note to Embedding.

    Parameters
    ----------
    embedding_dim : int
        Dimension of the note embedding vectors
    """

    def __init__(self, embed_dim=64):
        super().__init__()
        self.embed_dim = embed_dim
        assert self.embed_dim % 8 == 0

        self.octave_embedding = nn.Embedding(
            num_embeddings=8,
            embedding_dim=embed_dim // 8,
            norm_type=2,
            max_norm=1,
        )
        self.pitch_embedding = nn.Embedding(
            num_embeddings=12,
            embedding_dim=embed_dim // 8,
            norm_type=2,
            max_norm=1,
        )
        self.short_duration_embedding = nn.Embedding(
            num_embeddings=10,
            embedding_dim=embed_dim // 8,
            padding_idx=0,
            norm_type=2,
            max_norm=1,
        )
        self.medium_duration_embedding = nn.Embedding(
            num_embeddings=10,
            embedding_dim=embed_dim // 8,
            padding_idx=0,
            norm_type=2,
            max_norm=1,
        )
        self.long_duration_embedding = nn.Embedding(
            num_embeddings=10,
            embedding_dim=embed_dim // 8,
            padding_idx=0,
            norm_type=2,
            max_norm=1,
        )
        self.velocity_embedding = nn.Embedding(
            num_embeddings=16,
            embedding_dim=embed_dim // 8,
            padding_idx=0,
            norm_type=2,
            max_norm=1,
        )
        self.short_shift_embedding = nn.Embedding(
            num_embeddings=20,
            embedding_dim=embed_dim // 8,
            padding_idx=0,
            norm_type=2,
            max_norm=1,
        )
        self.long_shift_embedding = nn.Embedding(
            num_embeddings=10,
            embedding_dim=embed_dim // 8,
            padding_idx=0,
            norm_type=2,
            max_norm=1,
        )

    def forward(self, x):
        """Forward Pass of Note Embedding

        Parameters
        ----------
        x : torch.LongTensor
            Input Tensor of shape (batch_size, num_patches, patch_notes, 8)

        Returns
        -------
        emb : torch.FloatTensor
            Note Embedding of shape (batch_size, num_patches, patch_notes, embedding_dim*8)
        """
        octave = self.octave_embedding(x[..., 0])
        pitch = self.pitch_embedding(x[..., 1])
        short_note = self.short_duration_embedding(x[..., 2])
        medium_note = self.medium_duration_embedding(x[..., 3])
        long_note = self.long_duration_embedding(x[..., 4])
        velocity = self.velocity_embedding(x[..., 5])
        short_shift = self.short_shift_embedding(x[..., 6])
        long_shift = self.long_shift_embedding(x[..., 7])

        emb = torch.concat(
            [
                octave,
                pitch,
                short_note,
                medium_note,
                long_note,
                velocity,
                short_shift,
                long_shift,
            ],
            dim=-1,
        )
        return emb


class Encoder(nn.Module):
    """Transformer Encoder"""

    def __init__(
        self,
        embed_dim=32,
        num_heads=8,
        patch_size=64,
        depth=4,
        seq_length=1024,
        mlp_drop=0.0,
        attn_drop=0.0,
        proj_drop=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert depth % 2 == 0
        self.patch_size = patch_size
        self.num_patch = seq_length // patch_size
        self.depth = depth
        self.embed_dim = embed_dim
        self.note_embed = NoteEmbed(embed_dim=embed_dim)
        self.norm_layer1 = norm_layer([self.num_patch, self.patch_size, self.embed_dim])
        self.norm_layer2 = norm_layer(
            [self.num_patch * self.depth // 2, self.embed_dim]
        )
        self.norm_layer3 = norm_layer(4 * self.embed_dim)
        self.local_blocks = nn.ModuleList(
            [
                LocalBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_drop=mlp_drop,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                )
                for _ in range(self.depth)
            ]
        )
        self.patch_blocks = nn.ModuleList(
            [
                PatchBlock(
                    dim=embed_dim,
                    patch_size=patch_size,
                    num_patch=self.num_patch,
                    mlp_drop=mlp_drop,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                )
                for _ in range(self.depth)
            ]
        )
        self.global_block = GlobalBlock(
            embed_dim=embed_dim,
            num_features=self.num_patch * depth // 2,
            mlp_drop=mlp_drop,
            act_layer=act_layer,
        )

    def prepare_tokens(self, x):
        B, N, C = x.shape
        assert N % self.patch_size == 0
        x = x.reshape(B, N // self.patch_size, self.patch_size, C)

        # TODO: CLS TOKEN
        # TODO: POSITION ENCODING

        return x

    def shift(self, x, value):
        B, N, P, C = x.shape
        x = x.reshape(B, N * P, C)
        x = torch.roll(x, shifts=-value, dims=1)
        x = x.reshape(B, N, P, C)
        return x

    def forward(self, x):
        x = self.prepare_tokens(x)
        x = self.note_embed(x)
        res = x
        patches = []
        for i in range(self.depth):
            patch = self.patch_blocks[i](x)
            x = self.local_blocks[i](x)
            if (i + 1) > self.depth // 2:
                patches.append(patch)
            if (i + 1) % (self.depth // 2) != 0:
                x = self.shift(x + res, self.patch_size // (self.depth // 2))
                res = self.shift(res, self.patch_size // (self.depth // 2))
            else:
                x = self.shift(
                    x + res, -self.patch_size + self.patch_size // (self.depth//2)
                )
                res = self.shift(res, -self.patch_size + self.patch_size // (self.depth//2))
            x = self.norm_layer1(x)

        patches = torch.stack(patches, dim=2).flatten(start_dim=1, end_dim=2)
        patches = self.norm_layer2(patches)
        global_embed = torch.squeeze(self.global_block(patches), dim=-1)

        return x, patches, global_embed

    def debug(self, x):
        x = self.prepare_tokens(x)
        x = self.note_embed(x)
        res = x
        patches = []
        for i in range(self.depth):
            patch = self.patch_blocks[i](x)
            if i == 4:
                x, attn = self.local_blocks[i](x, return_attention=True)
                emb=torch.clone(x)
                emb=torch.clone(attn)
            else:
                x = self.local_blocks[i](x)
            if (i + 1) > self.depth // 2:
                patches.append(patch)
            if (i + 1) % (self.depth // 2) != 0:
                x = self.shift(x + res, self.patch_size // (self.depth // 2))
                res = self.shift(res, self.patch_size // (self.depth // 2))
            else:
                x = self.shift(
                    x + res, -self.patch_size + self.patch_size // (self.depth // 2)
                )
                res = self.shift(
                    res, -self.patch_size + self.patch_size // (self.depth // 2)
                )
            x = self.norm_layer1(x)

        # emb = torch.clone(x)
        emb = torch.clone(patch)
        patches = torch.stack(patches, dim=2).flatten(start_dim=1, end_dim=2)
        patches = self.norm_layer2(patches)
        # emb = torch.clone(patches)
        global_embed = self.global_block(patches)
        # emb = torch.clone(global_embed)
        return emb 

    def return_attention(self, x):
        x = self.prepare_tokens(x)
        x = self.note_embed(x)
        res = x
        patches = []
        for i in range(self.depth):
            patch = self.patch_blocks[i](x)
            if i == self.depth-1:
                x, attn = self.local_blocks[i](x, return_attention=True)
                return attn
            else:
                x = self.local_blocks[i](x)
            if (i + 1) > self.depth // 2:
                patches.append(patch)
            if (i + 1) % (self.depth // 2) != 0:
                x = self.shift(x + res, self.patch_size // (self.depth // 2))
                res = self.shift(res, self.patch_size // (self.depth // 2))
            else:
                x = self.shift(
                    x + res, -self.patch_size + self.patch_size // (self.depth//2)
                )
                res = self.shift(res, -self.patch_size + self.patch_size // (self.depth//2))
            x = self.norm_layer1(x)