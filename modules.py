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

    def __init__(self, embedding_dim, num_heads=8, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        assert embedding_dim % self.num_heads == 0
        self.head_dim = embedding_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embedding_dim, embedding_dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embedding_dim, embedding_dim)
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
            Output shape of shape (batch_size, num_notes, dim)

        attn : torch.Tensor
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


# class Transformer(nn.Module):
#     def __init__(self, dim, num_attn, num_heads=8, attn_drop=0.0, proj_drop=0.0):
#         super().__init__()
#         self.attns = nn.ModuleList(
#             [Attention(dim, num_heads, attn_drop, proj_drop) for _ in range(num_attn)]
#         )

#     def forward(self, x):
#         for attn in self.attns:
#             x = attn(x)[0]
#         return x

class NoteEmbed(nn.Module):
    """Note to Embedding.

    Parameters
    ----------
    embed_dim : int
        Dimension of the note embedding vectors containing 8 features
        Each feature embedded by {embed_dim // 8}-d vector
    """

    def __init__(self, embedding_dim=64):
        super().__init__()
        self.embed_dim = embedding_dim
        assert self.embed_dim % 8 == 0

        self.octave_embedding = nn.Embedding(
            num_embeddings=8 + 3,
            embedding_dim=embedding_dim // 8,
            norm_type=2,
            max_norm=1,
        )
        self.pitch_embedding = nn.Embedding(
            num_embeddings=12 + 3,
            embedding_dim=embedding_dim // 8,
            norm_type=2,
            max_norm=1,
        )
        self.short_duration_embedding = nn.Embedding(
            num_embeddings=10 + 3,
            embedding_dim=embedding_dim // 8,
            norm_type=2,
            max_norm=1,
        )
        self.medium_duration_embedding = nn.Embedding(
            num_embeddings=10 + 3,
            embedding_dim=embedding_dim // 8,
            norm_type=2,
            max_norm=1,
        )
        self.long_duration_embedding = nn.Embedding(
            num_embeddings=10 + 3,
            embedding_dim=embedding_dim // 8,
            norm_type=2,
            max_norm=1,
        )
        self.velocity_embedding = nn.Embedding(
            num_embeddings=16 + 3,
            embedding_dim=embedding_dim // 8,
            norm_type=2,
            max_norm=1,
        )
        self.short_shift_embedding = nn.Embedding(
            num_embeddings=20 + 3,
            embedding_dim=embedding_dim // 8,
            norm_type=2,
            max_norm=1,
        )
        self.long_shift_embedding = nn.Embedding(
            num_embeddings=10 + 3,
            embedding_dim=embedding_dim // 8,
            norm_type=2,
            max_norm=1,
        )

    def forward(self, x):
        """Forward Pass of Note Embedding

        Parameters
        ----------
        x : torch.LongTensor
            Input Tensor of shape (batch_size, num_notes, 8)

        Returns
        -------
        emb : torch.FloatTensor
            Note Embedding of shape (batch_size, num_notes, embedding_dim)
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
