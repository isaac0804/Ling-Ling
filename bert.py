import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from modules import MLP, Attention, NoteEmbed
from preprocess import MidiDataset
import math


class Transformer(nn.Module):
    def __init__(self, embedding_dim, seq_length, num_heads) -> None:
        super().__init__()
        self.attn = Attention(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    attn_drop=0.1,
                    proj_drop=0.1,
                )
        self.norm = nn.LayerNorm([seq_length, embedding_dim])
        self.mlp = MLP(
                    [embedding_dim, embedding_dim * 2, embedding_dim],
                    nn.GELU,
                    drop=0.1,
                )
        self.norm2 = nn.LayerNorm([seq_length, embedding_dim])
    
    def forward(self, x):
        y, attn = self.attn(x)
        x = self.norm(y + x)
        y = self.mlp(x)
        x = self.norm2(y + x)
        return x, attn

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout = 0.1, max_len = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        import matplotlib.pyplot as plt
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x.transpose(0, 1)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x).transpose(0, 1)

class PianoBERT(nn.Module):
    def __init__(self, embedding_dim, seq_length, num_heads, num_layers) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.seq_length = seq_length
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            Transformer(embedding_dim, seq_length, num_heads)
            for _ in range(num_layers)
        ])
        self.note_embed = NoteEmbed(embedding_dim=embedding_dim)
        self.output_heads = nn.ModuleList([
            MLP([embedding_dim, embedding_dim*4, embedding_dim, num], drop=0.1)
            for num in [10, 14, 12, 12, 12, 18, 22, 12]
        ])

    def forward(self, notes):
        # Note embedding
        x = self.note_embed(notes)
        for layer in self.layers:
            x = layer(x)[0]
        outputs = []
        for output_head in self.output_heads:
            outputs.append(output_head(x))
        return outputs


if __name__ == "__main__":
    dataset = MidiDataset(seq_len=512)
    loader = DataLoader(dataset, batch_size=2)
    X, Y, mask = next(iter(loader))
    model = PianoBERT(embedding_dim=64, seq_length=512, num_heads=4, num_layers=6)
    outputs = model(X)
    EMBEDDING_DIM_LEN = [10, 14, 12, 12, 12, 18, 22, 12] 
    for ii, output in enumerate(outputs):
        _mask = torch.stack([mask] * EMBEDDING_DIM_LEN[ii], dim=-1)
        output = output.masked_select(_mask).view(-1, EMBEDDING_DIM_LEN[ii])
        target = Y[...,ii].masked_select(mask)
        print(target.shape)
        print(output.shape)
        loss = F.cross_entropy(output, target)