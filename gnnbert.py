import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from bert_config import BERTCONFIG
from config import GNNBERTCONFIG
from graph import GCN
from modules import MLP, Attention, NoteEmbed
from preprocess import MidiDataset
import math
from utils import emb_distance


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

class PianoGNNBERT(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.gnn = GCN()
        self.layers = nn.ModuleList([
            Transformer(config.embedding_dim, config.seq_length, config.num_heads)
            for _ in range(config.num_layers)
        ])

    def forward(self, notes):
        # Note embedding
        x = self.gnn()
        for layer in self.layers:
            x = layer(x)[0]
        outputs = []
        for output_head in self.output_heads:
            outputs.append(output_head(x))
        return outputs


if __name__ == "__main__":
    config = GNNBERTCONFIG()
    model = PianoGNNBERT(config)
    outputs = model(X)