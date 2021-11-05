import torch
import torch.nn.functional as F
from torch import nn


class GlobalBlock(nn.Module):
    """
    Global Block 
    Input: (N//p, p, d_global)
    Output: (N//p, d_global)
    """
    def __init__(self):
        super().__init__()
        self.emb_layer = [
            nn.Linear(128, 16),
            nn.Linear(16, 1),
        ]
        self.MLP = [
            nn.Linear(16, 16),
            nn.Linear(16, 16),
        ]
        self.MHA = nn.MultiheadAttention(16, 4)
        self.layer_norm = nn.LayerNorm(16)

    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))
        for layer in self.emb_layer:
            x = F.relu(layer(x))
        x = torch.unsqueeze(torch.squeeze(x), 0)
        attn, _ = self.MHA(x, x, x)
        x = self.layer_norm(attn+x)
        for layer in self.MLP:
            x = F.relu(layer(x)) + x
        return torch.squeeze(x)

class LocalBlock(nn.Module):
    """
    Local Block
    Input: (16, 128, 16)
    """
    def __init__(self):
        super().__init__()
        self.MLP = [
            nn.Linear(16, 16),
            nn.Linear(16, 16),
        ]
        self.MHA = nn.MultiheadAttention(16, 4, batch_first=True)
        self.layer_norm = nn.LayerNorm(16)

    def forward(self, x, x_global):
        # not concat for now, pass global feature into query of MHA
        # x_global = torch.concat([x_global]*128, 0) # doubtful
        x_global = torch.unsqueeze(x_global, 1)
        # x = torch.concat([x, x_global], dim=-1) 
        attn, _ = self.MHA(x_global, x, x)
        x = self.layer_norm(attn+x)
        for layer in self.MLP:
            x = F.relu(layer(x)) + x
        return x
