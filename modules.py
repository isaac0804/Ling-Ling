import torch
import torch.nn.functional as F
from torch import nn


class PatchBlock(nn.Module):
    """
    Patch Block 
    Input: (N//p, p, d_global)
    Output: (N//p, d_global)
    """
    def __init__(self):
        super().__init__()
        self.emb_layer = nn.ModuleList([
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        ])
        self.MLP = nn.ModuleList([
            nn.Linear(16, 16),
            nn.Linear(16, 16),
        ])
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
    Input: (16, 64, 16)
    """
    def __init__(self):
        super().__init__()
        self.emb_layer = nn.ModuleList([
            nn.Linear(32, 32),
            nn.Linear(32, 16),
        ])
        self.MLP = nn.ModuleList([
            nn.Linear(16, 16),
            nn.Linear(16, 16),
        ])
        self.MHA = nn.MultiheadAttention(16, 4, batch_first=True)
        self.layer_norm = nn.LayerNorm(16)

    def forward(self, x, x_global, return_attention=False):
        x_global = torch.concat([x_global]*64, -1) # doubtful
        x_global = torch.reshape(x_global, (16, 64, 16)) # doubtful
        x = torch.concat([x, x_global], dim=-1) 
        for layer in self.emb_layer:
            x = F.relu(layer(x))
        attn, attn_w = self.MHA(x, x, x)
        x = self.layer_norm(attn+x)
        for layer in self.MLP:
            x = F.relu(layer(x)) + x
        if not return_attention:
            return x
        else:
            return x, attn, attn_w
        
