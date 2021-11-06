import torch
import torch.nn.functional as F
from torch import nn

from modules import PatchBlock, LocalBlock


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_layers = 4
        self.num_patches = 16
        self.patch = 64 
        self.linear = nn.Linear(38, 16)
        self.patch_layers = nn.ModuleList([
            PatchBlock(),
            PatchBlock(),
            PatchBlock(),
            PatchBlock()
        ])
        self.local_layers = nn.ModuleList([
            LocalBlock(),
            LocalBlock(),
            LocalBlock(),
            LocalBlock()
        ])
        self.global_MLP = nn.ModuleList([
            nn.Linear(64, 16),
            nn.Linear(16, 8),
            nn.Linear(8, 1)
        ])
    
    def forward(self, x):
        x = self.linear(x)
        x_patches = []
        for i in range(self.num_layers):
            x_patch = self.patch_layers[i](x)
            x_patches.append(x_patch)
            x = self.local_layers[i](x, x_patch)
            if i != self.num_layers:
                x = self.shift(x, self.num_patches//self.num_layers)
            else:
                x = self.shift(x, self.num_patches*(-self.num_layers+1))

        x_patches = torch.stack(x_patches, dim=0)
        x_patches = torch.permute(x_patches, (1, 0, 2))
        x_patches = torch.reshape(x_patches, (64, 16))

        x_global = torch.clone(x_patches.T)
        for layer in self.global_MLP:
            x_global = F.relu(layer(x_global))
        x_global = torch.squeeze(x_global.T)
            
        return x, x_patch, x_global
    
    def shift(self, x, value):
        sections, p, emb_dim = x.shape
        x = x.view(-1, emb_dim)
        x = torch.concat([x[value:], x[:value]], 0)
        x = torch.reshape(x, (sections, p, emb_dim))
        return x

    def return_last_attention(self, x):
        x = self.linear(x)
        x_patches = []
        for i in range(self.num_layers):
            x_patch = self.patch_layers[i](x)
            x_patches.append(x_patch)
            if i != self.num_layers-1:
                x = self.local_layers[i](x, x_patch)
            else:
                x, attn, attn_w = self.local_layers[i](x, x_patch, return_attention=True)
            if i != self.num_layers:
                x = self.shift(x, self.num_patches//self.num_layers)
            else:
                x = self.shift(x, self.num_patches*(-self.num_layers+1))

        x_patches = torch.stack(x_patches, dim=0)
        x_patches = torch.permute(x_patches, (1, 0, 2))
        x_patches = torch.reshape(x_patches, (64, 16))

        x_global = torch.clone(x_patches.T)
        for layer in self.global_MLP:
            x_global = F.relu(layer(x_global))
        x_global = torch.squeeze(x_global.T)
            
        return attn, attn_w
