import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.normalization import LayerNorm

from modules import PatchBlock, LocalBlock


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_layers = 4
        self.num_patches = 16
        self.patch = 64 
        self.emb_layers = nn.ModuleList([
            nn.Linear(55, 16),
            # nn.Linear(32, 32),
            # nn.Linear(32, 16)
        ])
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
        self.dropout = nn.Dropout(0.5)
        self.norm = LayerNorm(16)
        # TODO Relavtive Position Encoding
    
    def forward(self, x):
        for layer in self.emb_layers:
            x = layer(x)
        skip = x
        x = self.dropout(x)
        x_patches = []
        for i in range(self.num_layers):
            x_patch = self.patch_layers[i](x)
            x_patch = self.dropout(x_patch)
            x_patches.append(x_patch)
            x = self.local_layers[i](x, x_patch)

            if i != self.num_layers-1:
                x = self.shift(x+skip, self.num_patches//self.num_layers)
                skip = self.shift(skip, self.num_patches//self.num_layers)
            else:
                x = self.shift(x+skip, self.num_patches*(-self.num_layers+1))
                skip = self.shift(skip, self.num_patches//self.num_layers)

        x_patches = torch.stack(x_patches, dim=0)
        x_patches = torch.permute(x_patches, (1, 0, 2))
        x_patches = torch.reshape(x_patches, (64, 16))

        x_global = torch.clone(x_patches.T)
        x_global = self.dropout(x_global)
        for layer in self.global_MLP:
            x_global = F.leaky_relu(layer(x_global))
        x_global = torch.squeeze(x_global.T)
            
        return x, x_patches, x_global
    
    def shift(self, x, value):
        sections, p, emb_dim = x.shape
        x = x.view(-1, emb_dim)
        x = torch.concat([x[value:], x[:value]], 0)
        x = torch.reshape(x, (sections, p, emb_dim))
        return x

    def debug(self, x):
        for layer in self.emb_layers:
            x = layer(x)
        skip = x
        x = self.dropout(x)
        x_patches = []
        for i in range(self.num_layers):
            if i == 0:
                x_patch, attn, attn_w, emb = self.patch_layers[i](x, return_attention=True)
            else:
                x_patch = self.patch_layers[i](x)
            x_patches.append(x_patch)
            if i != self.num_layers-1 and i != 0:
                x = self.local_layers[i](x, x_patch)
            elif i == 0:
                # emb = torch.clone(x_patch)
                # emb = torch.clone(x)
                x, attn, attn_w, _ = self.local_layers[i](x, x_patch, return_attention=True)
            else:
                _, attn, attn_w, _ = self.local_layers[i](x, x_patch, return_attention=True)
            if i != self.num_layers-1:
                x = self.shift(x+skip, self.num_patches//self.num_layers)
                skip = self.shift(skip, self.num_patches//self.num_layers)
            else:
                x = self.shift(x+skip, self.num_patches*(-self.num_layers+1))
                skip = self.shift(skip, self.num_patches//self.num_layers)

        x_patches = torch.stack(x_patches, dim=0)
        # emb = torch.clone(x_patches)
        x_patches = torch.permute(x_patches, (1, 0, 2))
        x_patches = torch.reshape(x_patches, (64, 16))

        x_global = torch.clone(x_patches.T)
        for layer in self.global_MLP:
            x_global = F.leaky_relu(layer(x_global))
        x_global = torch.squeeze(x_global.T)
            
        return attn, attn_w, emb
