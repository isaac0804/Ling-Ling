# import torch
# import torch.nn.functional as F
# from torch import nn


# class PatchBlock(nn.Module):
#     """
#     Patch Block 
#     Input: (N//p, p, d_global)
#     Output: (N//p, d_global)
#     """
#     def __init__(self):
#         super().__init__()
#         self.emb_layers = nn.ModuleList([
#             nn.Linear(16, 16),
#             nn.Linear(16, 16),
#         ])
#         self.MLP = nn.ModuleList([
#             nn.Linear(16, 16),
#             nn.Linear(16, 16),
#         ])
#         self.MHA = nn.MultiheadAttention(16, 4, batch_first=True)
#         self.layer_norm = nn.LayerNorm(16)
#         self.dropout = nn.Dropout(0.5)

#     def forward(self, x, return_attention=False):
#         x = torch.reshape(x, (16, 64, 16))
#         x = (torch.mean(x, dim=-2) - torch.min(x, dim=-2)[0])/(torch.max(x, dim=-2)[0] - torch.min(x, dim=-2)[0])
#         # temp = x
#         # for i in range(len(self.emb_layers)):
#         #     x = F.leaky_relu(self.emb_layers[i](x))
#         # x = self.dropout(x)
#         # x = x + temp 
#         emb = torch.clone(x)
#         x = torch.unsqueeze(x, 0)
#         attn, attn_w = self.MHA(x, x, x)
#         # emb = torch.clone(torch.squeeze(attn))
#         x = self.layer_norm(attn) + x
#         # emb = torch.clone(torch.squeeze(x))
#         temp = x
#         for layer in self.MLP:
#             x = F.leaky_relu(layer(x))
#         x = self.dropout(x)
#         x = x + temp
#         if not return_attention:
#             return torch.sigmoid(torch.squeeze(x))
#         else:
#             return torch.sigmoid(torch.squeeze(x)), attn, attn_w, emb

# class LocalBlock(nn.Module):
#     """
#     Local Block
#     Input: (16, 64, 16)
#     """
#     def __init__(self):
#         super().__init__()
#         self.emb_layers = nn.ModuleList([
#             nn.Linear(16, 16),
#             nn.Linear(16, 16),
#         ])
#         self.MLP = nn.ModuleList([
#             nn.Linear(16, 16),
#             nn.Linear(16, 16),
#         ])
#         self.MHA = nn.MultiheadAttention(16, 4, batch_first=True)
#         self.layer_norm = nn.LayerNorm(16)
#         self.dropout = nn.Dropout(0.5)
#         self.patch_weight = 0.2

#     def forward(self, x, x_patch, return_attention=False):
#         x_patch = torch.concat([x_patch]*64, -1) 
#         x_patch = torch.reshape(x_patch, (16, 64, 16)) 
#         x = (1-self.patch_weight)*x + self.patch_weight*x_patch
#         x = (x - torch.stack([torch.mean(x, dim=-2)]*64, dim=-2))/(torch.stack([torch.max(x, dim=-2)[0]]*64, dim=-2)-torch.stack([torch.min(x, dim=-2)[0]]*64, dim=-2))
#         # temp = x
#         # for layer in self.emb_layers:
#         #     x = F.leaky_relu(layer(x))
#         # x = self.dropout(x)
#         # x = x + temp
#         attn, attn_w = self.MHA(x, x, x)
#         x = self.layer_norm(attn) + x
#         temp = x 
#         emb = torch.clone(x)
#         for layer in self.MLP:
#             x = F.leaky_relu(layer(x))
#         x = self.dropout(x)
#         x = self.layer_norm(x) + temp
#         emb = torch.clone(torch.sigmoid(x))
#         if not return_attention:
#             return torch.sigmoid(x)
#         else:
#             return torch.sigmoid(x), attn, attn_w, emb 
        
