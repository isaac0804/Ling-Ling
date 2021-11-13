import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from encoder import *
from preprocess import MidiDataset

encoder = Encoder(
    embed_dim=32,
    num_heads=8,
    patch_size=64,
    depth=8,
    seq_length=1024,
    mlp_drop=0.4,
    attn_drop=0.4,
    proj_drop=0.4,
    act_layer=nn.GELU,
    norm_layer=nn.LayerNorm,
)
midiDataset = MidiDataset()

data = iter(midiDataset)
midi = next(data)

local, patches, global_embed = encoder(midi)

print(local)
print(local.shape)
# print(patch)
print(patches.shape)
# print(global_embed)
print(global_embed.shape)
emb = patches
fig, ax = plt.subplots(2)
sns.heatmap(emb.detach().numpy()[0], ax=ax[0])
sns.heatmap(emb.detach().numpy()[1], ax=ax[1])
plt.show()
