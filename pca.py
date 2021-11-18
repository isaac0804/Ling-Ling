import math
import os

import matplotlib.pyplot as plt
import muspy
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import Encoder
from preprocess import MidiDataset

model = Encoder(
    embed_dim=64,
    num_heads=8,
    patch_size=64,
    depth=16,
    seq_length=1024,
    mlp_drop=0.5,
    attn_drop=0.5,
    proj_drop=0.5,
    act_layer=nn.GELU,
    norm_layer=nn.LayerNorm,
)
model.load_state_dict(torch.load("checkpoints/model_epoch-296_loss-4.26.pt"))
model.eval()

pitchs = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
octaves = np.arange(8)
embed = model.note_embed.pitch_embedding.weight
# embed = model.note_embed.long_shift_embedding.weight
intervals = [0, 7]

for i in range(12):
    print(pitchs[i])
    print(embed[i].detach().numpy())

U, S, V = torch.pca_lowrank(embed)
test = torch.matmul(embed, V[:, :3])
x, y, z = test[:-1, 0].detach().numpy(), test[:-1, 1].detach().numpy(), test[:-1, 2].detach().numpy()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x, y, z)
for i, pitch in enumerate(pitchs):
    ax.text(x[i], y[i], z[i], '%s' % (pitch), size=10, zorder=1, color='k')
for i, pitch in enumerate(pitchs):
    a, b, c = [], [], []
    for interval in intervals:
        a.append(x[(i+interval)%12])
        b.append(y[(i+interval)%12])
        c.append(z[(i+interval)%12])
    plt.plot(a, b, c)
plt.show()