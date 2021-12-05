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
from vq_vae import Generator

model = Generator(
    embed_dim=32,
    window_size=16,
    num_heads=4,
    downsample_res=4,
    depth=[8, 6, 4, 2],
    codebook_size=[128, 64, 32],
    seq_length=1024,
    attn_drop=0.5,
    proj_drop=0.5,
    mlp_drop=0.5,
    commitment_loss=0.25,
    act_layer=nn.GELU,
    norm_layer=nn.LayerNorm,
)
checkpoints = torch.load("checkpoints/model_epoch-200.pt")
model.load_state_dict(checkpoints["gen_state_dict"])
model.eval()

# intervals = [0, 7]
pitchs = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
octave_embed = model.encoder.note_embed.octave_embedding.weight
pitch_embed = model.encoder.note_embed.pitch_embedding.weight
ret = []
for i, o in enumerate(octave_embed[:-1]):
    for j, p in enumerate(pitch_embed[:-1]):
        ret.append(torch.concat([o, p], dim=-1))
embed = torch.stack(ret)
print(embed.shape)

# for i in range(12):
#     print(pitchs[i])
#     print(embed[i].detach().numpy())

U, S, V = torch.pca_lowrank(embed)
test = torch.matmul(embed, V[:, :3])
x, y, z = test[:-1, 0].detach().numpy(), test[:-1, 1].detach().numpy(), test[:-1, 2].detach().numpy()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x, y, z)
for o in range(8):
    for i, pitch in enumerate(pitchs):
        # ax.text(x[8*o+i], y[8*o+i], z[8*o+i], f"{pitch}{o}", size=10, zorder=1, color='k')
        plt.plot(x[8*o+i], y[8*o+i], z[8*o+i])
        # ax.text(x[i], y[i], z[i], '%s' % (pitch), size=10, zorder=1, color='k')
# for i, pitch in enumerate(pitchs):
#     a, b, c = [], [], []
#     for interval in intervals:
#         a.append(x[(i+interval)%12])
#         b.append(y[(i+interval)%12])
#         c.append(z[(i+interval)%12])
#     plt.plot(a, b, c)
plt.show()