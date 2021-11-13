import math
import os

import matplotlib.pyplot as plt
import muspy
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from encoder import Encoder
from preprocess import MidiDataset
from utils import to_bits

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
model.load_state_dict(torch.load("checkpoints/model_epoch-1_loss-5.55.pt"))
model.eval()

dataset = MidiDataset()

midi = muspy.read_midi("midi/0130.mid")
notes = midi.tracks[0].notes
print(len(notes))
datas = []
for j in range(math.ceil(len(notes) / 1024)):
    data = []
    for i in range(min(j * 1024, len(notes) - 1024), min(j * 1024 + 1024, len(notes))):
        octave, pitch = (notes[i].pitch - 21) // 12, (notes[i].pitch - 21) % 12
        duration = notes[i].duration
        velocity = notes[i].velocity
        if i != 0:
            time_shift = notes[i].start - notes[i - 1].start
        else:
            time_shift = 0
        ret = [0] * 8
        ret[0] = octave
        ret[1] = pitch
        ret[2] = min(duration // 20, 9)
        ret[3] = min(duration // 200, 9)
        ret[4] = min(duration // 2000, 9)
        ret[5] = velocity // 8
        ret[6] = min(time_shift // 20, 19)
        ret[7] = min(time_shift // 400, 9)
        data.append(ret)
    datas.append(data)
datas = torch.LongTensor(datas)
datas = torch.reshape(datas, (math.ceil(len(notes) / 1024), 1024, 8))

attns = []
attn_weights = []
for i in range(math.ceil(len(notes) / 1024)):
    model.eval()
    emb = model.debug(datas)
    # local, patch, global_emb = model(datas[i])

    # attn = attn.detach().numpy()
    # attn_weight = attn_weight.detach().numpy()
    # attns.append(attn)
    # attn_weights.append(attn_weight)
    if i == 0:
        print(emb.shape)
        if emb.ndim == 2:
            fig, ax = plt.subplots()
            # sns.heatmap(torch.nn.LayerNorm(16)(emb).detach().numpy(), ax=ax)
            sns.heatmap(emb.detach().numpy(), ax=ax)
            plt.show()
        elif emb.ndim == 3:
            # emb = emb[0]
            fig, ax = plt.subplots(2, 2)
            sns.heatmap(emb.detach().numpy()[0], ax=ax[0, 0])
            sns.heatmap(emb.detach().numpy()[1], ax=ax[0, 1])
            sns.heatmap(emb.detach().numpy()[2], ax=ax[1, 0])
            sns.heatmap(emb.detach().numpy()[3], ax=ax[1, 1])
            plt.show()
        else:
            emb = emb[0]
            fig, ax = plt.subplots(2, 2)
            sns.heatmap(emb.detach().numpy()[0], ax=ax[0, 0])
            sns.heatmap(emb.detach().numpy()[1], ax=ax[0, 1])
            sns.heatmap(emb.detach().numpy()[2], ax=ax[1, 0])
            sns.heatmap(emb.detach().numpy()[3], ax=ax[1, 1])
            plt.show()
    # if i != math.ceil(len(notes)/1024)-1:
    #     attn_weights.append(attn_weight)
    # else:
    #     attn_weights.append(attn_weight[-(len(notes)%1024):])

# attn_weights = np.concatenate(attn_weights)
# print(attns[0].shape)
# print(attn_weights[0].shape)

# NOTES
# fig, ax = plt.subplots(2, 2)
# sns.heatmap(datas[0][0], ax=ax[0, 0])
# sns.heatmap(datas[0][1], ax=ax[0, 1])
# sns.heatmap(datas[0][2], ax=ax[1, 0])
# sns.heatmap(datas[0][3], ax=ax[1, 1])
# plt.show()

# # fig, ax = plt.subplots()
# # mean_attns = np.mean(attns[-1], axis=0)
# # sns.heatmap(mean_attns, ax=ax)
# fig, ax = plt.subplots(2, 2)
# sns.heatmap(attns[0][0], ax=ax[0, 0])
# sns.heatmap(attns[0][1], ax=ax[0, 1])
# sns.heatmap(attns[0][2], ax=ax[1, 0])
# sns.heatmap(attns[0][3], ax=ax[1, 1])
# plt.show()

# # fig, ax = plt.subplots()
# # mean_attn_weights = np.mean(attn_weights[-1], axis=0)
# # sns.heatmap(mean_attn_weights, ax=ax)
# fig, ax = plt.subplots(2,2)
# sns.heatmap(attn_weights[0][0], ax=ax[0, 0])
# sns.heatmap(attn_weights[0][1], ax=ax[0, 1])
# sns.heatmap(attn_weights[0][2], ax=ax[1, 0])
# sns.heatmap(attn_weights[0][3], ax=ax[1, 1])
# plt.show()

# mean = np.mean(attn_weights)
# max_ = max(attn_weights)
# min_ = min(attn_weights)
# values = []

# for i in range(len(notes)):
#     value = int(127*max((attn_weights[i] - min_) / (max_ - min_), 0))

#     if (value > 0):
#         print(notes[i])
#         print(value)
#     values.append(value)
#     notes[i].velocity = value

# midi.write_midi("0130_enc.mid")
# plt.plot(values)
# plt.show()
