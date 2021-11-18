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
    embed_dim=32,
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
model.load_state_dict(torch.load("checkpoints/model_epoch-29_loss-4.85.pt"))
# model.load_state_dict(torch.load("experiments/experiment4/model_epoch-296_loss-4.26.pt"))
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

    if i == 0:
        print(emb.shape)
        is_attn = True
        is_attn = False 
        if is_attn == True:
            emb = (emb[:16][0]).detach().numpy()
            # emb = emb[0].detach().numpy()
            # fig, ax = plt.subplots()
            fig, ax = plt.subplots(2, 4)
            for i in range(8):
                sns.heatmap(emb[i], ax=ax[i // 4, i % 4])
                ax[i // 4, i % 4].set_aspect("equal")
                ax[i // 4, i % 4].set_title(f"Head {i+1}")
            plt.show()
            break

        if emb.ndim == 2:
            fig, ax = plt.subplots()
            sns.heatmap(emb.detach().numpy(), ax=ax)
            plt.show()
        elif emb.ndim == 3:
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
