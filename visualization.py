import math
import os

import matplotlib.pyplot as plt
import muspy
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from model import Model
from preprocess import MidiDataset
from utils import to_bits

model = Model()
# model.load_state_dict(torch.load("trained_models/model_epoch-5_loss-6.4894.pt"))
model.load_state_dict(torch.load("checkpoints/model_epoch-6_loss-2.45.pt"))

dataset = MidiDataset()

midi = muspy.read_midi("midi/0130.mid")
notes = midi.tracks[0].notes
datas = []
for j in range(math.ceil(len(notes)/1024)):
    data = []
    for i in range(min(j*1024, len(notes)-1024), min(j*1024+1024, len(notes))):
        octave, pitch = (notes[i].pitch-21)//12, (notes[i].pitch-21)%12
        duration = notes[i].duration//10
        velocity = notes[i].velocity
        prev_notes = [0.0]*32
        for j in range(min(i, 8)):
            prev_notes[4*(7-j)] = ((notes[i-j].start- notes[i-j-1].start)//100)/100
            prev_notes[4*(7-j)+1] = ((notes[i-j].start- notes[i-j-1].start)%100)/100
            prev_notes[4*(7-j)+2] = ((notes[i-j].pitch - notes[i-j-1].pitch)//12)/8
            prev_notes[4*(7-j)+3] = ((notes[i-j].pitch - notes[i-j-1].pitch)%12)/12

        # ret = [0]*38
        # ret[:3] = to_bits(octave, 3)
        # ret[3:7] = to_bits(pitch, 4)
        # ret[7:21] = to_bits(min(int(duration), 16383), 14)
        # ret[21:34] = to_bits(min(int(time_shift), 8191), 13)
        # ret[34:38]ret = [0.0]*25
        ret = [0.0]*55
        ret[octave] = 1.0
        ret[8+pitch] = 1.0
        ret[20] = velocity/128
        ret[21] = (duration//100)/100
        ret[22] = (duration%100)/100
        ret[23:55] = prev_notes 
        data.append(ret)
    datas.append(data)
datas = torch.tensor(datas).float()
datas = torch.reshape(datas, (math.ceil(len(notes)/1024), 16, 64, 55))

attns = []
attn_weights = []
for i in range(math.ceil(len(notes)/1024)):
    model.eval()
    attn, attn_weight, emb = model.debug(datas[i])
    local, patch, global_emb = model(datas[i])
    
    attn = attn.detach().numpy()
    attn_weight = attn_weight.detach().numpy()
    attns.append(attn)
    attn_weights.append(attn_weight)

    if i == 0:
        # fig, ax = plt.subplots()
        # sns.heatmap(patch.detach().numpy(), ax=ax)
        # plt.show()
        if emb.ndim == 2:
            fig, ax = plt.subplots()
            # sns.heatmap(torch.nn.LayerNorm(16)(emb).detach().numpy(), ax=ax)
            sns.heatmap(emb.detach().numpy(), ax=ax)
        else:
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
