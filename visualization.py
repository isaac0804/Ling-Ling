import math
import os
import matplotlib.pyplot as plt
import muspy
import seaborn as sns
import torch
from torch.utils.data import DataLoader
import numpy as np

from model import Model
from preprocess import MidiDataset
from utils import to_bits

model = Model()
# model.load_state_dict(torch.load("trained_models/model_epoch-5_loss-6.4894.pt"))
model.load_state_dict(torch.load("trained_models/model_epoch-110_loss-5.9413.pt"))

dataset = MidiDataset()

midi = muspy.read_midi("midi/0130.mid")
notes = midi.tracks[0].notes
datas = []
for j in range(math.ceil(len(notes)/1024)):
    data = []
    for i in range(min(j*1024, len(notes)-1024), min(j*1024+1024, len(notes))):
        octave, pitch = (notes[i].pitch-21)//12, (notes[i].pitch-21)%12
        duration = notes[i].duration
        velocity = notes[i].velocity
        if i != 0:
            time_shift = (notes[i].start- notes[i-1].start)//10
        else:
            time_shift = 0

        ret = [0]*38
        ret[:3] = to_bits(octave, 3)
        ret[3:7] = to_bits(pitch, 4)
        ret[7:21] = to_bits(min(int(duration), 16383), 14)
        ret[21:34] = to_bits(min(int(time_shift), 8191), 13)
        ret[34:38] = to_bits(min(velocity//8, 15), 4)
        data.append(ret)
    datas.append(data)
datas = torch.tensor(datas).float()
datas = torch.reshape(datas, (math.ceil(len(notes)/1024), 16, 64, 38))

attns = []
attn_weights = []
for i in range(math.ceil(len(notes)/1024)):
    model.eval()
    attn, attn_weight = model.return_last_attention(datas[i])

    attn = torch.squeeze(attn).detach().numpy()
    attn_weight = torch.squeeze(attn_weight).detach().numpy().reshape((1024))
    attns.append(attn)
    if i != math.ceil(len(notes)/1024)-1:
        attn_weights.append(attn_weight)
    else:
        attn_weights.append(attn_weight[-(len(notes)%1024):])

attn_weights = np.concatenate(attn_weights)
print(attns[0].shape)
print(attn_weights.shape)

fig, ax = plt.subplots(2, 2)
sns.heatmap(attns[0], ax=ax[0, 0])
sns.heatmap(attns[1], ax=ax[0, 1])
sns.heatmap(attns[2], ax=ax[1, 0])
sns.heatmap(attns[3], ax=ax[1, 1])
plt.show()

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