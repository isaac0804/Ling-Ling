import math
import muspy
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from swin_encoder import SwinEncoder
from preprocess import MidiDataset

show_attention = True

model = SwinEncoder(
    embed_dim=64,
    window_size=16,
    num_heads=8,
    num_classes=None,
    downsample_res=4,
    depth=[8, 4, 2, 2],
    seq_length=1024,
    mlp_drop=0.5,
    attn_drop=0.5,
    proj_drop=0.5,
    act_layer=nn.GELU,
    norm_layer=nn.LayerNorm,
)

model.load_state_dict(torch.load("checkpoints/model_epoch-100_loss-6.24.pt"))
model.eval()

dataset = MidiDataset()

midi = muspy.read_midi("midi/0130.mid")
midi_notes = midi.tracks[0].notes
samples = []
for j in range(math.ceil(len(midi_notes) / 1024)):
    sample = []
    for i in range(min(j * 1024, len(midi_notes) - 1024), min(j * 1024 + 1024, len(midi_notes))):
        octave, pitch = (midi_notes[i].pitch - 21) // 12, (midi_notes[i].pitch - 21) % 12
        duration = midi_notes[i].duration
        velocity = midi_notes[i].velocity
        if i != 0:
            time_shift = midi_notes[i].start - midi_notes[i - 1].start
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
        sample.append(ret)
    samples.append(sample)
samples = torch.LongTensor(samples)
samples = torch.reshape(samples, (math.ceil(len(midi_notes) / 1024), 1024, 8))

print(f"Shape of samples   : {samples.shape}")
embed = model.return_attention(samples)
print(f"Shape of embedding : {embed.shape}")

print(sum(p.numel() for p in model.parameters()))

# if embed.ndim == 2:
#     fig, ax = plt.subplots()
#     sns.heatmap(embed, ax=ax)
# elif embed.ndim == 3:
#     embed = embed[0]
#     fig, ax = plt.subplots()
#     sns.heatmap(embed, ax=ax)
# else:
#     embed = embed[0]
#     fig, ax = plt.subplots(2, 2)
#     sns.heatmap(embed[0], ax=ax[0, 0])
#     sns.heatmap(embed[1], ax=ax[0, 1])
#     sns.heatmap(embed[2], ax=ax[1, 0])
#     sns.heatmap(embed[3], ax=ax[1, 1])
# plt.show()

thres = 0.35
embed = torch.max(embed, dim=-1)[0]
embed = embed.reshape(samples.shape[0], 8, samples.shape[1])
embed = embed.transpose(0, 1)
embed = embed.reshape(8, samples.shape[0]*samples.shape[1])
embed = torch.concat([embed[..., :-1024], embed[..., -(len(midi_notes)%1024):]], dim=-1)
assert embed.shape[-1] == len(midi_notes)
embed = embed.detach().numpy()
embed = embed * (embed > thres*(np.max(embed)+np.min(embed)))

fig, ax = plt.subplots()
sns.heatmap(embed, ax=ax)

# fig, ax = plt.subplots(2, 4)
# for i in range(8):
#     sns.heatmap(embed[i], ax=ax[i // 4, i % 4])
#     # ax[i // 4, i % 4].set_aspect("equal")
#     ax[i // 4, i % 4].set_title(f"Head {i+1}")
# plt.show()

for head in range(8):
    midi = muspy.read_midi("midi/0130.mid")
    midi_notes = copy.deepcopy(midi.tracks[0].notes)
    for i in range(len(midi_notes)):
        if embed[head, i] == 0:
            midi.tracks[0].notes.remove(midi_notes[i])
    print(len(midi.tracks[0].notes))
    midi.write_midi(f"0130_head{head}.mid")
plt.show()