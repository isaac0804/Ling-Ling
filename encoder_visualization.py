import math
import muspy
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt

from encoder import Encoder
from preprocess import MidiDataset

show_attention = True

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

model.load_state_dict(torch.load("experiments/experiment4/model_epoch-296_loss-4.26.pt"))
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

notes, patches, embed = model(samples)
print(f"Shape of notes   : {notes.shape}")
print(f"Shape of patches : {patches.shape}")
print(f"Shape of embed   : {embed.shape}")
# fig, ax = plt.subplots(2, 2)
# patches = patches.detach().numpy()
# sns.heatmap(patches[0], ax=ax[0, 0])
# sns.heatmap(patches[1], ax=ax[0, 1])
# sns.heatmap(patches[2], ax=ax[1, 0])
# sns.heatmap(patches[3], ax=ax[1, 1])
# plt.show()
# fig, ax = plt.subplots()
# embed = embed.detach().numpy()
# sns.heatmap(embed, ax=ax)
# plt.show()

attention = model.return_attention(samples)
print(attention.shape)
fig, ax = plt.subplots(2, 4)
for i in range(8):
    sns.heatmap(attention[i], ax=ax[i // 4, i % 4])
    ax[i // 4, i % 4].set_aspect("equal")
    ax[i // 4, i % 4].set_title(f"Head {i+1}")
plt.show()