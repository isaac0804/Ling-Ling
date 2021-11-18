import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from swin_encoder import SwinEncoder
from preprocess import MidiDataset

encoder = SwinEncoder(
    embed_dim=64,
    window_size=16,
    num_heads=8,
    num_classes=None,
    downsample_res=4,
    depth=[4, 4, 2, 2],
    seq_length=1024,
    mlp_drop=0.5,
    attn_drop=0.5,
    proj_drop=0.5,
    act_layer=nn.GELU,
    norm_layer=nn.LayerNorm,
)
midiDataset = MidiDataset()

data = iter(midiDataset)
midi = next(data)

output = encoder(midi)
print(output.shape)

fig, ax = plt.subplots()
sns.heatmap(output.detach().numpy(), ax=ax)
# fig, ax = plt.subplots(2)
# sns.heatmap(output.detach().numpy()[0], ax=ax[0])
# sns.heatmap(output.detach().numpy()[1], ax=ax[1])
plt.show()