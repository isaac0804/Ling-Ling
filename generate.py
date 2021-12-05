import matplotlib.pyplot as plt
import muspy
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import emb_to_index, fix_random_seeds
from vq_vae import Generator

seed = 128
fix_random_seeds(seed)
    
model = Generator(
    embed_dim=16,
    window_size=16,
    num_heads=4,
    downsample_res=4,
    depth=[4, 4, 2, 2],
    codebook_size=[64, 32, 16],
    seq_length=1024,
    attn_drop=0.5,
    proj_drop=0.5,
    mlp_drop=0.5,
    commitment_loss=0.25,
    act_layer=nn.GELU,
    norm_layer=nn.LayerNorm,
)
checkpoints = torch.load("checkpoints/model_epoch-40.pt")
model.load_state_dict(checkpoints["gen_state_dict"])
model.eval()

high_cb = model.high_vq.embedding.weight
mid_cb = model.mid_vq.embedding.weight
low_cb = model.low_vq.embedding.weight
decoder = model.decoder

high_emb = high_cb[torch.randint(0, 16, size=(1, 16))]
mid_emb = mid_cb[torch.randint(0, 32, size=(1, 64))]
low_emb = low_cb[torch.randint(0, 64, size=(1, 256))]

outputs = decoder(high_emb, mid_emb, low_emb)
# outputs = F.normalize(outputs, dim=-1)
outputs = emb_to_index(outputs, model)
outputs = outputs[0]

# midi = muspy.Music(tempos=[muspy.Tempo(0, 60)], resolution=1000)
# midi.tracks.append(muspy.Track())
# current_time = 0
# for i, output in enumerate(outputs.detach().numpy()):
#     octave = output[0]
#     pitch = output[1]
#     s_duration = output[2] * 20
#     m_duration = output[3] * 200
#     l_duration = output[4] * 2000
#     velocity = output[5] * 8 + 4
#     s_shift = output[6] * 20
#     l_shift = output[7] * 400

#     note = muspy.Note(
#         current_time,
#         20 + octave * 12 + pitch,
#         s_duration + m_duration + l_duration,
#         velocity,
#     )
#     if i < 1000:
#         print(note)
#     midi.tracks[0].append(note)
#     current_time += s_shift + l_shift
# midi.write_midi(f"Generated-{seed}.mid")

fig, ax = plt.subplots()
sns.heatmap(outputs.detach().numpy(), ax=ax)
plt.show()