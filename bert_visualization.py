import os
import random
import time

import matplotlib.pyplot as plt
import muspy
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from bert import PianoBERT
from config import BERTCONFIG
from preprocess import MidiDataset
from utils import emb_to_index

if __name__ == "__main__":
    vis_attn = True
    gen_midi = True
    config = BERTCONFIG()
    device = torch.device("cpu")
    model = PianoBERT(
        embedding_dim=config.EMBEDDING_DIM,
        seq_length=config.SEQ_LENGTH,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
    ).to(device)
    model.load_state_dict(
        torch.load("checkpoints/checkpoints_07_CE_loss/model_epoch-2000.pt")[
            "model_state_dict"
        ]
    )

    dataset = MidiDataset(seq_len=512, mask_prob=0.15)
    loader = DataLoader(dataset, shuffle=True, batch_size=1, drop_last=True)
    X, Y, mask = next(iter(loader))

    # Forward pass
    outputs = model(X)
    indices = []
    for output in outputs:
        index = torch.argmax(output, dim=-1)
        indices.append(index)
    indices = torch.stack(indices, dim=-1).squeeze().numpy()

    # Attention Visualization
    if vis_attn:
        target_layer = 12
        x = model.note_embed(X)
        for ii in range(target_layer):
            x, attn = model.layers[ii](x)
        attn = F.normalize(attn)
        # print(attn.shape)
        attn = attn.detach().numpy()[0]
        fig, ax = plt.subplots(2, 4)
        for ii in range(8):
            sns.heatmap(attn[ii], ax=ax[ii // 4, ii % 4], square=True, cmap="YlGnBu")
            ax[ii // 4, ii % 4].set_title(f"Head {ii+1}")
        plt.show()

    # Generate music
    if gen_midi:
        midi = muspy.Music(tempos=[muspy.Tempo(0, 60)], resolution=1000)
        midi.tracks.append(muspy.Track())
        current_time = 0
        for i, output in enumerate(indices):
            octave = output[0]
            pitch = output[1]
            s_duration = output[2] * 20
            m_duration = output[3] * 200
            l_duration = output[4] * 2000
            velocity = output[5] * 8 + 4
            s_shift = output[6] * 20
            l_shift = output[7] * 400

            note = muspy.Note(
                current_time,
                20 + octave * 12 + pitch,
                s_duration + m_duration + l_duration,
                velocity,
            )
            if i < 1000:
                print(note)
            midi.tracks[0].append(note)
            current_time += s_shift + l_shift
        midi.write_midi(f"Generated-{time.time():2.2f}.mid")
