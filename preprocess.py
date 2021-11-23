import copy
import os

import time
import muspy
import numpy as np
from numpy import random
import torch
from torch.utils.data import DataLoader, Dataset
import time


class MidiDataset(Dataset):
    def __init__(self, global_local=None, random_mask=None):
        self.filenames = []
        self.dir = "midi/"
        for _, _, filename in os.walk(self.dir):
            self.filenames = filename
        self.global_local = global_local
        self.random_mask = random_mask

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        midi = muspy.read_midi(os.path.join(self.dir, filename))

        time_scale = 0.5/midi.resolution
        notes = midi.tracks[0].notes

        pitch_range = [note.pitch - 21 for note in notes]
        max_pitch, min_pitch = max(pitch_range), min(pitch_range)
        indices = np.random.randint(0, len(notes) - 1024 + 1, size=1)
        pitch_tranpose = np.random.randint(-min_pitch, 88 - max_pitch, size=1)
        tempo_change = np.random.random(size=1) * 0.2 + 0.90
        velocity_change = np.random.random(size=1) * 0.2 + 0.90

        sample = []
        for i in range(indices[0], indices[0] + 1024):
            octave, pitch = (notes[i].pitch - 21 + pitch_tranpose[0]) // 12, (
                notes[i].pitch - 21 + pitch_tranpose[0]
            ) % 12
            duration = notes[i].duration * tempo_change[0] * time_scale
            velocity = notes[i].velocity * velocity_change[0]
            if i != 0:
                time_shift = (notes[i].start - notes[i - 1].start) * tempo_change[0] * time_scale
            else:
                time_shift = 0

            ret = [0] * 8
            ret[0] = octave
            ret[1] = pitch
            ret[2] = min(duration // 20, 9)
            ret[3] = min(duration // 200, 9)
            ret[4] = min(duration // 2000, 9)
            ret[5] = min(velocity // 8, 15)
            ret[6] = min(time_shift // 20, 19)
            ret[7] = min(time_shift // 400, 9)
            sample.append(ret)
        sample = np.array(sample)
        sample = torch.LongTensor(sample)
        return sample

# dataset = MidiDataset(global_local=None, random_mask=0.2)
# a = iter(dataset)
# data = next(a)