import os
import time

import muspy
import numpy as np
import random
from numpy.core.fromnumeric import shape
import torch
from torch._C import dtype
from torch.utils.data import Dataset, DataLoader


# class MidiDataset(Dataset):
#     def __init__(self, global_local=None, random_mask=None):
#         self.filenames = []
#         self.dir = "midi/"
#         for _, _, filename in os.walk(self.dir):
#             self.filenames = filename
#         self.global_local = global_local
#         self.random_mask = random_mask

#     def __len__(self):
#         return len(self.filenames)

#     def __getitem__(self, index):
#         filename = self.filenames[index]
#         midi = muspy.read_midi(os.path.join(self.dir, filename))

#         time_scale = 1000 * 0.5/midi.resolution
#         notes = midi.tracks[0].notes

#         pitch_range = [note.pitch - 21 for note in notes]
#         max_pitch, min_pitch = max(pitch_range), min(pitch_range)
#         indices = np.random.randint(0, len(notes) - 1024 + 1, size=1)
#         pitch_tranpose = np.random.randint(-min_pitch, 88 - max_pitch, size=1)
#         tempo_change = np.random.random(size=1) * 0.2 + 0.90
#         velocity_change = np.random.random(size=1) * 0.2 + 0.90

#         sample = []
#         for i in range(indices[0], indices[0] + 1024):
#             octave, pitch = (notes[i].pitch - 21 + pitch_tranpose[0]) // 12, (
#                 notes[i].pitch - 21 + pitch_tranpose[0]
#             ) % 12
#             duration = notes[i].duration * tempo_change[0] * time_scale
#             velocity = notes[i].velocity * velocity_change[0]
#             if i != 0:
#                 time_shift = (notes[i].start - notes[i - 1].start) * tempo_change[0] * time_scale
#             else:
#                 time_shift = 0

#             ret = [0] * 8
#             ret[0] = octave
#             ret[1] = pitch
#             ret[2] = min(duration // 20, 9)
#             ret[3] = min(duration // 200, 9)
#             ret[4] = min(duration // 2000, 9)
#             ret[5] = min(velocity // 8, 15)
#             ret[6] = min(time_shift // 20, 19)
#             ret[7] = min(time_shift // 400, 9)
#             sample.append(ret)
#         sample = np.array(sample)
#         sample = torch.LongTensor(sample)
#         return sample


class MidiDataset(Dataset):
    def __init__(self, seq_len=1024, mask_prob=0.15):
        self.filenames = []
        self.dir = "midi/"
        self.seq_len = seq_len
        for _, _, filename in os.walk(self.dir):
            self.filenames = filename
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """
        Return
        ------
        X : torch.tensor
            of shape (seq_len, 8) masked with 'mask'
        
        Y : torch.tensor
            of shape (seq_len, 8) unmasked 

        mask : torch.tensor
            of shape (seq_len) bool tensor
        """
        filename = self.filenames[index]
        midi = muspy.read_midi(os.path.join(self.dir, filename))

        time_scale = 1000 * 0.5/midi.resolution
        notes = midi.tracks[0].notes

        SOS = [8, 12, 10, 10, 10, 16, 20, 10]
        EOS = [9, 13, 11, 11, 11, 17, 21, 11]
        MASK = [10, 14, 12, 12, 12, 18, 22, 12]

        pitch_range = [note.pitch - 21 for note in notes]
        max_pitch, min_pitch = max(pitch_range), min(pitch_range)
        indices = np.random.randint(0, len(notes) - self.seq_len + 1, size=1)
        pitch_tranpose = np.random.randint(-min_pitch, 88 - max_pitch, size=1)
        tempo_change = np.random.random(size=1) * 0.2 + 0.90
        velocity_change = np.random.random(size=1) * 0.2 + 0.90
        mask = np.bool8(np.zeros(shape=(self.seq_len)))

        X, Y = [], []
        for i in range(indices[0], indices[0] + self.seq_len):
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
            ret[2] = int(min(duration // 20, 9))
            ret[3] = int(min(duration // 200, 9))
            ret[4] = int(min(duration // 2000, 9))
            ret[5] = int(min(velocity // 8, 15))
            ret[6] = int(min(time_shift // 20, 19))
            ret[7] = int(min(time_shift // 400, 9))

            
            if random.random() < self.mask_prob:
                if random.random() < 0.8:
                    mask[len(X)] = True
                    X.append(MASK)
                elif 0.8 < random.random() < 0.9:
                    mask[len(X)] = True
                    if len(X) != 0:
                        X.append(random.sample(X, 1)[0])
                    else:
                        X.append(ret)
                else:
                    mask[len(X)] = True
                    X.append(ret)
            else:
                X.append(ret)
            Y.append(ret)

        if indices[0] == 0:
            X.pop()
            X.insert(0, SOS)
            Y.pop()
            Y.insert(0, SOS)
        elif indices[0] == len(notes) - self.seq_len:
            X.pop(0)
            X.append(EOS)
            Y.pop(0)
            Y.append(EOS)
        assert len(X) == self.seq_len
        assert len(Y) == self.seq_len
        X = np.array(X, dtype=np.int16)
        X = torch.LongTensor(X)
        Y = np.array(Y, dtype=np.int16)
        Y = torch.LongTensor(Y)
        # MASK = torch.tensor(MASK)
        # MASK = torch.stack([MASK]*self.seq_len, dim=0)
        # mask = torch.rand(self.seq_len) < 0.15
        # stacked_mask = torch.stack([mask]*8, dim=-1)
        # masked_sample = torch.where(stacked_mask, MASK, sample)
        return X, Y, mask

if __name__ == "__main__":
    dataset = MidiDataset()
    loader = DataLoader(dataset, batch_size=1)
    start = time.perf_counter()
    for ii, (X, Y, mask) in enumerate(loader):
        # print(X.shape)
        # print(X[:10])
        # print(Y[:10])
        # print(Y.shape)
        # print(mask.shape)
        # print(X.masked_select(mask).shape)
        # print(mask)
        break
    end = time.perf_counter()
    print(end - start)