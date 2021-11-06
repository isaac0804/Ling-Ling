import copy

import muspy
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils import to_bits


class MidiDataset(Dataset):
    def __init__(self, is_train=True):
        return
        # self.filenames = []
        # self.dir = "midi/"
        # for _, _, filename in os.walk(self.dir):
        #   self.filenames = filename
        # if is_train:
        #   self.filenames = [file for file in self.filenames if not file.endswith("9.mid")]
        # else: 
        #   self.filenames = [file for file in self.filenames if file.endswith("9.mid")]
        # print(f"loading {len(self.filenames)} datas")

    def __len__(self):
        return 1
        # return len(self.filenames)

    def __getitem__(self, index):
        # filename = self.filenames[index]
        # midi_obj = muspy.read_midi(os.path.join(self.dir,filename))
        # notes = copy.deepcopy(midi_obj.tracks[0].notes)
        samples = []
        midi = muspy.read_midi("midi_0000.mid")
        notes =  midi.tracks[0].notes

        pitch_range = [note.pitch-21 for note in notes]
        max_pitch, min_pitch = max(pitch_range), min(pitch_range)
        indices = np.random.randint(0, len(notes)-2048, size=2)
        pitch_tranpose = np.random.randint(-min_pitch, 88-max_pitch, size=2) 
        tempo_change = np.random.random(size=2)*0.1 + 0.95

        # mask
        mask = np.zeros(2048, dtype=np.bool_)
        global_mask = np.zeros(64, dtype=np.bool_)
        if not np.abs(indices[1]-indices[0]) >= 2048:
            mask[:indices[1]-indices[0]] = True
            global_mask[:(indices[1]-indices[0]+16)//32] = True

        for s in range(2):
            sample = []
            for i in range(indices[s], indices[s]+2048):
                octave, pitch = (notes[i].pitch-21+pitch_tranpose[s])//12, (notes[i].pitch-21+pitch_tranpose[s])%12
                duration = notes[i].duration*tempo_change[s]
                velocity = notes[i].velocity
                if i != 0:
                    time_shift = tempo_change[s]*(notes[i].start- notes[i-1].start)//10
                else:
                    time_shift = 0

                ret = [0]*38
                ret[:3] = to_bits(octave, 3)
                ret[3:7] = to_bits(pitch, 4)
                ret[7:21] = to_bits(min(int(duration), 16383), 14)
                ret[21:34] = to_bits(min(int(time_shift), 8191), 13)
                ret[34:38] = to_bits(min(velocity//8, 15), 4)
                sample.append(ret)
            samples.append(sample)

        samples = torch.tensor(samples)
        samples = torch.reshape(samples, (2, 16, 128, 38))

        return samples.type(torch.FloatTensor), (mask, np.flip(mask).copy()), (global_mask, np.flip(global_mask).copy())
