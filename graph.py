import math
import os

import matplotlib.pyplot as plt
import muspy
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import dtype
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GNNExplainer

MAJOR_SIXTH = 9
MINOR_SIXTH = 8
PERFECT_FIFTH = 7
PERFECT_FOURTH = 5
MAJOR_THIRD = 4
MINOR_THIRD = 3

NOTE_SKIP = 4
MIN_TIME_INTERVAL = 0.02


"""Tonnetz structure

B  F#  C#  G#  D#
 D# A# F  C  G 
C  G  D  A  E 
 E  B  F#  C#  G#
"""

def get_edge_index():
    edge_index = []
    for pitch in range(12):
        edge_index.append([pitch, (pitch + MAJOR_SIXTH) % 12])
        edge_index.append([pitch, (pitch + MINOR_SIXTH) % 12])
        edge_index.append([pitch, (pitch + PERFECT_FIFTH) % 12])
        edge_index.append([pitch, (pitch + PERFECT_FOURTH) % 12])
        edge_index.append([pitch, (pitch + MAJOR_THIRD) % 12])
        edge_index.append([pitch, (pitch + MINOR_THIRD) % 12])
    return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

class PianoRollDataset(InMemoryDataset):
    def __init__(self):
        super().__init__()
        self.filenames = []
        self.dir = "midi/"
        for _, _, filename in os.walk(self.dir):
            self.filenames = filename
        self.edge_index = get_edge_index()

    def len(self):
        return len(self.filenames)

    def get(self, idx):

        filename = self.filenames[idx]
        midi = muspy.read_midi(os.path.join(self.dir, filename))
        midi.adjust_resolution(int(0.5/MIN_TIME_INTERVAL))
        # 88 piano key, but use 117-21=96 for better shape
        piano_roll = midi.to_pianoroll_representation()[:,21:117] / 128
        piano_roll = torch.tensor(piano_roll, dtype=torch.float).view(-1, 8, 12).transpose(-1, -2)

        data = Data(x=piano_roll, edge_index=self.edge_index, num_nodes=12)

        return data


# GCN Network
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(8, 16)
        self.conv2 = GCNConv(16, 16)
        self.conv3 = GCNConv(16, 8)
    
    def forward(self, x, edge_index, batch):
        print(x.shape)
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        return x

if __name__ == "__main__":
    # Data preprocess
    dataset = PianoRollDataset()
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(dataset.num_node_features)
    for ii, data in enumerate(loader):
        print(data)
        print(data.num_graphs)
        break
    
    # Forward
    model = GCN()
    output = model(data.x, data.edge_index, data.batch)
    print(output.shape)

    # explainer = GNNExplainer(model, epochs=100)
    # node_idx = 10
    # node_feat_mask, edge_mask = explainer.explain_node(node_idx, x, edge_index)
    # ax, G = explainer.visualize_subgraph(node_idx, edge_index, edge_mask, y=data.y)
    # plt.show()
