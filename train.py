import torch
from torch.utils.data import DataLoader, Dataset

from global_local import GlobalBlock
from model import Model
from preprocess import MidiDataset

epochs = 10

dataset = MidiDataset()
train_loader = DataLoader(dataset, shuffle=True, batch_size=None)

data = iter(train_loader)
midi = next(data)
midi = torch.squeeze(midi)

model = Model()
output, output_global, output_general = model(midi[0])
optimizer = torch.optim.Adam(model.parameters())

