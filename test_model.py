import torch
from torch import nn
from model import Model

model = Model()

notes = torch.rand((16, 128, 16))
output = model(notes)
print("========= Output =========")
print(f"Size: {output.shape}")