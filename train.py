import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from loss import GlobalLoss, Loss
from model import Model
from preprocess import MidiDataset

epochs = 10
device = torch.device('cpu')

dataset = MidiDataset()
train_loader = DataLoader(dataset, shuffle=True, batch_size=None)

data = iter(train_loader)
midi, local_mask, patch_mask = next(data)
midi = torch.squeeze(midi)

student = Model()
teacher = Model()
student, teacher = student.to(device), teacher.to(device)

teacher.load_state_dict(student.state_dict())
for p in teacher.parameters():
    p.requires_grad = False

# output of model: notes, patches, patch
s_local, s_patch, s_patch = student(midi[0])
t_local, t_patch, t_patch = teacher(midi[0])

# global Criterion Loss
global_criterion = GlobalLoss(out_dim=16)
global_loss= global_criterion(torch.unsqueeze(s_patch,0), torch.unsqueeze(t_patch,0))
print(f"Global Loss: {global_loss.item()}")

# patch Criterion Loss
patch_criterion = Loss(out_dim=16)
masked_s_patch = torch.masked_select(s_patch, torch.stack([patch_mask[0]]*16).T)
masked_s_patch = masked_s_patch.view(-1, 16)
masked_t_patch = torch.masked_select(t_patch, torch.stack([patch_mask[1]]*16).T)
masked_t_patch = masked_t_patch.view(-1, 16)
patch_loss = patch_criterion(masked_s_patch, masked_t_patch)
print(f"Patch Loss: {patch_loss.item()}")

# local Criterion Loss
local_criterion = Loss(out_dim=16)
masked_s = torch.masked_select(s_local, torch.stack([torch.reshape(local_mask[0], (16, 128))]*16, dim=-1))
masked_s = masked_s.view(-1, 16)
masked_t = torch.masked_select(t_local, torch.stack([torch.reshape(local_mask[1], (16, 128))]*16, dim=-1))
masked_t = masked_t.view(-1, 16)
local_loss = local_criterion(masked_s, masked_t)
print(f"Local Loss: {local_loss.item()}")