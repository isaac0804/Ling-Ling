import numpy as np
import torch
import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from loss import GlobalLoss, PairLoss
from model import Model
from preprocess import MidiDataset

epochs = 200
learning_rate = 0.005
# device = torch.device('cuda')
device = torch.device('cpu')

dataset = MidiDataset()
loader = DataLoader(dataset, shuffle=True, batch_size=None)

student = Model()
teacher = Model()
student, teacher = student.to(device), teacher.to(device)
optimizer = Adam(student.parameters(), lr=learning_rate)
global_criterion = GlobalLoss(out_dim=16).to(device)
patch_criterion = PairLoss(out_dim=16).to(device)
local_criterion = PairLoss(out_dim=16).to(device)

teacher.load_state_dict(student.state_dict())
for p in teacher.parameters():
    p.requires_grad = False

torch.autograd.set_detect_anomaly(True)
best_loss = 1000000.0
for epoch in range(1, epochs+1):
    running_loss = 0.0
    n_term = 0.0
    for i,data in tqdm.tqdm(enumerate(loader, 0), total=len(loader)):

        midi, local_mask, patch_mask = data
        midi = midi.to(device)

        optimizer.zero_grad()

        # output of model: notes, patches, patch
        s_local, s_patch, s_patch = student(midi[0])
        t_local, t_patch, t_patch = teacher(midi[1])

        # global Criterion Loss
        global_loss= global_criterion(torch.unsqueeze(s_patch,0), torch.unsqueeze(t_patch,0))
        loss = global_loss
        n_term += 1

        # patch Criterion Loss
        patch_loss = torch.tensor(0.0)
        if torch.sum(patch_mask[0]) != 0:
            masked_s_patch = torch.masked_select(s_patch, torch.stack([patch_mask[0]]*16).T.to(device))
            masked_s_patch = masked_s_patch.view(-1, 16)
            masked_t_patch = torch.masked_select(t_patch, torch.stack([patch_mask[1]]*16).T.to(device))
            masked_t_patch = masked_t_patch.view(-1, 16)
            patch_loss = patch_criterion(masked_s_patch, masked_t_patch)
            loss += patch_loss
            n_term += 1

        # local Criterion Loss
        local_loss = torch.tensor(0.0)
        if torch.sum(local_mask[0]) != 0:
            masked_s = torch.masked_select(s_local, torch.stack([torch.reshape(local_mask[0], (16, 64))]*16, dim=-1).to(device))
            masked_s = masked_s.view(-1, 16)
            masked_t = torch.masked_select(t_local, torch.stack([torch.reshape(local_mask[1], (16, 64))]*16, dim=-1).to(device))
            masked_t = masked_t.view(-1, 16)
            local_loss = local_criterion(masked_s, masked_t)
            loss += local_loss
            n_term += 1

        # backward 
        loss.backward()

        # optimizer
        optimizer.step()

        running_loss += loss.item()

        with torch.no_grad():
            for student_p, teacher_p in zip(student.parameters(), teacher.parameters()):
                teacher_p.data.mul_(0.995)
                teacher_p.data.add_((0.005)*student_p.detach().data)
    if best_loss < running_loss/n_term:
        best_loss = running_loss/n_term
        torch.save(teacher.state_dict(), f"checkpoints/model_epoch-{epoch}_loss-{running_loss/n_term:4.2f}.pt")
    print(f"Epoch: {epoch},  Loss: {running_loss/n_term}")
