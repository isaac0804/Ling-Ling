import numpy as np
import torch
import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from loss import GlobalLoss, PairLoss
from model import Model
from preprocess import MidiDataset

epochs = 20
learning_rate = 0.01
teacher_momentum = 0.98
# device = torch.device('cuda')
device = torch.device('cpu')

dataset = MidiDataset()
loader = DataLoader(dataset, shuffle=True, batch_size=None)

student = Model()
teacher = Model()
student, teacher = student.to(device), teacher.to(device)
student.train()
teacher.train()
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
    for i,data in tqdm.tqdm(enumerate(loader, 0), total=len(loader), ncols=100, desc="Progress"):

        midi, local_mask, patch_mask = data
        midi = midi.to(device)

        optimizer.zero_grad()

        # output of model: notes, patches, patch
        s_local_0, s_patch_0, s_global_0 = student(midi[0])
        t_local_0, t_patch_0, t_global_0= teacher(midi[1])

        s_local_1, s_patch_1, s_global_1 = student(midi[1])
        t_local_1, t_patch_1, t_global_1 = teacher(midi[0])

        # global Criterion Loss
        global_loss = global_criterion(torch.unsqueeze(s_global_0,0), torch.unsqueeze(t_global_0,0)) + global_criterion(torch.unsqueeze(s_global_1,0), torch.unsqueeze(t_global_1,0))
        loss = global_loss
        n_term += 2

        # patch Criterion Loss
        patch_loss = torch.tensor(0.0)
        if torch.sum(patch_mask[0]) != 0:
            mask_0 = torch.stack([patch_mask[0]]*16).T.to(device)
            mask_1 = torch.stack([patch_mask[1]]*16).T.to(device)
            masked_s_patch_0 = torch.masked_select(s_patch_0, mask_0).view(-1, 16)
            masked_t_patch_0 = torch.masked_select(t_patch_0, mask_1).view(-1, 16)

            masked_s_patch_1 = torch.masked_select(s_patch_1, mask_1).view(-1, 16)
            masked_t_patch_1 = torch.masked_select(t_patch_1, mask_0).view(-1, 16)

            patch_loss = patch_criterion(masked_s_patch_0, masked_t_patch_0) + patch_criterion(masked_s_patch_1, masked_t_patch_1)
            loss += patch_loss
            n_term += 2

        # local Criterion Loss
        local_loss = torch.tensor(0.0)
        if torch.sum(local_mask[0]) != 0:
            mask_0 = torch.stack([torch.reshape(local_mask[0],(16,64))]*16, dim=-1).T.to(device)
            mask_1 = torch.stack([torch.reshape(local_mask[1],(16,64))]*16, dim=-1).T.to(device)
            masked_s_local_0 = torch.masked_select(s_local_0, mask_0).view(-1, 16)
            masked_t_local_0 = torch.masked_select(t_local_0, mask_1).view(-1, 16)

            masked_s_local_1 = torch.masked_select(s_local_1, mask_1).view(-1, 16)
            masked_t_local_1 = torch.masked_select(t_local_1, mask_0).view(-1, 16)

            local_loss = local_criterion(masked_s_local_0, masked_t_local_0) + local_criterion(masked_s_local_1, masked_t_local_1)
            loss += local_loss
            n_term += 2
        else: 
            continue

        # backward 
        loss.backward()

        # optimizer
        optimizer.step()

        running_loss += loss.item()

        with torch.no_grad():
            for student_p, teacher_p in zip(student.parameters(), teacher.parameters()):
                teacher_p.data.mul_(teacher_momentum)
                teacher_p.data.add_((1-teacher_momentum)*student_p.detach().data)
    if best_loss > running_loss/n_term:
        best_loss = running_loss/n_term
        torch.save(teacher.state_dict(), f"checkpoints/model_epoch-{epoch}_loss-{running_loss/n_term:4.2f}.pt")
    elif epoch % 5 == 0:
        torch.save(teacher.state_dict(), f"checkpoints/model_epoch-{epoch}_loss-{running_loss/n_term:4.2f}.pt")
    print(f"Epoch: {epoch},  Loss: {running_loss/n_term}")
