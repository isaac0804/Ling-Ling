import torch
import torch.nn as nn
import tqdm
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader

from encoder import Encoder
from loss import GlobalLoss, gradient_clipping
from preprocess import MidiDataset

epochs = 300
learning_rate = 5e-5
teacher_momentum = 0.99
device = torch.device('cuda')
# device = torch.device("cpu")

dataset = MidiDataset()
loader = DataLoader(dataset, shuffle=True, batch_size=None)

student = Encoder(
    embed_dim=64,
    num_heads=8,
    patch_size=64,
    depth=16,
    seq_length=1024,
    mlp_drop=0.5,
    attn_drop=0.5,
    proj_drop=0.5,
    act_layer=nn.GELU,
    norm_layer=nn.LayerNorm,
)
teacher = Encoder(
    embed_dim=64,
    num_heads=8,
    patch_size=64,
    depth=16,
    seq_length=1024,
    mlp_drop=0.5,
    attn_drop=0.5,
    proj_drop=0.5,
    act_layer=nn.GELU,
    norm_layer=nn.LayerNorm,
)
student, teacher = student.to(device), teacher.to(device)
student.train()
teacher.train()
optimizer = Adam(student.parameters(), lr=learning_rate)
global_criterion = GlobalLoss(out_dim=256).to(device)
# patch_criterion = PairLoss(out_dim=32).to(device)
# local_criterion = PairLoss(out_dim=32).to(device)

teacher.load_state_dict(student.state_dict())
for p in teacher.parameters():
    p.requires_grad = False

torch.autograd.set_detect_anomaly(True)
best_loss = 1000000.0
for epoch in range(1, epochs + 1):
    running_loss = 0.0
    n_term = 0.0
    for i, midi in tqdm.tqdm(
        enumerate(loader, 0), total=len(loader), ncols=100, desc="Progress"
    ):
        midi = midi.to(device)

        optimizer.zero_grad()

        # output of model: notes, patches, patch
        s_local, s_patch, s_global = student(midi)
        t_local, t_patch, t_global = teacher(midi)

        # global Criterion Loss
        global_loss = global_criterion(s_global, t_global)
        loss = global_loss
        n_term += 1

        # patch Criterion Loss
        # patch_loss = torch.tensor(0.0)
        # if torch.sum(patch_mask[0]) != 0:
        #     print(patch_mask[0].shape)
        #     print(patch_mask[1].shape)
        #     mask_0 = torch.stack([patch_mask[0]]*16).T.to(device)
        #     mask_1 = torch.stack([patch_mask[1]]*16).T.to(device)
        #     masked_s_patch_0 = torch.masked_select(s_patch[0], mask_0).view(-1, 16)
        #     masked_t_patch_0 = torch.masked_select(t_patch[1], mask_1).view(-1, 16)

        #     masked_s_patch_1 = torch.masked_select(s_patch[1], mask_1).view(-1, 16)
        #     masked_t_patch_1 = torch.masked_select(t_patch[0], mask_0).view(-1, 16)

        #     patch_loss = patch_criterion(masked_s_patch_0, masked_t_patch_0) + patch_criterion(masked_s_patch_1, masked_t_patch_1)
        #     loss += patch_loss
        #     n_term += 2

        # # local Criterion Loss
        # local_loss = torch.tensor(0.0)
        # if torch.sum(local_mask[0]) != 0:
        #     mask_0 = torch.stack([torch.reshape(local_mask[0],(16,64))]*16, dim=-1).T.to(device)
        #     mask_1 = torch.stack([torch.reshape(local_mask[1],(16,64))]*16, dim=-1).T.to(device)
        #     masked_s_local_0 = torch.masked_select(s_local_0, mask_0).view(-1, 16)
        #     masked_t_local_0 = torch.masked_select(t_local_0, mask_1).view(-1, 16)

        #     masked_s_local_1 = torch.masked_select(s_local_1, mask_1).view(-1, 16)
        #     masked_t_local_1 = torch.masked_select(t_local_1, mask_0).view(-1, 16)

        #     local_loss = local_criterion(masked_s_local_0, masked_t_local_0) + local_criterion(masked_s_local_1, masked_t_local_1)
        #     loss += local_loss
        #     n_term += 2

        # backward
        loss.backward()

        # clip gradient
        gradient_clipping(student, 2.0)

        # optimizer
        optimizer.step()

        running_loss += loss.item()

        with torch.no_grad():
            for student_p, teacher_p in zip(student.parameters(), teacher.parameters()):
                teacher_p.data.mul_(teacher_momentum)
                teacher_p.data.add_((1 - teacher_momentum) * student_p.detach().data)
    if best_loss > running_loss / n_term:
        best_loss = running_loss / n_term
        torch.save(
            teacher.state_dict(),
            f"checkpoints/model_epoch-{epoch}_loss-{running_loss/n_term:4.2f}.pt",
        )
    elif epoch % 5 == 0:
        torch.save(
            teacher.state_dict(),
            f"checkpoints/model_epoch-{epoch}_loss-{running_loss/n_term:4.2f}.pt",
        )
    print(f"Epoch: {epoch},  Loss: {running_loss/n_term}")
