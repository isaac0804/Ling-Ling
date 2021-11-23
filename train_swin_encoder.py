import torch
from torch import optim
import torch.nn as nn
import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader

from swin_encoder import SwinEncoder
from utils import cosine_scheduler, fix_random_seeds
from loss import GlobalLoss, gradient_clipping
from preprocess import MidiDataset

fix_random_seeds(seed=42)
dataset = MidiDataset(global_local=None, random_mask=0.4)
loader = DataLoader(dataset, shuffle=True, batch_size=None)

epochs = 100
optim_frequency = 5
learning_rate_scheduler = cosine_scheduler(
    base_value=1e-4,
    final_value=1e-5,
    epochs=epochs,
    niter_per_ep=len(loader),
    warmup_epochs=10,
    start_warmup_value=0,
)
teacher_momentum_scheduler = cosine_scheduler(
    base_value=0.99, final_value=1, epochs=epochs, niter_per_ep=len(loader)
)
device = torch.device("cuda")
# device = torch.device("cpu")

student = SwinEncoder(
    embed_dim=64,
    window_size=16,
    num_heads=8,
    num_classes=None,
    downsample_res=4,
    depth=[8, 4, 2, 2],
    seq_length=1024,
    mlp_drop=0.5,
    attn_drop=0.5,
    proj_drop=0.5,
    act_layer=nn.GELU,
    norm_layer=nn.LayerNorm,
)
teacher = SwinEncoder(
    embed_dim=64,
    window_size=16,
    num_heads=8,
    num_classes=None,
    downsample_res=4,
    depth=[8, 4, 2, 2],
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
optimizer = AdamW(student.parameters())
global_criterion = GlobalLoss(
    out_dim=512,
    warmup_teacher_temp=0.04, 
    teacher_temp=0.07,
    warmup_teacher_temp_epochs=10,
    n_epochs=epochs,
    student_temp=0.1,
    center_momentum=0.9,
).to(device)

teacher.load_state_dict(student.state_dict())
for p in teacher.parameters():
    p.requires_grad = False

torch.autograd.set_detect_anomaly(True)
best_loss = 1000000.0
for epoch in range(epochs):
    running_loss = 0.0
    n_term = 0.0
    for i, midi in tqdm.tqdm(
        enumerate(loader, 0), total=len(loader), ncols=100, desc="Progress"
    ):
        it = len(loader) * epoch + i
        for _, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = learning_rate_scheduler[it]
        midi = midi.to(device)

        # output of model: notes, patches, patch
        s_global = student(midi)
        t_global = teacher(midi[:2])

        # global Criterion Loss
        global_loss = global_criterion(s_global, t_global, epoch)
        loss = global_loss / optim_frequency
        n_term += 1

        # backward
        loss.backward()
        running_loss += loss.item()

        if it % optim_frequency == 0:
            # clip gradient
            gradient_clipping(student, 3.0)

            # optimizer
            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                m = teacher_momentum_scheduler[it]
                for student_p, teacher_p in zip(student.parameters(), teacher.parameters()):
                    teacher_p.data.mul_(m)
                    teacher_p.data.add_((1 - m) * student_p.detach().data)

    if best_loss > running_loss / n_term:
        best_loss = running_loss / n_term
        torch.save(
            teacher.state_dict(),
            f"checkpoints/model_epoch-{epoch+1}_loss-{running_loss*optim_frequency/n_term:4.2f}.pt",
        )
    elif (epoch+1) % 5 == 0:
        torch.save(
            teacher.state_dict(),
            f"checkpoints/model_epoch-{epoch+1}_loss-{running_loss*optim_frequency/n_term:4.2f}.pt",
        )
    print(f"Epoch               : {epoch+1}")
    print(f"Loss                : {running_loss*optim_frequency/n_term:4.9f}")
    print(f"Learning Rate       : {learning_rate_scheduler[it]:4.9f}")
    print(f"Teacher Momentum    : {teacher_momentum_scheduler[it]:4.9f}")
    print(f"Teacher Temperature : {global_criterion.teacher_temp_scheduler[epoch]:4.9f}")
