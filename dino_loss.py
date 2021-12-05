import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class GlobalLoss(nn.Module):
    def __init__(
        self,
        out_dim,
        warmup_teacher_temp,
        teacher_temp,
        warmup_teacher_temp_epochs,
        n_epochs,
        student_temp=0.1,
        center_momentum=0.8,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.teacher_temp_scheduler = np.concatenate(
            (
                np.linspace(
                    warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs
                ),
                np.ones(n_epochs - warmup_teacher_temp_epochs) * teacher_temp,
            )
        )

    def forward(self, student_output, teacher_output, epoch):
        student_temp = student_output / self.student_temp
        teacher_temp = (teacher_output - self.center) / self.teacher_temp_scheduler[epoch]

        student_sm = F.log_softmax(student_temp, dim=-1)
        teacher_sm = F.softmax(teacher_temp, dim=-1).detach()

        total_loss = torch.tensor(0.0).to(self.dummy_param.device)
        n_term = 0

        for t_ix, t in enumerate(teacher_sm):
            for s_ix, s in enumerate(student_sm):
                if t_ix == s_ix:
                    continue
                loss = torch.sum(-t * s, dim=-1)
                total_loss += loss.mean()
                n_term += 1

        total_loss /= n_term
        self.update_center(teacher_output)

        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        # batch_center = teacher_output.mean(dim=0, keepdim=True) this is use when teacher_output is 2d tensor
        batch_center = torch.unsqueeze(teacher_output, 0)
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum
        )


class PairLoss(nn.Module):
    def __init__(
        self, out_dim, student_temp=0.2, teacher_temp=0.04, center_momentum=0.8
    ):
        super().__init__()
        self.out_dim = out_dim
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, student_output, teacher_output):
        student_temp = student_output / self.student_temp
        teacher_temp = (teacher_output - self.center) / self.teacher_temp

        student_sm = F.log_softmax(student_temp, dim=-1)
        teacher_sm = F.softmax(teacher_temp, dim=-1).detach()

        total_loss = torch.tensor(0.0).to(self.dummy_param.device)
        n_term = 0

        for s, t in zip(student_sm, teacher_sm):
            loss = torch.sum(-t * s, dim=-1)
            total_loss += loss
            n_term += 1

        self.update_center(teacher_output)
        total_loss /= n_term + 1e-3
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum
        )


def gradient_clipping(model, clip=2.0):
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm()
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
