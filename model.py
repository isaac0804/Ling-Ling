import torch
import torch.nn.functional as F
from torch import nn

from global_local import GlobalBlock, LocalBlock


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_layers = 4
        self.num_patches = 16
        self.patch = 128
        self.linear = nn.Linear(38, 16)
        self.global_layers = [
            GlobalBlock(),
            GlobalBlock(),
            GlobalBlock(),
            GlobalBlock()
        ]
        self.local_layers = [
            LocalBlock(),
            LocalBlock(),
            LocalBlock(),
            LocalBlock()
        ]
        self.general_MLP = [
            nn.Linear(64, 16),
            nn.Linear(16, 8),
            nn.Linear(8, 1)
        ]
    
    def forward(self, x):
        x = self.linear(x)
        x_global = torch.zeros((self.num_layers, self.num_patches, 16))

        for i in range(self.num_layers):
            x_global[i] = self.global_layers[i](x)
            x = self.local_layers[i](x, x_global[i])
            if i != self.num_layers:
                x = self.shift(x, self.num_patches//self.num_layers)
            else:
                x = self.shift(x, self.num_patches*(-self.num_layers+1))

        x_global = torch.permute(x_global, (1, 0, 2))
        x_global = torch.reshape(x_global, (64, 16))

        x_general = torch.clone(x_global.T)
        for layer in self.general_MLP:
            x_general = F.relu(layer(x_general))
        x_general = torch.squeeze(x_general.T)
            
        return x, x_global, x_general
    
    def shift(self, x, value):
        sections, p, emb_dim = x.shape
        x = x.view(-1, emb_dim)
        x = torch.concat([x[value:], x[:value]], 0)
        x = torch.reshape(x, (sections, p, emb_dim))
        return x

class Loss(nn.Module):
    def __init__(self, out_dim, student_temp=0.1, teacher_temp=0.04, center_momentum=0.9):
        super().__init__()
        self.out_dim = out_dim
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer('center', torch.zeros(1, out_dim))
    
    def forward(self, student_output, teacher_output):
        student_temp = [s / self.student_temp for s in student_output]
        teacher_temp = [(t-self.center) / self.teacher_temp for t in teacher_output]

        student_sm = [F.log_softmax(s, dim=-1) for s in student_temp]
        teacher_sm = [F.log_softmax(t, dim=-1).detach() for t in teacher_temp]
         
        loss = torch.sum(-teacher_sm*student_sm, dim=-1)

        return loss 
    
    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = teacher_output.mean(dim=0, keepdim=True)

        self.center = self.center * self.center_momentum + batch_center * (1-self.center_momentum)