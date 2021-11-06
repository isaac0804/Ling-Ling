import torch
from torch import nn
import torch.nn.functional as F

class GlobalLoss(nn.Module):
    def __init__(self, out_dim, student_temp=0.1, teacher_temp=0.04, center_momentum=0.9):
        super().__init__()
        self.out_dim = out_dim
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer('center', torch.zeros(1, out_dim))
    
    def forward(self, student_output, teacher_output):
        student_temp = student_output / self.student_temp
        teacher_temp = (teacher_output-self.center) / self.teacher_temp

        student_sm = F.log_softmax(student_temp, dim=-1) 
        teacher_sm = F.softmax(teacher_temp, dim=-1).detach() 
         
        loss = torch.sum(-teacher_sm*student_sm, dim=-1)
        self.update_center(teacher_output)
        return loss 
    
    @torch.no_grad()
    def update_center(self, teacher_output):
        # batch_center = teacher_output.mean(dim=0, keepdim=True) this is use when teacher_output is 2d tensor
        batch_center = torch.unsqueeze(teacher_output, 0)
        self.center = self.center * self.center_momentum + batch_center * (1-self.center_momentum)


class Loss(nn.Module):
    def __init__(self, out_dim, student_temp=0.1, teacher_temp=0.04, center_momentum=0.9):
        super().__init__()
        self.out_dim = out_dim
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer('center', torch.zeros(1, out_dim))
    
    def forward(self, student_output, teacher_output):
        student_temp = student_output / self.student_temp
        teacher_temp = (teacher_output-self.center) / self.teacher_temp

        student_sm = F.log_softmax(student_temp, dim=-1) 
        teacher_sm = F.softmax(teacher_temp, dim=-1).detach() 

        total_loss = 0.0
        n_term = 0
         
        for s in student_sm:
            for t in teacher_sm:
                loss = torch.sum(-t*s, dim=-1) 
                total_loss += loss
                n_term += 1

        self.update_center(teacher_output)
        if n_term == 0:
            return torch.tensor(total_loss)
        total_loss /= n_term
        return total_loss
    
    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = teacher_output.mean(dim=0, keepdim=True) 
        self.center = self.center * self.center_momentum + batch_center * (1-self.center_momentum)