import torch

from dino_loss import GlobalLoss, PairLoss

global_loss = GlobalLoss(16)
pair_loss = PairLoss(16)

a = torch.ones(16)
b = torch.randn(16)

print(a)
print(b)

print("global")
print(global_loss(a, a))
print(global_loss(a, b))
print(global_loss(b, a))
print(global_loss(b, b))
print("pair")
print(global_loss(torch.unsqueeze(a,0), torch.unsqueeze(a, 0)))
print(global_loss(torch.unsqueeze(a,0), torch.unsqueeze(b, 0)))