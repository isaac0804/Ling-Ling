import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


torch.random.manual_seed(42)
a = torch.rand(50)

def sharp(a):
    value_s = 0.4
    value_t = 0.04
    s = -F.log_softmax(a/value_s, dim=-1)
    t = F.softmax((a-torch.mean(a))/value_t, dim=-1)
    print(torch.sum(t*s))
    return s, t

s, t = sharp(a)

# print(a, s, t)
plt.plot(a)
plt.plot(s)
plt.plot(t)
plt.legend(["a", "s", "t"])
plt.show()