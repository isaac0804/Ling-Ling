import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn

from modules import LocalBlock, PatchBlock

patch_block = PatchBlock()
local_block  = LocalBlock()
x = torch.ones(16, 64, 16)
patch_output = patch_block(x)
print("==========Patch Output==========")
print(f"Size: {patch_output.shape}")

local_output = local_block(x, patch_output)
print("==========Local Output==========")
print(f"Size: {local_output.shape}")


local_output, attn, attn_w = local_block(x, patch_output, return_attention=True)
print(attn.shape)
print(attn_w.shape)