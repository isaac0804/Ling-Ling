from platform import java_ver
from global_local import GlobalBlock, LocalBlock
import torch
from torch import nn

global_block = GlobalBlock()
local_block  = LocalBlock()
x = torch.ones(16, 128, 16)
global_output = global_block(x)
print("==========Global Output==========")
print(f"Size: {global_output.shape}")
# print(global_output)

local_output = local_block(x, global_output)
print("==========Local Output==========")
print(f"Size: {local_output.shape}")
# print(local_output)