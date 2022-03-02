import tqdm
import time
import numpy as np
import torch
a=torch.randn(5,4*3*3,512,512)
unfold=torch.nn.Unfold(3,1,0,3)

print(unfold(a).shape)


