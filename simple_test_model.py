import torch
from model import suremi
import numpy as np

input = torch.from_numpy(np.zeros((7, 64, 256, 256))).float()
model = suremi().float()
output = model(input)