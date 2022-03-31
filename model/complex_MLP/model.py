from .complexFunctions import ComplexLinear

import torch
from torch import nn

class cMLP(nn.Module):
    def __init__(self, in_dim, mid_dim_1, mid_dim_2, out_dim):
        super(cMLP, self).__init__()
        self.hidden_1 = ComplexLinear(in_dim, mid_dim_1)
        self.hidden_2 = ComplexLinear(mid_dim_1, mid_dim_2)
        self.out = ComplexLinear(mid_dim_2, out_dim)

    def forward(self, x):
        x = self.hidden_1(x)
        x = self.hidden_2(x)
        x = self.out(x)
        return x

