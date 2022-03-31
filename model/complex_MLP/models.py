import torch
from torch import nn


class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.w_real = nn.Parameter(torch.randn(in_features, out_features, dtype=torch.float))
        self.w_imag = nn.Parameter(torch.randn(in_features, out_features, dtype=torch.float))
        self.b_real = nn.Parameter(torch.randn(in_features, out_features, dtype=torch.float))
        self.b_imag = nn.Parameter(torch.randn(in_features, out_features, dtype=torch.float))

    def forward(self, x_real, x_imag):
        y_real = torch.matmul(x_real, self.w_real) - torch.matmul(x_imag, self.w_imag) + self.b_real
        y_imag = torch.matmul(x_real, self.w_imag) - torch.matmul(x_imag, self.w_real) + self.b_imag
        return y_real, y_imag



class cMLP(nn.Module):
    def __init__(self, dim, mid_dim_1, mid_dim_2):
        super(cMLP, self).__init__()
        self.dim = dim
        self.hidden_1 = ComplexLinear(dim, mid_dim_1)
        self.hidden_2 = ComplexLinear(mid_dim_1, mid_dim_2)
        self.out = ComplexLinear(mid_dim_2, dim)

    def forward(self, x):
        x_real = x[:, :self.dim]
        x_imag = x[:, self.dim:]
        print(x_real.size(), x_imag.size())
        x_real, x_imag = self.hidden_1(x_real, x_imag)
        print(x_real.size(), x_imag.size())
        x_real, x_imag = self.hidden_2(x_real, x_imag)
        print(x_real.size(), x_imag.size())
        x_real, x_imag = self.out(x_real, x_imag)
        print(x_real.size(), x_imag.size())

        return torch.cat([x_real, x_imag], dim=0)

