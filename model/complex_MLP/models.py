import torch
from torch import nn


class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.w_real = nn.Parameter(torch.randn(in_features, out_features, dtype=torch.float))
        self.w_imag = nn.Parameter(torch.randn(in_features, out_features, dtype=torch.float))
        self.b_real = nn.Parameter(torch.randn(out_features, dtype=torch.float))
        self.b_imag = nn.Parameter(torch.randn(out_features, dtype=torch.float))
        self.weights_init()

    def weights_init(self):
        nn.init.normal_(self.w_real)
        nn.init.normal_(self.w_imag)
        nn.init.normal_(self.b_real)
        nn.init.normal_(self.b_imag)        

    def forward(self, x_real, x_imag):
        y_real = torch.matmul(x_real, self.w_real) - torch.matmul(x_imag, self.w_imag) + self.b_real.unsqueeze(0)
        y_imag = torch.matmul(x_real, self.w_imag) - torch.matmul(x_imag, self.w_real) + self.b_imag.unsqueeze(0)
        return y_real, y_imag



class cMLP(nn.Module):
    def __init__(self, dim, mid_dim):
        super(cMLP, self).__init__()
        self.dim = dim
        self.mid_dim = mid_dim
        self.in_fc = ComplexLinear(in_features=self.dim, out_features=self.mid_dim[0])
        self.hidden = nn.ModuleList([])
        self.hidden.extend([
            ComplexLinear(
                in_features=self.mid_dim[idx], 
                out_features=self.mid_dim[idx+1]
            )
            for idx in range(len(self.mid_dim)-1)
        ])
        self.out_fc = ComplexLinear(in_features=self.mid_dim[-1], out_features=self.dim)

    def forward(self, x):
        x_real = x[:, :self.dim]
        x_imag = x[:, self.dim:]
        x_real, x_imag = self.in_fc(x_real, x_imag)
        for idx in range(len(self.mid_dim)-1):
            x_real, x_imag = self.hidden[idx](x_real, x_imag)
        x_real, x_imag = self.out_fc(x_real, x_imag)

        return torch.cat([x_real, x_imag], dim=-1)

