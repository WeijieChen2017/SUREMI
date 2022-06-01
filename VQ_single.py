import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import glob

import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

class VQ(nn.Module):
    def __init__(self, 
                 in_channels=1, 
                 num_hiddens=256, 
                 cube_size=32, 
                 embedding_dim=64, 
                 num_embeddings=512,
                 commitment_cost=0.25):
        super(VQ, self).__init__()
        # N,C,D,H,W
#         self.conv_weight = nn.Parameter(torch.randn(num_hiddens, in_channels, cube_size, cube_size, cube_size))
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        
        self.conv1 = nn.Conv3d(in_channels, num_hiddens//2, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(num_hiddens//2, num_hiddens, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv3d(num_hiddens, num_hiddens, kernel_size=4, stride=2, padding=1)
        
        self.deconv1 = nn.ConvTranspose3d(num_hiddens, num_hiddens, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose3d(num_hiddens, num_hiddens//2, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose3d(num_hiddens//2, num_hiddens//2, kernel_size=4, stride=2, padding=1)
        
        self.outconv = nn.Conv3d(num_hiddens//2, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
#         x = nn.functional.conv3d(x, self.conv_weight, bias=None, stride=32, padding=0, dilation=1, groups=1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        input_shape = x.shape

        # Flatten input
        flat_input = x.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), x)
        q_latent_loss = F.mse_loss(quantized, x.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        # convert quantized from BHWC -> BCHW
        quantized = x + (quantized - x).detach()
        quantized = quantized.permute(0, 4, 1, 2, 3).contiguous()
        print(input_shape, quantized.size())
        
        recon = self.deconv1(quantized)
        recon = F.relu(recon)
        recon = self.deconv2(quantized)
        recon = F.relu(recon)
        recon = self.deconv3(quantized)
        recon = self.outconv(recon)
        
        return eq_loss, recon

gpu_list = ','.join(str(x) for x in [7])
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

file_CT = nib.load("./data_dir/Iman_CT/norm/02472.nii.gz")
file_MR = nib.load("./data_dir/Iman_MR/norm/02472.nii.gz")
data_CT = file_CT.get_fdata()
data_MR = file_MR.get_fdata()
print(data_CT.shape, data_MR.shape)

model = VQ().to(device)
x = torch.from_numpy(np.expand_dims(data_CT, (0, 1))).float().to(device)
ans = model(x)[1].detach().numpy()
print(ans.shape)