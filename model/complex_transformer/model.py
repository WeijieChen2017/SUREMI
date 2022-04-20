import os,sys,inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0,parentdir) 

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .modules.transformer import TransformerEncoder, TransformerDecoder
from .modules.complex_ops import *
# from models import *
# from utils import count_parameters

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TransformerModel(nn.Module):
    def __init__(self, time_step, input_dims, hidden_size, embed_dim, output_dim, num_heads, attn_dropout, relu_dropout, res_dropout, out_dropout, layers, attn_mask=False):
        """
        Construct a basic Transfomer model.
        
        :param input_dims: The input dimensions of the various modalities.
        :param hidden_size: The hidden dimensions of the fc layer.
        :param embed_dim: The dimensions of the embedding layer.
        :param output_dim: The dimensions of the output (128 in MuiscNet).
        :param num_heads: The number of heads to use in the multi-headed attention. 
        :param attn_dropout: The dropout following self-attention sm((QK)^T/d)V.
        :param relu_droput: The dropout for ReLU in residual block.
        :param res_dropout: The dropout of each residual block.
        :param out_dropout: The dropout of output layer.
        :param layers: The number of transformer blocks.
        :param attn_mask: A boolean indicating whether to use attention mask (for transformer decoder).
        """
        super(TransformerModel, self).__init__()
        self.conv = ComplexSequential(
            ComplexConv1d(in_channels=1, out_channels=16, kernel_size=6, stride=1),
            ComplexBatchNorm1d(16),
            ComplexReLU(),
            ComplexMaxPool1d(2, stride=2),

            ComplexConv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
            ComplexBatchNorm1d(32),
            ComplexReLU(),
            ComplexMaxPool1d(2, stride=2),

            ComplexConv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            ComplexBatchNorm1d(64),
            ComplexReLU(),
            ComplexMaxPool1d(2, stride=2),

            ComplexConv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            ComplexBatchNorm1d(64),
            ComplexReLU(),
            ComplexMaxPool1d(2, stride=2),

            ComplexConv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            ComplexBatchNorm1d(128),
            ComplexReLU(),
            ComplexMaxPool1d(2, stride=2),
            ComplexFlatten(),
            )
        [self.orig_d_a, self.orig_d_b] = input_dims
        assert self.orig_d_a == self.orig_d_b
        channels = ((((((((((self.orig_d_a -6)//1+1 -2)//2+1 -3)//1+1 -2)//2+1 
            -3)//1+1 -2)//2+1 -3)//1+1 -2)//2+1 -3)//1+1 -2)//2+1
        self.d_a, self.d_b = 128*channels, 128*channels
        final_out = embed_dim * 2
        h_out = hidden_size
        self.num_heads = num_heads
        self.layers = layers
        self.attn_dropout = attn_dropout
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.attn_mask = attn_mask
        self.embed_dim = embed_dim
        
        # Transformer networks
        self.trans = self.get_network()
        print("Encoder Model size: {0}".format(count_parameters(self.trans)))
        # Projection layers
        self.proj = ComplexLinear(self.d_a, self.embed_dim)
        
        self.out_fc1 = nn.Linear(final_out, h_out)
        
        self.out_fc2 = nn.Linear(h_out, output_dim)
        
        self.out_dropout = nn.Dropout(out_dropout)
    def get_network(self):
        
        return TransformerEncoder(embed_dim=self.embed_dim, num_heads=self.num_heads, layers=self.layers, attn_dropout=self.attn_dropout,
            relu_dropout=self.relu_dropout, res_dropout=self.res_dropout, attn_mask=self.attn_mask)
            
    def forward(self, x):
        """
        x should have dimension [seq_len, batch_size, n_features] (i.e., L, N, C).
        """
        time_step, batch_size, n_features = x.shape
        input_a = x[:, :, :n_features//2].view(-1, 1, n_features//2)
        input_b = x[:, :, n_features//2:].view(-1, 1, n_features//2)
        input_a, input_b = self.conv(input_a, input_b)
        input_a = input_a.reshape(time_step, batch_size, self.d_a)
        input_b = input_b.reshape(time_step, batch_size, self.d_b)
        input_a, input_b = self.proj(input_a, input_b)
        # Pass the input through individual transformers
        h_as, h_bs = self.trans(input_a, input_b)
        h_concat = torch.cat([h_as, h_bs], dim=-1)
        output = self.out_fc2(self.out_dropout(F.relu(self.out_fc1(h_concat))))
        # No sigmoid because we use BCEwithlogitis which contains sigmoid layer and more stable
        return output

class TransformerGenerationModel(nn.Module):
    def __init__(self, input_dims, hidden_size, embed_dim, output_dim, num_heads, attn_dropout, relu_dropout, res_dropout, out_dropout, layers, attn_mask=False, src_mask=False, tgt_mask=False):
        super(TransformerGenerationModel, self).__init__()
        # self.conv = ComplexSequential(
        #     ComplexConv1d(in_channels=1, out_channels=16, kernel_size=6, stride=1),
        #     ComplexBatchNorm1d(16),
        #     ComplexReLU(),
        #     ComplexMaxPool1d(2, stride=2),

        #     ComplexConv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
        #     ComplexBatchNorm1d(32),
        #     ComplexReLU(),
        #     ComplexMaxPool1d(2, stride=2),

        #     ComplexConv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
        #     ComplexBatchNorm1d(64),
        #     ComplexReLU(),
        #     ComplexMaxPool1d(2, stride=2),

        #     ComplexConv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
        #     ComplexBatchNorm1d(64),
        #     ComplexReLU(),
        #     ComplexMaxPool1d(2, stride=2),

        #     ComplexConv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
        #     ComplexBatchNorm1d(128),
        #     ComplexReLU(),
        #     ComplexMaxPool1d(2, stride=2),
        #     ComplexFlatten(),
        #     )
        [self.orig_d_a, self.orig_d_b] = input_dims
        assert self.orig_d_a == self.orig_d_b
        # channels = ((((((((((self.orig_d_a -6)//1+1 -2)//2+1 -3)//1+1 -2)//2+1 
        #     -3)//1+1 -2)//2+1 -3)//1+1 -2)//2+1 -3)//1+1 -2)//2+1
        # self.d_a, self.d_b = 128*channels, 128*channels
        self.d_a = self.orig_d_a
        self.d_b = self.orig_d_b
        final_out = embed_dim * 2
        h_out = hidden_size
        self.num_heads = num_heads
        self.layers = layers
        self.attn_dropout = attn_dropout
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.attn_mask = attn_mask
        self.embed_dim = embed_dim
        
        # Transformer networks
        self.trans_encoder = self.get_encoder_network()
        self.trans_decoder = self.get_decoder_network()

        print("Encoder Model size: {0}".format(count_parameters(self.trans_encoder)))
        print("Decoder Model size: {0}".format(count_parameters(self.trans_decoder)))
        
        # Projection layers
        self.proj_enc = ComplexLinear(self.d_a, self.embed_dim)
        self.proj_dec = ComplexLinear(self.orig_d_a, self.embed_dim)
        
        self.out_fc1 = nn.Linear(final_out, h_out)
        
        self.out_fc2 = nn.Linear(h_out, output_dim)
        
        self.out_dropout = nn.Dropout(out_dropout)

    def get_encoder_network(self):
        
        return TransformerEncoder(embed_dim=self.embed_dim, num_heads=self.num_heads, layers=self.layers, attn_dropout=self.attn_dropout,
            relu_dropout=self.relu_dropout, res_dropout=self.res_dropout, attn_mask=self.attn_mask)

    def get_decoder_network(self): 
        return TransformerDecoder(embed_dim=self.embed_dim, num_heads=self.num_heads, layers=self.layers, src_attn_dropout=self.attn_dropout, 
            relu_dropout=self.relu_dropout, res_dropout=self.res_dropout, tgt_attn_dropout=self.attn_dropout)
            
    def forward(self, x, y=None, max_len=None):
        """
        x should have dimension [seq_len, batch_size, n_features] (i.e., L, N, C).
        """
        time_step, batch_size, n_features = x.shape
        input_a = x[:, :, :n_features//2]
        input_b = x[:, :, n_features//2:]
        # input_a = x[:, :, :n_features//2].view(-1, 1, n_features//2)
        # input_b = x[:, :, n_features//2:].view(-1, 1, n_features//2)
        # print(input_a.size(), input_b.size()) # torch.Size([384, 1, 32768])
        # input_a, input_b = self.conv(input_a, input_b)
        # print(input_a.size(), input_b.size()) # torch.Size([384, 130688])
        # input_a = input_a.reshape(-1, batch_size, self.d_a)
        # input_b = input_b.reshape(-1, batch_size, self.d_b)
        # print("Line 200", input_a.size(), input_b.size())
        # print(input_a.size(), input_b.size()) # torch.Size([384, 1, 130688])
        input_a, input_b = self.proj_enc(input_a, input_b)
        # print(input_a.size(), input_b.size()) # torch.Size([384, 1, 240])
        # Pass the input through individual transformers
        h_as, h_bs = self.trans_encoder(input_a, input_b)
        # print("Line 205:", "h_as", h_as.size(), "h_bs", h_bs.size())

        if y is not None:
            seq_len, batch_size, n_features2 = y.shape 
            n_features = n_features2 // 2

            y_a = y[:, :, :self.orig_d_a]                               # truncate last target 
            y_b = y[:, :, self.orig_d_a: self.orig_d_a + self.orig_d_b] # truncate last target 

            # sos_a = torch.zeros(1, batch_size, n_features).to(x.device)
            # sos_b = torch.zeros(1, batch_size, n_features).to(x.device)
            # y_a = torch.cat([sos_a, y_a], dim=0)    # add <sos> to front 
            # y_b = torch.cat([sos_b, y_b], dim=0)    # add <sos> to front 

            y_a, y_b = self.proj_dec(y_a, y_b)
            out_as, out_bs = self.trans_decoder(input_A=y_a, input_B=y_b, enc_A=h_as, enc_B=h_bs)
            out_concat = torch.cat([out_as, out_bs], dim=-1)
            
            output = self.out_fc2(self.out_dropout(F.relu(self.out_fc1(out_concat))))

        elif max_len is not None:
            dec_a = torch.zeros(1, batch_size, n_features//2).to(x.device)
            dec_b = torch.zeros(1, batch_size, n_features//2).to(x.device)
            dec_a, dec_b = self.proj_dec(dec_a, dec_b)

            dec_a, dec_b = self.trans_decoder(input_A=dec_a, input_B=dec_b, enc_A=h_as, enc_B=h_bs) 
            y_a, y_b = dec_a, dec_b

            for i in range(max_len):
                # print(i)
                # print(y_a.size(), y_b.size())
                dec_a, dec_b = self.trans_decoder(input_A=y_a, input_B=y_b, enc_A=h_as, enc_B=h_bs)
                # print(y_a.size(), dec_a.unsqueeze(0).size(), y_b.size(), dec_b.unsqueeze(0).size())
                # y_a, y_b = torch.cat([y_a, dec_a.unsqueeze(0)], dim=0), torch.cat([y_b, dec_b.unsqueeze(0)], dim=0)
                y_a, y_b = torch.cat([y_a, dec_a], dim=0), torch.cat([y_b, dec_b], dim=0)
            out_concat = torch.cat([y_a, y_b], dim=-1)
            
            output = self.out_fc2(self.out_dropout(F.relu(self.out_fc1(out_concat))))

        else:
            print("Only one of y and max_len should be input.")
            assert False

        return output
