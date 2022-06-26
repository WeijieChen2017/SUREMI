import torch
import torch.nn as nn

def weighted_L1Loss(y_true, y_pred, weights):
    diff = torch.abs(y_true - y_pred)
    loss = torch.mul(diff, weights)
    return torch.mean(loss)