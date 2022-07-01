import torch
import torch.nn.functional as F

# from .. import _reduction as _Reduction

# class _Loss(Module):
#     reduction: str

#     def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
#         super(_Loss, self).__init__()
#         if size_average is not None or reduce is not None:
#             self.reduction: str = _Reduction.legacy_get_string(size_average, reduce)
#         else:
#             self.reduction = reduction

# class weighted_L1Loss(_Loss):

#     __constants__ = ['reduction']

#     def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
#         super(weighted_L1Loss, self).__init__(size_average, reduce, reduction)

#     def forward(self, y_pred: Tensor, y_true: Tensor, weight: Tensor) -> Tensor:
#         diff = torch.abs(y_true - y_pred)
#         loss = torch.mul(diff, weight)
#         return torch.mean(loss)

def weighted_L1Loss(y_true, y_pred, weight):
    diff = torch.abs(y_pred - y_true)
    loss = torch.mul(diff, F.sigmoid(weight))
    return torch.mean(loss)