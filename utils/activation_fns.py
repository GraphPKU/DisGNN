
import torch.nn as nn
import torch
from functools import partial
from torch.nn import functional as F


class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift
    
    
class ScaledSiLU(torch.nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.scale_factor = 1 / 0.6
        self._activation = torch.nn.SiLU(inplace)

    def forward(self, x):
        return self._activation(x) * self.scale_factor

activation_fn_map = {
    "ssp": ShiftedSoftplus(),
    "silu": nn.SiLU(False),
    "relu": partial(nn.ReLU, inplace=False),
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "selu": partial(nn.SELU, inplace=False),
    "identity": nn.Identity,
    "ssilu": ScaledSiLU(False)
}



