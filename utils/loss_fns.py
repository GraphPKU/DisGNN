import torch.nn as nn
import torch

def rmse(targets: torch.Tensor, pred: torch.Tensor):
    return torch.mean(torch.norm((pred - targets), p=2, dim=2), dim=(0,1))

loss_fn_map = {
    "l1": nn.functional.l1_loss,
    "rmse": rmse,
    "l2": nn.functional.mse_loss,
}