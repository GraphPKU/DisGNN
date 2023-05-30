import torch
import torch.nn as nn
from layers.basic_layers import Residual
from functools import partial
from torch import Tensor

class ThreeOrderOutputBlock(nn.Module):
    def __init__(self, hidden_dim, activation_fn, pooling_level="low", **kwargs) -> None:
        super().__init__()
        pattern_num = 2 if pooling_level == "middle" else 5 if pooling_level == "high" else 1
        self.global_mlp = nn.Sequential(
                Residual(
                    mlp_num=2,
                    hidden_dim=pattern_num * hidden_dim,
                    activation_fn=activation_fn
                    ),
                Residual(
                    mlp_num=2,
                    hidden_dim=pattern_num * hidden_dim,
                    activation_fn=activation_fn,
                    add_end_activation=False
                    ),
                nn.Linear(
                    in_features=pattern_num * hidden_dim, 
                    out_features=1,
                    bias=False
                    ) 
            )
        self.pooling_fn = partial(threeOrderSumpool, pooling_level=pooling_level)
        
    def forward(self, 
                kemb: torch.Tensor,
                **kwargs
                ):
        output = self.global_mlp(
            self.pooling_fn(kemb)
            )
            
        return output






def threeOrderSumpool(kemb: Tensor, 
                      pooling_level: str = "middle",
                      ):
    '''
    kemb: (B, N, N, N, d)
    ea: (B, N, N, 1)
    '''
    
    N = kemb.shape[1]
    
    idx = torch.arange(N, device=kemb.device)
    
    if pooling_level == "middle":
        x1 = torch.sum(kemb[:, idx, idx, idx], dim=1) #(B, d)
        x2 = torch.sum(kemb, dim=(1, 2, 3)) - x1# (B, d)
        out_emb = torch.cat((x1, x2), dim=-1)
    elif pooling_level == "high":
        x1 = torch.sum(kemb[:, idx, idx, idx], dim=1) #(B, d)
        x2 = torch.sum(kemb[:, :, idx, idx], dim=(1, 2)) - x1# (B, d)
        x3 = torch.sum(kemb[:, idx, :, idx], dim=(0, 2)) - x1# (B, d)
        x4 = torch.sum(kemb[:, idx, idx, :], dim=(1, 2)) - x1# (B, d)
        x5 = torch.sum(kemb, dim=(1, 2, 3)) - x1 - x2 - x3 - x4# (B, d)
        out_emb = torch.cat((x1, x2, x3, x4, x5), dim=-1)
    else:
        out_emb = torch.sum(kemb, dim=(1, 2, 3))
    return out_emb


    