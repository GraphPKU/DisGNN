import torch.nn as nn
from layers.basic_layers import Residual, Dense
import torch


class TwoOrderOutputBlock(nn.Module):
    def __init__(self, 
                 hidden_dim: int, 
                 activation_fn: nn.Module = nn.SiLU(),
                 **kargs):
        super().__init__()
        self.output_fn = nn.Sequential(
            Residual(
                mlp_num=2,
                hidden_dim=2 * hidden_dim,
                activation_fn=activation_fn,
                bias=True,
                ),
            Residual(
                mlp_num=2,
                hidden_dim=2 * hidden_dim,
                activation_fn=activation_fn,
                bias=True,
                add_end_activation=False
                ),
            Dense(
                in_features=2 * hidden_dim,
                out_features=1,
                bias=False
            )
        ) 
        self.sum_pooling = twoOrderSumpool
        
        
            
        
    def forward(self,
                kemb: torch.Tensor
                ):

        output = self.output_fn(
            self.sum_pooling(kemb=kemb)
            )
            
        return output
    







def twoOrderSumpool(kemb: torch.Tensor):

        
    N = kemb.shape[1]
    idx = torch.arange(N)
    
    kemb_diag = kemb[:, idx, idx, :]
    sum_kemb_diag = torch.sum(kemb_diag, 1)
    
    sum_kemb_offdiag = torch.sum(kemb, (1, 2)) - sum_kemb_diag
    
    
    output = torch.cat((sum_kemb_diag, sum_kemb_offdiag), dim=-1)
    
    return output

