import torch
import torch.nn as nn
from layers.basic_layers import Residual, Dense


class TwoFDisLayer(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 use_mult_lin,
                 activation_fn: nn.Module = nn.SiLU(),
                 **kwargs
                 ):
        super().__init__()
        
        self.emb_lins = nn.ModuleList(
            [
                nn.Sequential(
                    Dense(
                        in_features=hidden_dim,
                        out_features=hidden_dim,
                        activation_fn=activation_fn
                    ),
                    Dense(
                        in_features=hidden_dim,
                        out_features=hidden_dim,
                        activation_fn=activation_fn
                    )
                ) for _ in range(3)
            ] 
        )
        if use_mult_lin:
            self.mult_lin = nn.Sequential(
                Residual(
                    mlp_num=2,
                    hidden_dim=hidden_dim,
                    activation_fn=activation_fn,
                    ),
                Residual(
                    mlp_num=2,
                    hidden_dim=hidden_dim,
                    activation_fn=activation_fn,
                    ),
            )
        

        self.output_lin = Residual(
                mlp_num=2,
                hidden_dim=hidden_dim,
                activation_fn=activation_fn,
                )

        self.use_mult_lin = use_mult_lin

    def forward(self, 
                kemb: torch.Tensor,
                **kwargs
                ):
        '''
            kemb: (B, N, N, hidden_dim)
        '''
        
        
        self_message, kemb_0, kemb_1 = [self.emb_lins[i](kemb) for i in range(3)]
        
        kemb_0_p, kemb_1_p = (kemb_0.permute(0, 3, 1, 2), kemb_1.permute(0, 3, 1, 2))
        
        kemb_multed = torch.matmul(kemb_0_p, kemb_1_p).permute(0, 2, 3, 1)
        
        if self.use_mult_lin:
            kemb_multed = self.mult_lin(kemb_multed)

        kemb_out = self.output_lin(self_message * kemb_multed)
        
        return kemb_out




    
    
