import torch
import torch.nn as nn
from layers.basic_layers import Residual, Dense


class TwoFDisLayer(nn.Module):

    def __init__(self,
                 hidden_dim: int,
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
        


        if kwargs["interaction_residual"]:
            self.output_lin = Residual(
                    mlp_num=2,
                    hidden_dim=hidden_dim,
                    activation_fn=activation_fn,
                    )
        else:
            print("USE DENSE OUTPUT")
            self.output_lin = nn.Sequential(
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
                )
        

    def forward(self, 
                kemb: torch.Tensor,
                **kwargs
                ):
        '''
            kemb: (B, N, N, hidden_dim)
        '''
        
        
        self_message, kemb_0, kemb_1 = [self.emb_lins[i](kemb) for i in range(3)]
        
        kemb_0, kemb_1 = (kemb_0.permute(0, 3, 1, 2), kemb_1.permute(0, 3, 1, 2))
        
        kemb_multed = torch.matmul(kemb_0, kemb_1).permute(0, 2, 3, 1)

        kemb_out = self.output_lin(self_message * kemb_multed)
        
        return kemb_out




    
    
