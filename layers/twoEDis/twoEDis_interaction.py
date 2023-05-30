import torch
import torch.nn as nn
from layers.basic_layers import Residual, Dense


class TwoEDisLayer(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 ef_dim: int,
                 activation_fn: nn.Module = nn.SiLU(),
                 e_mode: str = 'E',
                 use_concat: bool = False,
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
        
        self.output_lin = Residual(
                mlp_num=2,
                hidden_dim=hidden_dim,
                activation_fn=activation_fn,
                )
        
        self.e_mode = e_mode
        self.use_concat = use_concat
        if e_mode == 'E':
            if use_concat:
                self.e_lin = nn.Sequential(
                    Dense(
                        in_features=ef_dim + hidden_dim,
                        out_features=ef_dim + hidden_dim,
                        activation_fn=activation_fn,
                    ),
                    Dense(
                        in_features=ef_dim + hidden_dim,
                        out_features=hidden_dim,
                        activation_fn=activation_fn,
                    ),
                )
            else:
                self.e_tuple_lin = Residual(
                    hidden_dim=hidden_dim,
                    mlp_num=2,
                    activation_fn=activation_fn
                    )
                self.e_lin = Dense(
                    in_features=ef_dim,
                    out_features=hidden_dim,
                    activation_fn=activation_fn,
                )
            
        elif e_mode == 'e':
            self.e_lin = Dense(
                in_features=ef_dim,
                out_features=hidden_dim,
                activation_fn=None,
                bias=False
            )
            
        self.merge_lin = nn.Sequential(
            Dense(
                in_features=2 * (hidden_dim),
                out_features=hidden_dim,
                activation_fn=activation_fn
            ),
            Residual(
                mlp_num=2,
                hidden_dim=hidden_dim,
                activation_fn=activation_fn
            ),
        )
        


    def forward(self, 
                kemb: torch.Tensor,
                ef: torch.Tensor,
                **kwargs
                ):
        
        
        self_message, kemb_0, kemb_1 = [self.emb_lins[i](kemb) for i in range(3)]
        
        if self.e_mode == 'E':
            if self.use_concat:
                e = self.e_lin(torch.cat([ef, kemb], dim=-1))
            else:
                e = self.e_lin(ef) * self.e_tuple_lin(kemb)
        elif self.e_mode == 'e':
            e = self.e_lin(ef)
        
        if self.e_mode == 'E' or self.e_mode == 'e':
            kemb_0 = torch.einsum('baid,bajd->bijd', e, kemb_0)
            kemb_1 = torch.einsum('bajd,biad->bijd', e, kemb_1)
        else:
            kemb_0 = torch.sum(kemb_0, dim=1, keepdim=True).repeat(1, kemb_0.shape[1], 1, 1)
            kemb_1 = torch.sum(kemb_1, dim=2, keepdim=True).repeat(1, 1, kemb_1.shape[2], 1)
        
        kemb = self_message * self.merge_lin(torch.cat([kemb_0, kemb_1], dim=-1))
        kemb_out = self.output_lin(kemb)
        
        return kemb_out




    
    
