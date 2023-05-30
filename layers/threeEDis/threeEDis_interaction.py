import torch
import torch.nn as nn
from torch import Tensor
from layers.basic_layers import Residual, Dense

class ThreeEDisLayer(nn.Module):
    def __init__(self, 
                 hidden_dim,
                 ef_dim,
                 activation_fn,
                 e_mode: str = 'E',
                 **kwargs
                 ):
        super().__init__()
                
        self.emb_lins = nn.ModuleList(
            [
                Residual(
                    hidden_dim=hidden_dim,
                    mlp_num=2,
                    activation_fn=activation_fn,
                ) for _ in range(4)
            ]
        )

        self.merge_lin = nn.Sequential(
            Dense(
                in_features=3 * (hidden_dim),
                out_features=hidden_dim,
                activation_fn=activation_fn
            ),
            Residual(
                mlp_num=2,
                hidden_dim=hidden_dim,
                activation_fn=activation_fn
            ),
        )

        self.output_lin = Residual(
            hidden_dim=hidden_dim,
            mlp_num=2,
            activation_fn=activation_fn
        )
        

        if e_mode == 'E':
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
        self.e_mode = e_mode

        


    def forward(
        self,
        kemb: Tensor,
        ef: Tensor,
        **kwargs
        ):
        '''
        kemb (B, N, N, N, d)
        ef (B, N, N, d)
        return: kemb_ijk <- kemb_ijk, { {(ef_ia,kemb_ajk)}, {(ef_ja, kemb_iak)}, {(ef_ka, kemb_ija)}| a\in V }
        '''
        
        kemb_0, kemb_1, kemb_2 = [self.emb_lins[i](kemb) for i in range(3)]
        
        self_message = self.emb_lins[3](kemb)
        
        if self.e_mode == 'E':
            e_emb = torch.sum(kemb, dim=3) + torch.sum(kemb, dim=2) + torch.sum(kemb, dim=1) # treat different multisets the same
            e = self.e_lin(ef) * self.e_tuple_lin(e_emb)
        elif self.e_mode == 'e':
            e = self.e_lin(ef)
        
        if self.e_mode == 'E' or self.e_mode == 'e':
            kemb_0 = torch.einsum("biad,bajkd->bijkd", e, kemb_0) 
            kemb_1 = torch.einsum("bjad,biakd->bijkd", e, kemb_1) 
            kemb_2 = torch.einsum("bkad,bijad->bijkd", e, kemb_2)
        else:
            kemb_0 = torch.sum(kemb_0, dim=1, keepdim=True).repeat(1, kemb.shape[1], 1, 1, 1)
            kemb_1 = torch.sum(kemb_1, dim=2, keepdim=True).repeat(1, 1, kemb.shape[2], 1, 1)
            kemb_2 = torch.sum(kemb_2, dim=3, keepdim=True).repeat(1, 1, 1, kemb.shape[3], 1)
        
        
        kemb = self_message * self.merge_lin(torch.cat((kemb_0, kemb_1, kemb_2), dim=-1))
        kemb = self.output_lin(kemb)
        
        return kemb


