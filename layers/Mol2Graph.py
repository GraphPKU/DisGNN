import torch.nn as nn
from torch import Tensor
import torch
from layers.basis_layers import rbf_class_mapping


class Mol2Graph(nn.Module):
    def __init__(self,
                 z_hidden_dim: int,
                 ef_dim: int,
                 rbf: str,
                 rbf_trainable: bool,
                 rbound_upper: float,
                 max_z: int,
                 **kwargs):
        super().__init__()
        self.rbf_fn = rbf_class_mapping[rbf](
                    num_rbf=ef_dim, 
                    rbound_upper=rbound_upper, 
                    rbf_trainable=rbf_trainable,
                    **kwargs
                )
        self.z_emb = nn.Embedding(max_z + 1, z_hidden_dim, padding_idx=0)
        

    def forward(self, z: Tensor, pos: Tensor, **kwargs):
        '''
            z (B, N)
            pos (B, N, 3)
        '''
        
        emb1 = self.z_emb(z) # (B, N, z_hidden_dim)
        
        B, N = z.shape[0], z.shape[1]
        ev = pos.unsqueeze(2) - pos.unsqueeze(1) # (B, N, N, 3)
        el = torch.norm(ev, dim=-1, keepdim=True) # (B, N, N, 1)
        ef = self.rbf_fn(el.reshape(-1, 1)).reshape(B, N, N, -1) # (B, N, N, ef_dim)
        
        return emb1, ef


