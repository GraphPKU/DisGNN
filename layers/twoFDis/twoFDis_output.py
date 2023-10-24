import torch.nn as nn
from layers.basic_layers import Residual, Dense
import torch
import ase

class TwoOrderOutputBlock(nn.Module):
    def __init__(self, 
                 hidden_dim: int, 
                 activation_fn: nn.Module = nn.SiLU(),
                 **kwargs
                 ):
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
                ),
            Residual(
                mlp_num=2,
                hidden_dim=2 * hidden_dim,
                activation_fn=activation_fn,
                add_end_activation=False,
                bias=True,
                ),
            Dense(
                in_features=2 * hidden_dim,
                out_features=1,
                bias=False
            )
        ) 


        self.sum_pooling = twoOrderSumpool
        
    def forward(self,
                kemb: torch.Tensor,
                **kwargs
                ):

        output = self.output_fn(self.sum_pooling(kemb=kemb))
        
        return output
    
    
def twoOrderSumpool(kemb):
    
    N = kemb.shape[1]
    idx = torch.arange(N)
    
    kemb_diag = kemb[:, idx, idx, :]
    sum_kemb_diag = torch.sum(kemb_diag, 1)
    
    sum_kemb_offdiag = torch.sum(kemb, (1, 2)) - sum_kemb_diag
    
    
    output = torch.cat((sum_kemb_diag, sum_kemb_offdiag), dim=-1)
    
    return output


    
    
class TwoOrderDipOutputBlock(nn.Module):
    def __init__(self, 
                 hidden_dim: int, 
                 activation_fn: nn.Module = nn.SiLU(),
                 **kwargs
                 ):
        super().__init__()
        
        self.output_fn = nn.Sequential(
            Residual(
                mlp_num=2,
                hidden_dim=hidden_dim,
                activation_fn=activation_fn,
                bias=True,
                # use_layer_norm=True 
                ),
            Residual(
                mlp_num=2,
                hidden_dim=hidden_dim,
                activation_fn=activation_fn,
                bias=True,
                # use_layer_norm=True
                ),
            Residual(
                mlp_num=2,
                hidden_dim=hidden_dim,
                activation_fn=activation_fn,
                bias=True,
                add_end_activation=False
                # use_layer_norm=True
                ),
            Dense(
                in_features=hidden_dim,
                out_features=1,
                bias=False
            )
        ) 

        
        
    def forward(self,
                kemb: torch.Tensor,
                pos: torch.Tensor,
                **kwargs
                ):
        
        pos = pos - pos.mean(dim=1, keepdim=True) # (B, N, 3)
        
        node_emb = kemb.sum(-2)
        
        q = self.output_fn(node_emb) # (B, N, 1)
        
        q = q - q.mean(dim=1, keepdim=True) # (B, N, 1)
        
        output = torch.sum(q * pos, dim=1) # (B, 3)
        
        output = torch.norm(output, dim=-1, keepdim=True) # (B, 1)
        
        return output
    

    
class TwoOrderElcOutputBlock(nn.Module):
    def __init__(self, 
                 hidden_dim: int, 
                 activation_fn: nn.Module = nn.SiLU(),
                 **kwargs
                 ):
        super().__init__()
        self.act = nn.Softplus()
        
        self.output_fn = nn.Sequential(
            Residual(
                mlp_num=2,
                hidden_dim=hidden_dim,
                activation_fn=activation_fn,
                bias=True,
                use_layer_norm=True 
                ),
            Residual(
                mlp_num=2,
                hidden_dim=hidden_dim,
                activation_fn=activation_fn,
                bias=True,
                use_layer_norm=True
                ),
            Residual(
                mlp_num=2,
                hidden_dim=hidden_dim,
                activation_fn=activation_fn,
                bias=True,
                use_layer_norm=True
                ),
            Dense(
                in_features=hidden_dim,
                out_features=1,
                bias=False
            )
        ) 
        
        atomic_mass = torch.from_numpy(ase.data.atomic_masses).float()
        atomic_mass[0] = 0
        self.register_buffer("atomic_mass", atomic_mass)

        self.atom_ref = nn.Embedding(100, 1, padding_idx=0)
        
    def forward(self,
                kemb: torch.Tensor,
                pos: torch.Tensor,
                z: torch.Tensor,
                **kwargs
                ):
        
        node_emb = kemb.sum(-2) # (B, N, D)
        q = self.output_fn(node_emb).squeeze(-1) # (B, N)
        ref = self.atom_ref(z).squeeze(-1) # (B, N)
        q = q + ref # (B, N)
        
        q = self.act(q) # (B, N)
        
        mass = self.atomic_mass[z].unsqueeze(-1) # (B, N, 1)

        pos = pos - pos.mean(dim=1, keepdim=True) # (B, N, 3)
        full_mass = torch.sum(mass, dim=1, keepdim=True) # (B, 1, 1)
        center = torch.sum(mass * pos, dim=1, keepdim=True) / full_mass # (B, 1, 3)
        centered_pos = pos - center # (B, N, 3)
        pos_powersum = torch.sum(torch.square(centered_pos), dim=-1) # (B, N)
        
        output = torch.sum(q * pos_powersum, dim=1, keepdim=True) # (B, 1)
        
        
        return output