import torch.nn as nn
import torch
from layers.basic_layers import Residual, Dense


class ThreeEDisInit(nn.Module):
    def __init__(self,
                 z_hidden_dim: int,
                 ef_dim: int,
                 k_tuple_dim: int,
                 activation_fn: nn.Module = nn.SiLU(),
                 **kwargs):
        super().__init__()
        self.z_lins = nn.ModuleList(
                [
                    Dense(
                        in_features=z_hidden_dim,
                        out_features=k_tuple_dim,
                        activation_fn=activation_fn
                        ) for _ in range(3)
                    ]
                )
        
        self.ef_lins = nn.ModuleList(
            [
                Dense(
                    in_features=ef_dim,
                    out_features=k_tuple_dim,
                    bias=False
                    )  for _ in range(3)
                ]
            )
        
        self.pattern_embedding = nn.Embedding(
            num_embeddings=6,
            embedding_dim=k_tuple_dim,
            padding_idx=0
        )
        
        self.mix_lin = Residual(
            hidden_dim=k_tuple_dim,
            activation_fn=activation_fn,
            mlp_num=2
        )
        

    def forward(self,
                emb1: torch.Tensor,
                ef: torch.Tensor,
                **kwargs
                ):


        
        ef0, ef1, ef2 = [self.ef_lins[i](ef) for i in range(3)]
        z0, z1, z2 =  [self.z_lins[i](emb1) for i in range(3)]

        z_mixed = z0[:, None, None, :, :] * z1[:, :, None, None, :] * z2[:, None, :, None, :]
        ef_mixed = ef0[:, None, :, :, :] * ef1[:, :, None, :, :] * ef2[:, :, :, None, :] 
            
        B = z_mixed.shape[0]
        N = z_mixed.shape[1]
        idx = torch.arange(N)
        tuple_pattern = torch.ones(size=(B, N, N, N), dtype=torch.int, device=z_mixed.device)
        tuple_pattern[:, idx, idx, idx] = 2
        tuple_pattern[:, idx, :, idx] = 3
        tuple_pattern[:, :, idx, idx] = 4
        tuple_pattern[:, idx, idx, :] = 5
        tuple_pattern = self.pattern_embedding(tuple_pattern)
        
        
        emb3 = z_mixed * ef_mixed * tuple_pattern
        
        emb3 = self.mix_lin(emb3)

        
        
        return emb3