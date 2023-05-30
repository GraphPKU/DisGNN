import torch.nn as nn
import torch
from layers.basic_layers import Residual, Dense



class TwoEDisInit(nn.Module):
    def __init__(self,
                 z_hidden_dim: int,
                 ef_dim: int,
                 k_tuple_dim: int,
                 activation_fn: nn.Module = nn.SiLU(),
                 **kwargs):
        super().__init__()
        
        self.z_lins = nn.ModuleList(
                [
                    nn.Sequential(
                        Dense(
                            in_features=z_hidden_dim,
                            out_features=k_tuple_dim,
                            activation_fn=activation_fn
                        ),
                    ) for _ in range(2)
                    ]
                )

        self.ef_lin = Dense(
            in_features=ef_dim,
            out_features=k_tuple_dim,
            bias=False,
            activation_fn=None
        )

        
        
        self.pattern_embedding = nn.Embedding(
            num_embeddings=3,
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
        
        z0, z1 =  [self.z_lins[i](emb1) for i in range(2)]
        ef0 = self.ef_lin(ef)
        
        z_mixed = z0[:, None, :, :] * z1[:, :, None, :]
        ef_mixed = ef0
        
        B = z_mixed.shape[0]
        N = z_mixed.shape[1]
        
        idx = torch.arange(N)
        tuple_pattern = torch.ones(size=(B, N, N), dtype=torch.int, device=z_mixed.device)
        tuple_pattern[:, idx, idx] = 2
        tuple_pattern = self.pattern_embedding(tuple_pattern)
        
        
        emb2 = z_mixed * ef_mixed * tuple_pattern
        
        emb2 = self.mix_lin(emb2)
        
        
        return emb2