import torch.nn as nn
import torch
from utils.initializers import he_orthogonal_init
    
class Dense(torch.nn.Module):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias: bool = True, 
        activation_fn: torch.nn.Module = None,
        use_layer_norm = False
    ):
        super().__init__()

        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
        self.in_features = in_features
        self.reset_parameters()
        self.weight = self.linear.weight
        self.bias = self.linear.bias

        if activation_fn is None:
            self._activation = torch.nn.Identity()
        else:
            self._activation = activation_fn
            
        self.layer_norm = nn.LayerNorm(out_features) if use_layer_norm else nn.Identity()

    def reset_parameters(self):
        if not self.in_features == 1:
            he_orthogonal_init(self.linear.weight)
        if self.linear.bias is not None:
            self.linear.bias.data.fill_(0)

    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        x = self.layer_norm(x)
        x = self._activation(x)
        return x


class Residual(nn.Module):
    def __init__(
        self,
        mlp_num: int,
        hidden_dim: int,
        activation_fn: torch.nn.Module = None,
        bias: bool = True,
        add_end_activation: bool = True,
        use_layer_norm = False,
    ):
        super().__init__()
        assert mlp_num > 0
                
        end_activation_fn = activation_fn if add_end_activation else None
        
        self.mlps = nn.Sequential(
            *[
                Dense(hidden_dim, hidden_dim, bias=bias, activation_fn=activation_fn, use_layer_norm=use_layer_norm)
                for _ in range(mlp_num - 1)
            ],
            Dense(hidden_dim, hidden_dim, bias=bias, activation_fn=end_activation_fn, use_layer_norm=use_layer_norm)
            )
            
    def forward(self, x: torch.Tensor):
        return self.mlps(x) + x
    
    

