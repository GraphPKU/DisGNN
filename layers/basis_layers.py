import torch.nn as nn
import torch
import math


class SincRadialBasis(nn.Module):
    # copied from Painn
    def __init__(self, num_rbf, rbound_upper, rbf_trainable=False, **kwargs):
        super().__init__()
        if rbf_trainable:
            self.register_parameter("n", nn.parameter.Parameter(torch.arange(
                1, num_rbf + 1, dtype=torch.float).unsqueeze(0)/rbound_upper))
        else:
            self.register_buffer("n", torch.arange(
                1, num_rbf + 1, dtype=torch.float).unsqueeze(0)/rbound_upper)

    def forward(self, r):
        n = self.n
        output = (math.pi) * n * torch.sinc(n * r)
        return output

class BesselBasisLayer(torch.nn.Module):
    def __init__(self, num_rbf, rbound_upper, rbound_lower=0.0, rbf_trainable=False, **kwargs):
        super().__init__()
        freq = torch.arange(
            1, num_rbf + 1, dtype=torch.float).unsqueeze(0)*math.pi/rbound_upper
        if not rbf_trainable:
            self.register_buffer("freq", freq)
        else:
            self.register_parameter("freq", nn.parameter.Parameter(freq))
            
        self.rbound_upper = rbound_upper

        

    def forward(self, dist):
        '''
        dist (B, 1)
        '''
        return ((self.freq * dist).sin() / (dist + 1e-7)) * ((2 / self.rbound_upper) ** 0.5)

class GaussianSmearing(nn.Module):
    def __init__(self,
                 num_rbf,
                 rbound_upper,
                 rbound_lower=0.0,
                 rbf_trainable=False,
                 **kwargs):
        super(GaussianSmearing, self).__init__()
        self.rbound_lower = rbound_lower
        self.rbound_upper = rbound_upper
        self.num_rbf = num_rbf
        self.rbf_trainable = rbf_trainable

        offset, coeff = self._initial_params()
        if rbf_trainable:
            self.register_parameter("coeff", nn.parameter.Parameter(coeff))
            self.register_parameter("offset", nn.parameter.Parameter(offset))
        else:
            self.register_buffer("coeff", coeff)
            self.register_buffer("offset", offset)

    def _initial_params(self):
        offset = torch.linspace(self.rbound_lower, self.rbound_upper,
                                self.num_rbf)
        coeff = -0.5 / (offset[1] - offset[0])**2
        return offset, coeff

    def reset_parameters(self):
        offset, coeff = self._initial_params()
        self.offset.data.copy_(offset)
        self.coeff.data.copy_(coeff)

    def forward(self, dist):
        return torch.exp(self.coeff * torch.square(dist - self.offset))

class NewExpNormalSmearing(nn.Module):
    '''
    modified: delete cutoff with r
    '''
    def __init__(self, num_rbf, rbound_upper, rbound_lower=0.0, rbf_trainable=False, **kwargs):
        super().__init__()
        self.rbound_upper = rbound_upper
        self.rbound_lower = rbound_lower
        self.num_rbf = num_rbf
        self.rbf_trainable = rbf_trainable
        means, betas = self._initial_params()
        if rbf_trainable:
            self.register_parameter("means", nn.parameter.Parameter(means))
            self.register_parameter("betas", nn.parameter.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        start_value = torch.exp(torch.scalar_tensor(-self.rbound_upper))
        end_value = torch.exp(torch.scalar_tensor(-self.rbound_lower))
        means = torch.linspace(start_value, end_value, self.num_rbf)
        betas = torch.tensor([(2 / self.num_rbf * (end_value - start_value))**-2] *
                             self.num_rbf)
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        return torch.exp(-self.betas * torch.square((torch.exp(-dist) - self.means)))

class CosineCutoff(nn.Module):
    def __init__(self, rbound_upper=5.0):
        super().__init__()
        self.register_buffer("rbound_upper", torch.tensor(rbound_upper))
        #self.rbound_upper = rbound_upper

    def forward(self, distances):
        ru = self.rbound_upper
        rbounds = 0.5 * \
            (torch.cos(distances * math.pi / ru) + 1.0)
        rbounds = rbounds * (distances < ru).float()
        return rbounds

rbf_class_mapping = {
    "gauss": GaussianSmearing,
    "nexpnorm": NewExpNormalSmearing,
    "sinc": SincRadialBasis,
    "bessel": BesselBasisLayer,
}


