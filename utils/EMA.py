import torch

class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device="cpu", use_buffers=False):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param
        self.use_buffers=use_buffers
        super().__init__(model, device, ema_avg, use_buffers=self.use_buffers)
        
    def update_parameters(self, model, exclude="ema_model"):
        origin_parameters = []
        origin_buffers = []
        for name, param in model.named_parameters():
            if exclude in name:
                continue
            else:
                origin_parameters.append(param)
        if self.use_buffers:
            for name, buffer in model.named_buffers():
                if exclude in name:
                    continue
                else:
                    origin_buffers.append(buffer)
        super().update_parameters(SimplifiedModel(origin_parameters, origin_buffers))
        
class SimplifiedModel():
    def __init__(self, parameters, buffers) -> None:
        self._parameters = parameters
        self._buffers = buffers
        
    def parameters(self):
        return self._parameters
    def buffers(self):
        return self._buffers
    