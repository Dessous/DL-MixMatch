import torch

class EMAOptim(object):
    def __init__(self, model, ema_model, wd, lr=0.002, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = wd

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype is not torch.int64:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * (1.0 - self.alpha))
                param.mul_(1 - self.wd)
