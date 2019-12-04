import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def interleave(xy, batch):
    nu = len(xy) - 1
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]

class MixMatchMetric(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def value(self):
        return self.avg

class MixMatchLoss(object):
    def __init__(self, lambda_u=75):
        self.lambda_u = lambda_u
        self.mse = torch.nn.MSELoss()
        
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u):
        probs_u = torch.softmax(outputs_u, dim=1)
        loss_lab = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        return loss_lab + self.lambda_u * self.mse(probs_u, targets_u)
