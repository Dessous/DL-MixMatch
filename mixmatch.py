from augmentations import Augmentor
import torch
import numpy as np
import torch.nn.functional as F


class MixMatchLoss:
    def __init__(self, config, model, model_ema):
        self.T = config.mixmatch.sharp_temperature
        self.K = config.mixmatch.n_aug
        self.mixup_mode = config.mixmatch.mixup_mode
        self.mixup_alpha = config.mixmatch.mixup_alpha
        self.augmentor = Augmentor(config)
        self.n_classes = 10 if config.dataset.name == 'cifar10' else 100
        self.model = model
        self.model_ema = model_ema
        self.loss = _CombinedLoss(config.mixmatch.lmbd_u, config.mixmatch.lmbd_rampup_length)
        print('Use MixMatchLoss')

    def guess_labels(self, unlabeled_batches):
        batch_size = unlabeled_batches[0].size(0)
        with torch.no_grad():
            guessed_y = torch.zeros(batch_size, self.n_classes)
            if unlabeled_batches[0].is_cuda:
                guessed_y = guessed_y.cuda()
            for batch in unlabeled_batches:
                if self.model_ema is None:
                    output = self.model(batch)
                else:
                    output = self.model_ema(batch)  
                guessed_y += torch.softmax(output, dim=1)
            guessed_y /= len(unlabeled_batches)
            guessed_y = guessed_y**(1 / self.T)
            guessed_y = guessed_y / guessed_y.sum(dim=1, keepdim=True)
        return guessed_y.detach()

    def mixup_permutation(self, labeled_size, unlabeled_size):
        if self.mixup_mode == 'full':
            return torch.randperm(labeled_size + unlabeled_size)
        elif self.mixup_mode == 'labeled':
            return torch.cat(
                [torch.randperm(labeled_size),
                 torch.arange(labeled_size, labeled_size + unlabeled_size)]
            )
        elif self.mixup_mode == 'unlabeled':
            return torch.cat(
                [torch.arange(labeled_size),
                 torch.randperm(unlabeled_size) + labeled_size]
            )

    def __call__(self, x_l, y, x_u, epoch):
        batch_size = x_l.size(0)
        labeled = self.augmentor(x_l)
        ohe_y_labeled = torch.zeros(batch_size, self.n_classes).cuda().scatter_(1, y.view(-1, 1), 1)
        if x_l.is_cuda:
            ohe_y_labeled = ohe_y_labeled.cuda()

        unlabeled = [self.augmentor(x_u) for i in range(self.K)]

        y_unlabeled = self.guess_labels(unlabeled)

        all_x = torch.cat([labeled] + unlabeled, dim=0)
        all_y = torch.cat([ohe_y_labeled] + [y_unlabeled] * self.K, dim=0)
        perm = self.mixup_permutation(len(labeled), len(unlabeled))
        shuffled_x = all_x[perm]
        shuffled_y = all_y[perm]

        lmbd = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        lmbd = max(lmbd, 1 - lmbd)
        mixup_x = lmbd * all_x + (1 - lmbd) * shuffled_x
        mixup_y = lmbd * all_y + (1 - lmbd) * shuffled_y

        mixup_x = _interleave(list(torch.split(mixup_x, batch_size)), batch_size)
        logits = [self.model(x) for x in mixup_x]
        logits = _interleave(logits, batch_size)
        logits_labeled = logits[0]
        logits_unlabeled = torch.cat(logits[1:])
        y_labeled = mixup_y[:batch_size]
        y_unlabeled = mixup_y[batch_size:]
        return self.loss(logits_labeled, logits_unlabeled, y_labeled, y_unlabeled, epoch)


class _CombinedLoss:
    def __init__(self, lmbd, rampup_length):
        self.lmbd = lmbd
        self.rampup_length = rampup_length

    def __call__(self,  logits_labeled, logits_unlabeled, y_labeled, y_unlabeled, epoch):
        unlabeled_prob = torch.softmax(logits_unlabeled, dim=1)
        l_x = -torch.mean(torch.sum(F.log_softmax(logits_labeled, dim=1) * y_labeled, dim=1))
        l_u = torch.mean((unlabeled_prob - y_unlabeled) ** 2)
        return l_x + l_u * self.lmbd * _rampup(epoch, self.rampup_length)

class MixUpLoss:
    def __init__(self, config, model):
        self.alpha = config.mixup.alpha
        self.xent = torch.nn.CrossEntropyLoss()
        self.model = model
        print('Use MixUpLoss')

    def _augment(self, x, y):
        if self.alpha == 1:
            lmbd = 1
        else:
            lmbd = np.random.beta(self.alpha, self.alpha)

        idx = torch.randperm(x.size(0)).cuda()
        mixup_x = lmbd * x + (1 - lmbd) * x[idx]
        y_a = y
        y_b = y[idx]
        return mixup_x, y_a, y_b, lmbd

    def __call__(self, x, y):
        mixup_x, y_a, y_b, lmbd = self._augment(x, y)
        pred = self.model(mixup_x)
        loss = lmbd * self.xent(pred, y_a) + (1 - lmbd) * self.xent(pred, y_b)
        return loss


def _interleave_offsets(batch_size, n_groups):
    groups = [batch_size // n_groups] * n_groups
    for x in range(batch_size - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch_size
    return offsets


def _interleave(batches, batch_size):
    offsets = _interleave_offsets(batch_size, len(batches))
    out = [[batch[offsets[i]:offsets[i+1]] for i in range(len(batches))] for batch in batches]
    for i in range(1, len(batches)):
        out[0][i], out[i][i] = out[i][i], out[0][i]
    out = [torch.cat(row) for row in out]
    return out


def _rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)
