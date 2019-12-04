import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from wideresnet import WideResNet28
from itertools import cycle
from mixmatch_utils import *

def one_hot_vector(num_class, label):
    vec = np.zeros((label.shape[0], num_class))
    vec[np.arange(label.shape[0]), label] = 1
    return torch.Tensor(vec)

def train(train_labeled_loader, train_unlabeled_loader, model,
          optimizer, train_criterion, epoch, augmentor,
          lr=0.002, num_iter=1024, T=1, alpha=0.75):
    
    model.train(True)
    dataloaders = {
        'labeled': cycle(train_labeled_loader),
        'unlabeled': cycle(train_unlabeled_loader),
    }
    
    mix_match_loss = MixMatchMetric()

    for i in range(num_iter):
        labeled, target = next(dataloaders['labeled'])
        unlabeled1 = next(dataloaders['unlabeled'])
        unlabeled2 = augmentor(unlabeled1)
        unlabeled1 = augmentor(unlabeled1)
        assert not torch.all(torch.eq(unlabeled1, unlabeled2))

        target = one_hot_vector(10, target)

        labeled = labeled.cuda()
        target = target.cuda(non_blocking=True)  # ???
        unlabeled1 = unlabeled1.cuda()
        unlabeled2 = unlabeled2.cuda()

        with torch.no_grad():
            u1_pred = model(unlabeled1)
            u2_pred = model(unlabeled2)
            p = (torch.softmax(u1_pred, dim=1) + torch.softmax(u2_pred, dim=1)) / 2
            pt = p**(1/T)
            u1_target = (pt / pt.sum(dim=1, keepdim=True)).detach()

        input_concat = torch.cat([labeled, unlabeled1, unlabeled2], dim=0)
        target_concat = torch.cat([target, u1_target, u1_target], dim=0)

        lambd = np.random.beta(alpha, alpha)
        lambd = max(lambd, 1 - lambd)

        idx = torch.randperm(input_concat.size(0))

        input_a, input_b = input_concat, input_concat[idx]
        target_a, target_b = target_concat, target_concat[idx]

        mixed_input = lambd * input_a + (1 - lambd) * input_b
        mixed_target = lambd * target_a + (1 - lambd) * target_b

        batch_size = labeled.size(0)
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = interleave(mixed_input, batch_size)

        logits = [model(input) for input in mixed_input]

        logits = interleave(logits, batch_size)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)

        total_loss = train_criterion(logits_x,
                                     mixed_target[:batch_size],
                                     logits_u,
                                     mixed_target[batch_size:])

        mix_match_loss.update(total_loss.item(), batch_size)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
    return mix_match_loss.value()

def test(loader, model, criterion):
    losses = MixMatchMetric()
    accuracy = MixMatchMetric()

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch_size = batch[0].size(0)
            pred = model(batch[0].cuda())
            loss = criterion(pred, batch[1].cuda())
            acc = (torch.argmax(pred, dim=1) == batch[1].cuda()).float().sum()
            losses.update(loss.item(), batch_size)
            accuracy.update(acc, batch_size)

    return losses.value(), accuracy.value()

def train_mixmatch(train_labeled_loader, train_unlabeled_loader, test_loader, augmentor, config):
    assert (augmentor is not None)

    writer = SummaryWriter(config.train.logdir)
    model = WideResNet28(10)
    model = model.cuda()
    train_criterion = MixMatchLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)

    for epoch in range(20):
        train_loss = train(train_labeled_loader, train_unlabeled_loader, model,
                           optimizer, train_criterion, epoch, augmentor)
        print(train_loss)
        writer.add_scalar('train_loss', train_loss, epoch)
        
        _, train_acc = test(train_labeled_loader, model, criterion)
        test_loss, test_acc = test(test_loader, model, criterion)
        
        
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)        
        writer.add_scalar('test_acc', test_acc, epoch)
