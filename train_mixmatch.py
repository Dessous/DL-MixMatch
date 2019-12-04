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
          optimizer, train_criterion, criterion, epoch, augmentor, writer,
          num_iter=1024, T=1, alpha=0.75):
    
    model.train(True)
    dataloaders = {
        'labeled': cycle(train_labeled_loader),
        'unlabeled': cycle(train_unlabeled_loader),
    }
    
    mix_match_loss = MixMatchMetric()

    for i in range(num_iter):
        labeled, target = next(dataloaders['labeled'])
        labeled = augmentor(labeled)
        unlabeled1 = next(dataloaders['unlabeled'])
        unlabeled2 = augmentor(unlabeled1)
        unlabeled1 = augmentor(unlabeled1)
        assert not torch.all(torch.eq(unlabeled1, unlabeled2))

        target_ohe = one_hot_vector(10, target)

        labeled = labeled.cuda()
        target_ohe = target_ohe.cuda(non_blocking=True)  # ???
        unlabeled1 = unlabeled1.cuda()
        unlabeled2 = unlabeled2.cuda()

        with torch.no_grad():
            u1_pred = model(unlabeled1)
            u2_pred = model(unlabeled2)
            p = (torch.softmax(u1_pred, dim=1) + torch.softmax(u2_pred, dim=1)) / 2
            pt = p**(1/T)
            u1_target = (pt / pt.sum(dim=1, keepdim=True)).detach()

        input_concat = torch.cat([labeled, unlabeled1, unlabeled2], dim=0)
        target_concat = torch.cat([target_ohe, u1_target, u1_target], dim=0)

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
        writer.add_scalar('train_loss', mix_match_loss.value(), i + epoch * num_iter)

        pred_lab = model(labeled.cuda())
        acc = (torch.argmax(pred_lab, dim=1) == target.cuda()).float().sum() / batch_size
        writer.add_scalar('train_acc', acc, i + epoch * num_iter)
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
    return mix_match_loss.value()

def test(loader, model, criterion):
    test_loss = 0.0
    test_acc = 0.0
    n = len(loader)

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch_size = batch[0].size(0)
            pred = model(batch[0].cuda())
            test_loss += criterion(pred, batch[1].cuda())
            test_acc += (torch.argmax(pred, dim=1) == batch[1].cuda()).float().sum() / batch_size

    return test_loss / n, test_acc / n

def train_mixmatch(train_labeled_loader, train_unlabeled_loader, test_loader, augmentor, config):
    assert (augmentor is not None)

    writer = SummaryWriter(config.train.logdir)
    model = WideResNet28(10)
    model = model.cuda()
    train_criterion = MixMatchLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)

    for epoch in range(config.train.num_epoch):
        train_loss = train(train_labeled_loader, train_unlabeled_loader, model,
                           optimizer, train_criterion, criterion, epoch, augmentor, writer,
                           config.train.num_iter, config.train.T, config.train.alpha)
        
        print('Avg loss at {} epoch is {}'.format(epoch, train_loss))
        test_loss, test_acc = test(test_loader, model, criterion)
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)
