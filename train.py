import torch 
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from wideresnet import WideResNet28
from itertools import cycle
from ema import EMAOptim
from mixmatch import MixMatchLoss, MixUpLoss


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


def train_epoch(train_labeled_loader, train_unlabeled_loader, model, augmentor,
          optimizer, ema_optimizer, criterion, epoch, writer, config):
    
    model.train(True)

    labeled_cycle = cycle(train_labeled_loader)
    unlabeled_cycle = None
    if config.train.use_mixmatch:
        unlabeled_cycle = cycle(train_unlabeled_loader)
        mixmatch = MixMatchLoss(config, model)
    else:
        mixup = MixUpLoss(config, model)

    num_iter = config.train.num_iter
    for i in range(num_iter):
        labeled, target = next(labeled_cycle)
        labeled = augmentor(labeled)
        mix_loss = None
        if config.train.use_mixmatch:
            unlabeled = next(unlabeled_cycle)
            mix_loss = mixmatch(labeled.cuda(), target.cuda(), unlabeled.cuda(), epoch)
        else:
            mix_loss = mixup(labeled.cuda(), target.cuda())

        pred = model(labeled.cuda())
        loss = criterion(pred, target.cuda())
        acc = (torch.argmax(pred, dim=1) == target.cuda()).float().sum() / labeled.size(0)
        writer.add_scalar('train/loss', loss.item(), i + epoch * num_iter)
        writer.add_scalar('train/acc', acc, i + epoch * num_iter)
        
        optimizer.zero_grad()
        mix_loss.backward()
        optimizer.step()
        if config.train.use_ema:
            ema_optimizer.step()


def train(train_labeled_loader, train_unlabeled_loader, test_loader, logger, augmentor, config):
    writer = logger.writer

    model = WideResNet28(config.dataset.num_classes)
    model = model.cuda()
    ema_optimizer = None
    if config.train.use_ema:
        model_ema = WideResNet28(config.dataset.num_classes)
        model_ema = model_ema.cuda()
        for param in model_ema.parameters():
            param.detach_()
        ema_optimizer = EMAOptim(model, model_ema)

    criterion = nn.CrossEntropyLoss()
    criterion.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)

    for epoch in range(config.train.num_epoch):
        print('EPOCH {}'.format(epoch))
        train_epoch(train_labeled_loader, train_unlabeled_loader, model, augmentor,
              optimizer, ema_optimizer, criterion, epoch, writer, config)
        
        if config.train.use_ema:
            test_loss, test_acc = test(test_loader, model_ema, criterion)
        else:
            test_loss, test_acc = test(test_loader, model, criterion)
        writer.add_scalar('test/loss', test_loss, epoch)
        writer.add_scalar('test/acc', test_acc, epoch)
