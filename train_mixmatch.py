import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from wideresnet import WideResNet28
from itertools import cycle
from mixmatch import MixMatch


def train(train_labeled_loader, train_unlabeled_loader, model,
          optimizer, criterion, epoch, writer, config):
    
    model.train(True)
    dataloaders = {
        'labeled': cycle(train_labeled_loader),
        'unlabeled': cycle(train_unlabeled_loader),
    }
    mixmatch = MixMatch(config, model)

    num_iter = config.train.num_iter
    for i in range(num_iter):
        labeled, target = next(dataloaders['labeled'])
        unlabeled = next(dataloaders['unlabeled'])
        mixmatch_loss = mixmatch(labeled.cuda(), target.cuda(), unlabeled.cuda(), epoch)

        pred = model(labeled.cuda())
        loss = criterion(pred, target.cuda())
        acc = (torch.argmax(pred, dim=1) == target.cuda()).float().sum() / labeled.size(0)
        writer.add_scalar('train_acc', acc, i + epoch * num_iter)
        writer.add_scalar('train_loss', loss.item(), i + epoch * num_iter)
        
        optimizer.zero_grad()
        mixmatch_loss.backward()
        optimizer.step()


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


def train_mixmatch(train_labeled_loader, train_unlabeled_loader, test_loader, logger, augmentor, config):
    assert (augmentor is not None)
    writer = logger.writer
    model = WideResNet28(10)
    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)

    for epoch in range(config.train.num_epoch):
        print('EPOCH {}'.format(epoch))
        train(train_labeled_loader, train_unlabeled_loader, model,
              optimizer, criterion, epoch, writer, config)
        
        test_loss, test_acc = test(test_loader, model, criterion)
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)
