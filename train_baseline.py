import torch 
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from wideresnet import WideResNet28
from itertools import cycle
from ema import EMAOptim


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


def train_baseline(train_loader, val_loader, logger, augmentor=None, lr=0.002, num_epoch=20, num_iter=1024):
    model = WideResNet28(10)
    model = model.cuda()
    model_ema = WideResNet28(10)
    model_ema = model_ema.cuda()
    for param in model_ema.parameters():
        param.detach_()
    ema_optimizer = EMAOptim(model, model_ema)

    criterion = nn.CrossEntropyLoss()
    criterion.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    writer = logger.writer
    train_loader_cycle = cycle(train_loader)

    for k in range(num_epoch):
        print("EPOCH ", k)
        model.train(True)
        for i in range(num_iter):
            batch = next(train_loader_cycle)
            optimizer.zero_grad()
            pred = None
            if augmentor is not None:
                aug_batch = augmentor(batch[0])
                pred = model(aug_batch.cuda())
            else:
                pred = model(batch[0].cuda())
            loss = criterion(pred, batch[1].cuda())
            loss.backward()
            optimizer.step()
            ema_optimizer.step()
            writer.add_scalar('loss-train', loss.item(), i + k * num_iter)
            writer.add_scalar('Accuracy-train',
                              (torch.argmax(pred, dim=1) == batch[1].cuda()).float().sum() / 32,
                              i + k * num_iter)
        loss_val, acc_val = test(val_loader, model, criterion)
        writer.add_scalar('Loss-test', loss_val, k)
        writer.add_scalar('Accuracy- test', acc_val, k)


