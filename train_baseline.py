import torch 
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from wideresnet import WideResNet28
from itertools import cycle

def train_baseline(train_loader, val_loader, augmentor=None, lr=0.002, num_epoch=20, num_iter=1024):
    model = WideResNet28(10)
    model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    criterion.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    writer = SummaryWriter('logs-baseline')
    loaders = {'train': cycle(train_loader), 'val': cycle(val_loader)}

    for k in range(num_epoch):
        print("EPOCH ", k)
        loss_val = 0.0
        acc_val = 0.0
        for stage, loader in loaders.items():
            if stage == 'train':
                model.train(True)
            else:
                model.eval()

            for i in range(num_iter):
                batch = next(loader)
                optimizer.zero_grad()
                pred = None
                if augmentor is not None:
                    aug_batch = augmentor(batch[0])
                    pred = model(aug_batch.cuda())
                else:
                    pred = model(batch[0].cuda())
                loss = criterion(pred, batch[1].cuda())
                loss.backward()
                if stage == 'train':
                    optimizer.step()
                    writer.add_scalar('Loss-' + stage, loss.item(), i + k * num_iter)
                    writer.add_scalar('Accuracy-' + stage,
                                      (torch.argmax(pred, dim=1) == batch[1].cuda()).float().sum() / 32,
                                      i + k * num_iter)
                else:
                    loss_val += loss.item()
                    acc_val += (torch.argmax(pred, dim=1) == batch[1].cuda()).float().sum() / 32
            if stage == 'val':    
                writer.add_scalar('Loss-' + stage, loss_val / len(val_loader), k)
                writer.add_scalar('Accuracy-' + stage, acc_val / len(val_loader), k)


