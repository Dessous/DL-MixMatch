import torch 
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from wideresnet import WideResNet28

def train_baseline(train_loader, val_loader, lr=0.002, num_epoch=20):
    model = WideResNet28(10)
    model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    criterion.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    writer = SummaryWriter('logs-baseline')
    loaders = {'train': train_loader, 'val': val_loader}

    i = 0
    for k in range(num_epoch):
        print("EPOCH ", k)
        loss_val = 0.0
        acc_val = 0.0
        for stage, loader in loaders.items():
            if stage == 'train':
                model.train(True)
            else:
                model.train(False)

            for batch in loader:
                optimizer.zero_grad()
                pred = model(batch[0].cuda())
                loss = criterion(pred, batch[1].cuda())
                loss.backward()
                if stage == 'train':
                    optimizer.step()
                    writer.add_scalar('Loss-' + stage, loss.item(), i)
                    writer.add_scalar('Accuracy-' + stage,
                                      (torch.argmax(pred, dim=1) == batch[1].cuda()).float().sum() / 32,
                                      i)
                    i += 1
                else:
                    loss_val += loss.item()
                    acc_val += (torch.argmax(pred, dim=1) == batch[1].cuda()).float().sum() / 32
            if stage == 'val':    
                writer.add_scalar('Loss-' + stage, loss_val / len(val_loader), k)
                writer.add_scalar('Accuracy-' + stage, acc_val / len(val_loader), k)


