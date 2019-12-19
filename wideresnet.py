import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

class WideBlock(nn.Module):
    def __init__(self, in_size, out_size, relu, dropout=0.0, stride=1):
        super(WideBlock, self).__init__()
        self.batch_norm1 = nn.BatchNorm2d(in_size, momentum=0.001)
        self.relu = nn.LeakyReLU(relu, inplace=True)
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3,
                               padding=1, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm2 = nn.BatchNorm2d(out_size, momentum=0.001)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3,
                               stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(self.relu(self.batch_norm1(x))))
        out = self.conv2(self.relu(self.batch_norm2(out)))
        out += self.shortcut(x)
        return out

def weigths_init(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()

class WideResNet28(nn.Module):
    def __init__(self, n_classes, filters=16, widen_factor=1, relu=0.01, dropout=0.0):
        super(WideResNet28, self).__init__()
        DEPTH = 28
        blocks_count = (DEPTH - 4) // 6
        
        self.conv = nn.Conv2d(3, filters, kernel_size=3,
                               stride=1, padding=1, bias=True)
        self.block1 = self.create_wide_block(filters, filters*widen_factor, blocks_count,
                                relu, dropout, 1)
        self.block2 = self.create_wide_block(filters*widen_factor, 2*filters*widen_factor, blocks_count,
                                relu, dropout, 2)
        self.block3 = self.create_wide_block(2*filters*widen_factor, 4*filters*widen_factor, blocks_count,
                                relu, dropout, 2)
        
        self.batch_norm = nn.BatchNorm2d(4*filters*widen_factor, momentum=0.001)
        self.relu = nn.LeakyReLU(relu, inplace=True)
        self.linear = nn.Linear(4*filters*widen_factor, n_classes)
        weigths_init(self.modules())
        
    def create_wide_block(self, in_size, out_size, blocks_count, relu, dropout, stride):
        layers = [
            WideBlock(in_size, out_size, relu, dropout, stride),
        ]
        for _ in range(blocks_count):
            layers.append(WideBlock(out_size, out_size, relu, dropout, 1))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.batch_norm(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        return self.linear(out)
