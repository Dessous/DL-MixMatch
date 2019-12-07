import datetime
from tensorboardX import SummaryWriter
import os


class Logger:
    def __init__(self, path):
        self.path = path
        self.name_suffix = '_' + datetime.datetime.now().strftime("%I_%M%p_%B_%d_%Y")
        self.log_file = open(self.path + 'log' + self.name_suffix + '.txt', 'w+')
        os.mkdir(self.path + self.name_suffix[1:])
        self.writer = SummaryWriter(self.path + self.name_suffix[1:])

    def add_row(self, s):
        self.log_file.write(s + '\n')