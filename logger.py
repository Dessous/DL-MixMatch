import datetime


class Logger:
    def __init__(self, path):
        self.path = path
        self.name_suffix = '_' + datetime.datetime.now().strftime("%I_%M%p_%B_%d_%Y")
        self.log_file = open(self.path + 'log' + self.name_suffix + '.txt', 'w+')

    def add_row(self, s):
        self.log_file.write(s + '\n')
