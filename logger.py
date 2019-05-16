import sys
from datetime import datetime

class Logger(object):
    def __init__(self, datestring):
        self.terminal = sys.stdout
        #datestring = datetime.strftime(datetime.now(), '%m%d%Y(%H%M)')

        self.log = open(datestring + '_logfile' + '.txt', 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

