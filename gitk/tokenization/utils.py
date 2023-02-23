import time
import sys
import select
import os
import numpy as np
import shutil

def prRed(skk): print("\033[91m{}\033[00m" .format(skk)) 
def prGreen(skk): print("\033[92m{}\033[00m" .format(skk)) 
def prYellow(skk): print("\033[93m{}\033[00m" .format(skk)) 
def prLightPurple(skk): print("\033[94m{}\033[00m" .format(skk)) 
def prPurple(skk): print("\033[95m{}\033[00m" .format(skk)) 
def prCyan(skk): print("\033[96m{}\033[00m" .format(skk)) 
def prLightGray(skk): print("\033[97m{}\033[00m" .format(skk)) 
def prBlack(skk): print("\033[98m{}\033[00m" .format(skk)) 

_log_path = None

def set_log_path(path):
    global _log_path
    _log_path = path

class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v

def time_str(t):
    if t >= 3600:
        return '{:.2f}h'.format(t / 3600)
    if t >= 60:
        return '{:.2f}m'.format(t / 60)
    return '{:.2f}s'.format(t)

def timed_response(prompt, wait_time, default):
    print(prompt, end='',flush=True)
    i, o, e = select.select( [sys.stdin], [], [], wait_time)
    if i:
        ans = sys.stdin.readline().strip()
        if ans not in ['y','n']:
            print('\033[91m{}\033[00m'.format(default))
            return default
        else:
            return ans
    else:
        print('\033[91m{}\033[00m'.format(default))
        return default

def log(obj, filename='log.txt'):
    print(obj,flush=True)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)

class lr_scheduler:
    def __init__(self, init_lr, end_lr, epochs, lr_info=None, mode='linear'):
        self.lr = init_lr
        self.end_lr = end_lr
        self.init_lr = init_lr
        self.mode = mode
        self.epochs = epochs
        self.lr_info = lr_info
        self.count = 0
        if mode == 'linear':
            self.freq = lr_info['freq']
    def step(self):
        self.count += 1
        if self.mode == 'linear':
            if self.count % self.freq == 0:
                self.lr = self.init_lr - (self.init_lr-self.end_lr)/self.epochs * self.count
        elif self.mode == 'milestone':
            milestones = np.array(self.lr_info['milestones'])
            power = (milestones <= self.count).sum()
            self.lr = self.init_lr * np.power(self.lr_info['ratio'], float(power))
            if self.lr < self.end_lr:
                self.lr = self.end_lr
        return self.lr

def ensure_dir(path, default='y'):
    if os.path.exists(path):
        if default == 'y':
            prompt = '\033[91m{} exists,remove?([y]/n):\033[00m '.format(path)
        else:
            prompt = '\033[91m{} exists,remove?(y/[n]):\033[00m '.format(path)
        ans = timed_response(prompt, 5, default)
        if ans != 'n':
            shutil.rmtree(path)
        else:
            return
    os.makedirs(path,exist_ok=True)

