"""Cosine Scheduler.

Cosine LR schedule with warmup.
"""

import numpy as np


class CosineLRScheduler():
    """Linear learning rate decay scheduler."""

    def __init__(self,
                 optimizer,
                 base_lr,
                 epochs,
                 warmup_length,
                 end_lr=0,
                 logger=None):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.epochs = epochs
        self.warmup_length = warmup_length
        self.end_lr = end_lr

    def step(self, epoch):
        if epoch < self.warmup_length:
            lr = self.base_lr * (epoch + 1) / self.warmup_length
        else:
            e = epoch - self.warmup_length
            es = self.epochs - self.warmup_length
            lr = self.end_lr + (0.5 * (1 + np.cos(np.pi * e / es)) *
                                (self.base_lr - self.end_lr))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
