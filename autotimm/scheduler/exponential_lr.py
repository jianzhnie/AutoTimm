"""Exponential Scheduler.

Exponential LR schedule with warmup.
"""
import math

import numpy as np


class ExponentialLRScheduler():
    """Exponential learning rate decay scheduler."""

    def __init__(self,
                 optimizer,
                 base_lr,
                 epochs,
                 warmup_length,
                 final_multiplier=0.001,
                 decay_factor=None,
                 decay_step=1,
                 logger=None):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.epochs = epochs
        self.warmup_length = warmup_length
        self.final_multiplier = final_multiplier
        self.decay_factor = decay_factor
        self.decay_step = decay_step

    def step(self, epoch):
        es = self.epochs - self.warmup_length

        if self.decay_factor is not None:
            epoch_decay = self.decay_factor
        else:
            epoch_decay = np.power(
                2,
                np.log2(self.final_multiplier) /
                math.floor(es / self.decay_step))
        if epoch < self.warmup_length:
            lr = self.base_lr * (epoch + 1) / self.warmup_length
        else:
            e = epoch - self.warmup_length
            lr = self.base_lr * (epoch_decay**math.floor(e / self.decay_step))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
