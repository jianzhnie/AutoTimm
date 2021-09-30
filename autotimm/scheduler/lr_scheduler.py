import math

import numpy as np


def lr_policy(lr_fn, logger=None):

    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return _alr


def lr_step_policy(base_lr, steps, decay_factor, warmup_length, logger=None):

    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            lr = base_lr
            for s in steps:
                if epoch >= s:
                    lr *= decay_factor
        return lr

    return lr_policy(_lr_fn, logger=logger)


def lr_linear_policy(base_lr, warmup_length, epochs, logger=None):

    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = base_lr * (1 - (e / es))
        return lr

    return lr_policy(_lr_fn, logger=logger)


def lr_cosine_policy(base_lr, warmup_length, epochs, end_lr=0, logger=None):

    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = end_lr + (0.5 * (1 + np.cos(np.pi * e / es)) *
                           (base_lr - end_lr))
        return lr

    return lr_policy(_lr_fn, logger=logger)


def lr_exponential_policy(base_lr,
                          warmup_length,
                          epochs,
                          final_multiplier=0.001,
                          decay_factor=None,
                          decay_step=1,
                          logger=None):
    """Exponential lr policy.

    Setting decay factor parameter overrides final_multiplier
    """
    es = epochs - warmup_length

    if decay_factor is not None:
        epoch_decay = decay_factor
    else:
        epoch_decay = np.power(
            2,
            np.log2(final_multiplier) / math.floor(es / decay_step))

    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            lr = base_lr * (epoch_decay**math.floor(e / decay_step))
        return lr

    return lr_policy(_lr_fn, logger=logger)
