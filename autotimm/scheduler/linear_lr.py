class LinearLRScheduler():
    """Linear learning rate decay scheduler."""

    def __init__(self, optimizer, base_lr, epochs, warmup_length, logger=None):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.epochs = epochs
        self.warmup_length = warmup_length

    def step(self, epoch):
        if epoch < self.warmup_length:
            lr = self.base_lr * (epoch + 1) / self.warmup_length
        else:
            e = epoch - self.warmup_length
            es = self.epochs - self.warmup_length
            lr = self.base_lr * (1 - (e / es))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
