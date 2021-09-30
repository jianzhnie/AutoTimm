class StepLRScheduler():
    """step learning rate scheduler."""

    def __init__(self,
                 optimizer,
                 base_lr,
                 steps,
                 decay_factor,
                 warmup_length,
                 logger=None):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.steps = steps
        self.decay_factor = decay_factor
        self.warmup_length = warmup_length

    def step(self, epoch):
        if epoch < self.warmup_length:
            lr = self.base_lr * (epoch + 1) / self.warmup_length
        else:
            lr = self.base_lr
            for s in self.steps:
                if epoch >= s:
                    lr *= self.decay_factor
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
