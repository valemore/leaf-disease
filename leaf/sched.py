import warnings

from torch.optim.lr_scheduler import CyclicLR, LambdaLR, StepLR, _LRScheduler

def get_warmup_scheduler(optimizer, min_lr, max_lr, warmup_steps):
    if float(min_lr) == 0.0:
        min_lr = 1e-12
    scheduler = CyclicLR(optimizer, base_lr=min_lr, max_lr=max_lr, step_size_up=warmup_steps)
    return scheduler


class LinearLR(_LRScheduler):
    def __init__(self, optimizer, start_lr, stop_lr, num_steps, last_epoch=-1, verbose=False):
        self.start_lr = start_lr
        self.stop_lr = stop_lr
        self.num_steps = num_steps
        self.step_size = (stop_lr - start_lr) / num_steps
        self.optimizer = optimizer
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        return [base_lr + self.step_size * self.last_epoch for base_lr in self.base_lrs]


def get_up_scheduler(optimizer, start_lr, stop_lr, n_steps):
    scheduler = CyclicLR(optimizer, base_lr=start_lr, max_lr=stop_lr, step_size_up=n_steps)
    return scheduler


def fix_optimizer_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg["lr"] = lr
        pg["initial_lr"] = lr
