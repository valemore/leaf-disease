from torch.optim.lr_scheduler import CyclicLR

def get_warmup_scheduler(optimizer, min_lr, max_lr, warmup_steps):
    if float(min_lr) == 0.0:
        min_lr = 1e-12
    scheduler = CyclicLR(optimizer, base_lr=min_lr, max_lr=max_lr, step_size_up=warmup_steps)
    return scheduler
