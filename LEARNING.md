- Configure: argparse, configuration dict
- How to best save? With tensorboard? Neptune? something else?
- exponential moving averages for loss, acc
- reduction can be done with pytorch function

https://github.com/Kaixhin/grokking-pytorch


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr