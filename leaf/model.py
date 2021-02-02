import contextlib
from datetime import datetime
import subprocess
import time

import numpy as np
import pandas as pd
from pathlib import Path

import timm
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from leaf.dta import LeafDataLoader, get_img_id
from leaf.label_smoothing import TaylorCrossEntropyLoss
from leaf.cutmix_utils import CutMixCrossEntropyLoss


def path_maybe(pth):
    if pth is None:
        return None
    assert isinstance(pth, (str, Path)), "Expected string or Path object!"
    return Path(pth)


def ema(ma, x, alpha):
    if ma is None:
        return x
    return (alpha * x + (1 - alpha) * ma)


def save_model(leaf_model, epoch_name, global_step, loss, fname):
    """Save model & optimizer state together with epoch, global step and running loss to fname."""
    if leaf_model.optimizer:
        optimizer_state_dict = leaf_model.optimizer.state_dict()
    else:
        optimizer_state_dict = None
    if leaf_model.scheduler:
        scheduler_state_dict = leaf_model.scheduler.state_dict()
    else:
        scheduler_state_dict = None

    torch.save({
        'epoch_name': epoch_name,
        'global_step': global_step,
        'model_state_dict': leaf_model.model.state_dict(),
        'optimizer_state_dict': optimizer_state_dict,
        'scheduler_state_dict': scheduler_state_dict,
        'loss': loss
    }, fname)


def log_commit(fname):
    git_cmd_result = subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE)
    with open(fname, "w") as f:
        f.write(git_cmd_result.stdout.decode("utf-8"))

class LeafModel(object):
    def __init__(self, CFG, model_prefix=None, output_dir=None, pretrained=True):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self._model_prefix = model_prefix if model_prefix is not None else f"{CFG.arch}_hero_{datetime.now().strftime('%b%d_%H-%M-%S')}"
        self.output_top_dir = path_maybe(output_dir)
        self.acc_logging = True
        if CFG.loss_fn == "CrossEntropyLoss":
            self.loss_fn = nn.CrossEntropyLoss().to(self.device)
        elif CFG.loss_fn == "TaylorCrossEntropyLoss":
            self.loss_fn = TaylorCrossEntropyLoss(smoothing=CFG.smoothing, target_size=CFG.num_classes).to(self.device)
        elif CFG.loss_fn == "CutMixCrossEntropyLoss":
            self.loss_fn = CutMixCrossEntropyLoss().to(self.device)
            self.acc_logging = False
        self.optimizer = None
        self.scheduler = None

        self.model = timm.create_model(model_name=CFG.arch, num_classes=CFG.num_classes, pretrained=pretrained)

    @property
    def model_prefix(self):
        return self._model_prefix

    @model_prefix.setter
    def model_prefix(self, value):
        self._model_prefix = value
        self.output_model_dir = self.output_top_dir / self.model_prefix

    @property
    def output_top_dir(self):
        return self._output_top_dir

    @output_top_dir.setter
    def output_top_dir(self, value):
        self._output_top_dir = value
        if self._output_top_dir is not None:
            self.output_model_dir = self._output_top_dir / self._model_prefix
        else:
            self.output_model_dir = None

    def update_optimizer_scheduler(self, optimizer, scheduler):
        self.optimizer = optimizer
        self.scheduler = scheduler

    def _prepare_output_dir(self):
        assert self.output_model_dir is not None, "Output directory is not defined!"
        self.output_top_dir.mkdir(exist_ok=True)
        self.output_model_dir.mkdir(exist_ok=True)

    def get_learning_rate_momentum(self):
        return self.optimizer.param_groups[0]["lr"], self.optimizer.param_groups[0].get("momentum", None)

    def log_training_ema(self, loss, acc, gn, step, neptune, verbose=False):
        if verbose:
            print('Step %5d EMA train loss: %.3f, train acc: %.3f, grad norm %.3f' %
                  (step, loss, acc if acc is not None else -1.0, gn))
        neptune.log_metric("loss/train", y=loss, x=step)
        if acc is not None:
            neptune.log_metric("acc/train", y=acc, x=step)
        lr, momentum = self.get_learning_rate_momentum()
        neptune.log_metric("lr", y=lr, x=step)
        if momentum is not None:
            neptune.log_metric("momentum", y=momentum, x=step)
        if gn is not None:
            neptune.log_metric("grad", y=gn, x=step)


    def log_validation(self, val_loss, val_acc, step, neptune, verbose=True):
        if verbose:
            print("Validation after step %5d: val loss %.3f, val acc %.3f" % (step, val_loss, val_acc))
        neptune.log_metric("loss/val", y=val_loss, x=step)
        neptune.log_metric("acc/val", y=val_acc, x=step)

    def save_checkpoint(self, checkpoint_name, epoch=None, global_step=None, loss=None):
        self._prepare_output_dir()
        log_commit(self.output_model_dir / "commit.txt")
        save_model(self, epoch, global_step, loss, self.output_model_dir / checkpoint_name)

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(self.output_model_dir / checkpoint_name)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    def load_checkpoint_from_file(self, fname):
        checkpoint = torch.load(fname)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    def load_model_state_dict(self, fname):
        model_state_dict = torch.load(fname)
        self.model.load_state_dict(model_state_dict)
        self.model = self.model.to(self.device)


def train_one_epoch(leaf_model: LeafModel, data_loader: LeafDataLoader, log_steps=None, val_data_loader=None, save_at_log_steps=False, epoch_name="", max_steps=None, ema_steps=None, epoch=None, neptune=None, fp16=True, grad_norm=None):
    print(f"Training one epoch ({epoch_name}) with a total of {len(data_loader)} steps...")
    if epoch is None:
        epoch = 1

    loss_list, acc_list = [], []

    # Logging setup
    if log_steps:
        if not isinstance(ema_steps, int):
            assert isinstance(log_steps, int), "log_steps must be int if ema_steps is not given!"
            ema_steps = log_steps
        alpha = 2.0 / (ema_steps + 1)

        running_loss = None
        running_acc = None
        running_gn = None
        gn = None

    if save_at_log_steps:
        epoch_name = f"{datetime.now().strftime('%b%d_%H-%M-%S')}" if epoch_name == "" else epoch_name

    if fp16:
        scaler = torch.cuda.amp.GradScaler()
        input_dtype = torch.half
    else:
        input_dtype = torch.float

    if isinstance(leaf_model.loss_fn, CutMixCrossEntropyLoss):
        label_dtype = torch.half if fp16 else torch.float
    else:
        label_dtype = torch.long

    leaf_model.model.train()
    leaf_model.model = leaf_model.model.to(leaf_model.device)
    step = 1

    tic = time.time()
    for imgs, labels, _ in tqdm(data_loader):
        if max_steps and step > max_steps:
            break
        imgs = imgs.to(leaf_model.device, dtype=input_dtype, non_blocking=True)
        labels = labels.to(leaf_model.device, dtype=label_dtype, non_blocking=True)

        leaf_model.optimizer.zero_grad()

        if fp16:
            with torch.cuda.amp.autocast():
                logits = leaf_model.model.forward(imgs)
                loss = leaf_model.loss_fn(logits, labels)
            scaler.scale(loss).backward()
            if grad_norm is not None:
                gn = torch.nn.utils.clip_grad_norm_(leaf_model.model.parameters(), grad_norm)
            scaler.step(leaf_model.optimizer)
            scaler.update()
        else:
            logits = leaf_model.model.forward(imgs)
            loss = leaf_model.loss_fn(logits, labels)
            loss.backward()
            if grad_norm is not None:
                gn = torch.nn.utils.clip_grad_norm_(leaf_model.model.parameters(), grad_norm)
            leaf_model.optimizer.step()

        if leaf_model.scheduler is not None:
            leaf_model.scheduler.step()

        # Metrics and logging
        if log_steps:
            log_step = len(data_loader) * (epoch - 1) + step
            with torch.no_grad():
                running_loss = ema(running_loss, loss.mean().item(), alpha)
                preds = logits.argmax(axis=-1)
                loss_list.append(running_loss)
                if leaf_model.acc_logging:
                    acc = (preds == labels).sum().item() / preds.shape[0]
                    running_acc = ema(running_acc, acc, alpha)
                    acc_list.append(running_acc)
                running_gn = ema(running_gn, gn, alpha)
                if neptune is not None and step >= ema_steps: # Only log in neptune once EMA is smooth enough
                    leaf_model.log_training_ema(running_loss, running_acc, running_gn, log_step, neptune)

            if isinstance(log_steps, int) and step % log_steps == 0:
                print('Step %5d EMA train loss: %.3f, EMA train acc: %.3f' % (step, running_loss, running_acc if running_acc is not None else -1.0))
                if val_data_loader is not None:
                    val_loss, val_acc = validate_one_epoch(leaf_model, val_data_loader)
                    print(f"Validation after step {step * epoch}: loss {val_loss}, acc {val_acc}")
                    if neptune is not None:
                        neptune.log_metric("loss/val", y=val_loss, x=log_step)
                        neptune.log_metric("acc/val", y=val_acc, x=log_step)

                # Save model
                if save_at_log_steps:
                    checkpoint_name = f"checkpoint_{epoch_name}_{step}"
                    leaf_model.save_checkpoint(checkpoint_name, epoch_name, step * epoch, running_loss)

                print(f"Time to step {step} took {(time.time() - tic):.1f} sec")
        step += 1

    return loss_list, acc_list # can be empty


def softmax(x):
    return np.exp(x)/sum(np.exp(x))


def validate_one_epoch(leaf_model: LeafModel, data_loader):
    leaf_model.model.eval()
    with torch.no_grad():
        logits_all = torch.zeros((data_loader.dataset_len, 5), dtype=float, device=leaf_model.device)
        labels_all = torch.zeros((data_loader.dataset_len), dtype=int, device=leaf_model.device)
        i = 0
        for imgs, labels, idxs in tqdm(data_loader):
            imgs = imgs.to(leaf_model.device)
            labels = labels.to(leaf_model.device)
            bs = imgs.shape[0]
            logits_all[i:(i + bs), :] = leaf_model.model.forward(imgs)
            labels_all[i:(i + bs)] = labels
            i += bs

        logits_all  = logits_all[:i]
        labels_all = labels_all[:i]

        loss = leaf_model.loss_fn(logits_all, labels_all)
        preds_all = logits_all.argmax(axis=-1)
        acc = (labels_all == preds_all).sum().item() / i

        return loss, acc



