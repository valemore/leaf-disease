from datetime import datetime
import logging
import subprocess
import time

import numpy as np
from pathlib import Path

import timm
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from leaf.dta import LeafDataLoader


num_classes = 5


def path_maybe(pth):
    return pth if isinstance(pth, str) else Path(pth)


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
    def __init__(self, arch, model_prefix=None, save_dir=None, logging_dir=None):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model_prefix = model_prefix if model_prefix is not None else f"model_{datetime.now().strftime('%b%d_%H-%M-%S')}"
        self.save_dir = save_dir
        if logging_dir:
            self.logging_dir = path_maybe(logging_dir) / self.model_prefix
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = None
        self.scheduler = None

        if arch in ["tf_efficientnet_b4_ns"]:
            self.model = timm.create_model(model_name=arch, num_classes=num_classes, pretrained=True)
        else:
            raise Exception(f"Unknown architecture name {arch}!")


    def update_optimizer_scheduler(self, optimizer, scheduler):
        self.optimizer = optimizer
        self.scheduler = scheduler


    def _prepare_logging(self):
        assert self.logging_dir is not None, "Logging directory is not defined!"
        self.logging_dir.mkdir(exist_ok=True)

    def _prepare_output_dir(self):
        assert self.save_dir is not None, "Output directory is not defined!"
        self.save_dir.mkdir(exist_ok=True)


    def log_training(self, running_loss, training_acc, logging_steps, global_step, optimizer, scheduler, tb_writer):
        """Logs training error and f1 using tensorboard"""
        logging_loss, logging_acc = running_loss / logging_steps, training_acc / logging_steps
        print('Step %5d running train loss: %.3f, running train acc: %.3f' %
              (global_step, logging_loss, training_acc))
        tb_writer.add_scalar("loss/train", logging_loss, global_step)
        tb_writer.add_scalar("acc/train", training_acc, global_step)
        if scheduler is not None:
            tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
        else:
            tb_writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)

    def log_validation(self, val_loss, val_acc, step, tb_writer):
        print("Validation after step %5d: val loss %.3f, val acc %.3f" % (step, val_loss, val_acc))
        tb_writer.add_scalar("loss/val", val_loss, step)
        tb_writer.add_scalar("acc/val", val_acc, step)

    def save_checkpoint(self, checkpoint_name, optimizer=None, epoch=None, global_step=None, loss=None):
        self._prepare_output_dir()
        output_dir = self.save_dir / self.model_prefix
        output_dir.mkdir(exist_ok=True)
        log_commit(output_dir / "commit.txt")
        save_model(self, epoch, global_step, loss, output_dir / checkpoint_name)





def train_one_epoch(leaf_model: LeafModel, data_loader: LeafDataLoader, log_steps=None, val_data_loader=None, save_at_log_steps=False, epoch_name="", max_steps=None):
    if log_steps is not None:
        assert val_data_loader is not None, "Need to provide additional data loader when performing logging and validation during training!"
        leaf_model._prepare_logging()
    print(f"Training one epoch ({epoch_name}) with a total of {len(data_loader)} steps...")

    # Logging setup
    if log_steps:
        running_loss = 0.0
        running_i = 0
        running_preds = torch.zeros(log_steps * data_loader.batch_size, dtype=int)
        running_labels = torch.zeros(log_steps * data_loader.batch_size, dtype=int)
        logging_prefix = f"train_{epoch_name}_{datetime.now().strftime('%b%d_%H-%M-%S')}"
        leaf_model._prepare_logging()
        log_commit(leaf_model.logging_dir / "commit.txt")
        tb_writer = SummaryWriter(log_dir=leaf_model.logging_dir / logging_prefix)
    if save_at_log_steps:
        epoch_name = f"{datetime.now().strftime('%b%d_%H-%M-%S')}" if epoch_name == "" else epoch_name

    leaf_model.model.train()
    leaf_model.model = leaf_model.model.to(leaf_model.device)
    step = 1

    tic = time.time()
    for imgs, labels in tqdm(data_loader):
        if max_steps and step > max_steps:
            break
        imgs = imgs.to(leaf_model.device)
        labels = labels.to(leaf_model.device)

        leaf_model.optimizer.zero_grad()

        logits = leaf_model.model.forward(imgs)
        loss = leaf_model.loss_fn(logits, labels)
        loss.backward()
        leaf_model.optimizer.step()
        if leaf_model.scheduler is not None:
            leaf_model.scheduler.step()

        # Metrics and logging
        if log_steps:
            running_loss += loss.mean().item()
            with torch.no_grad():
                running_preds[running_i:(running_i + logits.shape[0])] = logits.argmax(axis=-1)
                running_labels[running_i:(running_i + logits.shape[0])] = labels
                running_i += logits.shape[0]
                if log_steps is not None and step % log_steps == 0:
                    # Record loss & acc on training set (over last logging_steps) and validation set
                    training_acc = (running_preds == running_labels).sum().item() / running_i
                    leaf_model.log_training(running_loss, training_acc, log_steps, step, leaf_model.optimizer, leaf_model.scheduler, tb_writer)
                    running_loss, running_i = 0.0, 0
                    running_preds.fill_(0)
                    running_labels.fill_(0)

                    val_loss, val_acc = validate_one_epoch(leaf_model, val_data_loader)
                    leaf_model.log_validation(val_loss, val_acc, step, tb_writer)

                    # Save model
                    if save_at_log_steps:
                        checkpoint_name = f"checkpoint_{epoch_name}_{step}"
                        leaf_model.save_checkpoint(checkpoint_name, leaf_model.optimizer, epoch_name, step, running_loss)

            print(f"Time to step {step} took {(time.time() - tic):.1f} sec")
        step += 1


def validate_one_epoch(leaf_model: LeafModel, data_loader):
    leaf_model.model.eval()
    with torch.no_grad():
        logits_all = torch.zeros((data_loader.num_padded_samples, 5), dtype=float, device=leaf_model.device)
        labels_all = torch.zeros((data_loader.num_padded_samples), dtype=int, device=leaf_model.device)
        i = 0
        for imgs, labels in tqdm(data_loader):
            imgs = imgs.to(leaf_model.device)
            labels = labels.to(leaf_model.device)
            bs = imgs.shape[0]
            logits_all[i:(i + bs), :] = leaf_model.model.forward(imgs)
            labels_all[i:(i + bs)] = labels
            i += bs

        val_loss = leaf_model.loss_fn(logits_all, labels_all)
        preds_all = logits_all.argmax(axis=-1)
        val_acc = (labels_all == preds_all).sum().item() / i

        return val_loss, val_acc


def warmup(leaf_model: LeafModel, data_loader: LeafDataLoader, n_steps, learning_rate, log_warmup=False):
    print(f"Doing warmup for {n_steps} steps and learning rate {learning_rate}")
    leaf_model.model.train()
    leaf_model.model = leaf_model.model.to(leaf_model.device)

    if log_warmup:
        logging_prefix = f"warmup_{datetime.now().strftime('%b%d_%H-%M-%S')}"
        leaf_model._prepare_logging()
        tb_writer = SummaryWriter(log_dir=leaf_model.logging_dir / logging_prefix)

    optimizer = torch.optim.Adam(leaf_model.model.parameters(), lr=learning_rate)
    tic = time.time()
    steps = 0
    for imgs, labels in tqdm(data_loader, total=min(len(data_loader), n_steps)):
        if steps == n_steps:
            break
        leaf_model.model.train()
        imgs = imgs.to(leaf_model.device)
        labels = labels.to(leaf_model.device)

        optimizer.zero_grad()

        logits = leaf_model.model.forward(imgs)
        loss = leaf_model.loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        if log_warmup:
            tb_writer.add_scalar("loss/warmup", loss, steps)

        steps += 1

    print(f"Warmed up for {steps} steps using learning rate {learning_rate}, taking {(time.time() - tic):.1f} sec")

