from datetime import datetime
import logging
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
from torch.utils.tensorboard import SummaryWriter

from leaf.dta import LeafDataLoader, get_img_id, get_patch_id


num_classes = 5


def path_maybe(pth):
    return pth if isinstance(pth, str) else Path(pth)


def ema(ma, x, alpha, step):
    if ma is None:
        return x
    return (alpha * x + (1 - alpha) * ma) # / (1 - (1 - alpha) ** step)
    #return (beta * v + (1 - beta) * x) / (1 - beta ** step)


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
    def __init__(self, arch, model_prefix=None, output_dir=None, logging_dir=None, ragged_batches=False):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self._model_prefix = model_prefix if model_prefix is not None else f"model_{datetime.now().strftime('%b%d_%H-%M-%S')}"
        self.output_top_dir = path_maybe(output_dir)
        self.logging_top_dir = path_maybe(logging_dir)
        if not ragged_batches:
            self.loss_fn = nn.CrossEntropyLoss().to(self.device)
        else:
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1).to(self.device)
        self.optimizer = None
        self.scheduler = None

        #if arch in ["tf_efficientnet_b4_ns"]:
        self.model = timm.create_model(model_name=arch, num_classes=num_classes, pretrained=True)
        # else:
        #     raise Exception(f"Unknown architecture name {arch}!")

    @property
    def model_prefix(self):
        return self._model_prefix

    @model_prefix.setter
    def model_prefix(self, value):
        self._model_prefix = value
        self.output_model_dir = self.output_top_dir / self.model_prefix
        self.logging_model_dir = self.logging_top_dir / self.model_prefix

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

    @property
    def logging_top_dir(self):
        return self._logging_top_dir

    @logging_top_dir.setter
    def logging_top_dir(self, value):
        self._logging_top_dir = value
        if self._logging_top_dir is not None:
            self.logging_model_dir = self._logging_top_dir / self._model_prefix
        else:
            self.logging_model_dir = None

    def update_optimizer_scheduler(self, optimizer, scheduler):
        self.optimizer = optimizer
        self.scheduler = scheduler

    def _prepare_logging(self):
        assert self.logging_model_dir is not None, "Logging directory is not defined!"
        self.logging_top_dir.mkdir(exist_ok=True)
        self.logging_model_dir.mkdir(exist_ok=True)

    def _prepare_output_dir(self):
        assert self.output_model_dir is not None, "Output directory is not defined!"
        self.output_top_dir.mkdir(exist_ok=True)
        self.output_model_dir.mkdir(exist_ok=True)

    def get_learning_rate(self):
        return self.optimizer.param_groups[0]["lr"]
        # if self.scheduler is not None:
        #     return self.scheduler.get_last_lr()[0]
        # else:
        #     return self.optimizer.param_groups[0]["lr"]


    def log_training(self, running_loss, training_acc, logging_steps, global_step, tb_writer):
        """Logs training error and f1 using tensorboard"""
        logging_loss, logging_acc = running_loss / logging_steps, training_acc / logging_steps
        print('Step %5d running train loss: %.3f, running train acc: %.3f' %
              (global_step, logging_loss, training_acc))
        tb_writer.add_scalar("loss/train", logging_loss, global_step)
        tb_writer.add_scalar("acc/train", training_acc, global_step)
        tb_writer.add_scalar("lr", self.get_learning_rate(), global_step)

    def log_training_ema(self, loss, acc, step, tb_writer, verbose=False):
        if verbose:
            print('Step %5d EMA train loss: %.3f, EMA train acc: %.3f' %
                  (step, loss, acc))
        tb_writer.add_scalar("loss/train", loss, step)
        tb_writer.add_scalar("acc/train", acc, step)
        tb_writer.add_scalar("lr", self.get_learning_rate(), step)

    def log_validation(self, val_loss, val_acc, step, tb_writer, verbose=True):
        if verbose:
            print("Validation after step %5d: val loss %.3f, val acc %.3f" % (step, val_loss, val_acc))
        tb_writer.add_scalar("loss/val", val_loss, step)
        tb_writer.add_scalar("acc/val", val_acc, step)

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


def train_one_epoch(leaf_model: LeafModel, data_loader: LeafDataLoader, log_steps=None, val_data_loader=None, save_at_log_steps=False, epoch_name="", max_steps=None, ema_steps=None):
    if log_steps is not None:
        leaf_model._prepare_logging()
    print(f"Training one epoch ({epoch_name}) with a total of {len(data_loader)} steps...")

    # Logging setup
    if log_steps:
        logging_prefix = f"{epoch_name}"
        #logging_prefix = f"{epoch_name}_{datetime.now().strftime('%b%d_%H-%M-%S')}"
        leaf_model._prepare_logging()
        log_commit(leaf_model.logging_model_dir / "commit.txt")
        tb_writer = SummaryWriter(log_dir=leaf_model.logging_model_dir / logging_prefix)
        if isinstance(log_steps, int):
            alpha = 2.0 / (log_steps + 1)
        else:
            alpha = 2.0 / (ema_steps + 1)
        #beta = min(1.0 - 1.0 / log_steps, 0.9)
        running_loss = None
        running_acc = None

    if save_at_log_steps:
        epoch_name = f"{datetime.now().strftime('%b%d_%H-%M-%S')}" if epoch_name == "" else epoch_name

    leaf_model.model.train()
    leaf_model.model = leaf_model.model.to(leaf_model.device)
    step = 1

    tic = time.time()
    for imgs, labels, idxs in tqdm(data_loader):
        if max_steps and step > max_steps:
            break
        imgs = imgs.to(leaf_model.device, non_blocking=True)
        labels = labels.to(leaf_model.device, non_blocking=True)

        leaf_model.optimizer.zero_grad()

        logits = leaf_model.model.forward(imgs)
        loss = leaf_model.loss_fn(logits, labels)
        loss.backward()
        leaf_model.optimizer.step()
        if leaf_model.scheduler is not None:
            leaf_model.scheduler.step()

        # Metrics and logging
        if log_steps:
            with torch.no_grad():
                running_loss = ema(running_loss, loss.mean().item(), alpha, step)
                preds = logits.argmax(axis=-1)
                acc = (preds == labels).sum().item() / preds.shape[0]
                running_acc = ema(running_acc, acc, alpha, step)
                leaf_model.log_training_ema(running_loss, running_acc, step, tb_writer)

            if isinstance(log_steps, int) and step % log_steps == 0:
                print('Step %5d EMA train loss: %.3f, EMA train acc: %.3f' % (step, running_loss, running_acc))
                if val_data_loader is not None:
                    if data_loader.max_samples_per_image > 1:
                        val_loss, val_raw_acc, val_logits_mean_acc, val_probs_mean_acc = validate_one_epoch(leaf_model, val_data_loader)
                        print(f"Validation after step %5d: loss {val_loss}, raw acc {val_raw_acc}, logits mean acc {val_logits_mean_acc}, probs mean acc {val_probs_mean_acc}")
                        tb_writer.add_scalar("loss/val", val_loss, step)
                        tb_writer.add_scalar("raw_acc/val", val_raw_acc, step)
                        tb_writer.add_scalar("logits_mean_acc/val", val_logits_mean_acc, step)
                        tb_writer.add_scalar("probs_mean_acc/val", val_probs_mean_acc, step)
                    else:
                        val_loss, val_acc = validate_one_epoch(leaf_model, val_data_loader)
                        print(f"Validation after step %5d: loss {val_loss}, acc {val_acc}")
                        tb_writer.add_scalar("loss/val", val_loss, step)
                        tb_writer.add_scalar("acc/val", val_acc, step)

                # Save model
                if save_at_log_steps:
                    checkpoint_name = f"checkpoint_{epoch_name}_{step}"
                    leaf_model.save_checkpoint(checkpoint_name, epoch_name, step, running_loss)

                print(f"Time to step {step} took {(time.time() - tic):.1f} sec")
        step += 1


def softmax(x):
    return np.exp(x)/sum(np.exp(x))


def validate_one_epoch(leaf_model: LeafModel, data_loader):
    ensemble_patches = data_loader.max_samples_per_image > 1
    leaf_model.model.eval()
    with torch.no_grad():
        logits_all = torch.zeros((data_loader.num_padded_samples, 5), dtype=float, device=leaf_model.device)
        labels_all = torch.zeros((data_loader.num_padded_samples), dtype=int, device=leaf_model.device)
        if ensemble_patches:
            img_ids = np.zeros(data_loader.num_padded_samples, dtype=int)
        #patch_ids = np.zeros(data_loader.num_padded_samples, dtype=int)
        i = 0
        for imgs, labels, idxs in tqdm(data_loader):
            imgs = imgs.to(leaf_model.device)
            labels = labels.to(leaf_model.device)
            bs = imgs.shape[0]
            logits_all[i:(i + bs), :] = leaf_model.model.forward(imgs)
            labels_all[i:(i + bs)] = labels
            if ensemble_patches:
                img_ids[i:(i + bs)] = np.array([data_loader.dataset.img_ids[idx] for idx in idxs])
            #patch_ids[i:(i + bs)] = np.array([get_patch_id(data_loader.dataset.fnames[idx]) for idx in idxs])
            i += bs

        loss = leaf_model.loss_fn(logits_all, labels_all)
        raw_preds_all = logits_all.argmax(axis=-1)
        raw_acc = (labels_all == raw_preds_all).sum().item() / i

        if ensemble_patches:
            logits_df = pd.DataFrame({
                "img_id": img_ids,
                #"patch_idx": patch_ids,
                "label": labels_all.cpu().numpy().astype(int),
                "logits_0": logits_all.cpu().numpy()[:, 0],
                "logits_1": logits_all.cpu().numpy()[:, 1],
                "logits_2": logits_all.cpu().numpy()[:, 2],
                "logits_3": logits_all.cpu().numpy()[:, 3],
                "logits_4": logits_all.cpu().numpy()[:, 4]
                # "raw_pred": raw_preds_all.cpu().numpy().astype(int)
            })

            logits_mean_df = logits_df.groupby("img_id").agg(
                {"label": "first", "logits_0": "mean", "logits_1": "mean", "logits_2": "mean", "logits_3": "mean", "logits_4": "mean"}).reset_index()
            logits_mean_preds = logits_mean_df.loc[:, ["logits_0", "logits_1", "logits_2", "logits_3", "logits_4"]].values.argmax(-1)
            logits_mean_acc = (logits_mean_df["label"].values == logits_mean_preds).sum() / len(logits_mean_preds)

            probs_df = pd.concat([logits_df.loc[:, ["img_id", "label"]],
                                  pd.DataFrame(np.apply_along_axis(softmax, 1, logits_df.loc[:, ["logits_0", "logits_1", "logits_2", "logits_3", "logits_4"]].values))], axis=1)
            probs_df.rename(columns={i: f"probs_{i}" for i in range(5)}, inplace=True)
            probs_mean_df = probs_df.groupby("img_id").agg(
                {"label": "first", "probs_0": "mean", "probs_1": "mean", "probs_2": "mean", "probs_3": "mean", "probs_4": "mean"}).reset_index()
            probs_mean_preds = probs_mean_df.loc[:, ["probs_0", "probs_1", "probs_2", "probs_3", "probs_4"]].values.argmax(-1)
            probs_mean_acc = (probs_mean_df["label"].values == probs_mean_preds).sum() / len(probs_mean_preds)

            return loss, raw_acc, logits_mean_acc, probs_mean_acc

        return loss, raw_acc



def warmup(leaf_model: LeafModel, data_loader: LeafDataLoader, n_steps, log_warmup=False):
    tic = time.time()
    print(f"Doing warmup for {n_steps} steps and learning rate {leaf_model.get_learning_rate()}")
    train_one_epoch(leaf_model, data_loader, log_steps=n_steps, epoch_name=f"warmup_{datetime.now().strftime('%b%d_%H-%M-%S')}", max_steps=n_steps)
    print(f"Warmed up for {n_steps} steps using learning rate {leaf_model.get_learning_rate()}, taking {(time.time() - tic):.1f} sec")

