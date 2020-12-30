from pathlib import Path
import time
from datetime import datetime
from tqdm import tqdm
#from docopt import docopt
import json
import subprocess
import shutil
import math

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

import timm
# from efficientnet_pytorch import EfficientNet

from leaf.dta import LeafDataset, LeafIterableDataset, GetPatches, TransformPatches, RandomGreen, LeafDataLoader

# Transforms with normalizations for imagenet
data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        RandomGreen(224, 224),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def save_model(model, optimizer, epoch, global_step, loss, fname):
    """Save model & optimizer state together with epoch, global step and running loss to fname."""
    torch.save({
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, fname)


def log_training(running_loss, training_acc, logging_steps, global_step, learning_rate, scheduler, lr_policy, tb_writer):
    """Logs training error and f1 using tensorboard"""
    logging_loss, logging_acc = running_loss / logging_steps, training_acc / logging_steps
    print('Global step %5d running train loss: %.3f, running train acc: %.3f' %
          (global_step, logging_loss, training_acc))
    tb_writer.add_scalar("loss/train", logging_loss, global_step)
    tb_writer.add_scalar("acc/train", training_acc, global_step)
    if scheduler is not None:
        tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
    elif learning_rate is None:
        raise


def validate_model(model, val_dataloader, global_step, training_start_time, tb_writer):
    """Runs model on validation set and writes results using tensorboard"""
    model.eval()
    logits_all = torch.zeros((val_dataloader.num_padded_samples, 5), dtype=float)
    labels_all = torch.zeros((val_dataloader.num_padded_samples), dtype=int)
    i = 0
    for imgs, labels in tqdm(val_dataloader):
        imgs = imgs.to(device)
        labels = labels.to(device)
        bs = imgs.shape[0]
        logits_all[i:(i+bs), :] = model.forward(imgs)
        labels_all[i:(i+bs)] = labels
        i += bs

    val_loss = F.cross_entropy(logits_all, labels_all)
    preds_all = logits_all.cpu().numpy().argmax(axis=-1)
    labels_all = labels_all.cpu().numpy()
    val_acc = np.sum(labels_all == preds_all) / i

    print("Time to global step %d took %.1f sec, val loss %.3f, val acc %.3f" % (
        global_step, time.time() - training_start_time, val_loss, val_acc))

    tb_writer.add_scalar("loss/val", val_loss, global_step)
    tb_writer.add_scalar("acc/val", val_acc, global_step)


if __name__ == "__main__":
    save_dir = Path("/mnt/hdd/leaf-disease-outputs")
    save_dir.mkdir(exist_ok=True)
    logging_dir = Path("/mnt/hdd/leaf-disease-runs")
    logging_dir.mkdir(exist_ok=True)

    n_epochs = 5
    T_0 = 10
    batch_size = 16
    val_batch_size = 32
    learning_rate = 1e-4
    min_learning_rate = 1e-6
    final_layers_lr = 0.0
    weight_decay = 1e-6
    final_layers_wd = 0.0
    timm_args ={
        "model_name": 'tf_efficientnet_b4_ns',
        "num_classes": 5
    }
    # efficientnet_args = {
    #     "model_name": 'efficientnet-b4',
    #     "num_classes": 5
    # }

    num_workers = 4

    logging_steps = 700
    save_checkpoints = True

    train_dset = LeafDataset("./data/train_images", "./data/train_images/labels.csv", transform=data_transforms["train"])
    val_dset = LeafDataset("./data/patches_val", "./data/patches_val/labels.csv", extended_labels=True, transform=data_transforms["val"])

    train_dataloader = LeafDataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = LeafDataLoader(val_dset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device("cuda")

    model = timm.create_model(**timm_args, pretrained=True)
    #model = EfficientNet.from_pretrained(**efficientnet_args)

    # Allow for different learning rates/regularization strenghts for final layers
    # TODO

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    lr_policy = "CosineAnnealingWarmRestarts"
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=1, eta_min=min_learning_rate, last_epoch=-1)
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=len(train_dataloader), epochs=n_epochs)

    # Output and loggind directories
    model_prefix = f"{datetime.now().strftime('%b%d_%H-%M-%S')}"
    output_dir = save_dir / model_prefix
    output_dir.mkdir(exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)


    # Set up logging
    cfg_dict = {
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "min_learning_rate": min_learning_rate,
        "l2_weight_decay": weight_decay,
        "timm_args": timm_args
    }
    with open(output_dir / "cfg_dict.json", "w") as f:
        json.dump(cfg_dict, f)

    script_path = Path(__file__)
    shutil.copy(Path(__file__), output_dir / script_path.name)

    git_cmd_result = subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE)

    with open(output_dir / "commit.txt", "w") as f:
        f.write(git_cmd_result.stdout.decode("utf-8"))

    tb_writer = SummaryWriter(log_dir=logging_dir / model_prefix)

    running_loss = 0.0
    running_i = 0
    running_preds = np.zeros(logging_steps * batch_size, dtype=int)
    running_labels = np.zeros(logging_steps * batch_size, dtype=int)

    model.train()
    model = model.to(device)
    global_step = 1
    for epoch in range(n_epochs):
        tic = time.time()
        for imgs, labels in tqdm(train_dataloader):
            model.train()
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model.forward(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Metrics and logging
            running_loss += loss.mean().item()
            with torch.no_grad():
                running_preds[running_i:(running_i + logits.shape[0])] = logits.detach().cpu().numpy().argmax(axis=-1)
                running_labels[running_i:(running_i + logits.shape[0])] = labels.detach().cpu().numpy()
                running_i += logits.shape[0]
                if global_step % logging_steps == 0:
                    # Record loss & acc on training set (over last logging_steps) and validation set
                    training_acc = np.sum((running_preds == running_labels)) / running_i
                    log_training(running_loss, training_acc, logging_steps, global_step, learning_rate, scheduler, lr_policy,
                                 tb_writer)
                    running_loss, running_i = 0.0, 0
                    running_preds.fill(0)
                    running_labels.fill(0)

                    validate_model(model, val_dataloader, global_step, tic, tb_writer)

                    # Save model
                    if save_checkpoints:
                        save_model(model, optimizer, epoch, global_step, running_loss,
                                   output_dir / "checkpoints" / f"{global_step}.pt")
            global_step += 1


        # Validation at end of each epoch
        with torch.no_grad():
            validate_model(model, val_dataloader, global_step, tic, tb_writer)
        # Save model
        if save_checkpoints:
            save_model(model, optimizer, epoch, global_step, running_loss, output_dir / "checkpoints" / f"{global_step}.pt")

    tb_writer.close()

    # Save model
    save_model(model, optimizer, epoch, global_step, running_loss, output_dir / f"final.pt")
