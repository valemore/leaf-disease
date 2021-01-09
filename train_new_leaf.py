# This may be completely obsolete and superseded by the notebooks

from pathlib import Path
from tqdm import tqdm
import math

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import transforms

from leaf.dta import LeafDataset, GetPatches, TransformPatches, RandomGreen, LeafDataLoader
from leaf.model import LeafModel, train_one_epoch, validate_one_epoch, warmup


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
    'warmup': transforms.Compose([
        transforms.ToTensor(),
        RandomGreen(224, 224),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'patches': transforms.Compose([
        transforms.ToTensor()
    ])
}


if __name__ == "__main__":
    save_dir = Path("/mnt/hdd/leaf-disease-outputs")
    save_dir.mkdir(exist_ok=True)
    logging_dir = Path("/mnt/hdd/leaf-disease-runs")
    logging_dir.mkdir(exist_ok=True)

    batch_size = 16
    val_batch_size = 32

    num_workers = 4

    learning_rate = 1e-6
    weight_decay = 1e-6
    warmup_lr = 1e-10

    # n_epochs = 5
    # T_0 = 10
    # batch_size = 16
    # val_batch_size = 32
    # learning_rate = 1e-4
    # min_learning_rate = 1e-6
    # final_layers_lr = 0.0
    # weight_decay = 1e-6
    # final_layers_wd = 0.0
    # timm_args ={
    #     "model_name": 'tf_efficientnet_b4_ns',
    #     "num_classes": 5
    # }
    # # efficientnet_args = {
    # #     "model_name": 'efficientnet-b4',
    # #     "num_classes": 5
    # # }


    patches_logging_steps = 1000
    save_checkpoints = True

    train_dset = LeafDataset("./data/train_images", "./data/train_images/labels.csv", transform=data_transforms["train"])
    patches_dset = LeafDataset("./data/patches_train", "./data/patches_train/labels.csv", extended_labels=True, transform=data_transforms["patches"])
    #warmup_dset = LeafDataset("./data/train_images", "./data/train_images/labels.csv", transform=data_transforms["warmup"])
    val_dset = LeafDataset("./data/patches_val", "./data/patches_val/labels.csv", extended_labels=True, transform=data_transforms["val"])

    train_dataloader = LeafDataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    patches_dataloader = LeafDataLoader(patches_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = LeafDataLoader(val_dset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)

    leaf_model = LeafModel("tf_efficientnet_b4_ns", output_dir=save_dir, logging_dir=logging_dir)

    optimizer = Adam(leaf_model.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=1, eta_min=min_learning_rate, last_epoch=-1)
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=len(train_dataloader), epochs=n_epochs)
    leaf_model.update_optimizer_scheduler(optimizer, None)

    warmup(leaf_model, train_dataloader, 500, warmup_lr, log_warmup=True)
    train_one_epoch(leaf_model, patches_dataloader, log_steps=patches_logging_steps, val_data_loader=val_dataloader, save_at_log_steps=save_checkpoints, epoch_name=1)


