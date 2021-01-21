from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torchvision import transforms

from leaf.dta import LeafDataset, LeafIterableDataset, LeafDataLoader, LeafDataLoader, get_leaf_splits
from leaf.model import LeafModel, train_one_epoch, validate_one_epoch

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CyclicLR

from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":

    # Transforms with normalizations for imagenet
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=418), # 10 % bigger than input size 380
            transforms.CenterCrop(size=380),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=418), # 10 % bigger than input size 380
            transforms.CenterCrop(size=380),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    output_dir = Path("/mnt/hdd/leaf-disease-outputs")
    output_dir.mkdir(exist_ok=True)
    logging_dir = Path("/mnt/hdd/leaf-disease-runs")
    logging_dir.mkdir(exist_ok=True)

    num_workers = 4

    batch_size = 6
    val_batch_size = 12

    log_steps = 500

    num_epochs = 10
    min_lr = 1e-2
    max_lr = 1
    weight_decay = 0.0
    momentum = 0.9

    train_dset = LeafDataset("data/train_images", "data/train_images/labels.csv", transform=data_transforms["train"])
    val_dset = LeafDataset("data/val_images", "data/val_images/labels.csv", transform=data_transforms["val"])

    train_dataloader = LeafDataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = LeafDataLoader(val_dset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)

    leaf_model = LeafModel("tf_efficientnet_b4_ns", model_prefix="tf_efficientnet_b4_ns", output_dir=output_dir, logging_dir=logging_dir)
    optimizer = SGD(leaf_model.model.parameters(), lr=min_lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = CyclicLR(optimizer, base_lr=min_lr, max_lr=max_lr, step_size_up=num_epochs*len(train_dataloader), mode='triangular')
    leaf_model.update_optimizer_scheduler(optimizer, scheduler)

    #torch.save(leaf_model.model.state_dict(), "inital_state_dict.pth")

    run_name = f"{leaf_model.model_prefix}_find-lr-big-momentum"
    for epoch in range(1, num_epochs+1):
        epoch_name = f"{run_name}-{epoch}"
        #leaf_model.load_model_state_dict("inital_state_dict.pth")
        train_one_epoch(leaf_model, train_dataloader, log_steps=log_steps, epoch_name=epoch_name)
        val_loss, val_acc = validate_one_epoch(leaf_model, val_dataloader)
        print(f"Validation after epoch {epoch}: loss {val_loss}, acc {val_acc}")
        tb_writer = SummaryWriter(log_dir=leaf_model.logging_model_dir / epoch_name)
        val_step = len(train_dataloader)
        tb_writer.add_scalar("loss/val", val_loss, val_step)
        tb_writer.add_scalar("acc/val", val_acc, val_step)
        tb_writer.close()
        leaf_model.save_checkpoint(f"{epoch_name}", epoch=epoch)
