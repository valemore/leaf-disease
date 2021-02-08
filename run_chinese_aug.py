from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd

import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torchvision import transforms

from leaf.dta import LeafDataset, LeafIterableDataset, LeafDataLoader, LeafDataLoader, get_leaf_splits
from leaf.model import LeafModel, train_one_epoch, validate_one_epoch

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CyclicLR

from torch.utils.tensorboard import SummaryWriter

from albumentations import (
    Compose,
    Resize,
    OneOf,
    RandomBrightness,
    RandomContrast,
    MotionBlur,
    MedianBlur,
    GaussianBlur,
    VerticalFlip,
    HorizontalFlip,
    ShiftScaleRotate,
    Normalize,
    RandomBrightnessContrast
)

from albumentations.pytorch.transforms import ToTensorV2

if __name__ == "__main__":
    img_height = 380
    img_width = 380

    # Transforms with normalizations for imagenet adapted from https://github.com/alipay/cvpr2020-plant-pathology
    train_transforms = Compose(
        [
            Resize(height=img_height, width=img_width),
            OneOf([RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.0, p=1), RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=0.1, p=1)]),
            # OneOf([RandomBrightness(limit=0.1, p=1), RandomContrast(limit=0.1, p=1)]),
            OneOf([MotionBlur(blur_limit=3), MedianBlur(blur_limit=3), GaussianBlur(blur_limit=3),], p=0.5,),
            VerticalFlip(p=0.5),
            HorizontalFlip(p=0.5),
            ShiftScaleRotate(
                shift_limit=0.2,
                scale_limit=0.2,
                rotate_limit=20,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT_101,
                p=1,
            ),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
            ToTensorV2()
        ]
    )

    val_transforms = Compose(
        [
            Resize(height=img_height, width=img_width),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
            ToTensorV2()
        ]
    )

    output_dir = Path("/home/jupyter/outputs")
    output_dir.mkdir(exist_ok=True)
    logging_dir = Path("/home/jupyter/runs")
    logging_dir.mkdir(exist_ok=True)

    num_workers = 0

    batch_size = 18
    val_batch_size = 36

    log_steps = 500

    num_epochs = 36
    min_lr = 1e-4
    max_lr = 1e-2
    weight_decay = 1e-5
    momentum = 0.9

    train_dset = LeafDataset("data/train_images", "data/train_images/labels.csv", transform=train_transforms)
    val_dset = LeafDataset("data/val_images", "data/val_images/labels.csv", transform=val_transforms)

    train_dataloader = LeafDataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = LeafDataLoader(val_dset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)

    leaf_model = LeafModel("tf_efficientnet_b4_ns", model_prefix="tf_efficientnet_b4_ns", output_dir=output_dir, logging_dir=logging_dir)
    optimizer = SGD(leaf_model.model.parameters(), lr=min_lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = CyclicLR(optimizer, base_lr=min_lr, max_lr=max_lr, step_size_up=2*len(train_dataloader), mode='triangular')
    leaf_model.update_optimizer_scheduler(optimizer, scheduler)

    #torch.save(leaf_model.model.state_dict(), "inital_state_dict.pth")

    run_name = f"{leaf_model.model_prefix}_chinese_augs_Jan24_wd1e5"
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
        leaf_model.save_checkpoint(f"{epoch_name}", epoch_name=epoch)
