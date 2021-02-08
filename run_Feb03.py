from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import os
import random

import numpy as np

from torch.optim import SGD

from leaf.dta import LeafDataset, LeafIterableDataset, LeafDataLoader, LeafDataLoader, get_leaf_splits
from leaf.model import LeafModel, train_one_epoch, validate_one_epoch
from leaf.sched import get_warmup_scheduler

from torch.optim.lr_scheduler import OneCycleLR

import neptune

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


if __name__ == "__main__":
    on_gcp = os.getcwd() == "/home/jupyter/leaf-disease"
    print(f"Running on {'GCP' if on_gcp else 'local'} machine!")

    @dataclass
    class CFG:
        num_classes: int = 5
        img_size: int = 380
        arch: str = "tf_efficientnet_b4_ns"
        loss_fn: str = "CrossEntropyLoss"
        smoothing: float = 0.00
        crop_para: int = 0

        def __repr__(self):
            return json.dumps(self.__dict__)
    cfg = CFG()

    output_dir = Path("/home/jupyter/outputs") if on_gcp else Path("/mnt/hdd/leaf-disease-outputs")
    output_dir.mkdir(exist_ok=True)

    num_workers = 4

    batch_size = 18 if on_gcp else 12
    val_batch_size = 36 if on_gcp else 24

    log_steps = 200

    min_lr = 8.055822378718028e-4
    #max_lr = 0.015
    max_lr = 0.06190499161193587
    weight_decay = 0.0
    momentum = 0.95

    grad_norm = None
    
    num_epochs = 7

    for crop_para in [0, 40, 80, 120]:
        cfg.crop_para = crop_para
        train_transforms = A.Compose(
            [
                A.RandomSizedCrop(min_max_height=(cfg.img_size, cfg.img_size + crop_para), height=cfg.img_size, width=cfg.img_size),
                # A.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=15, p=0.5),
                # A.RandomCrop(cfg.img_size, cfg.img_size),
                #A.SmallestMaxSize(cfg.img_size),
                # A.RandomGridShuffle(),
                # A.RandomBrightnessContrast(brightness_limit=0.07, contrast_limit=0.07, p=1.0),
                # A.RGBShift(p=1.0),
                # A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=20, val_shift_limit=(-10, 20), p=1.0),
                # A.GaussNoise(p=1.0),
                #A.ISONoise(p=1.0),
                # Grid !!!
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
                ToTensorV2()
            ]
        )

        val_transforms = A.Compose(
            [
                A.CenterCrop(cfg.img_size, cfg.img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
                ToTensorV2()
            ]
        )

        train_dset = LeafDataset("data/stratified/train_images", "data/stratified/train_images/labels.csv", transform=train_transforms)
        val_dset = LeafDataset("data/stratified/val_images", "data/stratified/val_images/labels.csv", transform=val_transforms)

        train_dataloader = LeafDataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_dataloader = LeafDataLoader(val_dset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)

        model_prefix = f"{cfg.arch}_hero_{datetime.now().strftime('%b%d_%H-%M-%S')}"
        leaf_model = LeafModel(cfg, model_prefix=model_prefix, output_dir=output_dir)

        optimizer = SGD(leaf_model.model.parameters(), lr=min_lr, momentum=0.0, weight_decay=weight_decay)
        div_factor = max_lr / min_lr
        scheduler = OneCycleLR(optimizer, epochs=num_epochs, steps_per_epoch=len(train_dataloader), max_lr=max_lr, div_factor=div_factor)
        leaf_model.update_optimizer_scheduler(optimizer, scheduler)

        neptune.init(project_qualified_name='vmorelli/leaf')
        params_dict = {
            param: eval(param) for param in ["cfg", "train_transforms", "val_transforms", "batch_size", "num_epochs", "min_lr", "max_lr", "weight_decay", "optimizer", "scheduler", "grad_norm"]
        }
        neptune.create_experiment(name=model_prefix, params=params_dict, upload_source_files=['*.py', 'leaf/*.py', 'environment.yml'],
                                  description="Testing cropping augmentations without resizing base")
        str_params_dict = {p: str(pv) for p, pv in params_dict.items()}
        neptune.log_text("params", f"{json.dumps(str_params_dict)}")

        for epoch in range(1, num_epochs+1):
            epoch_name = f"{model_prefix}-{epoch}"
            train_one_epoch(leaf_model, train_dataloader, log_steps=log_steps, epoch_name=epoch_name, epoch=epoch, neptune=neptune, grad_norm=grad_norm)
            val_loss, val_acc = validate_one_epoch(leaf_model, val_dataloader)
            print(f"Validation after epoch {epoch}: loss {val_loss}, acc {val_acc}")
            val_step = len(train_dataloader) * epoch
            neptune.log_metric("loss/val", y=val_loss, x=val_step)
            neptune.log_metric("acc/val", y=val_acc, x=val_step)
        leaf_model.save_checkpoint(f"{epoch_name}", epoch_name=epoch)
