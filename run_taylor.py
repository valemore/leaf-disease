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

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import neptune

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


if __name__ == "__main__":
    on_gcp = os.getcwd() == "/home/jupyter/leaf-disease"
    print(f"Running on {'GCP' if on_gcp else 'local'} machine!")

    img_size = 380

    train_transforms = A.Compose(
        [
            A.SmallestMaxSize(500),
            A.RandomSizedCrop(min_max_height=(300, 460), height=img_size, width=img_size),
            A.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=15, p=0.5),
            # A.RandomCrop(img_size, img_size),
            #A.SmallestMaxSize(img_size),
            #A.RandomGridShuffle(),
            A.RandomBrightnessContrast(brightness_limit=0.07, contrast_limit=0.07, p=1.0),
            # A.RGBShift(p=1.0),
            A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=20, val_shift_limit=(-10, 20), p=1.0),
            A.GaussNoise(p=1.0),
            #A.ISONoise(p=1.0),
            # Grid !!!
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
            ToTensorV2()
        ]
    )

    val_transforms = A.Compose(
        [
            A.SmallestMaxSize(500),
            A.CenterCrop(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
            ToTensorV2()
        ]
    )

    output_dir = Path("/home/jupyter/outputs") if on_gcp else Path("/mnt/hdd/leaf-disease-outputs")
    output_dir.mkdir(exist_ok=True)

    num_workers = 4

    batch_size = 18 if on_gcp else 12
    val_batch_size = 36 if on_gcp else 24

    log_steps = 200

    min_lr = 0.00001
    #max_lr = 0.015
    max_lr = 0.002
    weight_decay = 0.0
    momentum = 0.97

    grad_norm = None
    
    cosine_epochs = 15
    warmup_epochs = 1
    num_epochs = warmup_epochs + cosine_epochs

    class CFG:
        num_classes = 5
        arch = "tf_efficientnet_b4_ns"
        loss_fn = "TaylorCrossEntropyLoss"
        smoothing = 0.05
        def __repr__(self):
            return json.dumps(self.__dict__)

    train_dset = LeafDataset("data/stratified/train_images", "data/stratified/train_images/labels.csv", transform=train_transforms)
    val_dset = LeafDataset("data/stratified/val_images", "data/stratified/val_images/labels.csv", transform=val_transforms)

    train_dataloader = LeafDataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = LeafDataLoader(val_dset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)

    model_prefix = f"{CFG.arch}_hero_{datetime.now().strftime('%b%d_%H-%M-%S')}"
    leaf_model = LeafModel(CFG, model_prefix=model_prefix, output_dir=output_dir)
    optimizer = SGD(leaf_model.model.parameters(), lr=min_lr, momentum=momentum, weight_decay=weight_decay)

    # LR Schedule
    #scheduler_cosine = CosineAnnealingWarmRestarts(optimizer, T_0=len(train_dataloader) * cosine_epochs, eta_min=min_lr)
    scheduler = get_warmup_scheduler(optimizer, 0.0, max_lr, len(train_dataloader))
    leaf_model.update_optimizer_scheduler(optimizer, scheduler)

    neptune.init(project_qualified_name='vmorelli/leaf')
    params_dict = {
        param: eval(param) for param in ["CFG", "img_size", "train_transforms", "val_transforms", "batch_size", "num_epochs", "warmup_epochs", "cosine_epochs", "min_lr", "max_lr", "weight_decay", "optimizer", "scheduler", "grad_norm"]
    }
    neptune.create_experiment(name=model_prefix, params=params_dict, upload_source_files=['*.py', 'leaf/*.py', 'environment.yml'],
                              description="Runs with warmup + Cosine Annealing learning rate policy.")
    str_params_dict = {p: str(pv) for p, pv in params_dict.items()}
    neptune.log_text("params", f"{json.dumps(str_params_dict)}")

    reset_epoch_diff = 1
    reset_epoch = 2
    lr_decay_epochs = [3, 5, 9]
    for epoch in range(1, num_epochs+1):
        if epoch == 2:
            for pg in leaf_model.optimizer.param_groups:
                pg["lr"] = max_lr
                pg["initial_lr"] = max_lr
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=len(train_dataloader), T_mult=2, eta_min=min_lr)
            leaf_model.update_optimizer_scheduler(optimizer, scheduler)
        elif epoch in lr_decay_epochs:
            for pg in leaf_model.optimizer.param_groups:
                pg["lr"] /= 2
                pg["initial_lr"] /= 2
            leaf_model.scheduler.base_lrs = [base_lr / 2 for base_lr in leaf_model.scheduler.base_lrs]
            leaf_model.scheduler.eta_min /= 10
        epoch_name = f"{model_prefix}-{epoch}"
        train_one_epoch(leaf_model, train_dataloader, log_steps=log_steps, epoch_name=epoch_name, epoch=epoch, neptune=neptune, grad_norm=grad_norm)
        val_loss, val_acc = validate_one_epoch(leaf_model, val_dataloader)
        print(f"Validation after epoch {epoch}: loss {val_loss}, acc {val_acc}")
        val_step = len(train_dataloader) * epoch
        neptune.log_metric("loss/val", y=val_loss, x=val_step)
        neptune.log_metric("acc/val", y=val_acc, x=val_step)
        leaf_model.save_checkpoint(f"{epoch_name}", epoch=epoch)