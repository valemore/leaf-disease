from datetime import datetime
import json
from pathlib import Path
import os
import random

import numpy as np

from torch.optim import SGD

from leaf.dta import LeafDataset, LeafIterableDataset, LeafDataLoader, LeafDataLoader, get_leaf_splits
from leaf.model import LeafModel, train_one_epoch, validate_one_epoch

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CyclicLR

import neptune

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

if __name__ == "__main__":
    on_gcp = os.getcwd() == "/home/jupyter/leaf-disease"
    print(f"Running lr finder on {'GCP' if on_gcp else 'local'} machine!")

    img_size = 380

    train_transforms = A.Compose(
        [
            A.SmallestMaxSize(500),
            A.RandomSizedCrop(min_max_height=(300, 460), height=img_size, width=img_size),
            # A.RandomCrop(img_size, img_size),
            #A.SmallestMaxSize(img_size),
            A.RandomBrightnessContrast(brightness_limit=0.07, contrast_limit=0.07, p=1.0),
            A.RGBShift(p=1.0),
            A.GaussNoise(p=1.0),
            A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=20, val_shift_limit=(-10, 20), p=1.0),
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

    max_steps = 500
    log_steps = 50
    num_runs = 30

    min_lr = 1e-6
    max_lr = 1.0
    weight_decay_list = [0.0]
    momentum_list = [0.9, 0.95, 0.97, 0.99]

    train_dset = LeafDataset("data/stratified/train_images", "data/stratified/train_images/labels.csv", transform=train_transforms)
    val_dset = LeafDataset("data/stratified/val_images", "data/stratified/val_images/labels.csv", transform=val_transforms)

    train_dataloader = LeafDataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = LeafDataLoader(val_dset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)

    arch = "tf_efficientnet_b4_ns"
    model_prefix = f"{arch}_lr-finder_{datetime.now().strftime('%b%d_%H-%M-%S')}"

    #torch.save(leaf_model.model.state_dict(), "inital_state_dict.pth")

    neptune.init(project_qualified_name='vmorelli/leaf')
    params_dict = {
        param: eval(param) for param in ["arch", "img_size", "train_transforms", "val_transforms", "batch_size", "max_steps", "min_lr", "max_lr", "weight_decay_list", "momentum_list", "num_runs"]
    }
    neptune.create_experiment(name=model_prefix, params=params_dict, upload_source_files=['*.py', 'leaf/*.py', 'environment.yml'])

    for num_run in range(num_runs):
        lr_exp_range = (np.log2(min_lr), np.log2(max_lr))
        lr_exp = random.uniform(*lr_exp_range)
        lr = np.exp2(lr_exp)
        momentum = random.choice(momentum_list)
        weight_decay = random.choice(weight_decay_list)
        run_params = {param: eval(param) for param in ["lr", "momentum", "weight_decay"]}
        leaf_model = LeafModel(arch, model_prefix=model_prefix, output_dir=output_dir)
        optimizer = SGD(leaf_model.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        leaf_model.update_optimizer_scheduler(optimizer, None)
        train_losses, train_accs = train_one_epoch(leaf_model, train_dataloader, max_steps=max_steps, log_steps=log_steps, epoch_name=f"lr-finder_{json.dumps(run_params)}")
        neptune.log_text("loss-history", y=f"{json.dumps(train_losses)}", x=num_run)
        neptune.log_text("acc-history", y=f"{json.dumps(train_accs)}", x=num_run)
        neptune.log_metric("loss/train", y=train_losses[-1], x=num_run)
        neptune.log_metric("acc/train", y=train_accs[-1], x=num_run)
        val_loss, val_acc = validate_one_epoch(leaf_model, val_dataloader)
        neptune.log_metric("loss/val", y=val_loss, x=num_run)
        neptune.log_metric("acc/val", y=val_acc, x=num_run)
        print(f"Validation loss {val_loss}, acc {val_acc}")
        neptune.log_text("params", y=f"{json.dumps(run_params)}", x=num_run)

