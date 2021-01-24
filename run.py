from datetime import datetime
from pathlib import Path
import os

from torch.optim import SGD

from leaf.dta import LeafDataset, LeafIterableDataset, LeafDataLoader, LeafDataLoader, get_leaf_splits
from leaf.model import LeafModel, train_one_epoch, validate_one_epoch

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CyclicLR

import neptune

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

if __name__ == "__main__":
    on_gcp = os.getcwd() == "/home/jupyter/leaf-disease"
    print(f"Training on {'GCP' if on_gcp else 'local'} machine!")

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
    logging_dir = Path("/home/jupyter/runs") if on_gcp else Path("/mnt/hdd/leaf-disease-runs")
    logging_dir.mkdir(exist_ok=True)

    num_workers = 4

    batch_size = 18 if on_gcp else 12
    val_batch_size = 36 if on_gcp else 24

    log_steps = 200 if on_gcp else 200

    num_epochs = 7
    min_lr = 1e-4 if on_gcp else 1e-4
    max_lr = 1e-2 if on_gcp else 5e-3
    weight_decay = 5e-5
    momentum = 0.9

    train_dset = LeafDataset("data/stratified/train_images", "data/stratified/train_images/labels.csv", transform=train_transforms)
    val_dset = LeafDataset("data/stratified/val_images", "data/stratified/val_images/labels.csv", transform=val_transforms)

    train_dataloader = LeafDataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = LeafDataLoader(val_dset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)

    arch = "tf_efficientnet_b4_ns"
    model_prefix = f"{arch}_{datetime.now().strftime('%b%d_%H-%M-%S')}"
    leaf_model = LeafModel(arch, model_prefix=model_prefix, output_dir=output_dir, logging_dir=logging_dir)
    optimizer = SGD(leaf_model.model.parameters(), lr=max_lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=len(train_dataloader)+1, T_mult=2, eta_min=min_lr)
    leaf_model.update_optimizer_scheduler(optimizer, scheduler)

    #torch.save(leaf_model.model.state_dict(), "inital_state_dict.pth")

    neptune.init(project_qualified_name='vmorelli/leaf')
    params_dict = {
        param: eval(param) for param in ["arch", "img_size", "train_transforms", "val_transforms", "batch_size", "num_epochs", "min_lr", "max_lr", "weight_decay", "momentum", "optimizer", "scheduler"]
    }
    neptune.create_experiment(name=model_prefix, params=params_dict, upload_source_files=['*.py', 'leaf/*.py', 'environment.yml'])
    for epoch in range(1, num_epochs+1):
        epoch_name = f"{model_prefix}-{epoch}"
        train_one_epoch(leaf_model, train_dataloader, log_steps=log_steps, epoch_name=epoch_name, epoch=epoch, neptune=neptune)
        val_loss, val_acc = validate_one_epoch(leaf_model, val_dataloader)
        print(f"Validation after epoch {epoch}: loss {val_loss}, acc {val_acc}")
        val_step = len(train_dataloader) * epoch
        neptune.log_metric("loss/val", y=val_loss, x=val_step)
        neptune.log_metric("acc/val", y=val_acc, x=val_step)
        leaf_model.save_checkpoint(f"{epoch_name}", epoch=epoch)
