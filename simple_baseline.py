from pathlib import Path
import time
from datetime import datetime
from tqdm import tqdm
#from docopt import docopt

from skimage import io
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

import timm

# Transforms with normalizations for imagenet
data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize([0.0, 0.0, 0.0], [255.0, 255.0, 255.0]),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize([0.0, 0.0, 0.0], [255.0, 255.0, 255.0]),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}


class LeafDataset(Dataset):
    """Cassava Leaf Disease Classification dataset."""

    def __init__(self, img_dir, labels_csv=None, transform=None):
        self.img_dir = img_dir if isinstance(img_dir, Path) else Path(img_dir)
        self.transform = transform
        if labels_csv:
            df = pd.read_csv(labels_csv)
            self.fnames = df["image_id"].values
            self.labels = df["label"].values
        else:
            self.fnames = np.array([img.name for img in img_dir.glob("*.jpg")])
            self.labels = None
        self.dataset_len = len(self.fnames)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = io.imread(self.img_dir / self.fnames[idx])
        if self.transform:
            img = self.transform(img)
        if self.labels is not None:
            label = self.labels[idx]
        else:
            label = None
        return img, label


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
    if lr_policy == "onecycle":
        tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
    else:
        raise


def validate_model(model, global_step, training_start_time, tb_writer):
    """Runs model on validation set and writes results using tensorboard"""
    model.eval()
    logits_all = torch.zeros((len(val_dset), 5), dtype=float)
    labels_all = torch.zeros(len(val_dset), dtype=int)
    i = 0
    for imgs, labels in tqdm(val_dataloader):
        bs = len(imgs)
        imgs = imgs.to(device)
        labels = labels.to(device)
        logits_all[i:(i+bs), :] = model.forward(imgs)
        labels_all[i:(i+bs)] = labels
        i += bs

    val_loss = F.cross_entropy(logits_all, labels_all)
    preds_all = logits_all.cpu().numpy().argmax(axis=-1)
    val_acc = np.sum(labels_all.cpu().numpy() == preds_all) / i

    print("Time to global step %d took %.1f sec, val loss %.3f, val acc %.3f" % (
        global_step, time.time() - training_start_time, val_loss, val_acc))

    tb_writer.add_scalar("loss/val", val_loss, global_step)
    tb_writer.add_scalar("acc/val", val_acc, global_step)


output_dir = Path("./output/")
logging_dir = Path("./runs")

n_epochs = 10
batch_size = 4
learning_rate = 3e-4

logging_steps = 500
save_checkpoints = True

train_dset = LeafDataset("./data/train_images", "./data/train_images/labels.csv", transform=data_transforms["train"])
val_dset = LeafDataset("./data/val_images", "./data/val_images/labels.csv", transform=data_transforms["val"])

train_dataloader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dset, batch_size=batch_size, shuffle=False, num_workers=4)

device = torch.device("cuda")

model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=5)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)
lr_policy = "onecycle"
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=len(train_dset), epochs=n_epochs)


# Output and loggind directories
output_dir.mkdir(exist_ok=True)
(output_dir / "checkpoints").mkdir(exist_ok=True)
logging_dir.mkdir(exist_ok=True)

# Set up logging
tb_writer_dir = f"{datetime.now().strftime('%b%d_%H-%M-%S')}"
tb_writer = SummaryWriter(log_dir=logging_dir / tb_writer_dir)

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
                training_acc = np.sum(running_preds == running_labels) / running_i
                log_training(running_loss, training_acc, logging_steps, global_step, learning_rate, scheduler, lr_policy,
                             tb_writer)
                running_loss, running_i = 0.0, 0
                running_preds.fill(0)
                running_labels.fill(0)

                validate_model(model, global_step, tic, tb_writer)

                # Save model
                if save_checkpoints:
                    save_model(model, optimizer, epoch, global_step, running_loss,
                               output_dir / "checkpoints" / f"{global_step}.pt")
        global_step += 1


    # Validation at end of each epoch
    with torch.no_grad():
        validate_model(model, global_step, tic, tb_writer)
    # Save model
    if save_checkpoints:
        save_model(model, optimizer, epoch, global_step, running_loss, output_dir / "checkpoints" / f"{global_step}.pt")

tb_writer.close()

# Save model
save_model(model, optimizer, epoch, global_step, running_loss, output_dir / f"final.pt")
