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

# import timm
from efficientnet_pytorch import EfficientNet

from leaf.dta import LeafDataset, LeafIterableDataset, GetPatches, TransformPatches, RandomGreen, LeafDataLoader

# Transforms with normalizations for imagenet
data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        #RandomGreen(64, 64),
        #transforms.Resize((768, 576)),
        GetPatches(800, 600, 224),
        TransformPatches([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        transforms.Lambda(lambda patches: torch.stack(patches))
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        GetPatches(800, 600, 224),
        TransformPatches([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        transforms.Lambda(lambda patches: torch.stack(patches))
    ]),
}

# The bare minimum
# transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])


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


def validate_model(model, global_step, training_start_time, tb_writer, samples_per_image):
    """Runs model on validation set and writes results using tensorboard"""
    model.eval()
    logits_all = torch.zeros((len(val_dataloader), samples_per_image, 5), dtype=float)
    labels_all = torch.zeros((len(val_dataloader), samples_per_image), dtype=int)
    i = 0
    for imgs, labels in tqdm(val_dataloader):
        imgs = imgs.to(device)
        labels = labels.to(device)
        bs, n_patches, _ = imgs.shape
        assert n_patches == samples_per_image
        logits_all[i:(i+bs), :, :] = model.forward(imgs)
        labels_all[i:(i+bs), :] = labels
        i += bs

    val_loss = F.cross_entropy(logits_all.reshape(-1, 5), labels_all.reshape(-1, 5))
    preds_all = logits_all.reshape(-1, 5).cpu().numpy().argmax(axis=-1)
    val_acc = np.sum(labels_all.reshape(-1, 5).cpu().numpy() == preds_all) / i

    print("Time to global step %d took %.1f sec, val loss %.3f, val acc %.3f" % (
        global_step, time.time() - training_start_time, val_loss, val_acc))

    tb_writer.add_scalar("loss/val", val_loss, global_step)
    tb_writer.add_scalar("acc/val", val_acc, global_step)


if __name__ == "__main__":
    save_dir = Path("/mnt/hdd/leaf-disease-outputs")
    save_dir.mkdir(exist_ok=True)
    logging_dir = Path("/mnt/hdd/leaf-disease-runs")
    logging_dir.mkdir(exist_ok=True)

    n_epochs = 1
    batch_size = 4
    learning_rate = 1e-6
    final_layers_lr = 0.0
    weight_decay = 0.0
    final_layers_wd = 0.0
    efficientnet_args ={
        "model_name": 'efficientnet-b4',
        "num_classes": 5
    }

    logging_steps = 6
    save_checkpoints = True

    train_dset = LeafDataset("./data/train_images", "./data/train_images/labels.csv", transform=data_transforms["train"], tiny=True)
    val_dset = LeafDataset("./data/val_images", "./data/val_images/labels.csv", transform=data_transforms["val"], tiny=True)

    train_dataloader = LeafDataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=1, max_samples_per_image=4 * 3 + 1)
    val_dataloader = LeafDataLoader(val_dset, batch_size=batch_size, shuffle=False, num_workers=1, max_samples_per_image=4 * 3 + 1)

    device = torch.device("cpu")

    model = EfficientNet.from_pretrained(**efficientnet_args)

    # Allow for different learning rates/regularization strenghts for final layers
    final_layers = ['_fc.weight',
                    '_fc.bias']

    if final_layers_lr == -1.0:
        final_layers_lr = learning_rate
    if final_layers_wd == -1.0:
        final_layers_wd = weight_decay

    final_layer_params = [(n, p) for n, p in model.named_parameters() if n in final_layers]
    non_final_layer_params = [(n, p) for n, p in model.named_parameters() if n not in final_layers]

    no_decay = ['bias', 'LayerNorm.weight']
    final_layer_decaying_params = [p for n, p in final_layer_params if not any(nd in n for nd in no_decay)]
    final_layer_nondecaying_params = [p for n, p in final_layer_params if any(nd in n for nd in no_decay)]

    non_final_layer_decaying_params = [p for n, p in non_final_layer_params if not any(nd in n for nd in no_decay)]
    non_final_layer_nondecaying_params = [p for n, p in non_final_layer_params if any(nd in n for nd in no_decay)]

    optimizer_grouped_parameters = [
        {'params': final_layer_decaying_params,
         'lr': final_layers_lr,
         'weight_decay': final_layers_wd},
        {'params': final_layer_nondecaying_params,
         'lr': final_layers_lr,
         'weight_decay': 0.0},
        {'params': non_final_layer_decaying_params,
         'lr': learning_rate,
         'weight_decay': weight_decay},
        {'params': non_final_layer_nondecaying_params,
         'lr': learning_rate,
         'weight_decay': 0.0},
    ]

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(optimizer_grouped_parameters, lr=learning_rate, weight_decay=weight_decay)
    lr_policy = "onecycle"
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=len(train_dataloader), epochs=n_epochs)

    # Output and loggind directories
    model_prefix = f"{datetime.now().strftime('%b%d_%H-%M-%S')}"
    output_dir = save_dir / model_prefix
    output_dir.mkdir(exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)


    # Set up logging
    hyperparameters_dict = {
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "l2_weight_decay": weight_decay,
        "efficientnet_args": efficientnet_args
    }
    with open(output_dir / "hyperparamters_dict.json", "w") as f:
        json.dump(hyperparameters_dict, f)

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
                    training_acc = np.sum(running_preds == running_labels) / running_i
                    log_training(running_loss, training_acc, logging_steps, global_step, learning_rate, scheduler, lr_policy,
                                 tb_writer)
                    running_loss, running_i = 0.0, 0
                    running_preds.fill(0)
                    running_labels.fill(0)

                    validate_model(model, global_step, tic, tb_writer, samples_per_image=13)

                    # Save model
                    if save_checkpoints:
                        save_model(model, optimizer, epoch, global_step, running_loss,
                                   output_dir / "checkpoints" / f"{global_step}.pt")
            global_step += 1


        # Validation at end of each epoch
        with torch.no_grad():
            validate_model(model, global_step, tic, tb_writer, samples_per_image=13)
        # Save model
        if save_checkpoints:
            save_model(model, optimizer, epoch, global_step, running_loss, output_dir / "checkpoints" / f"{global_step}.pt")

    tb_writer.close()

    # Save model
    save_model(model, optimizer, epoch, global_step, running_loss, output_dir / f"final.pt")
