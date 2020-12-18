import torch


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
