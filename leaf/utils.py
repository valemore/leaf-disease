from pathlib import Path
import torch


def convert_checkpoint(checkpoint_fname, out_fname):
    checkpoint_fname = Path(checkpoint_fname) if isinstance(checkpoint_fname, str) else checkpoint_fname
    out_fname = Path(out_fname) if isinstance(out_fname, str) else out_fname
    checkpoint = torch.load(checkpoint_fname)
    torch.save(checkpoint['model_state_dict'], out_fname)


out_dir = Path("/home/v/v/leaf-disease/kaggle/leaf-model-folds/")
out_dir.mkdir(exist_ok=True)
checkpoint_fnames = [f"/mnt/hdd/leaf-disease-outputs/vit_base_patch16_224_lr1e4_fold{fold}/vit_base_patch16_224_lr1e4_fold{fold}_12-epochs-12" for fold in range(5)]
out_fnames = [out_dir / f"fold{fold}" for fold in range(5)]

for checkpoint, out in zip(checkpoint_fnames, out_fnames):
    convert_checkpoint(checkpoint, out)