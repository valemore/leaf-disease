import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from leaf.model import LeafModel, softmax

def predict(leaf_model: LeafModel, data_loader):
    ensemble_patches = data_loader.max_samples_per_image > 1
    leaf_model.model.eval()
    with torch.no_grad():
        logits_all = torch.zeros((data_loader.dataset_len, 5), dtype=float, device=leaf_model.device)
        if ensemble_patches:
            idxs_all = np.zeros(data_loader.dataset_len, dtype=int)
        i = 0
        for imgs, _, idxs in tqdm(data_loader):
            imgs = imgs.to(leaf_model.device)
            bs = imgs.shape[0]
            logits_all[i:(i + bs), :] = leaf_model.model.forward(imgs)
            idxs_all[i:(i + bs)] = idxs
            i += bs

        logits_all  = logits_all[:i]
        idxs_all = idxs_all[:i]

        img_ids = data_loader.dataset.img_ids[idxs_all]

        if ensemble_patches:
            logits_df = pd.DataFrame({
                "img_id": img_ids,
                #"patch_idx": patch_ids,
                "logits_0": logits_all.cpu().numpy()[:, 0],
                "logits_1": logits_all.cpu().numpy()[:, 1],
                "logits_2": logits_all.cpu().numpy()[:, 2],
                "logits_3": logits_all.cpu().numpy()[:, 3],
                "logits_4": logits_all.cpu().numpy()[:, 4]
                # "raw_pred": raw_preds_all.cpu().numpy().astype(int)
            })

            probs_df = pd.concat([logits_df.loc[:, ["img_id"]],
                                  pd.DataFrame(np.apply_along_axis(softmax, 1, logits_df.loc[:, ["logits_0", "logits_1", "logits_2", "logits_3", "logits_4"]].values))], axis=1)
            probs_df.rename(columns={i: f"probs_{i}" for i in range(5)}, inplace=True)
            probs_mean_df = probs_df.groupby("img_id").agg(
                {"probs_0": "mean", "probs_1": "mean", "probs_2": "mean", "probs_3": "mean", "probs_4": "mean"}).reset_index()

            img_ids = probs_mean_df.loc[:, ["img_id"]].values
            probs_mean = probs_mean_df.loc[:, ["probs_0", "probs_1", "probs_2", "probs_3", "probs_4"]].values
            probs_mean_preds = probs_mean.argmax(-1)

            return img_ids, probs_mean, probs_mean_preds

        probs_all = logits_all.softmax(axis=-1)
        preds_all = probs_all.argmax(axis=-1)

        return img_ids, probs_all.cpu().numpy(), preds_all.cpu().numpy()


def ensemble_models(model_fns, data_fns, inference_fns, weights=None):
    # TODO: implement other methods than soft-voting
    n_models = max(len(model_fns), len(data_fns), len(inference_fns))
    if len(model_fns) == 1:
        model_fns = [model_fns[0]] * n_models
    if len(data_fns) == 1:
        data_fns = [data_fns[0]] * n_models
    if len(inference_fns) == 1:
        inference_fns = [inference_fns[0]] * n_models
    if weights is None:
        weights = np.ones(len(model_fns))
    assert len(model_fns) == len(data_fns) == len(data_fns) == len(weights), "Lenghts of model, data, inference functions, and weights don't match!"

    img_ids = []
    probs = []

    for model_fn, data_fn, inference_fn, weight in zip(model_fns, data_fns, inference_fns, weights):
        model = model_fn()
        dataloader = data_fn()
        model_img_ids, model_probs, model_preds = inference_fn(model, dataloader)
        img_ids += [model_img_ids]
        probs += [weight * model_probs]

    img_ids = np.concatenate(img_ids, axis=0)
    probs = np.concatenate(probs, axis=0)

    ensemble_df = pd.concat([pd.DataFrame(img_ids).rename(columns={0: "img_id"}),
                             pd.DataFrame(probs).rename(columns={0: "probs_0", 1: "probs_1", 2: "probs_2", 3: "probs_3", 4: "probs_4"})], axis=1)

    ensemble_df = ensemble_df.groupby("img_id").agg(
        {"probs_0": "mean", "probs_1": "mean", "probs_2": "mean", "probs_3": "mean", "probs_4": "mean"}).reset_index()


    img_ids = ensemble_df.loc[:, ["img_id"]].values
    probs = ensemble_df.loc[:, ["probs_0", "probs_1", "probs_2", "probs_3", "probs_4"]].values
    preds = np.argmax(probs, axis=1)

    return img_ids, probs, preds
