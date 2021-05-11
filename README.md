# Cassava Leaf Disease Classification

My solution for the following Kaggle Competition:

Cassava Leaf Disease Classification
https://www.kaggle.com/c/cassava-leaf-disease-classification

## Get the data
`kaggle competitions download -c cassava-leaf-disease-classification`

I also used the images from the 2019 competition as additional training data:

`kaggle competitions download -c cassava-disease`

## Summary of approach
After extensive cross-validation settled on:
- Efficientnet B4
- Larger image size (446x446)
- Standard augmentations + cutmix (cutmix made a big difference)
- Self-distillation on out-of-fold predicted soft targets
- Cosine annealing learning rate schedule, model converges after 10 epochs

# Training
Two-step training `gcp_446.py` and `gcp_distillation_final10.py` for the final distillated model.

# Inference
In notebook `leaf_inference.ipynb`, usign light test-time augmentation.
