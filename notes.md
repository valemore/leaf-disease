# Ideas
## Download additional training images
Search for cassava + disease (disease acronym?) on Google Images. \
How to filter out some good ones, good quality and sizes? \
Manually sift through them to discard bogus ones. \
Compatible with Kaggle rules?

## Augmentation
What's the right way to do it?

### Test-time augmentation

### Sometimes leaf is only in one portion
Could tackle with TTA or Multi-Stage model

# Computer Vision Open Questions
* What's the "best" baseline right now?
* How to resize properly for the model? Keep aspect ratio and pad? Or distort aspect ratio?

# Multi-Stage model?
First detect/localize, then segment region?

# Oversampling?

# Old-school features
Like HOG and so on. Check scikit-image for ready-to-use implementations.
Other old-school features: SIFT, spatial pyramid, bag of visual words

# Various notes
* The average image resolution on ImageNet is 469x387 pixels.
* Need to use image net normalization for timm's models?
* Pooled or unpooled features for feature extractor?

# EfficientNotes
Where is the dropout?

# Pre-Training
Pre-train on similar problem/dataset?

# Cross-Validation, use whole training dataset

# Different learning rates for different parts / freezing-unfreezing strategy?

# Scheme for cropping at train / TTA
Divide each image into 6 200x200 patches, rescale them to ImageNet size, do the other augmentations
At train: Feed all patches (or only gree ones?)
At test time: Only feed in "green" patches, have a vote among all, have a vote among only the disease classes
(can play with thresholds here, e.g. minimum number of diseased patches required)