# Ideas
## Download additional training images
Search for cassava + disease (disease acronym?) on Google Images. \
How to filter out some good ones, good quality and sizes? \
Manually sift through them to discard bogus ones. \
Compatible with Kaggle rules?

## Augmentation
What's the right way to do it?

# Computer Vision Open Questions
* What's the "best" baseline right now?
* How to resize properly for the model? Keep aspect ratio and pad? Or distort aspect ratio?

# Old-school features
Like HOG and so on. Check scikit-image for ready-to-use implementations.

# Various notes
* The average image resolution on ImageNet is 469x387 pixels.
* Need to use image net normalization for timm's models?
* Pooled or unpooled features for feature extractor?