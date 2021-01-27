# HIGH PRIO
## Validation strategy
Is 5 fold really useful? Use my split, and then train on whole at the end.

# Experiment learning rate, schedule, (and possibly weight decay for mid-sized model)
Maybe b4?

# Localize relevant portions?
Important for image size considerations?
Look at dataset!

## Exampple parameters
EfficientNet Paper: RMSProp, 1e-5 weight decay and 0.264 learning rate that decays by ... every 2.4 epochs

Note the use of RMSProp instead of SGD. 

# Augmentation with mid-sized model
The regular ones
SnapMix, or something similar.
Mosaic strategy in wheat challenge?

Custom ideas or approaches, mixed with the above

# Pseudo Labeling / Something else or similar?

# Gradient clipping

# Better use of Tensorboard
Better logging in Tensorboard (log hyperparameters)


# LOW PRIO
## Use some of my custom augmentation?
If used: Read hsv from disk? Check first with profiler.
Experiment with test_colors hyperparameters

## Use additional training images from last competition

## Use additional training images that were downloaded
Manually sift through them.
Compatible with Kaggle rules?

## Old-school features
Like HOG and so on. Check scikit-image for ready-to-use implementations.
Other old-school features: SIFT, spatial pyramid, bag of visual words

## Pre-Training
Pre-train on similar problem/dataset?

# MID PRIO
# Balanced sampling (check out the efficient net baseline notebook on Kaggle)


# VARIOUS NOTES
* The average image resolution on ImageNet is 469x387 pixels.
* Need to use image net normalization for timm's models? Yes!
* Pooled or unpooled features for feature extractor?

## EfficientNet notes
Where is the dropout?

## Different learning rates for different parts / freezing-unfreezing strategy?
I have not seen this in recent literature.

## Loss going up at start of a new epoch 
Why does loss go up at start of new epoch? Too high learning rate, we forget a bit?

