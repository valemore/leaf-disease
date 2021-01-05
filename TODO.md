- Choose bigger patch size (but experiment later with the smaller ones and 2D interpolation)
  Maybe smaller patch sizes could work well with ViT
- Complete testing test_colors
- Add more augmentations (SnapMix, flip, color jitter)
- Training schedule: Once through whole dataset, then random images. Maybe warmup with random images?
- Weight decay?
- Learning rate, learning rate schedule, regularization: Look at top papers/Kagglers  
- Label smoothing or similar, especially relevant for the multi-patch approach
- Bigger batch size? Better to use smaller model with higher batch size on my PC?

Review RandomResizedCrop, quite useful

New model:
Try smaller EfficientNet with the above augmentation/training strategy.

Treat label 1 differently? Always use patch?

Pin memory
Maybe Do Lambda Trick after applying transforms to patches
Verify transforms

Why slow?
Why high memory use?
Compare with RandomCrop

Pre-process on disk?

Different schedules: Linear, Karpathy style, the other Cosine stuff from the forums

Topic of memory/batch size: Consider that in the GetPatches case effective batch size is x13
Compare with vanilla setup from before (speed, memory)
Look at PyTorch's implementation of the Crop transforms


# NEW TODO
- Validation logic (hand-crafted? using ML, NN, trees, SVM or something?)
- Learning rate scheduling
- Training machinery (warmup, train one epoch on GetPatches, then RandomGreen ad nauseam, valdiate on GetPatches using validation logic)
- Folds?
- Balance the classes during training


The run with GetPatches that went reasonably well
Dec30_23-03-08

# INFERENCE LOGIC
