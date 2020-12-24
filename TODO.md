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